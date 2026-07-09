import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))

from src.method.vllm_teacher import VLLMTeacherScorer  # noqa: E402


def expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            paths.append(pattern)
    return paths


def load_pt(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def normalize_mm_data(multi_modal_data: dict[str, Any] | None) -> dict[str, Any]:
    if multi_modal_data is None:
        return {}
    prompt_mm_data: dict[str, Any] = {}
    if "image" in multi_modal_data:
        prompt_mm_data["image"] = multi_modal_data["image"]
    elif "images" in multi_modal_data:
        prompt_mm_data["image"] = multi_modal_data["images"]
    if "video" in multi_modal_data:
        prompt_mm_data["video"] = multi_modal_data["video"]
    elif "videos" in multi_modal_data:
        prompt_mm_data["video"] = multi_modal_data["videos"]
    if "audio" in multi_modal_data:
        prompt_mm_data["audio"] = multi_modal_data["audio"]
    elif "audios" in multi_modal_data:
        prompt_mm_data["audio"] = multi_modal_data["audios"]
    return prompt_mm_data


def image_sig(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "mode") and hasattr(value, "size") and hasattr(value, "tobytes"):
        return {
            "type": "PIL",
            "mode": value.mode,
            "size": list(value.size),
            "md5": hashlib.md5(value.tobytes()).hexdigest(),
        }
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        return {
            "type": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "md5": hashlib.md5(tensor.numpy().tobytes()).hexdigest(),
        }
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def mm_sig(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: mm_sig(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [mm_sig(item) for item in value]
    return image_sig(value)


def prompt_kwargs_from_final(final_payload: dict[str, Any]) -> dict[str, Any]:
    prompt_kwargs = dict(final_payload.get("prompt_kwargs") or {})
    if not prompt_kwargs:
        prompt_kwargs["prompt_token_ids"] = final_payload["prompt_token_ids"]
        if "multi_modal_data" in final_payload:
            prompt_kwargs["multi_modal_data"] = final_payload["multi_modal_data"]
        if "mm_processor_kwargs" in final_payload:
            prompt_kwargs["mm_processor_kwargs"] = final_payload["mm_processor_kwargs"]
    return prompt_kwargs


def build_prompt_kwargs_from_teacher_request(
    request: dict[str, Any],
    *,
    image_token_id: int,
    video_token_id: int,
    dedup_mm_tokens: bool,
) -> tuple[dict[str, Any], list[int]]:
    token_ids = [int(token_id) for token_id in request["sequence_ids"]]
    if dedup_mm_tokens:
        prompt_token_ids, kept_indices = VLLMTeacherScorer._dedup_consecutive_mm_tokens(
            token_ids,
            image_token_id,
            video_token_id,
        )
    else:
        prompt_token_ids = token_ids
        kept_indices = list(range(len(token_ids)))

    prompt_kwargs: dict[str, Any] = {"prompt_token_ids": prompt_token_ids}
    prompt_kwargs["multi_modal_data"] = normalize_mm_data(request.get("multi_modal_data"))
    mm_processor_kwargs = request.get("mm_processor_kwargs")
    if mm_processor_kwargs:
        prompt_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
    return prompt_kwargs, kept_indices


def request_id(payload: dict[str, Any]) -> str | None:
    value = payload.get("request_id")
    return str(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CLight-built final vLLM prompt kwargs from VERL teacher-manager "
            "request dumps against VERL vLLM final prompt dumps."
        )
    )
    parser.add_argument("--teacher-requests", nargs="+", required=True)
    parser.add_argument("--final-prompts", nargs="+", required=True)
    parser.add_argument("--image-token-id", type=int, default=151655)
    parser.add_argument("--video-token-id", type=int, default=151656)
    parser.add_argument("--dedup-mm-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metrics-output", default=None)
    args = parser.parse_args()

    teacher_paths = expand_paths(args.teacher_requests)
    final_paths = expand_paths(args.final_prompts)
    teacher_payloads = [(path, load_pt(path)) for path in teacher_paths]
    final_payloads = [(path, load_pt(path)) for path in final_paths]

    final_by_upstream: dict[str, tuple[str, dict[str, Any]]] = {}
    for path, payload in final_payloads:
        upstream_id = payload.get("upstream_request_id")
        if upstream_id is not None:
            final_by_upstream[str(upstream_id)] = (path, payload)

    metrics_file = open(args.metrics_output, "w", encoding="utf-8") if args.metrics_output else None
    try:
        print("=== compare teacher request -> final vLLM prompt ===")
        print(f"teacher_request_count={len(teacher_payloads)}")
        print(f"final_prompt_count={len(final_payloads)}")
        print(f"dedup_mm_tokens={args.dedup_mm_tokens}")

        matched = 0
        bad = 0
        missing = 0
        for index, (teacher_path, teacher_payload) in enumerate(teacher_payloads):
            rid = request_id(teacher_payload)
            if rid is None or rid not in final_by_upstream:
                missing += 1
                bad += 1
                print(f"MISSING_FINAL index={index} teacher={Path(teacher_path).name} request_id={rid}")
                continue

            final_path, final_payload = final_by_upstream[rid]
            built_kwargs, kept_indices = build_prompt_kwargs_from_teacher_request(
                teacher_payload,
                image_token_id=args.image_token_id,
                video_token_id=args.video_token_id,
                dedup_mm_tokens=args.dedup_mm_tokens,
            )
            final_kwargs = prompt_kwargs_from_final(final_payload)

            built_ids = [int(token_id) for token_id in built_kwargs.get("prompt_token_ids", [])]
            final_ids = [int(token_id) for token_id in final_kwargs.get("prompt_token_ids", [])]
            ids_same = built_ids == final_ids
            mm_same = mm_sig(built_kwargs.get("multi_modal_data")) == mm_sig(final_kwargs.get("multi_modal_data"))
            mmkw_same = repr(built_kwargs.get("mm_processor_kwargs")) == repr(final_kwargs.get("mm_processor_kwargs"))

            ok = ids_same and mm_same and mmkw_same
            matched += int(ok)
            bad += int(not ok)

            record = {
                "index": index,
                "ok": ok,
                "teacher_request": teacher_path,
                "final_prompt": final_path,
                "request_id": rid,
                "built_prompt_len": len(built_ids),
                "final_prompt_len": len(final_ids),
                "original_prompt_len": len(teacher_payload["sequence_ids"]),
                "kept_len": len(kept_indices),
                "ids_same": ids_same,
                "multi_modal_data_same": mm_same,
                "mm_processor_kwargs_same": mmkw_same,
            }
            if metrics_file is not None:
                metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            status = "OK" if ok else "BAD"
            print(
                " | ".join(
                    [
                        status,
                        f"index={index}",
                        f"teacher={Path(teacher_path).name}",
                        f"final={Path(final_path).name}",
                        f"orig_len={len(teacher_payload['sequence_ids'])}",
                        f"built_len={len(built_ids)}",
                        f"final_len={len(final_ids)}",
                        f"ids_same={ids_same}",
                        f"mm_same={mm_same}",
                        f"mmkw_same={mmkw_same}",
                    ]
                )
            )

        print(f"matched={matched}")
        print(f"missing={missing}")
        print(f"bad={bad}")
        print(f"RESULT={'OK' if bad == 0 and missing == 0 else 'FAIL'}")
    finally:
        if metrics_file is not None:
            metrics_file.close()


if __name__ == "__main__":
    main()
