import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from replay_verl_opd_trace import load_trace  # noqa: E402


def expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)
        else:
            raise FileNotFoundError(f"No trace files matched: {pattern}")
    return sorted(dict.fromkeys(paths))


def item_at(value: Any, index: int) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
        return value[index]
    try:
        return value[index]
    except Exception:
        return None


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist") and not torch.is_tensor(value):
        maybe_list = value.tolist()
        return maybe_list if isinstance(maybe_list, list) else [maybe_list]
    return [value]


def active_sequence(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[int]:
    active = attention_mask.bool()
    return [int(token_id) for token_id in input_ids[active].tolist()]


def normalize_mm_processor_kwargs(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        return dict(value.items())
    return None


def normalize_teacher_mm_data(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for plural, singular in (("images", "image"), ("videos", "video"), ("audios", "audio")):
            if plural in value and value[plural] is not None:
                result[plural] = as_list(value[plural])
            elif singular in value and value[singular] is not None:
                result[plural] = as_list(value[singular])
    return result


def mm_data_summary(mm_data: dict[str, Any]) -> dict[str, int]:
    return {
        "image_count": len(mm_data.get("images") or []),
        "video_count": len(mm_data.get("videos") or []),
        "audio_count": len(mm_data.get("audios") or []),
    }


def stable_request_id(trace_path: str, dump_index: Any, global_steps: Any, row: int) -> str:
    raw = f"{Path(trace_path).name}|{dump_index}|{global_steps}|{row}".encode("utf-8")
    return hashlib.md5(raw, usedforsecurity=False).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct VERL teacher-manager vLLM requests from OPD trace dumps. "
            "This recreates the inputs passed to compute_teacher_logprobs_single: "
            "sequence_ids, multi_modal_data, mm_processor_kwargs, and sampling_params."
        )
    )
    parser.add_argument("traces", nargs="+", help="Trace dump file(s) or glob pattern(s).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--allow-vllm-images-fallback",
        action="store_true",
        help=(
            "Debug fallback only. The VERL teacher path uses non_tensor_batch['multi_modal_data']; "
            "vllm_images is derived from that field and is not used by default."
        ),
    )
    args = parser.parse_args()

    trace_paths = expand_paths(args.traces)
    if args.max_files > 0:
        trace_paths = trace_paths[: args.max_files]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.jsonl"

    sampling_params = {
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "prompt_logprobs": int(args.topk),
    }

    total = 0
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for trace_index, trace_path in enumerate(trace_paths):
            payload = load_trace(trace_path)
            batch = payload["batch"]
            non_tensor = payload.get("non_tensor_batch", {})
            input_ids = batch["input_ids"].cpu()
            attention_mask = batch["attention_mask"].cpu()
            sample_count = int(input_ids.shape[0])
            if args.max_rows_per_file > 0:
                sample_count = min(sample_count, args.max_rows_per_file)

            for row in range(sample_count):
                sequence_ids = active_sequence(input_ids[row], attention_mask[row])
                mm_data_raw = item_at(non_tensor.get("multi_modal_data"), row)
                mm_data = normalize_teacher_mm_data(mm_data_raw)
                used_vllm_images_fallback = False
                if args.allow_vllm_images_fallback and "images" not in mm_data:
                    vllm_images_raw = item_at(non_tensor.get("vllm_images"), row)
                    images = as_list(vllm_images_raw)
                    if images:
                        mm_data["images"] = images
                        used_vllm_images_fallback = True
                mm_processor_kwargs = normalize_mm_processor_kwargs(
                    item_at(non_tensor.get("mm_processor_kwargs"), row)
                )
                request_id = stable_request_id(
                    trace_path,
                    payload.get("dump_index"),
                    payload.get("global_steps"),
                    row,
                )

                summary = mm_data_summary(mm_data)
                request = {
                    "format": "reconstructed_verl_teacher_vllm_request_v1",
                    "source_format": payload.get("format"),
                    "trace_path": str(trace_path),
                    "trace_dump_index": payload.get("dump_index"),
                    "global_steps": payload.get("global_steps"),
                    "chunk_index": payload.get("chunk_index"),
                    "chunk_count": payload.get("chunk_count"),
                    "trace_row": row,
                    "request_index": total,
                    "request_id": request_id,
                    "teacher_key": "teacher_model",
                    "sequence_ids": sequence_ids,
                    "sequence_len": len(sequence_ids),
                    "sampling_params": sampling_params,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": mm_processor_kwargs,
                    "used_vllm_images_fallback": used_vllm_images_fallback,
                    **summary,
                }
                filename = (
                    f"teacher_request_trace{trace_index:04d}"
                    f"_dump{int(payload.get('dump_index', trace_index)):06d}"
                    f"_step{int(payload.get('global_steps', 0)):06d}"
                    f"_row{row:03d}.pt"
                )
                out_path = output_dir / filename
                torch.save(request, out_path)

                manifest_record = {
                    "path": str(out_path),
                    "trace_path": str(trace_path),
                    "trace_dump_index": payload.get("dump_index"),
                    "global_steps": payload.get("global_steps"),
                    "chunk_index": payload.get("chunk_index"),
                    "trace_row": row,
                    "request_index": total,
                    "request_id": request_id,
                    "sequence_len": len(sequence_ids),
                    "has_mm_processor_kwargs": mm_processor_kwargs is not None,
                    "used_vllm_images_fallback": used_vllm_images_fallback,
                    **summary,
                }
                manifest_file.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
                total += 1

            print(
                f"trace={trace_index} path={trace_path} rows={sample_count} "
                f"total_requests={total}",
                flush=True,
            )

    print(f"reconstructed_teacher_requests={total}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
