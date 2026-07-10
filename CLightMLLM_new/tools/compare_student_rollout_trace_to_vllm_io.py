import argparse
import glob
import json
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
            raise FileNotFoundError(f"No files matched: {pattern}")
    return sorted(dict.fromkeys(paths))


def item_at(value: Any, index: int) -> Any:
    if value is None:
        return None
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
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        out = value.tolist()
        return out if isinstance(out, list) else [out]
    return [value]


def first_active_index(mask: torch.Tensor) -> int:
    active = torch.nonzero(mask.bool(), as_tuple=False).flatten()
    if active.numel() == 0:
        return int(mask.numel())
    return int(active[0].item())


def dedup_consecutive_mm_tokens(
    token_ids: list[int],
    *,
    image_token_id: int | None,
    video_token_id: int | None,
) -> list[int]:
    mm_ids = {int(x) for x in (image_token_id, video_token_id) if x is not None}
    if not mm_ids:
        return list(token_ids)

    out: list[int] = []
    previous_was_mm = False
    for token_id in token_ids:
        token_id = int(token_id)
        current_is_mm = token_id in mm_ids
        if current_is_mm and previous_was_mm:
            continue
        out.append(token_id)
        previous_was_mm = current_is_mm
    return out


def response_ids_from_trace(batch: dict[str, torch.Tensor], row: int) -> list[int]:
    responses = batch["responses"][row].detach().cpu()
    response_mask = batch["response_mask"][row].detach().cpu().bool()
    return [int(x) for x in responses[response_mask].tolist()]


def prompt_ids_from_trace(
    batch: dict[str, torch.Tensor],
    row: int,
    *,
    pad_token_id: int | None,
) -> list[int]:
    prompts = batch["prompts"][row].detach().cpu()
    attention_mask = batch.get("attention_mask")

    if attention_mask is not None:
        prompt_width = int(prompts.numel())
        prompt_mask = attention_mask[row, :prompt_width].detach().cpu().bool()
        if prompt_mask.numel() == prompt_width and int(prompt_mask.sum().item()) > 0:
            return [int(x) for x in prompts[prompt_mask].tolist()]

    values = [int(x) for x in prompts.tolist()]
    if pad_token_id is None:
        return values
    start = 0
    while start < len(values) and values[start] == int(pad_token_id):
        start += 1
    return values[start:]


def get_nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def load_student_io(paths: list[str]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        generated = get_nested(payload, "output", "generated_token_ids")
        if generated is None:
            generated = payload.get("generated_token_ids")
        prompt_ids = get_nested(payload, "input", "prompt_token_ids")
        original_prompt_ids = get_nested(payload, "input", "original_prompt_token_ids")
        sampling_params = get_nested(payload, "input", "sampling_params") or payload.get("sampling_params") or {}
        out.append(
            {
                "path": path,
                "payload": payload,
                "generated": [int(x) for x in as_list(generated)],
                "prompt_ids": [int(x) for x in as_list(prompt_ids)],
                "original_prompt_ids": [int(x) for x in as_list(original_prompt_ids)],
                "sampling_params": sampling_params if isinstance(sampling_params, dict) else {},
            }
        )
    return out


def prefix_diff(left: list[int], right: list[int]) -> tuple[int, int | None, int | None]:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if int(left[idx]) != int(right[idx]):
            return idx, int(left[idx]), int(right[idx])
    if len(left) != len(right):
        return limit, left[limit] if limit < len(left) else None, right[limit] if limit < len(right) else None
    return -1, None, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Match VERL trace response rows to dumped student vLLM IO, then verify "
            "that trace prompts reconstruct the actual final vLLM prompt ids."
        )
    )
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--student-vllm-io", nargs="+", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pad-token-id", type=int, default=151643)
    parser.add_argument("--image-token-id", type=int, default=151655)
    parser.add_argument("--video-token-id", type=int, default=151656)
    args = parser.parse_args()

    trace_paths = expand_paths(args.traces)
    io_rows = load_student_io(expand_paths(args.student_vllm_io))
    used_io: set[int] = set()

    records: list[dict[str, Any]] = []
    missing = 0
    original_prompt_same = 0
    final_prompt_same = 0

    for trace_i, trace_path in enumerate(trace_paths):
        trace = load_trace(trace_path)
        batch = trace["batch"]
        row_count = int(batch["responses"].shape[0])
        for row in range(row_count):
            response_ids = response_ids_from_trace(batch, row)
            match_i = None
            for io_i, io_row in enumerate(io_rows):
                if io_i in used_io:
                    continue
                if response_ids == io_row["generated"][: len(response_ids)] and len(response_ids) == len(io_row["generated"]):
                    match_i = io_i
                    break
            if match_i is None:
                missing += 1
                records.append(
                    {
                        "trace": trace_path,
                        "trace_i": trace_i,
                        "row": row,
                        "matched": False,
                        "response_len": len(response_ids),
                    }
                )
                continue

            used_io.add(match_i)
            io_row = io_rows[match_i]
            trace_original_prompt = prompt_ids_from_trace(batch, row, pad_token_id=args.pad_token_id)
            trace_final_prompt = dedup_consecutive_mm_tokens(
                trace_original_prompt,
                image_token_id=args.image_token_id,
                video_token_id=args.video_token_id,
            )
            original_same = trace_original_prompt == io_row["original_prompt_ids"]
            final_same = trace_final_prompt == io_row["prompt_ids"]
            original_prompt_same += int(original_same)
            final_prompt_same += int(final_same)
            original_diff = prefix_diff(trace_original_prompt, io_row["original_prompt_ids"])
            final_diff = prefix_diff(trace_final_prompt, io_row["prompt_ids"])
            records.append(
                {
                    "trace": trace_path,
                    "trace_i": trace_i,
                    "row": row,
                    "matched": True,
                    "student_vllm_io": io_row["path"],
                    "response_len": len(response_ids),
                    "trace_original_prompt_len": len(trace_original_prompt),
                    "io_original_prompt_len": len(io_row["original_prompt_ids"]),
                    "trace_final_prompt_len": len(trace_final_prompt),
                    "io_final_prompt_len": len(io_row["prompt_ids"]),
                    "original_prompt_same": original_same,
                    "final_prompt_same": final_same,
                    "original_first_diff": original_diff[0],
                    "original_trace_token": original_diff[1],
                    "original_io_token": original_diff[2],
                    "final_first_diff": final_diff[0],
                    "final_trace_token": final_diff[1],
                    "final_io_token": final_diff[2],
                    "max_tokens": io_row["sampling_params"].get("max_tokens"),
                    "prompt_logprobs": io_row["sampling_params"].get("prompt_logprobs"),
                }
            )

    matched = len([r for r in records if r.get("matched")])
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"trace_files = {len(trace_paths)}")
    print(f"student_vllm_io_files = {len(io_rows)}")
    print(f"matched = {matched}")
    print(f"missing = {missing}")
    print(f"unused_io = {len(io_rows) - len(used_io)}")
    print(f"original_prompt_same = {original_prompt_same}/{matched}")
    print(f"final_prompt_same = {final_prompt_same}/{matched}")
    if args.output:
        print(f"output = {args.output}")
    print("first records:")
    for record in records[:10]:
        print(
            "trace_i=", record["trace_i"],
            "row=", record["row"],
            "matched=", record["matched"],
            "orig_same=", record.get("original_prompt_same"),
            "final_same=", record.get("final_prompt_same"),
            "trace_final_len=", record.get("trace_final_prompt_len"),
            "io_final_len=", record.get("io_final_prompt_len"),
            "file=", Path(record.get("student_vllm_io", "")).name,
        )
    print("RESULT =", "OK" if missing == 0 and final_prompt_same == matched and original_prompt_same == matched else "FAIL")


if __name__ == "__main__":
    main()
