import argparse
import glob
from pathlib import Path
from typing import Any

import torch


def load_pt(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def active_sequence(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[int]:
    active = attention_mask.bool()
    return [int(token_id) for token_id in input_ids[active].tolist()]


def dedup_consecutive_mm_tokens(
    token_ids: list[int],
    image_token_id: int | None,
    video_token_id: int | None,
) -> list[int]:
    mm_ids = {token_id for token_id in (image_token_id, video_token_id) if token_id is not None}
    if not mm_ids:
        return token_ids

    deduped: list[int] = []
    previous_was_mm = False
    for token_id in token_ids:
        current_is_mm = token_id in mm_ids
        if current_is_mm and previous_was_mm:
            continue
        deduped.append(token_id)
        previous_was_mm = current_is_mm
    return deduped


def common_prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        count += 1
    return count


def count_token(token_ids: list[int], token_id: int | None) -> int:
    if token_id is None:
        return 0
    return sum(1 for value in token_ids if value == token_id)


def mismatch_window(left: list[int], right: list[int], center: int, radius: int = 8) -> dict[str, Any]:
    start = max(center - radius, 0)
    end = min(max(len(left), len(right)), center + radius + 1)
    return {
        "start": start,
        "end": end,
        "trace": left[start:min(end, len(left))],
        "teacher": right[start:min(end, len(right))],
    }


def expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            paths.append(pattern)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare VERL teacher vLLM request sequence_ids with OPD trace batch input_ids."
    )
    parser.add_argument("--trace", required=True, help="A verl_opd_trace .pt file.")
    parser.add_argument("--teacher-requests", nargs="+", required=True, help="Teacher request dump .pt files or globs.")
    parser.add_argument("--image-token-id", type=int, default=151655)
    parser.add_argument("--video-token-id", type=int, default=151656)
    parser.add_argument("--max-rows", type=int, default=24)
    args = parser.parse_args()

    trace = load_pt(args.trace)
    batch = trace["batch"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    row_count = min(int(input_ids.shape[0]), args.max_rows)
    trace_sequences = [
        active_sequence(input_ids[row], attention_mask[row])
        for row in range(row_count)
    ]
    trace_dedup_sequences = [
        dedup_consecutive_mm_tokens(seq, args.image_token_id, args.video_token_id)
        for seq in trace_sequences
    ]

    request_paths = expand_paths(args.teacher_requests)
    request_payloads = [load_pt(path) for path in request_paths]
    request_sequences = [payload["sequence_ids"] for payload in request_payloads]

    print("=== compare VERL teacher request with trace batch ===")
    print(f"trace={args.trace}")
    print(f"trace_rows={row_count}")
    print(f"teacher_request_count={len(request_sequences)}")
    print(f"image_token_id={args.image_token_id} video_token_id={args.video_token_id}")

    for row, trace_seq in enumerate(trace_sequences):
        if row >= len(request_sequences):
            break
        teacher_seq = request_sequences[row]
        trace_dedup = trace_dedup_sequences[row]
        exact = trace_seq == teacher_seq
        dedup_exact = trace_dedup == teacher_seq
        prefix = common_prefix_len(trace_seq, teacher_seq)
        dedup_prefix = common_prefix_len(trace_dedup, teacher_seq)
        first_mismatch = prefix if not exact else None

        print(
            " | ".join(
                [
                    f"row={row}",
                    f"teacher_file={Path(request_paths[row]).name}",
                    f"exact={exact}",
                    f"dedup_exact={dedup_exact}",
                    f"trace_len={len(trace_seq)}",
                    f"trace_dedup_len={len(trace_dedup)}",
                    f"teacher_len={len(teacher_seq)}",
                    f"prefix={prefix}",
                    f"dedup_prefix={dedup_prefix}",
                    f"trace_image_tokens={count_token(trace_seq, args.image_token_id)}",
                    f"dedup_image_tokens={count_token(trace_dedup, args.image_token_id)}",
                    f"teacher_image_tokens={count_token(teacher_seq, args.image_token_id)}",
                ]
            )
        )
        if first_mismatch is not None:
            print(f"  mismatch_window={mismatch_window(trace_seq, teacher_seq, first_mismatch)}")


if __name__ == "__main__":
    main()
