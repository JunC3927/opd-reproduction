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
from rescore_verl_teacher_requests import (  # noqa: E402
    active_sequence,
    compare_slices,
    extract_images,
    first_active_index,
    pad_sequences,
    request_mm_processor_kwargs,
    request_multi_modal_data,
    request_sequence,
)
from src.method.vllm_teacher_client import RemoteTeacherScorer  # noqa: E402


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


def load_request(path: str) -> dict[str, Any]:
    request = torch.load(path, map_location="cpu", weights_only=False)
    if request.get("format") != "reconstructed_verl_teacher_vllm_request_v1":
        raise ValueError(
            f"{path} is not a reconstructed VERL teacher request. "
            "Run tools/reconstruct_verl_teacher_requests_from_trace.py first."
        )
    if "trace_row" not in request:
        raise KeyError(f"{path} is missing trace_row.")
    if "trace_path" not in request:
        raise KeyError(f"{path} is missing trace_path.")
    return request


def trace_match_keys(trace_path: str) -> set[str]:
    path = Path(trace_path)
    return {str(path), path.name}


def request_trace_key(request: dict[str, Any]) -> str:
    return Path(str(request["trace_path"])).name


def group_requests_by_trace(request_paths: list[str]) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for path in request_paths:
        request = load_request(path)
        grouped.setdefault(request_trace_key(request), []).append((path, request))
    for items in grouped.values():
        items.sort(key=lambda item: int(item[1]["trace_row"]))
    return grouped


def response_start_from_trace(trace: dict[str, Any], teacher_shift_offset: int) -> int:
    batch = trace["batch"]
    input_ids = batch["input_ids"]
    response_mask = batch["response_mask"]
    prompts = batch.get("prompts")
    if torch.is_tensor(prompts):
        prompt_width = int(prompts.shape[1])
    else:
        prompt_width = int(input_ids.shape[1] - response_mask.shape[1])
    return prompt_width + int(teacher_shift_offset)


def build_sequence_to_row(trace: dict[str, Any]) -> dict[tuple[int, ...], int]:
    batch = trace["batch"]
    input_ids = batch["input_ids"].cpu()
    attention_mask = batch["attention_mask"].cpu()
    return {
        tuple(active_sequence(input_ids[row], attention_mask[row])): row
        for row in range(input_ids.shape[0])
    }


def locate_trace_row(
    *,
    request_path: str,
    request: dict[str, Any],
    trace: dict[str, Any],
    sequence_to_row: dict[tuple[int, ...], int],
) -> int:
    row = int(request["trace_row"])
    batch = trace["batch"]
    input_ids = batch["input_ids"].cpu()
    attention_mask = batch["attention_mask"].cpu()
    if 0 <= row < input_ids.shape[0]:
        trace_seq = active_sequence(input_ids[row], attention_mask[row])
        if trace_seq == request_sequence(request):
            return row

    fallback_row = sequence_to_row.get(tuple(request_sequence(request)))
    if fallback_row is None:
        raise RuntimeError(
            "Could not map reconstructed teacher request back to a trace row: "
            f"{request_path}"
        )
    return int(fallback_row)


def score_requests(
    *,
    scorer: RemoteTeacherScorer,
    requests: list[dict[str, Any]],
    pad_token_id: int,
    image_token_id: int,
    video_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    sequences = [request_sequence(request) for request in requests]
    input_batch, mask_batch = pad_sequences(sequences, pad_token_id)
    multi_modal_data_batch = [request_multi_modal_data(request) for request in requests]
    images_batch = [extract_images(request) for request in requests]
    kwargs_batch = [request_mm_processor_kwargs(request) for request in requests]
    logps, ids = scorer.score(
        sequences=input_batch,
        attention_mask=mask_batch,
        images_per_sample=images_batch,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        pad_token_id=pad_token_id,
        mm_processor_kwargs_per_sample=kwargs_batch,
        multi_modal_data_per_sample=multi_modal_data_batch,
    )
    local_starts = [max(input_batch.shape[1] - len(seq), 0) for seq in sequences]
    active_lengths = [max(len(seq) - 1, 0) for seq in sequences]
    return logps.cpu().float(), ids.cpu().long(), local_starts, active_lengths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create Layer B OPD trace dumps by replacing VERL trace teacher_ids/"
            "teacher_logprobs with top-k values recomputed by a vLLM teacher."
        )
    )
    parser.add_argument("traces", nargs="+", help="Original VERL trace dump(s) or glob pattern(s).")
    parser.add_argument("--teacher-requests", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29577)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--teacher-shift-offset", type=int, default=-1)
    parser.add_argument("--image-token-id", type=int, default=151655)
    parser.add_argument("--video-token-id", type=int, default=151656)
    parser.add_argument("--pad-token-id", type=int, default=151643)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    trace_paths = expand_paths(args.traces)
    if args.max_files > 0:
        trace_paths = trace_paths[: args.max_files]
    request_paths = expand_paths(args.teacher_requests)
    grouped_requests = group_requests_by_trace(request_paths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = open(args.metrics_output, "w", encoding="utf-8") if args.metrics_output else None
    scorer = RemoteTeacherScorer(host=args.host, port=args.port, timeout=args.timeout, topk=args.topk)

    print("=== rescore VERL trace to Layer B ===", flush=True)
    print(f"trace_count={len(trace_paths)}", flush=True)
    print(f"request_count={len(request_paths)}", flush=True)
    print(f"teacher={args.host}:{args.port}", flush=True)
    print(f"output_dir={output_dir}", flush=True)

    total_rows = 0
    try:
        for trace_index, trace_path in enumerate(trace_paths):
            output_path = output_dir / Path(trace_path).name
            if args.skip_existing and output_path.exists():
                print(f"skip_existing trace={trace_index} output={output_path}", flush=True)
                continue

            trace = load_trace(trace_path)
            batch = trace["batch"]
            old_teacher_ids = batch["teacher_ids"].cpu()
            old_teacher_logps = batch["teacher_logprobs"].cpu()
            new_teacher_ids = old_teacher_ids.clone()
            new_teacher_logps = old_teacher_logps.clone()

            trace_requests = []
            for key in trace_match_keys(trace_path):
                trace_requests.extend(grouped_requests.get(key, []))
            # trace_match_keys includes exact path and basename; de-duplicate by file path.
            deduped: dict[str, dict[str, Any]] = {path: request for path, request in trace_requests}
            trace_requests = sorted(deduped.items(), key=lambda item: int(item[1]["trace_row"]))
            if not trace_requests:
                raise RuntimeError(f"No reconstructed teacher requests found for trace: {trace_path}")

            response_start = response_start_from_trace(trace, args.teacher_shift_offset)
            response_mask = batch["response_mask"].cpu().bool()
            trace_starts = [
                first_active_index(batch["attention_mask"][row].cpu())
                for row in range(batch["input_ids"].shape[0])
            ]
            sequence_to_row = build_sequence_to_row(trace)

            file_records = []
            for start in range(0, len(trace_requests), args.micro_batch_size):
                end = min(start + args.micro_batch_size, len(trace_requests))
                request_slice = trace_requests[start:end]
                paths_slice = [path for path, _request in request_slice]
                requests_slice = [request for _path, request in request_slice]
                rows = [
                    locate_trace_row(
                        request_path=path,
                        request=request,
                        trace=trace,
                        sequence_to_row=sequence_to_row,
                    )
                    for path, request in request_slice
                ]
                logps, ids, local_starts, active_lengths = score_requests(
                    scorer=scorer,
                    requests=requests_slice,
                    pad_token_id=args.pad_token_id,
                    image_token_id=args.image_token_id,
                    video_token_id=args.video_token_id,
                )

                for local_idx, row in enumerate(rows):
                    active_len = int(active_lengths[local_idx])
                    local_start = int(local_starts[local_idx])
                    trace_start = int(trace_starts[row])
                    if active_len <= 0:
                        continue
                    old_slice = slice(trace_start, trace_start + active_len)
                    new_slice = slice(local_start, local_start + active_len)
                    if old_slice.stop > new_teacher_ids.shape[1]:
                        raise RuntimeError(
                            f"New teacher slice exceeds trace tensor length: trace={trace_path}, row={row}"
                        )
                    if new_slice.stop > ids.shape[1]:
                        raise RuntimeError(
                            f"Teacher output slice exceeds returned tensor length: request={paths_slice[local_idx]}"
                        )

                    new_teacher_ids[row, old_slice, :] = ids[local_idx, new_slice, :].to(new_teacher_ids.dtype)
                    new_teacher_logps[row, old_slice, :] = logps[local_idx, new_slice, :].to(new_teacher_logps.dtype)

                    response_len = min(
                        int(response_mask.shape[1]),
                        old_teacher_ids.shape[1] - response_start,
                        active_len - max(response_start - trace_start, 0),
                    )
                    response_stats = {}
                    if response_len > 0:
                        active_response_start = max(response_start - trace_start, 0)
                        response_stats = compare_slices(
                            old_logps=old_teacher_logps[
                                row, response_start : response_start + response_len, :
                            ].unsqueeze(0),
                            old_ids=old_teacher_ids[
                                row, response_start : response_start + response_len, :
                            ].unsqueeze(0),
                            new_logps=new_teacher_logps[
                                row, response_start : response_start + response_len, :
                            ].unsqueeze(0),
                            new_ids=new_teacher_ids[
                                row, response_start : response_start + response_len, :
                            ].unsqueeze(0),
                            mask=response_mask[row, :response_len].unsqueeze(0),
                        )

                    record = {
                        "trace_index": trace_index,
                        "trace_path": trace_path,
                        "output_path": str(output_path),
                        "request_path": paths_slice[local_idx],
                        "trace_row": int(row),
                        "trace_start": trace_start,
                        "local_start": local_start,
                        "active_len": active_len,
                        "response": response_stats,
                    }
                    file_records.append(record)
                    if metrics_file is not None:
                        metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        metrics_file.flush()

            out_trace = dict(trace)
            out_batch = dict(batch)
            out_batch["teacher_ids"] = new_teacher_ids.to(old_teacher_ids.dtype)
            out_batch["teacher_logprobs"] = new_teacher_logps.to(old_teacher_logps.dtype)
            out_trace["batch"] = out_batch
            out_trace["trace_stage"] = "layerB_vllm_teacher_rescored"
            out_trace["layerB_teacher_rescore"] = {
                "format": "layerB_vllm_teacher_rescore_v1",
                "source_trace_path": trace_path,
                "request_count": len(trace_requests),
                "teacher": f"{args.host}:{args.port}",
                "topk": int(args.topk),
                "teacher_shift_offset": int(args.teacher_shift_offset),
            }
            torch.save(out_trace, output_path)
            total_rows += len(file_records)

            mean_overlap = 0.0
            mean_logp_abs = 0.0
            response_records = [r["response"] for r in file_records if r["response"]]
            if response_records:
                mean_overlap = sum(r["set_overlap"] for r in response_records) / len(response_records)
                mean_logp_abs = sum(r["logps_mean_abs"] for r in response_records) / len(response_records)
            print(
                " | ".join(
                    [
                        f"trace={trace_index}",
                        f"path={trace_path}",
                        f"rows={len(file_records)}",
                        f"output={output_path}",
                        f"mean_resp_set_overlap={mean_overlap:.6f}",
                        f"mean_resp_logps_abs={mean_logp_abs:.6e}",
                    ]
                ),
                flush=True,
            )
    finally:
        if metrics_file is not None:
            metrics_file.close()

    print(f"layerB_trace_files={len(trace_paths)}", flush=True)
    print(f"layerB_rows={total_rows}", flush=True)
    print("rescore_verl_trace_to_layerB_ok=True", flush=True)


if __name__ == "__main__":
    main()
