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
from src.method.vllm_teacher_client import RemoteTeacherScorer  # noqa: E402


def load_pt(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            paths.append(pattern)
    return paths


def active_sequence(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[int]:
    active = attention_mask.bool()
    return [int(token_id) for token_id in input_ids[active].tolist()]


def first_active_index(attention_mask: torch.Tensor) -> int:
    active = torch.nonzero(attention_mask.bool(), as_tuple=False).flatten()
    if active.numel() == 0:
        return int(attention_mask.numel())
    return int(active[0].item())


def extract_images(request: dict[str, Any]) -> list[Any]:
    prompt_kwargs = request.get("prompt_kwargs") or {}
    mm_data = prompt_kwargs.get("multi_modal_data") or request.get("multi_modal_data") or {}
    for key in ("images", "image"):
        value = mm_data.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]
    return []


def request_sequence(request: dict[str, Any]) -> list[int]:
    if "prompt_token_ids" in request:
        return [int(token_id) for token_id in request["prompt_token_ids"]]
    prompt_kwargs = request.get("prompt_kwargs") or {}
    if "prompt_token_ids" in prompt_kwargs:
        return [int(token_id) for token_id in prompt_kwargs["prompt_token_ids"]]
    if "sequence_ids" in request:
        return [int(token_id) for token_id in request["sequence_ids"]]
    raise KeyError("Request dump has neither prompt_token_ids nor sequence_ids.")


def request_multi_modal_data(request: dict[str, Any]) -> dict[str, Any] | None:
    prompt_kwargs = request.get("prompt_kwargs") or {}
    if "multi_modal_data" in prompt_kwargs:
        return prompt_kwargs["multi_modal_data"]
    if "multi_modal_data" in request:
        return request["multi_modal_data"]
    return None


def request_mm_processor_kwargs(request: dict[str, Any]) -> dict[str, Any] | None:
    prompt_kwargs = request.get("prompt_kwargs") or {}
    if "mm_processor_kwargs" in prompt_kwargs:
        return prompt_kwargs["mm_processor_kwargs"]
    return request.get("mm_processor_kwargs")


def pad_sequences(sequences: list[list[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("No sequences to pad.")
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, seq in enumerate(sequences):
        length = len(seq)
        start = max_len - length
        input_ids[row, start:] = torch.tensor(seq, dtype=torch.long)
        attention_mask[row, start:] = 1
    return input_ids, attention_mask


def topk_set_overlap(old_ids: torch.Tensor, new_ids: torch.Tensor, mask: torch.Tensor) -> float:
    overlaps = []
    old_cpu = old_ids.detach().cpu()
    new_cpu = new_ids.detach().cpu()
    mask_cpu = mask.detach().cpu().bool()
    for row in range(old_cpu.shape[0]):
        for pos in range(old_cpu.shape[1]):
            if not bool(mask_cpu[row, pos].item()):
                continue
            old_set = set(int(x) for x in old_cpu[row, pos].tolist())
            new_set = set(int(x) for x in new_cpu[row, pos].tolist())
            if old_set:
                overlaps.append(len(old_set & new_set) / len(old_set))
    if not overlaps:
        return 0.0
    return float(sum(overlaps) / len(overlaps))


def compare_slices(
    *,
    old_logps: torch.Tensor,
    old_ids: torch.Tensor,
    new_logps: torch.Tensor,
    new_ids: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    mask = mask.bool()
    mask3 = mask.unsqueeze(-1).expand_as(old_ids)
    delta = (old_logps.float() - new_logps.float()).abs()
    ids_equal = old_ids.long().eq(new_ids.long())
    old_mass = old_logps.float().exp().sum(dim=-1)
    new_mass = new_logps.float().exp().sum(dim=-1)
    denom = mask.float().sum().clamp_min(1.0)
    return {
        "active_tokens": float(mask.sum().item()),
        "ids_same": float(ids_equal[mask3].float().mean().item()) if mask3.any() else 0.0,
        "set_overlap": topk_set_overlap(old_ids, new_ids, mask),
        "logps_mean_abs": float(delta[mask3].mean().item()) if mask3.any() else 0.0,
        "logps_max_abs": float(delta[mask3].max().item()) if mask3.any() else 0.0,
        "old_mass": float((old_mass * mask.float()).sum().item() / denom.item()),
        "new_mass": float((new_mass * mask.float()).sum().item() / denom.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score dumped VERL teacher vLLM requests and compare against trace teacher tensors."
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--teacher-requests", nargs="+", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29577)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--teacher-shift-offset", type=int, default=-1)
    parser.add_argument("--image-token-id", type=int, default=151655)
    parser.add_argument("--video-token-id", type=int, default=151656)
    parser.add_argument("--pad-token-id", type=int, default=151643)
    parser.add_argument("--metrics-output", default=None)
    args = parser.parse_args()

    trace = load_trace(args.trace)
    batch = trace["batch"]
    input_ids = batch["input_ids"].cpu()
    attention_mask = batch["attention_mask"].cpu()
    response_mask = batch["response_mask"].cpu().bool()
    old_logps = batch["teacher_logprobs"].cpu().float()
    old_ids = batch["teacher_ids"].cpu().long()
    prompts = batch.get("prompts")
    prompt_width = int(prompts.shape[1]) if torch.is_tensor(prompts) else int(input_ids.shape[1] - response_mask.shape[1])
    response_start = prompt_width + int(args.teacher_shift_offset)

    trace_sequences = [active_sequence(input_ids[row], attention_mask[row]) for row in range(input_ids.shape[0])]
    trace_starts = [first_active_index(attention_mask[row]) for row in range(input_ids.shape[0])]
    sequence_to_row = {tuple(seq): row for row, seq in enumerate(trace_sequences)}

    request_paths = expand_paths(args.teacher_requests)
    requests = [load_pt(path) for path in request_paths]
    request_sequences = [request_sequence(request) for request in requests]
    request_rows = []
    for path, seq in zip(request_paths, request_sequences, strict=True):
        row = sequence_to_row.get(tuple(seq))
        if row is None:
            raise RuntimeError(f"Teacher request does not exactly match any trace row: {path}")
        request_rows.append(row)

    scorer = RemoteTeacherScorer(host=args.host, port=args.port, timeout=args.timeout, topk=args.topk)

    print("=== rescore VERL teacher request dumps ===", flush=True)
    print(f"trace={args.trace}", flush=True)
    print(f"request_count={len(requests)}", flush=True)
    print(f"teacher={args.host}:{args.port}", flush=True)
    print(f"request_rows={request_rows}", flush=True)

    metrics_file = open(args.metrics_output, "w", encoding="utf-8") if args.metrics_output else None
    try:
        all_old_active_logps = []
        all_old_active_ids = []
        all_new_active_logps = []
        all_new_active_ids = []
        all_active_masks = []
        all_old_resp_logps = []
        all_old_resp_ids = []
        all_new_resp_logps = []
        all_new_resp_ids = []
        all_resp_masks = []

        for start in range(0, len(requests), args.micro_batch_size):
            end = min(start + args.micro_batch_size, len(requests))
            seq_batch = request_sequences[start:end]
            row_batch = request_rows[start:end]
            input_batch, mask_batch = pad_sequences(seq_batch, args.pad_token_id)
            images_batch = [extract_images(request) for request in requests[start:end]]
            multi_modal_data_batch = [request_multi_modal_data(request) for request in requests[start:end]]
            kwargs_batch = [request_mm_processor_kwargs(request) for request in requests[start:end]]
            new_logps, new_ids = scorer.score(
                sequences=input_batch,
                attention_mask=mask_batch,
                images_per_sample=images_batch,
                image_token_id=args.image_token_id,
                video_token_id=args.video_token_id,
                pad_token_id=args.pad_token_id,
                mm_processor_kwargs_per_sample=kwargs_batch,
                multi_modal_data_per_sample=multi_modal_data_batch,
            )
            new_logps = new_logps.cpu().float()
            new_ids = new_ids.cpu().long()

            for local_idx, row in enumerate(row_batch):
                seq_len = len(seq_batch[local_idx])
                trace_start = trace_starts[row]
                active_len = max(seq_len - 1, 0)
                old_active_logps = old_logps[row, trace_start : trace_start + active_len, :]
                old_active_ids = old_ids[row, trace_start : trace_start + active_len, :]
                new_active_logps = new_logps[local_idx, :active_len, :]
                new_active_ids = new_ids[local_idx, :active_len, :]
                active_mask = torch.ones(active_len, dtype=torch.bool)

                active_response_start = max(response_start - trace_start, 0)
                response_len = min(
                    int(response_mask.shape[1]),
                    old_logps.shape[1] - response_start,
                    new_logps.shape[1] - active_response_start,
                )
                old_resp_logps = old_logps[row, response_start : response_start + response_len, :]
                old_resp_ids = old_ids[row, response_start : response_start + response_len, :]
                new_resp_logps = new_logps[local_idx, active_response_start : active_response_start + response_len, :]
                new_resp_ids = new_ids[local_idx, active_response_start : active_response_start + response_len, :]
                resp_mask = response_mask[row, :response_len]

                active_stats = compare_slices(
                    old_logps=old_active_logps.unsqueeze(0),
                    old_ids=old_active_ids.unsqueeze(0),
                    new_logps=new_active_logps.unsqueeze(0),
                    new_ids=new_active_ids.unsqueeze(0),
                    mask=active_mask.unsqueeze(0),
                )
                resp_stats = compare_slices(
                    old_logps=old_resp_logps.unsqueeze(0),
                    old_ids=old_resp_ids.unsqueeze(0),
                    new_logps=new_resp_logps.unsqueeze(0),
                    new_ids=new_resp_ids.unsqueeze(0),
                    mask=resp_mask.unsqueeze(0),
                )

                record = {
                    "request_index": start + local_idx,
                    "request_path": request_paths[start + local_idx],
                    "trace_row": row,
                    "sequence_len": seq_len,
                    "image_count": len(images_batch[local_idx]),
                    "has_mm_processor_kwargs": kwargs_batch[local_idx] is not None,
                    "active": active_stats,
                    "response": resp_stats,
                }
                if metrics_file is not None:
                    metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    metrics_file.flush()
                print(
                    " | ".join(
                        [
                            f"request={record['request_index']}",
                            f"trace_row={row}",
                            f"seq_len={seq_len}",
                            f"images={record['image_count']}",
                            f"resp_ids_same={resp_stats['ids_same']:.6f}",
                            f"resp_set_overlap={resp_stats['set_overlap']:.6f}",
                            f"resp_logps_mean_abs={resp_stats['logps_mean_abs']:.6e}",
                            f"resp_logps_max_abs={resp_stats['logps_max_abs']:.6e}",
                            f"resp_old_mass={resp_stats['old_mass']:.8f}",
                            f"resp_new_mass={resp_stats['new_mass']:.8f}",
                        ]
                    ),
                    flush=True,
                )

                all_old_active_logps.append(old_active_logps)
                all_old_active_ids.append(old_active_ids)
                all_new_active_logps.append(new_active_logps)
                all_new_active_ids.append(new_active_ids)
                all_active_masks.append(active_mask)
                all_old_resp_logps.append(old_resp_logps)
                all_old_resp_ids.append(old_resp_ids)
                all_new_resp_logps.append(new_resp_logps)
                all_new_resp_ids.append(new_resp_ids)
                all_resp_masks.append(resp_mask)

        active_summary = compare_slices(
            old_logps=torch.nn.utils.rnn.pad_sequence(all_old_active_logps, batch_first=True),
            old_ids=torch.nn.utils.rnn.pad_sequence(all_old_active_ids, batch_first=True, padding_value=args.pad_token_id),
            new_logps=torch.nn.utils.rnn.pad_sequence(all_new_active_logps, batch_first=True),
            new_ids=torch.nn.utils.rnn.pad_sequence(all_new_active_ids, batch_first=True, padding_value=args.pad_token_id),
            mask=torch.nn.utils.rnn.pad_sequence(all_active_masks, batch_first=True),
        )
        response_summary = compare_slices(
            old_logps=torch.nn.utils.rnn.pad_sequence(all_old_resp_logps, batch_first=True),
            old_ids=torch.nn.utils.rnn.pad_sequence(all_old_resp_ids, batch_first=True, padding_value=args.pad_token_id),
            new_logps=torch.nn.utils.rnn.pad_sequence(all_new_resp_logps, batch_first=True),
            new_ids=torch.nn.utils.rnn.pad_sequence(all_new_resp_ids, batch_first=True, padding_value=args.pad_token_id),
            mask=torch.nn.utils.rnn.pad_sequence(all_resp_masks, batch_first=True),
        )
        print(f"active_summary={json.dumps(active_summary, sort_keys=True)}", flush=True)
        print(f"response_summary={json.dumps(response_summary, sort_keys=True)}", flush=True)
    finally:
        if metrics_file is not None:
            metrics_file.close()

    print("rescore_verl_teacher_requests_ok=True", flush=True)


if __name__ == "__main__":
    main()
