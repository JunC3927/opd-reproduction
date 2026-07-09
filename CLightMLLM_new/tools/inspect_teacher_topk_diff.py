import argparse
import glob
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
    expand_paths,
    first_active_index,
    load_pt,
    request_id,
    request_row_match_sequence,
    request_sequence,
)
from src.method.vllm_teacher_client import RemoteTeacherScorer  # noqa: E402


def maybe_load_tokenizer(path: str | None):
    if not path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=False)


def token_text(tokenizer: Any | None, token_id: int) -> str:
    if tokenizer is None:
        return ""
    try:
        return repr(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        return "<decode_failed>"


def find_trace_rows(
    *,
    trace_sequences: list[list[int]],
    requests: list[dict[str, Any]],
    row_match_requests: list[dict[str, Any]] | None,
) -> list[int]:
    sequence_to_row = {tuple(seq): row for row, seq in enumerate(trace_sequences)}
    if not row_match_requests:
        rows = []
        for request in requests:
            row = sequence_to_row.get(tuple(request_row_match_sequence(request)))
            if row is None:
                raise RuntimeError(
                    "Teacher request does not exactly match a trace row. "
                    "For final vLLM prompt dumps, pass --row-match-requests."
                )
            rows.append(row)
        return rows

    row_by_request_id: dict[str, int] = {}
    for match_request in row_match_requests:
        rid = request_id(match_request)
        if rid is None:
            continue
        row = sequence_to_row.get(tuple(request_row_match_sequence(match_request)))
        if row is None:
            continue
        row_by_request_id[rid] = row

    rows = []
    for request in requests:
        rid = request_id(request)
        row = row_by_request_id.get(rid) if rid is not None else None
        if row is None:
            row = sequence_to_row.get(tuple(request_row_match_sequence(request)))
        if row is None:
            raise RuntimeError(f"Could not locate trace row for request_id={rid}.")
        rows.append(row)
    return rows


def index_by_token(ids: torch.Tensor, logps: torch.Tensor) -> dict[int, tuple[int, float]]:
    return {int(token_id.item()): (rank, float(logps[rank].item())) for rank, token_id in enumerate(ids)}


def print_position(
    *,
    tokenizer: Any | None,
    trace_row: int,
    response_offset: int,
    old_ids: torch.Tensor,
    old_logps: torch.Tensor,
    new_ids: torch.Tensor,
    new_logps: torch.Tensor,
    response_token_id: int | None,
    rank_limit: int,
) -> None:
    old_map = index_by_token(old_ids, old_logps)
    new_map = index_by_token(new_ids, new_logps)
    old_set = set(old_map)
    new_set = set(new_map)
    overlap = len(old_set & new_set) / max(len(old_set), 1)
    same_rank = old_ids.eq(new_ids).float().mean().item()

    print("=" * 120)
    print(
        f"trace_row={trace_row} response_offset={response_offset} "
        f"response_token_id={response_token_id} {token_text(tokenizer, response_token_id) if response_token_id is not None else ''}"
    )
    print(f"rank_same={same_rank:.6f} set_overlap={overlap:.6f}")
    print(
        "rank | old_id old_logp old_text | new_id new_logp new_text | "
        "same_rank | old_id_new_rank old_id_new_logp absdiff"
    )
    print("-" * 120)
    for rank in range(min(rank_limit, old_ids.numel(), new_ids.numel())):
        old_id = int(old_ids[rank].item())
        new_id = int(new_ids[rank].item())
        old_lp = float(old_logps[rank].item())
        new_lp = float(new_logps[rank].item())
        mapped = new_map.get(old_id)
        if mapped is None:
            mapped_rank = "NA"
            mapped_lp = "NA"
            diff = "NA"
        else:
            mapped_rank = str(mapped[0])
            mapped_lp = f"{mapped[1]: .8f}"
            diff = f"{abs(old_lp - mapped[1]): .8f}"
        print(
            f"{rank:>4} | "
            f"{old_id:>8} {old_lp: .8f} {token_text(tokenizer, old_id):<18} | "
            f"{new_id:>8} {new_lp: .8f} {token_text(tokenizer, new_id):<18} | "
            f"{str(old_id == new_id):<9} | "
            f"{mapped_rank:>15} {mapped_lp:>16} {diff:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print VERL teacher top-k vs re-scored vLLM teacher top-k for selected response tokens."
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--teacher-requests", nargs="+", required=True)
    parser.add_argument("--row-match-requests", nargs="*", default=None)
    parser.add_argument("--request-index", type=int, default=0)
    parser.add_argument("--response-token-index", type=int, default=0)
    parser.add_argument("--num-response-tokens", type=int, default=5)
    parser.add_argument("--rank-limit", type=int, default=32)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29577)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--teacher-shift-offset", type=int, default=-1)
    parser.add_argument("--pad-token-id", type=int, default=151643)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()

    trace = load_trace(args.trace)
    batch = trace["batch"]
    input_ids = batch["input_ids"].cpu()
    attention_mask = batch["attention_mask"].cpu()
    response_mask = batch["response_mask"].cpu().bool()
    responses = batch.get("responses")
    responses = responses.cpu() if torch.is_tensor(responses) else None
    old_logps_all = batch["teacher_logprobs"].cpu().float()
    old_ids_all = batch["teacher_ids"].cpu().long()
    prompts = batch.get("prompts")
    prompt_width = int(prompts.shape[1]) if torch.is_tensor(prompts) else int(input_ids.shape[1] - response_mask.shape[1])
    response_start = prompt_width + int(args.teacher_shift_offset)

    request_paths = expand_paths(args.teacher_requests)
    requests = [load_pt(path) for path in request_paths]
    if not (0 <= args.request_index < len(requests)):
        raise IndexError(f"--request-index {args.request_index} out of range [0, {len(requests) - 1}]")

    row_match_requests = None
    if args.row_match_requests:
        row_match_requests = [load_pt(path) for path in expand_paths(args.row_match_requests)]

    trace_sequences = [active_sequence(input_ids[row], attention_mask[row]) for row in range(input_ids.shape[0])]
    trace_starts = [first_active_index(attention_mask[row]) for row in range(input_ids.shape[0])]
    request_rows = find_trace_rows(
        trace_sequences=trace_sequences,
        requests=requests,
        row_match_requests=row_match_requests,
    )

    request = requests[args.request_index]
    trace_row = request_rows[args.request_index]
    request_path = request_paths[args.request_index]
    sequence = request_sequence(request)
    trace_start = trace_starts[trace_row]
    active_response_start = max(response_start - trace_start, 0)

    scorer = RemoteTeacherScorer(host=args.host, port=args.port, timeout=args.timeout, topk=args.topk)
    new_logps, new_ids, new_lengths = scorer.score_prompt_requests(
        requests=[request],
        pad_token_id=args.pad_token_id,
    )
    new_logps = new_logps[0].cpu().float()
    new_ids = new_ids[0].cpu().long()
    new_len = int(new_lengths[0].item())

    tokenizer = maybe_load_tokenizer(args.tokenizer)

    print("=== inspect teacher top-k diff ===")
    print(f"trace={args.trace}")
    print(f"request_path={request_path}")
    print(f"request_index={args.request_index}")
    print(f"trace_row={trace_row}")
    print(f"sequence_len={len(sequence)} new_prompt_logprobs_len={new_len}")
    print(f"trace_start={trace_start} prompt_width={prompt_width} response_start={response_start}")
    print(f"active_response_start_in_new={active_response_start}")
    print(f"request_id={request_id(request)}")
    print()

    for offset in range(args.response_token_index, args.response_token_index + args.num_response_tokens):
        if offset < 0 or offset >= response_mask.shape[1]:
            print(f"skip response_offset={offset}: outside response width {response_mask.shape[1]}")
            continue
        if not bool(response_mask[trace_row, offset].item()):
            print(f"skip response_offset={offset}: response_mask is false")
            continue
        old_pos = response_start + offset
        new_pos = active_response_start + offset
        if old_pos >= old_ids_all.shape[1] or new_pos >= new_ids.shape[0]:
            print(f"skip response_offset={offset}: old_pos={old_pos}, new_pos={new_pos} out of range")
            continue
        response_token_id = int(responses[trace_row, offset].item()) if responses is not None else None
        print_position(
            tokenizer=tokenizer,
            trace_row=trace_row,
            response_offset=offset,
            old_ids=old_ids_all[trace_row, old_pos],
            old_logps=old_logps_all[trace_row, old_pos],
            new_ids=new_ids[new_pos],
            new_logps=new_logps[new_pos],
            response_token_id=response_token_id,
            rank_limit=args.rank_limit,
        )


if __name__ == "__main__":
    main()
