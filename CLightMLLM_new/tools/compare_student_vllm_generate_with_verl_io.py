import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def to_builtin(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_builtin(val) for val in value]
    if isinstance(value, tuple):
        return tuple(to_builtin(val) for val in value)
    return value


def get_nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def load_tokens_prompt_cls() -> Any:
    try:
        from vllm.inputs import TokensPrompt

        return TokensPrompt
    except Exception:
        return None


def build_prompt(payload: dict[str, Any], tokens_prompt_cls: Any) -> Any:
    prompt_kwargs = get_nested(payload, "input", "prompt_kwargs")
    if not isinstance(prompt_kwargs, dict):
        prompt_kwargs = {}

    if "prompt_token_ids" not in prompt_kwargs:
        prompt_ids = get_nested(payload, "input", "prompt_token_ids")
        prompt_kwargs["prompt_token_ids"] = [int(x) for x in as_list(prompt_ids)]

    prompt_kwargs = to_builtin(prompt_kwargs)
    if tokens_prompt_cls is not None:
        try:
            return tokens_prompt_cls(**prompt_kwargs)
        except TypeError:
            pass
    return prompt_kwargs


def sampling_kwargs_from_dump(
    payload: dict[str, Any],
    *,
    default_top_p: float | None,
    default_top_k: int | None,
) -> dict[str, Any]:
    dumped = get_nested(payload, "input", "sampling_params")
    if not isinstance(dumped, dict):
        dumped = {}

    kwargs = {
        "max_tokens": dumped.get("max_tokens"),
        "temperature": dumped.get("temperature"),
        "top_p": dumped.get("top_p", default_top_p),
        "top_k": dumped.get("top_k", default_top_k),
        "logprobs": dumped.get("logprobs"),
        "prompt_logprobs": dumped.get("prompt_logprobs"),
        "repetition_penalty": dumped.get("repetition_penalty"),
        "ignore_eos": dumped.get("ignore_eos"),
    }
    return {key: val for key, val in kwargs.items() if val is not None}


def make_sampling_params(sampling_params_cls: Any, kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(sampling_params_cls)
    allowed = set(signature.parameters)
    filtered = {key: val for key, val in kwargs.items() if key in allowed}
    return sampling_params_cls(**filtered)


def prefix_match_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        count += 1
    return count


def compare_ids(new_ids: list[int], old_ids: list[int]) -> dict[str, Any]:
    common = min(len(new_ids), len(old_ids))
    same = sum(int(int(new_ids[i]) == int(old_ids[i])) for i in range(common))
    first_diff = -1
    for idx in range(common):
        if int(new_ids[idx]) != int(old_ids[idx]):
            first_diff = idx
            break
    if first_diff < 0 and len(new_ids) != len(old_ids):
        first_diff = common
    return {
        "new_len": len(new_ids),
        "old_len": len(old_ids),
        "common_len": common,
        "same_prefix_len": prefix_match_len(new_ids, old_ids),
        "same_ratio_common": same / max(common, 1),
        "exact_same": new_ids == old_ids,
        "first_diff": first_diff,
        "new_token_at_diff": new_ids[first_diff] if 0 <= first_diff < len(new_ids) else None,
        "old_token_at_diff": old_ids[first_diff] if 0 <= first_diff < len(old_ids) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CLight/vLLM student generation on matched VERL final prompts and compare token outputs."
    )
    parser.add_argument("--alignment", required=True, help="student_rollout_prompt_alignment.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=1537)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--default-top-p", type=float, default=1.0)
    parser.add_argument("--default-top-k", type=int, default=-1)
    parser.add_argument(
        "--order",
        choices=("trace", "vllm-dump"),
        default="trace",
        help="Replay rows in trace order or in the original VERL vLLM IO dump order.",
    )
    parser.add_argument(
        "--save-token-ids",
        action="store_true",
        help="Store generated token ids in the jsonl output for text diff inspection.",
    )
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    rows = [row for row in load_jsonl(args.alignment) if row.get("matched")]
    if args.order == "vllm-dump":
        row_payload_pairs = []
        for row in rows:
            payload = torch.load(row["student_vllm_io"], map_location="cpu", weights_only=False)
            row_payload_pairs.append((row, payload))
        row_payload_pairs.sort(
            key=lambda item: (
                get_nested(item[1], "node_rank") if get_nested(item[1], "node_rank") is not None else -1,
                get_nested(item[1], "replica_rank") if get_nested(item[1], "replica_rank") is not None else -1,
                get_nested(item[1], "dump_index") if get_nested(item[1], "dump_index") is not None else 10**18,
                str(item[0].get("student_vllm_io")),
            )
        )
        rows = [row for row, _ in row_payload_pairs]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("No matched alignment rows found.")

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs

    print("=== compare student vLLM generate with VERL IO ===", flush=True)
    print(f"alignment={args.alignment}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"samples={len(rows)}", flush=True)
    print(f"micro_batch_size={args.micro_batch_size}", flush=True)
    print(f"order={args.order}", flush=True)
    print(f"llm_kwargs={llm_kwargs}", flush=True)

    llm = LLM(**llm_kwargs)
    tokens_prompt_cls = load_tokens_prompt_cls()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exact_count = 0
    total = 0
    prefix_lens: list[int] = []
    same_ratios: list[float] = []

    with out_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(rows), args.micro_batch_size):
            chunk = rows[start : start + args.micro_batch_size]
            payloads = [
                torch.load(row["student_vllm_io"], map_location="cpu", weights_only=False)
                for row in chunk
            ]
            prompts = [build_prompt(payload, tokens_prompt_cls) for payload in payloads]
            # This smoke tool keeps one SamplingParams object per micro-batch. Use
            # micro_batch_size=1 for exact per-request SamplingParams replay.
            sampling_kwargs = sampling_kwargs_from_dump(
                payloads[0],
                default_top_p=args.default_top_p,
                default_top_k=args.default_top_k,
            )
            sampling_params = make_sampling_params(SamplingParams, sampling_kwargs)
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            for row, payload, output in zip(chunk, payloads, outputs, strict=True):
                new_ids = list(getattr(output.outputs[0], "token_ids", []) or []) if output.outputs else []
                old_ids = [int(x) for x in as_list(get_nested(payload, "output", "generated_token_ids"))]
                cmp_result = compare_ids(new_ids, old_ids)
                exact_count += int(cmp_result["exact_same"])
                total += 1
                prefix_lens.append(int(cmp_result["same_prefix_len"]))
                same_ratios.append(float(cmp_result["same_ratio_common"]))
                record = {
                    "trace": row.get("trace"),
                    "trace_i": row.get("trace_i"),
                    "row": row.get("row"),
                    "student_vllm_io": row.get("student_vllm_io"),
                    "sampling_kwargs": sampling_kwargs,
                    **cmp_result,
                }
                if args.save_token_ids:
                    record["new_generated_token_ids"] = [int(x) for x in new_ids]
                    record["old_generated_token_ids"] = [int(x) for x in old_ids]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"idx={total - 1} trace_i={record['trace_i']} row={record['row']} "
                    f"exact={cmp_result['exact_same']} prefix={cmp_result['same_prefix_len']} "
                    f"new_len={cmp_result['new_len']} old_len={cmp_result['old_len']} "
                    f"same_ratio={cmp_result['same_ratio_common']:.6f} "
                    f"file={Path(str(row.get('student_vllm_io'))).name}",
                    flush=True,
                )

    print(f"output={out_path}", flush=True)
    print(f"exact_same={exact_count}/{total}", flush=True)
    print(f"mean_prefix_len={sum(prefix_lens) / max(len(prefix_lens), 1):.6f}", flush=True)
    print(f"mean_same_ratio_common={sum(same_ratios) / max(len(same_ratios), 1):.6f}", flush=True)
    print("RESULT =", "OK" if exact_count == total else "DIFF", flush=True)


if __name__ == "__main__":
    main()
