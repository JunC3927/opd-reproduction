import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch

from compare_student_vllm_generate_with_verl_io import (
    build_prompt,
    compare_ids,
    get_nested,
    load_jsonl,
    load_tokens_prompt_cls,
    make_sampling_params,
    sampling_kwargs_from_dump,
)


def load_payload(row: dict[str, Any]) -> dict[str, Any]:
    return torch.load(row["student_vllm_io"], map_location="cpu", weights_only=False)


def make_prompt_from_payload(payload: dict[str, Any], tokens_prompt_cls: Any) -> Any:
    return build_prompt(copy.deepcopy(payload), tokens_prompt_cls)


def ids_from_output(output: Any) -> list[int]:
    if not getattr(output, "outputs", None):
        return []
    return [int(x) for x in (getattr(output.outputs[0], "token_ids", []) or [])]


def print_compare(name: str, left: list[int], right: list[int]) -> dict[str, Any]:
    result = compare_ids(left, right)
    print(
        f"{name}: exact={result['exact_same']} prefix={result['same_prefix_len']} "
        f"same_ratio={result['same_ratio_common']:.6f} "
        f"left_len={result['new_len']} right_len={result['old_len']} "
        f"first_diff={result['first_diff']}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether vLLM gives identical outputs for identical final prompts."
    )
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--context-count", type=int, default=3)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=1537)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-temperature", type=float, default=None)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--default-top-p", type=float, default=1.0)
    parser.add_argument("--default-top-k", type=int, default=-1)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    rows = [row for row in load_jsonl(args.alignment) if row.get("matched")]
    if not rows:
        raise RuntimeError("No matched rows found in alignment jsonl.")
    if args.sample_index < 0 or args.sample_index >= len(rows):
        raise IndexError(f"--sample-index {args.sample_index} out of range for {len(rows)} rows")

    target_row = rows[args.sample_index]
    context_rows = [row for idx, row in enumerate(rows) if idx != args.sample_index][: args.context_count]
    target_payload = load_payload(target_row)
    context_payloads = [load_payload(row) for row in context_rows]

    sampling_kwargs = sampling_kwargs_from_dump(
        target_payload,
        default_top_p=args.default_top_p,
        default_top_k=args.default_top_k,
    )
    if args.force_temperature is not None:
        sampling_kwargs["temperature"] = args.force_temperature
    if args.sampling_seed is not None:
        sampling_kwargs["seed"] = args.sampling_seed
    sampling_params = make_sampling_params(SamplingParams, sampling_kwargs)

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "seed": args.seed,
    }

    print("=== check vLLM same input consistency ===", flush=True)
    print(f"VLLM_BATCH_INVARIANT={os.environ.get('VLLM_BATCH_INVARIANT')}", flush=True)
    print(f"VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}", flush=True)
    print(f"alignment={args.alignment}", flush=True)
    print(f"sample_index={args.sample_index}", flush=True)
    print(f"student_vllm_io={target_row.get('student_vllm_io')}", flush=True)
    print(f"sampling_kwargs={sampling_kwargs}", flush=True)
    print(f"llm_kwargs={llm_kwargs}", flush=True)

    llm = LLM(**llm_kwargs)
    tokens_prompt_cls = load_tokens_prompt_cls()

    target_prompt_a = make_prompt_from_payload(target_payload, tokens_prompt_cls)
    target_prompt_b = make_prompt_from_payload(target_payload, tokens_prompt_cls)
    target_prompt_c = make_prompt_from_payload(target_payload, tokens_prompt_cls)
    target_prompt_d = make_prompt_from_payload(target_payload, tokens_prompt_cls)
    context_prompts = [make_prompt_from_payload(payload, tokens_prompt_cls) for payload in context_payloads]

    duplicate_outputs = llm.generate([target_prompt_a, target_prompt_b], sampling_params, use_tqdm=False)
    dup0 = ids_from_output(duplicate_outputs[0])
    dup1 = ids_from_output(duplicate_outputs[1])

    mixed_prompts = [target_prompt_c] + context_prompts + [target_prompt_d]
    mixed_outputs = llm.generate(mixed_prompts, sampling_params, use_tqdm=False)
    mix0 = ids_from_output(mixed_outputs[0])
    mix_last = ids_from_output(mixed_outputs[-1])

    seq_out_1 = ids_from_output(llm.generate([make_prompt_from_payload(target_payload, tokens_prompt_cls)], sampling_params, use_tqdm=False)[0])
    seq_out_2 = ids_from_output(llm.generate([make_prompt_from_payload(target_payload, tokens_prompt_cls)], sampling_params, use_tqdm=False)[0])

    records = {
        "env": {
            "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
            "VLLM_USE_V1": os.environ.get("VLLM_USE_V1"),
        },
        "sample_index": args.sample_index,
        "student_vllm_io": target_row.get("student_vllm_io"),
        "sampling_kwargs": sampling_kwargs,
        "duplicate_same_call": print_compare("duplicate_same_call", dup0, dup1),
        "mixed_same_call": print_compare("mixed_same_call", mix0, mix_last),
        "sequential_same_engine": print_compare("sequential_same_engine", seq_out_1, seq_out_2),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"output={out_path}", flush=True)
    ok = all(
        records[key]["exact_same"]
        for key in ("duplicate_same_call", "mixed_same_call", "sequential_same_engine")
    )
    print("RESULT =", "OK" if ok else "DIFF", flush=True)


if __name__ == "__main__":
    main()
