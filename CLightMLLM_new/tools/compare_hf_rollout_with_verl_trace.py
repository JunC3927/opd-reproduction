import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from transformers import GenerationConfig

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from replay_verl_opd_trace import (  # noqa: E402
    build_mm_kwargs,
    build_mm_token_type_ids,
    expand_paths,
    load_trace,
    normalize_mm_inputs,
    parse_yaml_args,
    slice_rows,
    sync_cuda,
)
from src.model import ModelTuner, load_vision_language_model  # noqa: E402


def active_response_ids(responses: torch.Tensor, response_mask: torch.Tensor) -> list[list[int]]:
    out = []
    for row, mask in zip(responses, response_mask, strict=True):
        out.append(row[mask.bool()].detach().cpu().tolist())
    return out


def clean_generated_ids(ids: torch.Tensor, *, pad_token_id: int | None, eos_token_id: int | None) -> list[int]:
    result = []
    for token in ids.detach().cpu().tolist():
        token = int(token)
        if pad_token_id is not None and token == int(pad_token_id):
            break
        result.append(token)
        if eos_token_id is not None and token == int(eos_token_id):
            break
    return result


def prefix_match_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        count += 1
    return count


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def make_generation_config(method_args: Any, tokenizer: Any, args: argparse.Namespace) -> GenerationConfig:
    do_sample = method_args.rollout_do_sample if args.do_sample is None else args.do_sample
    max_new_tokens = args.max_new_tokens or method_args.rollout_max_new_tokens
    temperature = method_args.rollout_temperature if args.temperature is None else args.temperature
    top_p = method_args.rollout_top_p if args.top_p is None else args.top_p
    top_k = method_args.rollout_top_k if args.top_k is None else args.top_k

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
    return GenerationConfig(**kwargs)


def decode(tokenizer: Any, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True).replace("\n", "\\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CLight HF generate rollout with verl trace vLLM rollout.")
    parser.add_argument("--config", required=True)
    parser.add_argument("traces", nargs="+", help="verl OPD trace dump file(s) or glob pattern(s).")
    parser.add_argument("--model-path", default=None, help="Optional model path override. Defaults to config model.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--output", default=None, help="Optional JSONL path with per-sample comparison rows.")
    args = parser.parse_args()

    os.chdir(ROOT)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    (
        _cl_sft_args,
        data_args,
        _loader_args,
        method_args,
        model_args,
        _optimizer_args,
        _trainer_args,
        tuning_args,
    ) = parse_yaml_args(args.config)
    if args.model_path:
        model_args = replace(model_args, model_name_or_path=args.model_path)
    model_args = replace(model_args, use_cache=True)

    paths = expand_paths(args.traces)
    if args.max_files is not None:
        paths = paths[: args.max_files]
    if not paths:
        raise FileNotFoundError(f"No trace files matched: {args.traces}")

    device = torch.device(args.device)
    model, _processor, tokenizer = load_vision_language_model(model_args, data_args.template)
    model = ModelTuner(tuning_args).apply(model)
    model.to(device)
    model.eval()
    generation_config = make_generation_config(method_args, tokenizer, args)
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    print("=== hf rollout vs verl trace rollout ===")
    print(f"config={args.config}")
    print(f"model={model_args.model_name_or_path}")
    print(f"trace_count={len(paths)}")
    print(f"device={device}")
    print(f"micro_batch_size={args.micro_batch_size}")
    print(f"generation_config={generation_config.to_dict()}")

    output = open(args.output, "w", encoding="utf-8") if args.output else None
    rows: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for file_idx, path in enumerate(paths):
                payload = load_trace(path)
                batch = payload["batch"]
                non_tensor = payload.get("non_tensor_batch", {})
                prompts_cpu = batch["prompts"]
                responses_cpu = batch["responses"]
                response_mask_cpu = batch["response_mask"]
                attention_mask_cpu = batch.get("attention_mask")
                mm_inputs = normalize_mm_inputs(non_tensor.get("multi_modal_inputs"))

                batch_size = int(prompts_cpu.shape[0])
                if args.max_samples is not None:
                    batch_size = min(batch_size, args.max_samples)
                prompt_width = int(prompts_cpu.shape[1])
                verl_active = active_response_ids(responses_cpu[:batch_size], response_mask_cpu[:batch_size])

                for start in range(0, batch_size, args.micro_batch_size):
                    end = min(start + args.micro_batch_size, batch_size)
                    prompt_ids = slice_rows(prompts_cpu, start, end, device)
                    if attention_mask_cpu is not None:
                        prompt_attention_mask = slice_rows(attention_mask_cpu[:, :prompt_width], start, end, device)
                    elif pad_token_id is not None:
                        prompt_attention_mask = prompt_ids.ne(int(pad_token_id)).long()
                    else:
                        prompt_attention_mask = torch.ones_like(prompt_ids)

                    forward_kwargs: dict[str, Any] = {
                        "input_ids": prompt_ids,
                        "attention_mask": prompt_attention_mask,
                    }
                    forward_kwargs.update(build_mm_kwargs(mm_inputs, start, end, device))
                    mm_token_type_ids = build_mm_token_type_ids(model, prompt_ids)
                    if mm_token_type_ids is not None and "image_grid_thw" in forward_kwargs:
                        forward_kwargs["mm_token_type_ids"] = mm_token_type_ids

                    sequences = model.generate(**forward_kwargs, generation_config=generation_config)
                    sync_cuda(device, f"file {file_idx} rows {start}:{end} generate")

                    for local_idx, sequence in enumerate(sequences):
                        row_idx = start + local_idx
                        hf_ids = clean_generated_ids(
                            sequence[prompt_width:],
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id,
                        )
                        verl_ids = verl_active[row_idx]
                        common = prefix_match_len(hf_ids, verl_ids)
                        row = {
                            "file_index": file_idx,
                            "path": path,
                            "global_step": payload.get("global_steps"),
                            "row": row_idx,
                            "hf_ids": hf_ids,
                            "verl_ids": verl_ids,
                            "hf_len": len(hf_ids),
                            "verl_len": len(verl_ids),
                            "len_diff": len(hf_ids) - len(verl_ids),
                            "first_token_same": bool(hf_ids and verl_ids and hf_ids[0] == verl_ids[0]),
                            "exact_same": hf_ids == verl_ids,
                            "prefix_match_len": common,
                            "prefix_match_ratio": common / max(min(len(hf_ids), len(verl_ids)), 1),
                            "hf_text": decode(tokenizer, hf_ids),
                            "verl_text": decode(tokenizer, verl_ids),
                        }
                        rows.append(row)
                        if output is not None:
                            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if output is not None:
            output.close()

    hf_lens = [float(row["hf_len"]) for row in rows]
    verl_lens = [float(row["verl_len"]) for row in rows]
    exact = [1.0 if row["exact_same"] else 0.0 for row in rows]
    first = [1.0 if row["first_token_same"] else 0.0 for row in rows]
    prefix = [float(row["prefix_match_ratio"]) for row in rows]
    len_abs = [abs(float(row["len_diff"])) for row in rows]

    print("=== summary ===")
    print(f"samples={len(rows)}")
    print(f"hf_len_mean={mean(hf_lens):.6f}")
    print(f"verl_len_mean={mean(verl_lens):.6f}")
    print(f"len_abs_diff_mean={mean(len_abs):.6f}")
    print(f"first_token_same_rate={mean(first):.6f}")
    print(f"exact_same_rate={mean(exact):.6f}")
    print(f"prefix_match_ratio_mean={mean(prefix):.6f}")

    print("=== examples ===")
    for row in rows[: args.examples]:
        print(
            " | ".join(
                [
                    f"file={row['file_index']}",
                    f"row={row['row']}",
                    f"hf_len={row['hf_len']}",
                    f"verl_len={row['verl_len']}",
                    f"first_same={row['first_token_same']}",
                    f"prefix={row['prefix_match_ratio']:.3f}",
                ]
            )
        )
        print(f"  hf:   {row['hf_text'][:300]}")
        print(f"  verl: {row['verl_text'][:300]}")

    print("compare_hf_rollout_with_verl_trace_ok=True")


if __name__ == "__main__":
    main()
