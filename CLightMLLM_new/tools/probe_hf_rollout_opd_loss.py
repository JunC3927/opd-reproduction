import argparse
import json
import os
import sys
from contextlib import nullcontext
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
    compute_topk_loss_from_logits,
    expand_paths,
    load_trace,
    normalize_mm_inputs,
    parse_yaml_args,
    sanitize_teacher_ids,
    slice_rows,
    sync_cuda,
)
from src.method.rollout import RolloutMixin  # noqa: E402
from src.method.vllm_teacher_client import RemoteTeacherScorer  # noqa: E402
from src.model import ModelTuner, load_vision_language_model  # noqa: E402


class _Probe(RolloutMixin):
    AUX_BATCH_KEYS = {
        "prompt_input_ids",
        "prompt_attention_mask",
        "reference_text",
    }

    def __init__(self, model: torch.nn.Module, tokenizer: Any, method_args: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.method_args = method_args

    def model_kwargs(self, batch: dict[str, Any], include_labels: bool = True) -> dict[str, Any]:
        return {key: value for key, value in batch.items() if key not in self.AUX_BATCH_KEYS}


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


def active_completion_mask(sequences: torch.Tensor, prompt_width: int, pad_token_id: int, eos_token_id: int | None) -> torch.Tensor:
    completion_ids = sequences[:, prompt_width:]
    mask = completion_ids.ne(int(pad_token_id))
    if eos_token_id is not None:
        eos_seen = completion_ids.eq(int(eos_token_id)).cumsum(dim=1).bool()
        before_or_at_first_eos = torch.cat(
            [torch.ones_like(mask[:, :1], dtype=torch.bool), ~eos_seen[:, :-1]],
            dim=1,
        )
        mask = mask & before_or_at_first_eos
    return mask


def build_prompt_batch(
    *,
    payload: dict[str, Any],
    start: int,
    end: int,
    prompt_width: int,
    device: torch.device,
) -> dict[str, Any]:
    batch = payload["batch"]
    non_tensor = payload.get("non_tensor_batch", {})
    mm_inputs = normalize_mm_inputs(non_tensor.get("multi_modal_inputs"))
    prompt_ids = slice_rows(batch["prompts"], start, end, device)
    attention_mask = slice_rows(batch["attention_mask"][:, :prompt_width], start, end, device)
    out: dict[str, Any] = {
        "prompt_input_ids": prompt_ids,
        "prompt_attention_mask": attention_mask,
    }
    out.update(build_mm_kwargs(mm_inputs, start, end, device))
    return out


def compute_loss_for_sequences(
    *,
    model: torch.nn.Module,
    probe: _Probe,
    teacher: RemoteTeacherScorer,
    batch: dict[str, Any],
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    log_prob_min_clamp: float | None,
    loss_max_clamp: float | None,
    amp_dtype: str,
) -> dict[str, torch.Tensor]:
    model_kwargs = probe.sequence_model_kwargs(batch, sequences, attention_mask)
    with torch.no_grad():
        teacher_topk_logps, teacher_topk_ids = teacher.score(
            sequences=sequences,
            attention_mask=attention_mask,
            images_per_sample=None,
            image_token_id=getattr(getattr(model, "config", None), "image_token_id", None),
            video_token_id=getattr(getattr(model, "config", None), "video_token_id", None),
            pad_token_id=probe.tokenizer.pad_token_id,
            model_kwargs=model_kwargs,
        )

    device = sequences.device
    if amp_dtype == "bf16":
        autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    elif amp_dtype == "fp16":
        autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")
    else:
        autocast_ctx = nullcontext()
    with torch.no_grad(), autocast_ctx:
        outputs = model(**model_kwargs)
    sync_cuda(device, "student forward for hf rollout loss")

    loss_outputs = compute_topk_loss_from_logits(
        student_logits=outputs.logits[:, :-1, :],
        teacher_logps=teacher_topk_logps,
        teacher_ids=teacher_topk_ids,
        response_mask=response_mask,
        log_prob_min_clamp=log_prob_min_clamp,
        loss_max_clamp=loss_max_clamp,
    )
    token_count = response_mask.sum().clamp_min(1.0)
    return {
        "loss": loss_outputs["loss_num"] / token_count,
        "teacher_mass": (loss_outputs["teacher_mass"] * response_mask).sum() / token_count,
        "student_mass": (loss_outputs["student_mass"] * response_mask).sum() / token_count,
        "topk_overlap": ((loss_outputs["overlap_count"].float() / teacher_topk_ids.shape[-1]) * response_mask).sum()
        / token_count,
        "tokens": response_mask.sum(),
    }


def compute_trace_loss(
    *,
    model: torch.nn.Module,
    payload: dict[str, Any],
    start: int,
    end: int,
    device: torch.device,
    prompt_width: int,
    teacher_shift_offset: int,
    log_prob_min_clamp: float | None,
    loss_max_clamp: float | None,
    amp_dtype: str,
) -> dict[str, torch.Tensor]:
    batch = payload["batch"]
    non_tensor = payload.get("non_tensor_batch", {})
    mm_inputs = normalize_mm_inputs(non_tensor.get("multi_modal_inputs"))
    sequences = slice_rows(batch["input_ids"], start, end, device)
    attention_mask = slice_rows(batch["attention_mask"], start, end, device)
    response_mask = slice_rows(batch["response_mask"].float(), start, end, device)
    response_start = prompt_width + teacher_shift_offset
    current_response_len = min(
        response_mask.shape[1],
        sequences.shape[1] - 1 - response_start,
        batch["teacher_ids"].shape[1] - response_start,
    )
    response_mask = response_mask[:, :current_response_len]

    forward_kwargs: dict[str, Any] = {
        "input_ids": sequences,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    forward_kwargs.update(build_mm_kwargs(mm_inputs, start, end, device))
    mm_token_type_ids = build_mm_token_type_ids(model, sequences)
    if mm_token_type_ids is not None and "image_grid_thw" in forward_kwargs:
        forward_kwargs["mm_token_type_ids"] = mm_token_type_ids

    if amp_dtype == "bf16":
        autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    elif amp_dtype == "fp16":
        autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")
    else:
        autocast_ctx = nullcontext()
    with torch.no_grad(), autocast_ctx:
        outputs = model(**forward_kwargs)
    sync_cuda(device, "student forward for trace loss")

    teacher_ids = batch["teacher_ids"][start:end, response_start : response_start + current_response_len, :]
    teacher_ids = sanitize_teacher_ids(
        teacher_ids=teacher_ids.long(),
        response_mask=response_mask.cpu(),
        vocab_size=int(getattr(model.config, "vocab_size", model.get_input_embeddings().weight.shape[0])),
        file_idx=int(payload.get("dump_index") or 0),
        row_start=start,
        response_start=response_start,
    ).to(device)
    teacher_logps = batch["teacher_logprobs"][start:end, response_start : response_start + current_response_len, :].to(device)
    loss_outputs = compute_topk_loss_from_logits(
        student_logits=outputs.logits[:, :-1, response_start : response_start + current_response_len, :]
        if False
        else outputs.logits[:, :-1, :][:, response_start : response_start + current_response_len, :],
        teacher_logps=teacher_logps,
        teacher_ids=teacher_ids,
        response_mask=response_mask,
        log_prob_min_clamp=log_prob_min_clamp,
        loss_max_clamp=loss_max_clamp,
    )
    token_count = response_mask.sum().clamp_min(1.0)
    return {
        "loss": loss_outputs["loss_num"] / token_count,
        "teacher_mass": (loss_outputs["teacher_mass"] * response_mask).sum() / token_count,
        "student_mass": (loss_outputs["student_mass"] * response_mask).sum() / token_count,
        "topk_overlap": ((loss_outputs["overlap_count"].float() / teacher_ids.shape[-1]) * response_mask).sum() / token_count,
        "tokens": response_mask.sum(),
    }


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HF rollout, score it with teacher server, and compare OPD loss to verl trace.")
    parser.add_argument("--config", required=True)
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher-host", default=None)
    parser.add_argument("--teacher-port", type=int, default=None)
    parser.add_argument("--teacher-timeout", type=float, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--teacher-shift-offset", type=int, default=-1)
    parser.add_argument("--log-prob-min-clamp", type=float, default=-10.0)
    parser.add_argument("--loss-max-clamp", type=float, default=10.0)
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--output", default=None)
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
    model_args = replace(model_args, use_cache=False)
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
    probe = _Probe(model=model, tokenizer=tokenizer, method_args=method_args)
    generation_config = make_generation_config(method_args, tokenizer, args)
    teacher = RemoteTeacherScorer(
        host=args.teacher_host or method_args.opd_teacher_server_host,
        port=args.teacher_port or method_args.opd_teacher_server_port,
        timeout=args.teacher_timeout or method_args.opd_teacher_server_timeout,
        topk=method_args.opd_topk,
    )

    print("=== hf rollout opd loss probe ===")
    print(f"config={args.config}")
    print(f"trace_count={len(paths)}")
    print(f"device={device}")
    print(f"teacher={teacher.host}:{teacher.port}")
    print(f"micro_batch_size={args.micro_batch_size}")
    print(f"generation_config={generation_config.to_dict()}")

    output = open(args.output, "w", encoding="utf-8") if args.output else None
    rows: list[dict[str, Any]] = []
    try:
        for file_idx, path in enumerate(paths):
            payload = load_trace(path)
            batch = payload["batch"]
            batch_size = int(batch["prompts"].shape[0])
            if args.max_samples is not None:
                batch_size = min(batch_size, args.max_samples)
            prompt_width = int(batch["prompts"].shape[1])

            for start in range(0, batch_size, args.micro_batch_size):
                end = min(start + args.micro_batch_size, batch_size)
                prompt_batch = build_prompt_batch(
                    payload=payload,
                    start=start,
                    end=end,
                    prompt_width=prompt_width,
                    device=device,
                )
                with torch.no_grad():
                    sequences = model.generate(
                        **probe.prompt_model_kwargs(prompt_batch),
                        generation_config=generation_config,
                    )
                sync_cuda(device, f"file {file_idx} rows {start}:{end} generate")
                completion_mask = active_completion_mask(
                    sequences,
                    prompt_width,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                attention_mask = probe.sequence_attention_mask(prompt_batch, sequences, completion_mask)
                response_mask = probe.shift_completion_mask(
                    token_values=sequences[:, 1:],
                    completion_mask=completion_mask,
                    prompt_width=prompt_width,
                )
                hf_metrics = compute_loss_for_sequences(
                    model=model,
                    probe=probe,
                    teacher=teacher,
                    batch=prompt_batch,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    response_mask=response_mask,
                    log_prob_min_clamp=args.log_prob_min_clamp,
                    loss_max_clamp=args.loss_max_clamp,
                    amp_dtype=args.amp_dtype,
                )
                trace_metrics = compute_trace_loss(
                    model=model,
                    payload=payload,
                    start=start,
                    end=end,
                    device=device,
                    prompt_width=prompt_width,
                    teacher_shift_offset=args.teacher_shift_offset,
                    log_prob_min_clamp=args.log_prob_min_clamp,
                    loss_max_clamp=args.loss_max_clamp,
                    amp_dtype=args.amp_dtype,
                )
                row = {
                    "format": "hf_rollout_opd_loss_probe_v1",
                    "file_index": file_idx,
                    "path": path,
                    "dump_index": payload.get("dump_index"),
                    "global_step": payload.get("global_steps"),
                    "row_start": start,
                    "row_end": end,
                    "hf_loss": scalar(hf_metrics["loss"]),
                    "trace_loss": scalar(trace_metrics["loss"]),
                    "hf_teacher_mass": scalar(hf_metrics["teacher_mass"]),
                    "trace_teacher_mass": scalar(trace_metrics["teacher_mass"]),
                    "hf_student_mass": scalar(hf_metrics["student_mass"]),
                    "trace_student_mass": scalar(trace_metrics["student_mass"]),
                    "hf_topk_overlap": scalar(hf_metrics["topk_overlap"]),
                    "trace_topk_overlap": scalar(trace_metrics["topk_overlap"]),
                    "hf_tokens": int(scalar(hf_metrics["tokens"])),
                    "trace_tokens": int(scalar(trace_metrics["tokens"])),
                }
                rows.append(row)
                print(
                    " | ".join(
                        [
                            f"file={file_idx}",
                            f"rows={start}:{end}",
                            f"hf_loss={row['hf_loss']:.6f}",
                            f"trace_loss={row['trace_loss']:.6f}",
                            f"hf_tokens={row['hf_tokens']}",
                            f"trace_tokens={row['trace_tokens']}",
                            f"hf_overlap={row['hf_topk_overlap']:.6f}",
                            f"trace_overlap={row['trace_topk_overlap']:.6f}",
                        ]
                    ),
                    flush=True,
                )
                if output is not None:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
    finally:
        if output is not None:
            output.close()

    print("=== summary ===")
    for key in (
        "hf_loss",
        "trace_loss",
        "hf_teacher_mass",
        "trace_teacher_mass",
        "hf_student_mass",
        "trace_student_mass",
        "hf_topk_overlap",
        "trace_topk_overlap",
        "hf_tokens",
        "trace_tokens",
    ):
        print(f"{key}_mean={mean([float(row[key]) for row in rows]):.6f}")
    if rows:
        diffs = [abs(float(row["hf_loss"]) - float(row["trace_loss"])) for row in rows]
        print(f"loss_abs_diff_mean={mean(diffs):.6f}")
        print(f"loss_abs_diff_max={max(diffs):.6f}")
    print("probe_hf_rollout_opd_loss_ok=True")


if __name__ == "__main__":
    main()
