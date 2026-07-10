import argparse
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replay_verl_opd_trace import (  # noqa: E402
    build_mm_kwargs,
    build_mm_token_type_ids,
    build_optimizer,
    build_position_ids,
    compute_topk_loss_from_logits,
    expand_paths,
    finish_swanlab,
    init_swanlab,
    load_trace,
    log_swanlab_metrics,
    model_grad_dtype_counts,
    model_param_dtype_counts,
    normalize_mm_inputs,
    optimizer_state_dtype_counts,
    parse_yaml_args,
    sanitize_teacher_ids,
    sync_cuda,
    trainable_parameter_summary,
    validate_token_ids,
)
from src.model import ModelTuner, load_vision_language_model  # noqa: E402


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def rank_print(*args: Any, **kwargs: Any) -> None:
    if is_rank0():
        print(*args, **kwargs)


def init_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP replay requires CUDA.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    return rank, world_size, local_rank, device


def split_contiguous_rows(batch_size: int, rank: int, world_size: int) -> tuple[int, int]:
    if batch_size % world_size != 0:
        raise ValueError(
            f"FSDP replay needs equal row counts on every rank to keep collectives aligned. "
            f"Got batch_size={batch_size}, world_size={world_size}. "
            "Use a world size that divides the trace sample count; chunk12 works with 3, 4, or 6 GPUs."
        )
    per_rank = batch_size // world_size
    return rank * per_rank, (rank + 1) * per_rank


def reduce_sum(value: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def save_fsdp_hf_model(model: FSDP, base_model: torch.nn.Module, processor: Any, output_dir: str) -> None:
    save_path = Path(output_dir)
    if is_rank0():
        save_path.mkdir(parents=True, exist_ok=True)

    dist.barrier()
    state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_cfg):
        state_dict = model.state_dict()

    if is_rank0():
        base_model.eval()
        base_model.save_pretrained(str(save_path), state_dict=state_dict, safe_serialization=True)
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(str(save_path))
        with (save_path / "clight_fsdp_replay_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "format": "clight_verl_trace_replay_fsdp_model_v1",
                    "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"saved_fsdp_replay_model={save_path}", flush=True)
    dist.barrier()


def select_fsdp_update_probes(
    model: torch.nn.Module,
    *,
    samples_per_param: int = 64,
    max_params: int = 1,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    probes: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        flat = param.detach().flatten()
        if flat.numel() == 0:
            continue
        sample_count = min(int(samples_per_param), int(flat.numel()))
        if sample_count == int(flat.numel()):
            idx = torch.arange(sample_count, device=flat.device, dtype=torch.long)
        else:
            idx = torch.linspace(0, flat.numel() - 1, steps=sample_count, device=flat.device).long().unique()
        probes.append((name, idx, flat[idx].float().clone()))
        if len(probes) >= max_params:
            break
    if not probes:
        raise RuntimeError("No trainable parameter found for FSDP update probe.")
    return probes


def compute_fsdp_update_stats(
    model: torch.nn.Module,
    probes: list[tuple[str, torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    if not probes:
        return {
            "param_update_max_abs": float("nan"),
            "param_update_mean_abs": float("nan"),
            "param_update_rel_mean": float("nan"),
        }
    params = dict(model.named_parameters())
    device = probes[0][1].device
    local_max = torch.tensor(0.0, device=device)
    local_sum = torch.tensor(0.0, device=device)
    local_before_abs_sum = torch.tensor(0.0, device=device)
    local_count = torch.tensor(0.0, device=device)

    for name, idx, before in probes:
        param = params.get(name)
        if param is None:
            continue
        after = param.detach().flatten()[idx].float()
        before = before.to(after.device)
        delta = (after - before).abs()
        local_max = torch.maximum(local_max, delta.max())
        local_sum += delta.sum()
        local_before_abs_sum += before.abs().sum()
        local_count += float(delta.numel())

    dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_before_abs_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

    mean_abs = local_sum / local_count.clamp_min(1.0)
    before_mean_abs = local_before_abs_sum / local_count.clamp_min(1.0)
    return {
        "param_update_max_abs": float(local_max.detach().cpu().item()),
        "param_update_mean_abs": float(mean_abs.detach().cpu().item()),
        "param_update_rel_mean": float((mean_abs / before_mean_abs.clamp_min(1e-12)).detach().cpu().item()),
    }


def format_update_probe_names(probes: list[tuple[str, torch.Tensor, torch.Tensor]]) -> str:
    if not probes:
        return ""
    return ",".join(name for name, _, _ in probes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay verl OPD trace dumps with FSDP-sharded CLight student.")
    parser.add_argument("--config", required=True)
    parser.add_argument("traces", nargs="+", help="Trace dump file(s) or glob pattern(s).")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--train", action="store_true", help="Run backward + optimizer step for each trace file.")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--position-ids-mode",
        choices=("none", "trace3", "trace4", "trace_batch4"),
        default="none",
        help="Use none+mm_token_type_ids for official HF, or pass verl-dumped position_ids.",
    )
    parser.add_argument("--teacher-shift-offset", type=int, default=-1)
    parser.add_argument("--log-prob-min-clamp", type=float, default=-10.0)
    parser.add_argument("--loss-max-clamp", type=float, default=10.0)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "bf16", "fp16"),
        default="bf16",
        help="CUDA autocast dtype for forward. bf16 is the expected A100 setting.",
    )
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--swanlab-project", default=None)
    parser.add_argument("--swanlab-experiment-name", default=None)
    parser.add_argument("--swanlab-workspace", default=None)
    parser.add_argument("--swanlab-mode", default=None)
    parser.add_argument("--swanlab-logdir", default=None)
    parser.add_argument("--save-model-dir", default=None, help="Optional HF directory for the replay-updated student.")
    parser.add_argument("--debug-dtypes", action="store_true")
    parser.add_argument(
        "--fsdp-min-num-params",
        type=int,
        default=10_000_000,
        help="Size-based auto-wrap threshold. Lower wraps more modules and lowers peak all-gather memory.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable model gradient checkpointing before FSDP wrapping.",
    )
    parser.add_argument(
        "--disable-update-probe",
        action="store_true",
        help="Skip sampled parameter-delta diagnostics. This does not affect training.",
    )
    parser.add_argument(
        "--update-probe-samples",
        type=int,
        default=64,
        help="Number of values sampled per FSDP local parameter shard for update diagnostics.",
    )
    parser.add_argument(
        "--update-probe-max-params",
        type=int,
        default=1,
        help="Maximum number of FSDP local parameter shards sampled for update diagnostics.",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rank, world_size, local_rank, device = init_distributed()

    (
        _cl_sft_args,
        data_args,
        _loader_args,
        method_args,
        model_args,
        optimizer_args,
        _trainer_args,
        tuning_args,
    ) = parse_yaml_args(args.config)
    if args.learning_rate is not None:
        optimizer_args = replace(optimizer_args, learning_rate=args.learning_rate)
    model_args = replace(model_args, use_cache=False)
    if args.gradient_checkpointing and hasattr(model_args, "gradient_checkpointing"):
        model_args = replace(model_args, gradient_checkpointing=True)

    paths = expand_paths(args.traces)
    if args.max_files is not None:
        paths = paths[: args.max_files]
    if not paths:
        raise FileNotFoundError(f"No trace files matched: {args.traces}")

    base_model, processor, _tokenizer = load_vision_language_model(model_args, data_args.template)
    base_model = ModelTuner(tuning_args).apply(base_model)
    base_model.train(args.train)
    trainable, total = trainable_parameter_summary(base_model)
    vocab_size = int(
        getattr(getattr(base_model, "config", None), "vocab_size", 0)
        or base_model.get_input_embeddings().weight.shape[0]
    )

    auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=args.fsdp_min_num_params)
    model = FSDP(
        base_model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        limit_all_gathers=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )
    sync_cuda(device, "FSDP wrap")
    optimizer = build_optimizer(model, optimizer_args) if args.train else None

    rank_print("=== replay verl opd trace fsdp ===")
    rank_print(f"config={args.config}")
    rank_print(f"trace_count={len(paths)}")
    rank_print(f"world_size={world_size}")
    rank_print(f"local_rank={local_rank}")
    rank_print(f"train={args.train}")
    rank_print(f"micro_batch_size={args.micro_batch_size}")
    rank_print(f"position_ids_mode={args.position_ids_mode}")
    rank_print(f"teacher_shift_offset={args.teacher_shift_offset}")
    rank_print(f"amp_dtype={args.amp_dtype}")
    rank_print(f"learning_rate={optimizer_args.learning_rate}")
    rank_print(f"fsdp_min_num_params={args.fsdp_min_num_params}")
    rank_print(f"gradient_checkpointing={args.gradient_checkpointing}")
    rank_print(f"trainable_params={trainable} total_params={total}")
    rank_print(f"student_vocab_size={vocab_size}")
    if args.debug_dtypes and is_rank0():
        print(f"[dtype] fsdp_param_dtypes={model_param_dtype_counts(model)}", flush=True)
        print(f"[dtype] fsdp_trainable_param_dtypes={model_param_dtype_counts(model, trainable_only=True)}", flush=True)
        if optimizer is not None:
            print(f"[dtype] optimizer_defaults={optimizer.defaults}", flush=True)

    metrics_output = None
    if args.metrics_output and is_rank0():
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_output = open(metrics_path, "w", encoding="utf-8")

    swanlab_run = None
    if is_rank0():
        swanlab_run = init_swanlab(
            args,
            {
                "config": args.config,
                "trace_count": len(paths),
                "world_size": world_size,
                "train": bool(args.train),
                "micro_batch_size": args.micro_batch_size,
                "position_ids_mode": args.position_ids_mode,
                "teacher_shift_offset": args.teacher_shift_offset,
                "amp_dtype": args.amp_dtype,
                "learning_rate": optimizer_args.learning_rate,
                "grad_clip": args.grad_clip,
                "fsdp_min_num_params": args.fsdp_min_num_params,
                "gradient_checkpointing": args.gradient_checkpointing,
                "trainable_params": trainable,
                "total_params": total,
            },
        )

    try:
        for file_idx, path in enumerate(paths):
            payload = load_trace(path)
            batch = payload["batch"]
            non_tensor = payload.get("non_tensor_batch", {})
            meta = payload.get("meta_info", {})

            input_ids_cpu = batch["input_ids"]
            attention_mask_cpu = batch["attention_mask"]
            response_mask_cpu = batch["response_mask"].float()
            teacher_ids_cpu = batch["teacher_ids"]
            teacher_logps_cpu = batch["teacher_logprobs"]
            position_ids_cpu = batch.get("position_ids")
            prompts_cpu = batch.get("prompts")
            mm_inputs = normalize_mm_inputs(non_tensor.get("multi_modal_inputs"))

            batch_size = int(input_ids_cpu.shape[0])
            seq_len = int(input_ids_cpu.shape[1])
            local_start, local_end = split_contiguous_rows(batch_size, rank, world_size)
            prompt_width = int(prompts_cpu.shape[1]) if torch.is_tensor(prompts_cpu) else seq_len - response_mask_cpu.shape[1]
            response_start = prompt_width + args.teacher_shift_offset
            response_len = int(response_mask_cpu.shape[1])
            if response_start < 0:
                raise ValueError(f"Invalid response_start={response_start} from prompt_width={prompt_width}.")

            validate_token_ids(f"file {file_idx} input_ids local", input_ids_cpu[local_start:local_end], vocab_size)
            teacher_check_len = min(response_len, teacher_ids_cpu.shape[1] - response_start)
            if teacher_check_len <= 0:
                raise ValueError(
                    f"Invalid teacher response slice for file {file_idx}: "
                    f"teacher_shape={tuple(teacher_ids_cpu.shape)}, response_start={response_start}"
                )
            active_teacher_ids = teacher_ids_cpu[
                local_start:local_end,
                response_start : response_start + teacher_check_len,
                :,
            ]
            active_teacher_mask = (
                response_mask_cpu[local_start:local_end, :teacher_check_len].bool().unsqueeze(-1).expand_as(active_teacher_ids)
            )
            validate_token_ids(f"file {file_idx} active teacher_ids local", active_teacher_ids, vocab_size, mask=active_teacher_mask)

            local_token_count = response_mask_cpu[local_start:local_end].sum().to(device)
            global_token_count = reduce_sum(local_token_count.detach().clone()).clamp_min(1.0)
            loss_num_metric = torch.tensor(0.0, device=device)
            teacher_mass_num = torch.tensor(0.0, device=device)
            student_mass_num = torch.tensor(0.0, device=device)
            overlap_num = torch.tensor(0.0, device=device)
            actual_token_count = torch.tensor(0.0, device=device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if args.disable_update_probe:
                    update_probes = []
                else:
                    update_probes = select_fsdp_update_probes(
                        model,
                        samples_per_param=args.update_probe_samples,
                        max_params=args.update_probe_max_params,
                    )
            else:
                update_probes = []

            for start in range(local_start, local_end, args.micro_batch_size):
                end = min(start + args.micro_batch_size, local_end)
                input_ids = input_ids_cpu[start:end].to(device)
                attention_mask = attention_mask_cpu[start:end].to(device)
                response_mask = response_mask_cpu[start:end].to(device)

                forward_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                }
                forward_kwargs.update(build_mm_kwargs(mm_inputs, start, end, device))
                mm_token_type_ids = build_mm_token_type_ids(base_model, input_ids)
                if mm_token_type_ids is not None and "image_grid_thw" in forward_kwargs:
                    forward_kwargs["mm_token_type_ids"] = mm_token_type_ids
                if position_ids_cpu is not None:
                    position_ids = build_position_ids(position_ids_cpu, args.position_ids_mode, start, end, device)
                    if position_ids is not None:
                        forward_kwargs["position_ids"] = position_ids

                if args.amp_dtype == "bf16":
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                elif args.amp_dtype == "fp16":
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
                else:
                    autocast_ctx = nullcontext()
                with autocast_ctx:
                    outputs = model(**forward_kwargs)
                sync_cuda(device, f"file {file_idx} rows {start}:{end} model forward")

                shifted_logits = outputs.logits[:, :-1, :]
                if args.debug_dtypes and file_idx == 0 and start == local_start and is_rank0():
                    debug_forward = {
                        "input_ids": str(input_ids.dtype),
                        "attention_mask": str(attention_mask.dtype),
                        "response_mask": str(response_mask.dtype),
                        "logits": str(outputs.logits.dtype),
                    }
                    for key, value in forward_kwargs.items():
                        if torch.is_tensor(value):
                            debug_forward[key] = str(value.dtype)
                    print(f"[dtype] first_forward={debug_forward}", flush=True)

                max_student_len = shifted_logits.shape[1] - response_start
                max_teacher_len = teacher_ids_cpu.shape[1] - response_start
                current_response_len = min(response_len, max_student_len, max_teacher_len)
                if current_response_len <= 0:
                    raise RuntimeError(
                        f"Empty response slice: response_start={response_start}, "
                        f"shifted_logits={tuple(shifted_logits.shape)}, teacher_ids={tuple(teacher_ids_cpu.shape)}"
                    )

                student_logits = shifted_logits[:, response_start : response_start + current_response_len, :]
                teacher_ids_cpu_slice = teacher_ids_cpu[start:end, response_start : response_start + current_response_len, :]
                teacher_logps_cpu_slice = teacher_logps_cpu[start:end, response_start : response_start + current_response_len, :]
                response_mask = response_mask[:, :current_response_len]
                teacher_ids_cpu_slice = sanitize_teacher_ids(
                    teacher_ids=teacher_ids_cpu_slice.long(),
                    response_mask=response_mask.cpu(),
                    vocab_size=vocab_size,
                    file_idx=file_idx,
                    row_start=start,
                    response_start=response_start,
                )
                teacher_ids = teacher_ids_cpu_slice.to(device)
                teacher_logps = teacher_logps_cpu_slice.to(device)

                loss_outputs = compute_topk_loss_from_logits(
                    student_logits=student_logits,
                    teacher_logps=teacher_logps,
                    teacher_ids=teacher_ids,
                    response_mask=response_mask,
                    log_prob_min_clamp=args.log_prob_min_clamp,
                    loss_max_clamp=args.loss_max_clamp,
                )
                loss_num = loss_outputs["loss_num"]
                loss_num_metric += loss_num.detach()
                # FSDP reduces gradients across ranks like data parallel training. Scale
                # local token sums by world size so the averaged gradient equals the
                # global token-normalized gradient for this trace file.
                micro_loss = loss_num * world_size / global_token_count

                if args.debug_dtypes and file_idx == 0 and start == local_start and is_rank0():
                    debug_loss = {
                        "student_logits": str(student_logits.dtype),
                        "teacher_logps": str(teacher_logps.dtype),
                        "teacher_ids": str(teacher_ids.dtype),
                        "loss_num": str(loss_num.dtype),
                        "micro_loss": str(micro_loss.dtype),
                        "token_loss": str(loss_outputs["token_loss"].dtype),
                        "student_mass": str(loss_outputs["student_mass"].dtype),
                        "teacher_mass": str(loss_outputs["teacher_mass"].dtype),
                    }
                    print(f"[dtype] first_loss={debug_loss}", flush=True)

                if args.train:
                    micro_loss.backward()
                    sync_cuda(device, f"file {file_idx} rows {start}:{end} backward")

                with torch.no_grad():
                    token_count = response_mask.sum()
                    actual_token_count += token_count
                    teacher_mass_num += (loss_outputs["teacher_mass"] * response_mask).sum()
                    student_mass_num += (loss_outputs["student_mass"] * response_mask).sum()
                    overlap_num += ((loss_outputs["overlap_count"] / teacher_ids.shape[-1]) * response_mask).sum()

                del outputs, shifted_logits, student_logits, teacher_ids, teacher_logps

            global_loss_num = reduce_sum(loss_num_metric.detach().clone())
            global_teacher_mass_num = reduce_sum(teacher_mass_num.detach().clone())
            global_student_mass_num = reduce_sum(student_mass_num.detach().clone())
            global_overlap_num = reduce_sum(overlap_num.detach().clone())
            global_actual_token_count = reduce_sum(actual_token_count.detach().clone()).clamp_min(1.0)
            loss_value = global_loss_num / global_token_count

            grad_value = torch.tensor(0.0, device=device)
            update_stats: dict[str, float] = {}
            if args.train:
                if args.debug_dtypes and file_idx == 0 and is_rank0():
                    print(f"[dtype] grad_dtypes_before_step={model_grad_dtype_counts(model)}", flush=True)
                if args.grad_clip is not None and args.grad_clip > 0:
                    grad_value = model.clip_grad_norm_(args.grad_clip).detach()
                optimizer.step()
                if args.debug_dtypes and file_idx == 0 and is_rank0():
                    print(f"[dtype] optimizer_state_dtypes_after_step={optimizer_state_dtype_counts(optimizer)}", flush=True)
                update_stats = compute_fsdp_update_stats(model, update_probes)

            if is_rank0():
                record = {
                    "format": "clight_verl_trace_replay_fsdp_metrics_v1",
                    "file_index": file_idx,
                    "replay_update_step": file_idx + 1,
                    "path": path,
                    "dump_index": payload.get("dump_index"),
                    "global_step": payload.get("global_steps"),
                    "chunk_index": payload.get("chunk_index"),
                    "chunk_count": payload.get("chunk_count"),
                    "source_sample_count": payload.get("source_sample_count"),
                    "samples": batch_size,
                    "local_samples_per_rank": local_end - local_start,
                    "world_size": world_size,
                    "tokens": int(global_actual_token_count.item()),
                    "loss": float(loss_value.detach().cpu().item()),
                    "teacher_mass": float((global_teacher_mass_num / global_actual_token_count).detach().cpu().item()),
                    "student_mass": float((global_student_mass_num / global_actual_token_count).detach().cpu().item()),
                    "topk_overlap": float((global_overlap_num / global_actual_token_count).detach().cpu().item()),
                    "train": bool(args.train),
                    "amp_dtype": args.amp_dtype,
                }
                if args.train:
                    record["grad_norm"] = float(grad_value.detach().cpu().item())
                    record["update_param"] = format_update_probe_names(update_probes)
                    record.update(update_stats)
                if meta:
                    global_token_num = meta.get("global_token_num")
                    if global_token_num is not None:
                        record["meta_global_token_num"] = global_token_num

                parts = [
                    f"file={file_idx}",
                    f"path={path}",
                    f"dump_index={payload.get('dump_index')}",
                    f"global_steps={payload.get('global_steps')}",
                    f"chunk_index={payload.get('chunk_index')}",
                    f"samples={batch_size}",
                    f"local_samples_per_rank={local_end - local_start}",
                    f"tokens={int(global_actual_token_count.item())}",
                    f"loss={record['loss']:.8f}",
                    f"teacher_mass={record['teacher_mass']:.8f}",
                    f"student_mass={record['student_mass']:.8f}",
                    f"topk_overlap={record['topk_overlap']:.8f}",
                ]
                if args.train:
                    parts.append(f"grad_norm={record['grad_norm']:.8f}")
                    parts.append(f"update_param={record['update_param']}")
                    for key in ("param_update_max_abs", "param_update_mean_abs", "param_update_rel_mean"):
                        if key in record:
                            parts.append(f"{key}={record[key]:.8e}")
                if "meta_global_token_num" in record:
                    parts.append(f"meta_global_token_num={record['meta_global_token_num']}")
                print(" | ".join(parts), flush=True)

                if metrics_output is not None:
                    metrics_output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    metrics_output.flush()
                if swanlab_run is not None:
                    log_swanlab_metrics(record, int(record["replay_update_step"]))

            dist.barrier()
        if args.save_model_dir:
            save_fsdp_hf_model(model, base_model, processor, args.save_model_dir)
    finally:
        if metrics_output is not None:
            metrics_output.close()
        if swanlab_run is not None:
            finish_swanlab()
        dist.barrier()
        dist.destroy_process_group()

    rank_print("replay_verl_opd_trace_fsdp_ok=True")


if __name__ == "__main__":
    main()
