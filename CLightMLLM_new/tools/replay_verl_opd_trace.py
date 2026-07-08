import argparse
import glob
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hparams import (  # noqa: E402
    CLSFTArguments,
    DataArguments,
    LoaderArguments,
    MethodArguments,
    ModelArguments,
    OptimizerArguments,
    TrainerArguments,
    TuningArguments,
)
from src.model import ModelTuner, load_vision_language_model  # noqa: E402


ARG_GROUPS = {
    "cl_sft": CLSFTArguments,
    "data": DataArguments,
    "loader": LoaderArguments,
    "method": MethodArguments,
    "model": ModelArguments,
    "optimizer": OptimizerArguments,
    "trainer": TrainerArguments,
    "tuning": TuningArguments,
}


def parse_yaml_args(path: str) -> tuple[Any, ...]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    unknown = sorted(set(config) - set(ARG_GROUPS))
    if unknown:
        raise KeyError(f"Unsupported config groups: {unknown}. Allowed groups: {sorted(ARG_GROUPS)}")

    hparams = []
    for group, group_cls in ARG_GROUPS.items():
        group_config = config.get(group) or {}
        allowed = {field.name for field in fields(group_cls) if field.init}
        unknown = sorted(set(group_config) - allowed)
        if unknown:
            raise KeyError(f"Unsupported {group_cls.__name__} config keys: {unknown}")
        hparams.append(group_cls(**group_config))
    return tuple(hparams)


def expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)
    return sorted(dict.fromkeys(paths))


def build_optimizer(model: torch.nn.Module, optimizer_args: OptimizerArguments) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    no_decay_marks = ("bias", "norm.weight", "layernorm.weight")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or any(mark in name.lower() for mark in no_decay_marks):
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": optimizer_args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=optimizer_args.learning_rate,
        betas=(optimizer_args.adam_beta1, optimizer_args.adam_beta2),
        eps=optimizer_args.adam_epsilon,
    )


def grad_norm(model: torch.nn.Module) -> torch.Tensor:
    total = None
    for param in model.parameters():
        if param.grad is None:
            continue
        part = param.grad.detach().float().pow(2).sum()
        total = part if total is None else total + part
    if total is None:
        return torch.tensor(0.0)
    return total.sqrt()


def select_update_probe(model: torch.nn.Module) -> tuple[str, torch.Tensor, torch.Tensor]:
    preferred = ("language_model", "lm_head", "model.embed_tokens")
    fallback: tuple[str, torch.Tensor, torch.Tensor] | None = None
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        flat = param.detach().flatten()
        if flat.numel() == 0:
            continue
        idx = torch.arange(min(4096, flat.numel()), device=flat.device, dtype=torch.long)
        before = flat[idx].float().clone()
        item = (name, idx, before)
        if fallback is None:
            fallback = item
        if any(mark in name for mark in preferred):
            return item
    if fallback is None:
        raise RuntimeError("No trainable parameter found.")
    return fallback


def compute_update_stats(model: torch.nn.Module, name: str, idx: torch.Tensor, before: torch.Tensor) -> dict[str, float]:
    for param_name, param in model.named_parameters():
        if param_name != name:
            continue
        after = param.detach().flatten()[idx].float()
        delta = (after - before.to(after.device)).abs()
        denom = before.to(after.device).abs().mean().clamp_min(1e-12)
        return {
            "param_update_max_abs": float(delta.max().item()),
            "param_update_mean_abs": float(delta.mean().item()),
            "param_update_rel_mean": float((delta.mean() / denom).item()),
        }
    raise KeyError(f"Parameter {name!r} not found after optimizer step.")


def move_tensor(value: Any, device: torch.device) -> Any:
    return value.to(device) if torch.is_tensor(value) else value


def slice_rows(value: torch.Tensor, start: int, end: int, device: torch.device) -> torch.Tensor:
    return value[start:end].to(device)


def sync_cuda(device: torch.device, label: str) -> None:
    if device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(f"CUDA failed after: {label}") from exc


def validate_token_ids(name: str, ids: torch.Tensor, vocab_size: int, *, mask: torch.Tensor | None = None) -> None:
    if mask is not None:
        ids = ids[mask.bool()]
    if ids.numel() == 0:
        return
    min_id = int(ids.min().item())
    max_id = int(ids.max().item())
    if min_id < 0 or max_id >= vocab_size:
        bad_mask = ids.lt(0) | ids.ge(vocab_size)
        bad_value = int(ids[bad_mask][0].item())
        raise ValueError(
            f"{name} has out-of-vocab token id: min={min_id}, max={max_id}, "
            f"first_bad={bad_value}, vocab_size={vocab_size}"
        )


def normalize_mm_inputs(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def build_mm_kwargs(mm_inputs: list[Any], start: int, end: int, device: torch.device) -> dict[str, torch.Tensor]:
    selected = mm_inputs[start:end]
    pixel_values = []
    image_grid_thw = []
    for item in selected:
        if item is None:
            continue
        if "pixel_values" in item:
            pixel_values.append(move_tensor(item["pixel_values"], device).float())
        if "image_grid_thw" in item:
            grid = move_tensor(item["image_grid_thw"], device).long()
            if grid.ndim == 1:
                grid = grid.unsqueeze(0)
            image_grid_thw.append(grid)

    kwargs: dict[str, torch.Tensor] = {}
    if pixel_values:
        kwargs["pixel_values"] = torch.cat(pixel_values, dim=0)
    if image_grid_thw:
        kwargs["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
    return kwargs


def build_mm_token_type_ids(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor | None:
    config = getattr(model, "config", None)
    image_token_id = getattr(config, "image_token_id", None)
    video_token_id = getattr(config, "video_token_id", None)
    if image_token_id is None and video_token_id is None:
        return None

    token_type_ids = torch.zeros_like(input_ids)
    if image_token_id is not None:
        token_type_ids = token_type_ids.masked_fill(input_ids.eq(int(image_token_id)), 1)
    if video_token_id is not None:
        token_type_ids = token_type_ids.masked_fill(input_ids.eq(int(video_token_id)), 2)
    return token_type_ids


def build_position_ids(position_ids: torch.Tensor, mode: str, start: int, end: int, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    sliced = position_ids[start:end].to(device)
    if mode == "trace_batch4":
        return sliced
    if mode == "trace4":
        return sliced.permute(1, 0, 2).contiguous()
    if mode == "trace3":
        if sliced.shape[1] == 4:
            sliced = sliced[:, 1:, :]
        return sliced.permute(1, 0, 2).contiguous()
    raise ValueError(f"Unsupported position_ids mode: {mode}")


def masked_sum(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum()


def compute_topk_loss_from_logits(
    *,
    student_logits: torch.Tensor,
    teacher_logps: torch.Tensor,
    teacher_ids: torch.Tensor,
    response_mask: torch.Tensor,
    log_prob_min_clamp: float | None,
    loss_max_clamp: float | None,
) -> dict[str, torch.Tensor]:
    topk = teacher_ids.shape[-1]
    logits = student_logits.float()
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    student_at_teacher = torch.gather(logits, dim=-1, index=teacher_ids.long()) - log_z
    student_topk_ids = torch.topk(logits, k=topk, dim=-1).indices

    teacher_logps = teacher_logps.float()
    student_mass = student_at_teacher.exp().sum(dim=-1)
    teacher_mass = teacher_logps.exp().sum(dim=-1)

    if log_prob_min_clamp is not None:
        student_at_teacher = student_at_teacher.clamp_min(log_prob_min_clamp)
        teacher_logps = teacher_logps.clamp_min(log_prob_min_clamp)

    token_loss = (teacher_logps.exp() * (teacher_logps - student_at_teacher)).sum(dim=-1)
    token_loss = token_loss.clamp_min(0.0)
    if loss_max_clamp is not None:
        token_loss = token_loss.clamp(min=-loss_max_clamp, max=loss_max_clamp)

    overlap_count = (teacher_ids.unsqueeze(-1).long() == student_topk_ids.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    loss_num = masked_sum(token_loss, response_mask)

    return {
        "loss_num": loss_num,
        "token_loss": token_loss,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count.float(),
    }


def sanitize_teacher_ids(
    *,
    teacher_ids: torch.Tensor,
    response_mask: torch.Tensor,
    vocab_size: int,
    file_idx: int,
    row_start: int,
    response_start: int,
) -> torch.Tensor:
    active = response_mask.bool().unsqueeze(-1)
    active_ids = teacher_ids[active.expand_as(teacher_ids)]
    if active_ids.numel() > 0:
        bad_active = active_ids.lt(0) | active_ids.ge(vocab_size)
        if bad_active.any():
            flat_bad = torch.nonzero(
                ((teacher_ids.lt(0) | teacher_ids.ge(vocab_size)) & active.expand_as(teacher_ids)),
                as_tuple=False,
            )[0]
            local_row = int(flat_bad[0].item())
            token_pos = int(flat_bad[1].item())
            topk_pos = int(flat_bad[2].item())
            bad_value = int(teacher_ids[local_row, token_pos, topk_pos].item())
            raise ValueError(
                "Active teacher_ids contains an out-of-vocab id: "
                f"file={file_idx}, row={row_start + local_row}, "
                f"model_pos={response_start + token_pos}, topk={topk_pos}, "
                f"id={bad_value}, vocab_size={vocab_size}. "
                "This usually means the teacher/student tokenizer vocabularies differ."
            )

    # Inactive response positions are ignored by the loss, but gather would still
    # read their ids. Replace them with a valid dummy id to avoid CUDA index asserts.
    return torch.where(active, teacher_ids, torch.zeros_like(teacher_ids))


def load_trace(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "batch" not in payload:
        raise KeyError(f"{path} is missing payload['batch']")
    return payload


def trainable_parameter_summary(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


def init_swanlab(args: argparse.Namespace, config: dict[str, Any]) -> Any:
    if not args.swanlab_project:
        return None
    try:
        import swanlab
    except ImportError as exc:
        raise ImportError(
            "`--swanlab-project` was set, but the `swanlab` package is not installed. "
            "Install it on the server with `pip install swanlab`."
        ) from exc

    init_kwargs = {
        "project": args.swanlab_project,
        "experiment_name": args.swanlab_experiment_name,
        "workspace": args.swanlab_workspace,
        "mode": args.swanlab_mode,
        "logdir": args.swanlab_logdir,
        "save_dir": args.swanlab_logdir,
        "config": config,
    }
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
    return swanlab.init(**init_kwargs)


def log_swanlab_metrics(record: dict[str, Any], step: int) -> None:
    try:
        import swanlab
    except ImportError:
        return

    metric_keys = (
        "loss",
        "teacher_mass",
        "student_mass",
        "topk_overlap",
        "grad_norm",
        "param_update_max_abs",
        "param_update_mean_abs",
        "param_update_rel_mean",
        "tokens",
        "samples",
    )
    metrics = {}
    for key in metric_keys:
        value = record.get(key)
        if isinstance(value, (int, float)):
            metrics[f"replay/{key}"] = value
    if metrics:
        swanlab.log(metrics, step=step)


def finish_swanlab() -> None:
    try:
        import swanlab
    except ImportError:
        return
    finish = getattr(swanlab, "finish", None)
    if callable(finish):
        finish()


def save_replay_model(model: torch.nn.Module, processor: Any, output_dir: str, metadata: dict[str, Any]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.eval()
    model_to_save.save_pretrained(str(path), safe_serialization=True)
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(path))
    with open(path / "clight_replay_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"saved_replay_model={path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay verl OPD trace dumps through CLight HF forward/loss.")
    parser.add_argument("--config", required=True)
    parser.add_argument("traces", nargs="+", help="Trace dump file(s) or glob pattern(s).")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
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
    parser.add_argument(
        "--teacher-shift-offset",
        type=int,
        default=-1,
        help="Teacher/logit response start = prompt_width + offset. verl response logprob convention is -1.",
    )
    parser.add_argument("--log-prob-min-clamp", type=float, default=-10.0)
    parser.add_argument("--loss-max-clamp", type=float, default=10.0)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "bf16", "fp16"),
        default="none",
        help="Optional CUDA autocast dtype for forward. Use bf16 to mimic mixed-precision training.",
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Optional JSONL path for per-trace replay metrics, for comparing with VERL_OPD_METRICS_DUMP.",
    )
    parser.add_argument("--swanlab-project", default=None, help="Optional SwanLab project for replay diagnostics.")
    parser.add_argument("--swanlab-experiment-name", default=None)
    parser.add_argument("--swanlab-workspace", default=None)
    parser.add_argument("--swanlab-mode", default=None)
    parser.add_argument("--swanlab-logdir", default=None)
    parser.add_argument("--save-model-dir", default=None, help="Optional HF directory for the replay-updated student.")
    args = parser.parse_args()

    os.chdir(ROOT)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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

    paths = expand_paths(args.traces)
    if args.max_files is not None:
        paths = paths[: args.max_files]
    if not paths:
        raise FileNotFoundError(f"No trace files matched: {args.traces}")

    device = torch.device(args.device)
    model, processor, _tokenizer = load_vision_language_model(model_args, data_args.template)
    model = ModelTuner(tuning_args).apply(model)
    model.to(device)
    sync_cuda(device, "model.to(device)")
    model.train(args.train)
    trainable, total = trainable_parameter_summary(model)
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or model.get_input_embeddings().weight.shape[0])

    optimizer = build_optimizer(model, optimizer_args) if args.train else None

    print("=== replay verl opd trace ===")
    print(f"config={args.config}")
    print(f"trace_count={len(paths)}")
    print(f"device={device}")
    print(f"train={args.train}")
    print(f"micro_batch_size={args.micro_batch_size}")
    print(f"position_ids_mode={args.position_ids_mode}")
    print(f"teacher_shift_offset={args.teacher_shift_offset}")
    print(f"amp_dtype={args.amp_dtype}")
    print(f"learning_rate={optimizer_args.learning_rate}")
    print(f"trainable_params={trainable} total_params={total}")
    print(f"student_vocab_size={vocab_size}")

    metrics_output = None
    if args.metrics_output:
        metrics_output = open(args.metrics_output, "w", encoding="utf-8")
    swanlab_run = init_swanlab(
        args,
        {
            "config": args.config,
            "trace_count": len(paths),
            "train": bool(args.train),
            "micro_batch_size": args.micro_batch_size,
            "position_ids_mode": args.position_ids_mode,
            "teacher_shift_offset": args.teacher_shift_offset,
            "amp_dtype": args.amp_dtype,
            "learning_rate": optimizer_args.learning_rate,
            "grad_clip": args.grad_clip,
            "trainable_params": trainable,
            "total_params": total,
            "save_model_dir": args.save_model_dir,
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
            prompt_width = int(prompts_cpu.shape[1]) if torch.is_tensor(prompts_cpu) else seq_len - response_mask_cpu.shape[1]
            response_start = prompt_width + args.teacher_shift_offset
            response_len = int(response_mask_cpu.shape[1])
            if response_start < 0:
                raise ValueError(f"Invalid response_start={response_start} from prompt_width={prompt_width}.")

            validate_token_ids(f"file {file_idx} input_ids", input_ids_cpu, vocab_size)
            teacher_check_len = min(response_len, teacher_ids_cpu.shape[1] - response_start)
            if teacher_check_len <= 0:
                raise ValueError(
                    f"Invalid teacher response slice for file {file_idx}: "
                    f"teacher_shape={tuple(teacher_ids_cpu.shape)}, response_start={response_start}"
                )
            active_teacher_ids = teacher_ids_cpu[
                :,
                response_start : response_start + teacher_check_len,
                :,
            ]
            active_teacher_mask = response_mask_cpu[:, :teacher_check_len].bool().unsqueeze(-1).expand_as(active_teacher_ids)
            validate_token_ids(f"file {file_idx} active teacher_ids", active_teacher_ids, vocab_size, mask=active_teacher_mask)

            loss_num_total = None
            token_count_total = response_mask_cpu.sum().to(device).clamp_min(1.0)
            sync_cuda(device, f"file {file_idx} token_count_total.to(device)")
            teacher_mass_num = torch.tensor(0.0, device=device)
            student_mass_num = torch.tensor(0.0, device=device)
            overlap_num = torch.tensor(0.0, device=device)
            actual_token_count = torch.tensor(0.0, device=device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                probe_name, probe_idx, probe_before = select_update_probe(model)
                sync_cuda(device, f"file {file_idx} select_update_probe")
            else:
                probe_name, probe_idx, probe_before = "", torch.empty(0), torch.empty(0)

            for start in range(0, batch_size, args.micro_batch_size):
                end = min(start + args.micro_batch_size, batch_size)
                input_ids = slice_rows(input_ids_cpu, start, end, device)
                sync_cuda(device, f"file {file_idx} rows {start}:{end} input_ids.to(device)")
                attention_mask = slice_rows(attention_mask_cpu, start, end, device)
                sync_cuda(device, f"file {file_idx} rows {start}:{end} attention_mask.to(device)")
                response_mask = slice_rows(response_mask_cpu, start, end, device)
                sync_cuda(device, f"file {file_idx} rows {start}:{end} response_mask.to(device)")

                forward_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                }
                mm_kwargs = build_mm_kwargs(mm_inputs, start, end, device)
                forward_kwargs.update(mm_kwargs)
                mm_token_type_ids = build_mm_token_type_ids(model, input_ids)
                if mm_token_type_ids is not None and "image_grid_thw" in forward_kwargs:
                    forward_kwargs["mm_token_type_ids"] = mm_token_type_ids
                if position_ids_cpu is not None:
                    position_ids = build_position_ids(position_ids_cpu, args.position_ids_mode, start, end, device)
                    if position_ids is not None:
                        forward_kwargs["position_ids"] = position_ids

                if args.amp_dtype == "bf16":
                    autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
                elif args.amp_dtype == "fp16":
                    autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")
                else:
                    autocast_ctx = nullcontext()
                with autocast_ctx:
                    outputs = model(**forward_kwargs)
                sync_cuda(device, f"file {file_idx} rows {start}:{end} model forward")
                shifted_logits = outputs.logits[:, :-1, :]
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
                sync_cuda(device, f"file {file_idx} rows {start}:{end} teacher_ids.to(device)")
                teacher_logps = teacher_logps_cpu_slice.to(device)
                sync_cuda(device, f"file {file_idx} rows {start}:{end} teacher_logps.to(device)")

                loss_outputs = compute_topk_loss_from_logits(
                    student_logits=student_logits,
                    teacher_logps=teacher_logps,
                    teacher_ids=teacher_ids,
                    response_mask=response_mask,
                    log_prob_min_clamp=args.log_prob_min_clamp,
                    loss_max_clamp=args.loss_max_clamp,
                )
                sync_cuda(device, f"file {file_idx} rows {start}:{end} compute_topk_loss")
                loss_num = loss_outputs["loss_num"]
                loss_num_total = loss_num if loss_num_total is None else loss_num_total + loss_num.detach()
                micro_loss = loss_num / token_count_total
                if args.train:
                    micro_loss.backward()
                    sync_cuda(device, f"file {file_idx} rows {start}:{end} backward")

                with torch.no_grad():
                    token_count = response_mask.sum()
                    actual_token_count += token_count
                    teacher_mass_num += masked_sum(loss_outputs["teacher_mass"], response_mask)
                    student_mass_num += masked_sum(loss_outputs["student_mass"], response_mask)
                    overlap_num += masked_sum(loss_outputs["overlap_count"] / teacher_ids.shape[-1], response_mask)

                del outputs, shifted_logits, student_logits, teacher_ids, teacher_logps

            loss_value = loss_num_total / token_count_total if loss_num_total is not None else torch.tensor(0.0, device=device)
            grad_value = torch.tensor(0.0)
            update_stats: dict[str, float] = {}
            if args.train:
                grad_value = grad_norm(model).detach().cpu()
                if args.grad_clip is not None and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                assert optimizer is not None
                optimizer.step()
                update_stats = compute_update_stats(model, probe_name, probe_idx, probe_before)

            denom = actual_token_count.clamp_min(1.0)
            record = {
                "format": "clight_verl_trace_replay_metrics_v1",
                "file_index": file_idx,
                "replay_update_step": file_idx + 1,
                "path": path,
                "dump_index": payload.get("dump_index"),
                "global_step": payload.get("global_steps"),
                "chunk_index": payload.get("chunk_index"),
                "chunk_count": payload.get("chunk_count"),
                "source_sample_count": payload.get("source_sample_count"),
                "samples": batch_size,
                "tokens": int(actual_token_count.item()),
                "loss": float(loss_value.detach().cpu().item()),
                "teacher_mass": float((teacher_mass_num / denom).detach().cpu().item()),
                "student_mass": float((student_mass_num / denom).detach().cpu().item()),
                "topk_overlap": float((overlap_num / denom).detach().cpu().item()),
                "train": bool(args.train),
                "amp_dtype": args.amp_dtype,
            }
            if args.train:
                record["grad_norm"] = float(grad_value.item())
                record["update_param"] = probe_name
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
                f"samples={batch_size}",
                f"tokens={int(actual_token_count.item())}",
                f"loss={record['loss']:.8f}",
                f"teacher_mass={record['teacher_mass']:.8f}",
                f"student_mass={record['student_mass']:.8f}",
                f"topk_overlap={record['topk_overlap']:.8f}",
            ]
            if args.train:
                parts.append(f"grad_norm={record['grad_norm']:.8f}")
                parts.append(f"update_param={probe_name}")
                for key in ("param_update_max_abs", "param_update_mean_abs", "param_update_rel_mean"):
                    parts.append(f"{key}={record[key]:.8e}")
            if "meta_global_token_num" in record:
                parts.append(f"meta_global_token_num={record['meta_global_token_num']}")
            print(" | ".join(parts), flush=True)
            if metrics_output is not None:
                metrics_output.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_output.flush()
            if swanlab_run is not None:
                log_swanlab_metrics(record, int(record["replay_update_step"]))
        if args.save_model_dir:
            save_replay_model(
                model,
                processor,
                args.save_model_dir,
                {
                    "format": "clight_verl_trace_replay_model_v1",
                    "config": args.config,
                    "trace_count": len(paths),
                    "train": bool(args.train),
                    "micro_batch_size": args.micro_batch_size,
                    "position_ids_mode": args.position_ids_mode,
                    "teacher_shift_offset": args.teacher_shift_offset,
                    "amp_dtype": args.amp_dtype,
                    "learning_rate": optimizer_args.learning_rate,
                    "grad_clip": args.grad_clip,
                    "metrics_output": args.metrics_output,
                },
            )
    finally:
        if metrics_output is not None:
            metrics_output.close()
        if swanlab_run is not None:
            finish_swanlab()

    print("replay_verl_opd_trace_ok=True")


if __name__ == "__main__":
    main()
