import argparse
import os
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.module import DatasetBuilder, TemplateFactory, VLCollator  # noqa: E402
from src.hparams import (  # noqa: E402
    CLSFTArguments,
    DataArguments,
    LoaderArguments,
    MethodArguments,
    ModelArguments,
    OptimizerArguments,
    TrainerArguments,
    TuningArguments,
    parse_torch_dtype,
)
from src.method.opd import OPDLearner  # noqa: E402
from src.model import load_vision_language_model  # noqa: E402


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


class _NoopStrategy:
    def barrier(self) -> None:
        return None


class _DummyTrainer:
    local_rank = 0
    global_rank = 0
    is_global_zero = True
    strategy = _NoopStrategy()


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


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


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


def select_updated_values(model: torch.nn.Module) -> tuple[str, torch.Tensor, torch.Tensor]:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        flat_grad = param.grad.detach().flatten()
        finite_nonzero = torch.isfinite(flat_grad) & flat_grad.ne(0)
        if not finite_nonzero.any():
            continue
        idx = torch.nonzero(finite_nonzero, as_tuple=False).flatten()[:1024]
        before = param.detach().flatten()[idx].float().clone()
        return name, idx, before
    raise RuntimeError("No trainable parameter has a finite non-zero gradient.")


def max_abs_update(param: torch.nn.Parameter, idx: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
    after = param.detach().flatten()[idx].float()
    return (after - before.to(after.device)).abs().max()


def find_parameter(model: torch.nn.Module, target_name: str) -> torch.nn.Parameter:
    for name, param in model.named_parameters():
        if name == target_name:
            return param
    raise KeyError(f"Parameter {target_name!r} not found after optimizer step.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one OPD optimizer step and verify that student weights update.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--student-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--teacher-device",
        default="cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--use-cache", action="store_true", help="Reuse existing HF datasets map cache.")
    args = parser.parse_args()

    os.chdir(ROOT)
    (
        cl_sft_args,
        data_args,
        _loader_args,
        method_args,
        model_args,
        optimizer_args,
        _trainer_args,
        _tuning_args,
    ) = parse_yaml_args(args.config)
    if not cl_sft_args.stages:
        raise ValueError("cl_sft.stages is empty.")
    if not method_args.opd_teacher_model_name_or_path:
        raise ValueError("method.opd_teacher_model_name_or_path is required.")
    if args.learning_rate is not None:
        optimizer_args = replace(optimizer_args, learning_rate=args.learning_rate)

    stage = cl_sft_args.stages[0]
    data_args = replace(
        data_args,
        dataset=stage.dataset,
        max_samples=args.max_samples,
        preprocessing_num_workers=args.num_workers,
        overwrite_cache=not args.use_cache,
        log_first_sample=False,
    )
    method_args = replace(
        method_args,
        rollout_max_new_tokens=args.max_new_tokens,
        rollout_do_sample=args.do_sample,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_num_generations=1,
    )

    student_device = torch.device(args.student_device)
    teacher_device = torch.device(args.teacher_device)

    student_model, processor, tokenizer = load_vision_language_model(model_args, data_args.template)
    student_model.to(student_device)
    student_model.train()

    teacher_model_args = replace(
        model_args,
        model_name_or_path=method_args.opd_teacher_model_name_or_path,
        gradient_checkpointing=False,
        use_cache=False,
    )
    teacher_model, _, _ = load_vision_language_model(teacher_model_args, data_args.template)
    teacher_model.to(teacher_device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    template = TemplateFactory.from_args(tokenizer, data_args)
    dataset = DatasetBuilder(
        template=template,
        model_args=model_args,
        data_args=data_args,
        tokenizer=tokenizer,
        processor=processor,
        trainer=_DummyTrainer(),
    ).build()
    if len(dataset) == 0:
        raise RuntimeError("No examples survived preprocessing.")

    sample_count = min(args.batch_size, len(dataset))
    samples = [dataset[i] for i in range(sample_count)]
    collator = VLCollator(
        template=template,
        model=student_model,
        tokenizer=tokenizer,
        processor=processor,
        pad_to_multiple_of=8,
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        torch_dtype=parse_torch_dtype(model_args.torch_dtype),
    )
    batch = move_batch_to_device(collator(samples), student_device)

    learner = OPDLearner(
        model=student_model,
        optimizer_args=optimizer_args,
        tokenizer=tokenizer,
        method_args=method_args,
        teacher_model=teacher_model,
    )
    learner.log_metric = lambda *unused_args, **unused_kwargs: None

    optimizer = build_optimizer(student_model, optimizer_args)
    optimizer.zero_grad(set_to_none=True)
    loss = learner.compute_loss(batch)
    loss.backward()

    grad_value = grad_norm(student_model)
    selected_name, selected_idx, selected_before = select_updated_values(student_model)
    if args.grad_clip is not None and args.grad_clip > 0:
        clipped_grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
    else:
        clipped_grad_norm = grad_value
    optimizer.step()

    selected_param = find_parameter(student_model, selected_name)
    update_value = max_abs_update(selected_param, selected_idx, selected_before)

    print("=== opd train step ===")
    print(f"student_device={student_device}")
    print(f"teacher_device={teacher_device}")
    print(f"batch_size={sample_count}")
    print(f"max_new_tokens={method_args.rollout_max_new_tokens}")
    print(f"topk={method_args.opd_topk}")
    print(f"learning_rate={optimizer_args.learning_rate}")
    print(f"loss={float(loss.detach().cpu())}")
    print(f"grad_norm={float(grad_value.detach().cpu())}")
    print(f"finite_grad_norm={bool(torch.isfinite(grad_value).item())}")
    print(f"clipped_grad_norm={float(clipped_grad_norm.detach().cpu())}")
    print(f"updated_param={selected_name}")
    print(f"max_abs_update={float(update_value.detach().cpu())}")
    print(f"weight_changed={bool(update_value.gt(0).item())}")
    print("opd_train_step_ok=True")


if __name__ == "__main__":
    main()
