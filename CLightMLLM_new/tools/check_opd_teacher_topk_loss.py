import argparse
import importlib.util
import os
import sys
from contextlib import nullcontext
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
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


def load_rollout_mixin():
    rollout_path = ROOT / "src" / "method" / "rollout.py"
    spec = importlib.util.spec_from_file_location("_clight_rollout_probe", rollout_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load rollout module from {rollout_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RolloutMixin


RolloutMixin = load_rollout_mixin()


class _NoopStrategy:
    def barrier(self) -> None:
        return None


class _DummyTrainer:
    local_rank = 0
    global_rank = 0
    is_global_zero = True
    strategy = _NoopStrategy()


class _RolloutProbe(RolloutMixin):
    AUX_BATCH_KEYS = {
        "prompt_input_ids",
        "prompt_attention_mask",
        "reference_text",
    }

    def __init__(self, model: torch.nn.Module, tokenizer: Any, method_args: MethodArguments) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.method_args = method_args

    def model_kwargs(self, batch: dict[str, Any], include_labels: bool = True) -> dict[str, Any]:
        kwargs = {key: value for key, value in batch.items() if key not in self.AUX_BATCH_KEYS}
        if not include_labels:
            kwargs.pop("labels", None)
        return kwargs


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
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value.lower() in {"none", "null", "off"}:
        return None
    return float(value)


def assert_completion_mask_alignment(
    token_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    prompt_width: int,
) -> None:
    if int(token_mask.sum().item()) != int(completion_mask.sum().item()):
        raise AssertionError(
            f"token_mask sum {int(token_mask.sum().item())} != completion_mask sum {int(completion_mask.sum().item())}"
        )

    for row_idx in range(completion_mask.shape[0]):
        comp_len = int(completion_mask[row_idx].sum().item())
        expected = torch.zeros_like(token_mask[row_idx])
        if comp_len > 0:
            start = max(prompt_width - 1, 0)
            expected[start : start + comp_len] = 1
        if not torch.equal(token_mask[row_idx].to(expected.dtype), expected):
            raise AssertionError(
                f"token_mask for row {row_idx} is not aligned to prompt_width={prompt_width}, comp_len={comp_len}"
            )


def shifted_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, :-1].float()


def compute_forward_kl_topk_loss(
    student_logits: torch.Tensor,
    teacher_topk_logps: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    response_mask: torch.Tensor,
    log_prob_min_clamp: float | None,
    loss_max_clamp: float | None,
) -> dict[str, torch.Tensor]:
    topk = teacher_topk_ids.shape[-1]
    student_logps = F.log_softmax(student_logits, dim=-1)
    student_topk_ids = torch.topk(student_logps, k=topk, dim=-1).indices
    student_at_teacher_topk = torch.gather(student_logps, dim=-1, index=teacher_topk_ids)

    student_mass = student_at_teacher_topk.exp().sum(dim=-1)
    teacher_mass = teacher_topk_logps.exp().sum(dim=-1)

    if log_prob_min_clamp is not None:
        student_at_teacher_topk = student_at_teacher_topk.clamp_min(log_prob_min_clamp)
        teacher_topk_logps = teacher_topk_logps.clamp_min(log_prob_min_clamp)

    teacher_probs = teacher_topk_logps.exp()
    raw_token_loss = (teacher_probs * (teacher_topk_logps - student_at_teacher_topk)).sum(dim=-1)
    token_loss = raw_token_loss.clamp_min(0.0)
    if loss_max_clamp is not None:
        token_loss = token_loss.clamp(min=-loss_max_clamp, max=loss_max_clamp)

    denom = response_mask.sum().clamp_min(1.0)
    loss = (token_loss * response_mask).sum() / denom

    overlap = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap.sum(dim=-1)

    return {
        "loss": loss,
        "raw_token_loss": raw_token_loss,
        "token_loss": token_loss,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "student_topk_logps": student_at_teacher_topk,
        "teacher_topk_logps": teacher_topk_logps,
        "overlap_count": overlap_count,
    }


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def masked_min(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask.bool()]
    return selected.min() if selected.numel() else torch.tensor(float("nan"), device=values.device)


def masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask.bool()]
    return selected.max() if selected.numel() else torch.tensor(float("nan"), device=values.device)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OPD teacher top-k forward_kl_topk loss on one mini-batch.")
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
    parser.add_argument("--teacher-device-map", default=None, help="Optional HF device_map, e.g. auto.")
    parser.add_argument("--teacher-input-device", default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--log-prob-min-clamp", default="-10.0")
    parser.add_argument("--loss-max-clamp", default="10.0")
    parser.add_argument("--backward", action="store_true", help="Also run loss.backward() and report grad norm.")
    parser.add_argument("--use-cache", action="store_true", help="Reuse existing HF datasets map cache.")
    args = parser.parse_args()

    os.chdir(ROOT)
    (
        cl_sft_args,
        data_args,
        _loader_args,
        method_args,
        model_args,
        _optimizer_args,
        _trainer_args,
        _tuning_args,
    ) = parse_yaml_args(args.config)
    if not cl_sft_args.stages:
        raise ValueError("cl_sft.stages is empty.")
    if not method_args.opd_teacher_model_name_or_path:
        raise ValueError("method.opd_teacher_model_name_or_path is required.")

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
    )

    topk = args.topk if args.topk is not None else method_args.opd_topk
    log_prob_min_clamp = parse_optional_float(args.log_prob_min_clamp)
    loss_max_clamp = parse_optional_float(args.loss_max_clamp)

    student_device = torch.device(args.student_device)
    student_model, processor, tokenizer = load_vision_language_model(model_args, data_args.template)
    student_model.eval()
    student_model.to(student_device)

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
    student_batch = move_batch_to_device(collator(samples), student_device)
    student_probe = _RolloutProbe(model=student_model, tokenizer=tokenizer, method_args=method_args)

    prompt_width = student_probe.prompt_width(student_batch)
    with torch.no_grad():
        sequences = student_probe.generate_rollout(student_batch)
        completion_mask = student_probe.completion_mask(sequences, prompt_width)
        attention_mask = student_probe.sequence_attention_mask(student_batch, sequences, completion_mask)
        _, response_mask = student_probe.sequence_token_logps(
            model=student_model,
            batch=student_batch,
            sequences=sequences,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            prompt_width=prompt_width,
        )

    assert_completion_mask_alignment(response_mask.detach().cpu(), completion_mask.detach().cpu(), prompt_width)
    response_token_count = int(response_mask.sum().item())
    if response_token_count == 0:
        raise RuntimeError("Generated completion has zero valid response tokens.")

    teacher_model_args = replace(
        model_args,
        model_name_or_path=method_args.opd_teacher_model_name_or_path,
        gradient_checkpointing=False,
        use_cache=False,
        device_map=args.teacher_device_map,
    )
    teacher_model, _, _ = load_vision_language_model(teacher_model_args, data_args.template)
    teacher_model.eval()
    if args.teacher_device_map is None:
        teacher_model.to(torch.device(args.teacher_device))

    teacher_input_device = (
        torch.device(args.teacher_input_device) if args.teacher_input_device is not None else first_parameter_device(teacher_model)
    )
    teacher_batch = move_batch_to_device(student_batch, teacher_input_device)
    teacher_sequences = sequences.to(teacher_input_device)
    teacher_attention_mask = attention_mask.to(teacher_input_device)
    teacher_probe = _RolloutProbe(model=teacher_model, tokenizer=tokenizer, method_args=method_args)

    with torch.no_grad():
        teacher_outputs = teacher_model(
            **teacher_probe.sequence_model_kwargs(
                teacher_batch,
                teacher_sequences,
                teacher_attention_mask,
            )
        )
        teacher_logps = F.log_softmax(shifted_logits(teacher_outputs.logits), dim=-1)
        if topk > teacher_logps.shape[-1]:
            raise ValueError(f"topk={topk} exceeds vocab size {teacher_logps.shape[-1]}.")
        teacher_topk_logps, teacher_topk_ids = torch.topk(teacher_logps, k=topk, dim=-1)

    del teacher_outputs, teacher_logps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher_topk_logps = teacher_topk_logps.to(student_device)
    teacher_topk_ids = teacher_topk_ids.to(student_device)
    response_mask = response_mask.to(student_device)

    student_model.train(args.backward)
    student_model.zero_grad(set_to_none=True)
    grad_context = nullcontext() if args.backward else torch.no_grad()
    with grad_context:
        student_outputs = student_model(
            **student_probe.sequence_model_kwargs(
                student_batch,
                sequences,
                attention_mask,
            )
        )
        loss_outputs = compute_forward_kl_topk_loss(
            student_logits=shifted_logits(student_outputs.logits),
            teacher_topk_logps=teacher_topk_logps,
            teacher_topk_ids=teacher_topk_ids,
            response_mask=response_mask,
            log_prob_min_clamp=log_prob_min_clamp,
            loss_max_clamp=loss_max_clamp,
        )
        loss = loss_outputs["loss"]

    grad_value = None
    if args.backward:
        loss.backward()
        grad_value = grad_norm(student_model).detach()

    mask = response_mask.float()
    completions = tokenizer.batch_decode(sequences[:, prompt_width:], skip_special_tokens=False)
    finite_teacher = torch.isfinite(loss_outputs["teacher_topk_logps"][mask.bool()]).all()
    finite_student = torch.isfinite(loss_outputs["student_topk_logps"][mask.bool()]).all()
    overlap_ratio = masked_mean(loss_outputs["overlap_count"].float() / topk, mask)

    print("=== teacher topk forward_kl_topk ===")
    print(f"student_device={student_device}")
    print(f"teacher_input_device={teacher_input_device}")
    print(f"teacher_device_map={args.teacher_device_map}")
    print(f"prompt_width={prompt_width}")
    print(f"sequences_shape={tuple(sequences.shape)}")
    print(f"response_mask_shape={tuple(response_mask.shape)} sum={response_token_count}")
    print(f"topk={topk}")
    print(f"log_prob_min_clamp={log_prob_min_clamp}")
    print(f"loss_max_clamp={loss_max_clamp}")
    print(f"teacher_topk_logps_shape={tuple(teacher_topk_logps.shape)}")
    print(f"teacher_topk_ids_shape={tuple(teacher_topk_ids.shape)}")
    print(f"finite_teacher_topk_logps={bool(finite_teacher.item())}")
    print(f"finite_student_topk_logps={bool(finite_student.item())}")
    print(f"teacher_mass_mean={float(masked_mean(loss_outputs['teacher_mass'], mask).detach().cpu())}")
    print(f"teacher_mass_min={float(masked_min(loss_outputs['teacher_mass'], mask).detach().cpu())}")
    print(f"teacher_mass_max={float(masked_max(loss_outputs['teacher_mass'], mask).detach().cpu())}")
    print(f"student_mass_mean={float(masked_mean(loss_outputs['student_mass'], mask).detach().cpu())}")
    print(f"student_mass_min={float(masked_min(loss_outputs['student_mass'], mask).detach().cpu())}")
    print(f"student_mass_max={float(masked_max(loss_outputs['student_mass'], mask).detach().cpu())}")
    print(f"raw_token_loss_mean={float(masked_mean(loss_outputs['raw_token_loss'], mask).detach().cpu())}")
    print(f"raw_token_loss_min={float(masked_min(loss_outputs['raw_token_loss'], mask).detach().cpu())}")
    print(f"raw_token_loss_max={float(masked_max(loss_outputs['raw_token_loss'], mask).detach().cpu())}")
    print(f"token_loss_mean={float(masked_mean(loss_outputs['token_loss'], mask).detach().cpu())}")
    print(f"overlap_ratio={float(overlap_ratio.detach().cpu())}")
    print(f"loss={float(loss.detach().cpu())}")
    if grad_value is not None:
        print(f"grad_norm={float(grad_value.cpu())}")
        print(f"finite_grad_norm={bool(torch.isfinite(grad_value).item())}")
    print("completion[0]:")
    print(completions[0] if completions else "")
    print("teacher_topk_loss_ok=True")


if __name__ == "__main__":
    main()
