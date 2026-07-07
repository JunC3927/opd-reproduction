import argparse
import importlib.util
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OPD student rollout and shifted logprob alignment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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

    device = torch.device(args.device)
    model, processor, tokenizer = load_vision_language_model(model_args, data_args.template)
    model.eval()
    model.to(device)

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
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        pad_to_multiple_of=8,
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        torch_dtype=parse_torch_dtype(model_args.torch_dtype),
    )
    batch = move_batch_to_device(collator(samples), device)

    probe = _RolloutProbe(model=model, tokenizer=tokenizer, method_args=method_args)
    prompt_width = probe.prompt_width(batch)
    with torch.no_grad():
        sequences = probe.generate_rollout(batch)
        completion_mask = probe.completion_mask(sequences, prompt_width)
        attention_mask = probe.sequence_attention_mask(batch, sequences, completion_mask)
        token_logps, token_mask = probe.sequence_token_logps(
            model=model,
            batch=batch,
            sequences=sequences,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            prompt_width=prompt_width,
        )

    assert_completion_mask_alignment(token_mask.detach().cpu(), completion_mask.detach().cpu(), prompt_width)
    if token_logps.shape != token_mask.shape:
        raise AssertionError(f"token_logps shape {token_logps.shape} != token_mask shape {token_mask.shape}")
    if attention_mask.shape != sequences.shape:
        raise AssertionError(f"attention_mask shape {attention_mask.shape} != sequences shape {sequences.shape}")

    completions = tokenizer.batch_decode(sequences[:, prompt_width:], skip_special_tokens=False)
    print("=== rollout alignment ===")
    print(f"device={device}")
    print(f"prompt_width={prompt_width}")
    print(f"sequences_shape={tuple(sequences.shape)}")
    print(f"completion_mask_shape={tuple(completion_mask.shape)} sum={int(completion_mask.sum().item())}")
    print(f"attention_mask_shape={tuple(attention_mask.shape)}")
    print(f"token_logps_shape={tuple(token_logps.shape)}")
    print(f"token_mask_shape={tuple(token_mask.shape)} sum={int(token_mask.sum().item())}")
    print(f"finite_token_logps={bool(torch.isfinite(token_logps[token_mask.bool()]).all().item()) if token_mask.any() else True}")
    print("completion[0]:")
    print(completions[0] if completions else "")
    print("alignment_ok=True")


if __name__ == "__main__":
    main()
