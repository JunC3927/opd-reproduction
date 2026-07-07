import argparse
import importlib.util
import os
import sys
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
    spec = importlib.util.spec_from_file_location("_clight_rollout_dump", rollout_path)
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
        "vllm_images",
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
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def move_tensors_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: move_tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors_to_cpu(item) for item in value]
    return value


def tensor_only_batch(batch: dict[str, Any]) -> dict[str, Any]:
    kept = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            kept[key] = value.detach().cpu()
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump one fixed OPD forward batch for HF/FSDP comparison.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-batch", default=None, help="Reuse batch/sequences from an existing dump and only rescore.")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--debug-topk", type=int, default=10)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
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
        rollout_backend="hf",
        rollout_max_new_tokens=args.max_new_tokens,
        rollout_do_sample=args.do_sample,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_num_generations=1,
    )

    device = torch.device(args.device)
    model, processor, tokenizer = load_vision_language_model(model_args, data_args.template)
    model.eval()
    model.to(device)

    probe = _RolloutProbe(model=model, tokenizer=tokenizer, method_args=method_args)

    reused_payload = None
    if args.reuse_batch is not None:
        try:
            reused_payload = torch.load(args.reuse_batch, map_location="cpu", weights_only=False)
        except TypeError:
            reused_payload = torch.load(args.reuse_batch, map_location="cpu")
        batch = move_batch_to_device(reused_payload["batch"], device)
        sequences = reused_payload["sequences"].to(device)
        attention_mask = reused_payload["attention_mask"].to(device)
        completion_mask = reused_payload["completion_mask"].to(device)
        response_mask = reused_payload["response_mask"].to(device)
        prompt_width = int(reused_payload["prompt_width"])
        sample_count = int(sequences.shape[0])
    else:
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
        prompt_width = probe.prompt_width(batch)

    with torch.no_grad():
        if reused_payload is None:
            sequences = probe.generate_rollout(batch)
            completion_mask = probe.completion_mask(sequences, prompt_width)
            attention_mask = probe.sequence_attention_mask(batch, sequences, completion_mask)
            response_mask = probe.shift_completion_mask(
                token_values=sequences[:, 1:],
                completion_mask=completion_mask,
                prompt_width=prompt_width,
            )
        sequence_kwargs = probe.sequence_model_kwargs(batch, sequences, attention_mask)
        sequence_position_ids = sequence_kwargs.get("position_ids")
        sequence_mm_token_type_ids = sequence_kwargs.get("mm_token_type_ids")
        sequence_outputs = model(**sequence_kwargs)
        clight_token_logps = probe.gather_token_logps(sequence_outputs.logits, sequences)
        clight_token_mask = probe.shift_completion_mask(clight_token_logps, completion_mask, prompt_width)
        clight_response_logps = clight_token_logps[response_mask.bool()].detach().cpu()
        response_shift_logits = sequence_outputs.logits[:, :-1].float()[response_mask.bool()]
        clight_response_topk_logps, clight_response_topk_ids = F.log_softmax(
            response_shift_logits,
            dim=-1,
        ).topk(args.debug_topk, dim=-1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "clight_opd_forward_batch_v1",
        "config": str(Path(args.config).resolve()),
        "reused_batch": str(Path(args.reuse_batch).resolve()) if args.reuse_batch else None,
        "model_name_or_path": model_args.model_name_or_path,
        "template": data_args.template,
        "torch_dtype": model_args.torch_dtype,
        "use_verl_monkey_patch": bool(model_args.use_verl_monkey_patch),
        "verl_monkey_patch_applied": bool(getattr(model, "_clight_verl_monkey_patched", False)),
        "verl_repo_path": model_args.verl_repo_path,
        "sequence_kwargs_keys": sorted(sequence_kwargs),
        "sequence_position_ids_shape": tuple(sequence_position_ids.shape) if sequence_position_ids is not None else None,
        "sequence_mm_token_type_ids_shape": (
            tuple(sequence_mm_token_type_ids.shape) if sequence_mm_token_type_ids is not None else None
        ),
        "prompt_width": int(prompt_width),
        "batch": tensor_only_batch(batch),
        "sequences": sequences.detach().cpu(),
        "attention_mask": attention_mask.detach().cpu(),
        "completion_mask": completion_mask.detach().cpu(),
        "response_mask": response_mask.detach().cpu(),
        "clight_token_logps": clight_token_logps.detach().cpu(),
        "clight_token_mask": clight_token_mask.detach().cpu(),
        "clight_response_logps": clight_response_logps,
        "clight_response_topk_logps": clight_response_topk_logps.detach().cpu(),
        "clight_response_topk_ids": clight_response_topk_ids.detach().cpu(),
        "prompt_text": (
            reused_payload.get("prompt_text")
            if reused_payload is not None and "prompt_text" in reused_payload
            else tokenizer.batch_decode(batch["prompt_input_ids"].detach().cpu(), skip_special_tokens=False)
        ),
        "completion_text": tokenizer.batch_decode(sequences[:, prompt_width:].detach().cpu(), skip_special_tokens=False),
    }
    torch.save(payload, output_path)

    print("=== dumped opd forward batch ===")
    print(f"output={output_path}")
    print(f"reuse_batch={args.reuse_batch}")
    print(f"config={Path(args.config).resolve()}")
    print(f"use_verl_monkey_patch={model_args.use_verl_monkey_patch}")
    print(f"verl_monkey_patch_applied={bool(getattr(model, '_clight_verl_monkey_patched', False))}")
    print(f"sequence_kwargs_keys={sorted(sequence_kwargs)}")
    print(f"sequence_position_ids_shape={tuple(sequence_position_ids.shape) if sequence_position_ids is not None else None}")
    print(
        "sequence_mm_token_type_ids_shape="
        f"{tuple(sequence_mm_token_type_ids.shape) if sequence_mm_token_type_ids is not None else None}"
    )
    print(f"first_clight_response_logps={clight_response_logps[: min(8, clight_response_logps.numel())].tolist()}")
    print(f"sample_count={sample_count}")
    print(f"prompt_width={prompt_width}")
    print(f"sequences_shape={tuple(sequences.shape)}")
    print(f"attention_mask_shape={tuple(attention_mask.shape)}")
    print(f"response_mask_shape={tuple(response_mask.shape)} sum={int(response_mask.sum().item())}")
    print(f"clight_response_logps_shape={tuple(clight_response_logps.shape)}")
    print(f"finite_clight_response_logps={bool(torch.isfinite(clight_response_logps).all().item())}")
    print("completion[0]:")
    print(payload["completion_text"][0] if payload["completion_text"] else "")
    print("dump_ok=True")


if __name__ == "__main__":
    main()
