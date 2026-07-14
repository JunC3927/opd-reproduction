#!/usr/bin/env python3
"""Stage-1 probe for on-policy student vLLM rollout weight sync.

This script deliberately stays outside Lightning/FSDP. It verifies the narrow
sequence needed before touching the training loop:

1. load the HF student and one real OPD prompt/image batch;
2. run HF ``generate`` on that batch;
3. load a student vLLM engine and run vLLM ``generate`` on the same prompt;
4. sync the HF student state dict into vLLM;
5. run vLLM ``generate`` again, then clean up.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import TemplateFactory  # noqa: E402
from src.data.module import DatasetBuilder, VLCollator  # noqa: E402
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
from src.method.rollout import RolloutMixin  # noqa: E402
from src.method.vllm_student import VLLMStudentRollout  # noqa: E402
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


class _NoopStrategy:
    def barrier(self) -> None:
        return None


class _RolloutProbe(RolloutMixin):
    def __init__(self, *, model: torch.nn.Module, tokenizer: Any, method_args: MethodArguments) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.method_args = method_args

    @staticmethod
    def model_kwargs(batch: dict[str, Any], include_labels: bool = True) -> dict[str, Any]:
        excluded = {
            "prompt_input_ids",
            "prompt_attention_mask",
            "reference_text",
            "vllm_images",
        }
        kwargs = {key: value for key, value in batch.items() if key not in excluded}
        if not include_labels:
            kwargs.pop("labels", None)
        return kwargs


def log(message: str) -> None:
    print(f"[student-vllm-probe] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe HF student -> vLLM student rollout sync.")
    parser.add_argument(
        "--config",
        default="config/opd/qwen3_vl_geo3k_hf.yaml",
        help="CLight YAML config to reuse for model/data/method settings.",
    )
    parser.add_argument("--stage-index", type=int, default=0, help="CL SFT stage index to sample from.")
    parser.add_argument("--sample-index", type=int, default=0, help="First preprocessed sample index.")
    parser.add_argument("--batch-size", type=int, default=1, help="Probe batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Short smoke rollout length.")
    parser.add_argument("--hf-device", default="cuda:0", help="Device for the HF student generate path.")
    parser.add_argument("--vllm-device", default="cuda:0", help="Device visible inside this process for vLLM.")
    parser.add_argument("--vllm-dtype", default="bfloat16", help="vLLM rollout dtype.")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--vllm-max-model-len", type=int, default=1536)
    parser.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--sync-backend",
        choices=["ipc", "direct", "none"],
        default="ipc",
        help="How to push HF weights into vLLM after the first vLLM generate.",
    )
    parser.add_argument("--ipc-bucket-size-mb", type=int, default=512)
    parser.add_argument("--ipc-use-shm", action="store_true")
    parser.add_argument("--ipc-timeout-sec", type=float, default=600.0)
    parser.add_argument(
        "--sync-dtype",
        default="bfloat16",
        help="Floating dtype used while syncing weights with the IPC backend; use 'none' to keep HF dtype.",
    )
    parser.add_argument(
        "--keep-gradient-checkpointing",
        action="store_true",
        help="Keep config gradient checkpointing enabled. The default disables it for this non-training probe.",
    )
    return parser.parse_args()


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


def resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(ROOT / candidate)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {value}, but CUDA is not available.")
    return device


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def fake_trainer() -> Any:
    return SimpleNamespace(
        local_rank=0,
        global_rank=0,
        is_global_zero=True,
        strategy=_NoopStrategy(),
    )


def load_probe_batch(
    *,
    data_args: DataArguments,
    loader_args: LoaderArguments,
    model_args: ModelArguments,
    model: torch.nn.Module,
    processor: Any,
    tokenizer: Any,
    sample_index: int,
    batch_size: int,
) -> dict[str, Any]:
    template = TemplateFactory.from_args(tokenizer, data_args)
    builder = DatasetBuilder(
        template=template,
        model_args=model_args,
        data_args=replace(
            data_args,
            max_samples=max(data_args.max_samples or 0, sample_index + batch_size),
            preprocessing_num_workers=1,
            log_first_sample=False,
        ),
        tokenizer=tokenizer,
        processor=processor,
        trainer=fake_trainer(),
    )
    log("dataset build start")
    dataset = builder.build()
    log(f"dataset build done: rows={len(dataset)}")
    if sample_index < 0 or sample_index + batch_size > len(dataset):
        raise IndexError(
            f"Requested sample range [{sample_index}, {sample_index + batch_size}) "
            f"from dataset with {len(dataset)} rows."
        )

    collator = VLCollator(
        template=template,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        torch_dtype=parse_torch_dtype(model_args.torch_dtype),
    )
    features = [dataset[int(idx)] for idx in range(sample_index, sample_index + batch_size)]
    batch = collator(features)
    prompt_shape = tuple(batch["prompt_input_ids"].shape)
    image_counts = [len(images) for images in batch.get("vllm_images", [])]
    log(f"batch ready: prompt_shape={prompt_shape}, image_counts={image_counts}")
    return batch


def decode_first_completion(
    *,
    tokenizer: Any,
    sequences: torch.Tensor,
    prompt_width: int,
    label: str,
) -> None:
    first = sequences[0].detach().cpu()
    completion = first[prompt_width:]
    text = tokenizer.decode(completion.tolist(), skip_special_tokens=False)
    text = text.replace("\n", "\\n")
    log(f"{label}: shape={tuple(sequences.shape)}, first_completion={text[:300]!r}")


def weight_items_from_model(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    items = []
    for name, tensor in model.state_dict().items():
        if torch.is_tensor(tensor):
            items.append((name, tensor.detach()))
    return items


def try_cleanup_vllm(rollout: VLLMStudentRollout | None) -> None:
    if rollout is None:
        return
    llm = getattr(rollout, "llm", None)
    for owner_path in ("llm", "llm.llm_engine", "llm.engine_core"):
        owner = llm
        if owner_path != "llm":
            owner = llm
            for attr in owner_path.split(".")[1:]:
                owner = getattr(owner, attr, None)
                if owner is None:
                    break
        if owner is None:
            continue
        for method_name in ("shutdown", "close"):
            method = getattr(owner, method_name, None)
            if callable(method):
                try:
                    log(f"cleanup calling {owner_path}.{method_name}()")
                    method()
                except Exception as exc:
                    log(f"cleanup {owner_path}.{method_name} failed: {type(exc).__name__}: {exc}")
    try:
        del rollout.llm
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    config_path = resolve_path(args.config)
    log("stage 1 student vLLM rollout sync probe")
    log(f"config={config_path}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    log(f"VLLM_WORKER_MULTIPROC_METHOD={os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')}")

    (
        cl_sft_args,
        data_args,
        loader_args,
        method_args,
        model_args,
        _optimizer_args,
        _trainer_args,
        tuning_args,
    ) = parse_yaml_args(config_path)

    if not cl_sft_args.stages:
        raise ValueError("cl_sft.stages is empty.")
    if args.stage_index < 0 or args.stage_index >= len(cl_sft_args.stages):
        raise IndexError(f"stage-index={args.stage_index} outside 0..{len(cl_sft_args.stages) - 1}")

    stage = cl_sft_args.stages[args.stage_index]
    data_args = replace(data_args, dataset=stage.dataset)
    loader_args = replace(
        loader_args,
        per_device_train_batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        shuffle=False,
    )
    method_args = replace(
        method_args,
        rollout_backend="hf",
        rollout_max_new_tokens=args.max_new_tokens,
    )
    if not args.keep_gradient_checkpointing:
        model_args = replace(model_args, gradient_checkpointing=False)

    hf_device = resolve_device(args.hf_device)
    log(
        "HF load start: "
        f"model={model_args.model_name_or_path}, dtype={model_args.torch_dtype}, "
        f"use_cache={model_args.use_cache}, gradient_checkpointing={model_args.gradient_checkpointing}"
    )
    start = time.time()
    model, processor, tokenizer = load_vision_language_model(model_args, data_args.template)
    model = ModelTuner(tuning_args).apply(model)
    model.eval()
    model.to(hf_device)
    log(f"HF load done: seconds={time.time() - start:.3f}, device={hf_device}")

    batch = load_probe_batch(
        data_args=data_args,
        loader_args=loader_args,
        model_args=model_args,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        sample_index=args.sample_index,
        batch_size=args.batch_size,
    )
    batch = move_batch_to_device(batch, hf_device)
    prompt_width = int(batch["prompt_input_ids"].shape[1])

    probe = _RolloutProbe(model=model, tokenizer=tokenizer, method_args=method_args)
    log("HF generate start")
    start = time.time()
    with torch.no_grad():
        hf_sequences = probe.generate_rollout(batch)
    if hf_device.type == "cuda":
        torch.cuda.synchronize(hf_device)
    log(f"HF generate done: seconds={time.time() - start:.3f}")
    decode_first_completion(
        tokenizer=tokenizer,
        sequences=hf_sequences,
        prompt_width=prompt_width,
        label="HF generate",
    )

    trust_remote_code = model_args.trust_remote_code if args.vllm_trust_remote_code is None else args.vllm_trust_remote_code
    log(
        "vLLM init start: "
        f"device={args.vllm_device}, dtype={args.vllm_dtype}, "
        f"gpu_memory_utilization={args.vllm_gpu_memory_utilization}"
    )
    start = time.time()
    rollout: VLLMStudentRollout | None = None
    try:
        rollout = VLLMStudentRollout(
            model_path=str(model_args.model_name_or_path),
            tokenizer=tokenizer,
            torch_dtype=args.vllm_dtype,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            enforce_eager=args.vllm_enforce_eager,
            device=args.vllm_device,
        )
        log(f"vLLM init done: seconds={time.time() - start:.3f}")

        config = getattr(model, "config", None)
        log("vLLM generate before sync start")
        start = time.time()
        before_sequences = rollout.generate(
            batch=batch,
            method_args=method_args,
            image_token_id=getattr(config, "image_token_id", None),
            video_token_id=getattr(config, "video_token_id", None),
            pad_token_id=tokenizer.pad_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        log(f"vLLM generate before sync done: seconds={time.time() - start:.3f}")
        decode_first_completion(
            tokenizer=tokenizer,
            sequences=before_sequences,
            prompt_width=prompt_width,
            label="vLLM before sync",
        )

        if args.sync_backend != "none":
            log(f"weight sync start: backend={args.sync_backend}")
            start = time.time()
            if args.sync_backend == "direct":
                rollout.sync_from_hf_model(model)
                sync_summary: Any = {"path": "direct_load_weights"}
            else:
                sync_dtype = None if args.sync_dtype.lower() == "none" else parse_torch_dtype(args.sync_dtype)
                weights = weight_items_from_model(model)
                log(f"weight sync collected: tensors={len(weights)}, sync_dtype={sync_dtype}")
                sync_summary = rollout.sync_from_weight_items_ipc(
                    weights,
                    bucket_size_mb=args.ipc_bucket_size_mb,
                    use_shm=args.ipc_use_shm,
                    timeout_sec=args.ipc_timeout_sec,
                    sync_dtype=sync_dtype,
                )
            log(f"weight sync done: seconds={time.time() - start:.3f}, summary={sync_summary}")
        else:
            log("weight sync skipped")

        log("vLLM generate after sync start")
        start = time.time()
        after_sequences = rollout.generate(
            batch=batch,
            method_args=method_args,
            image_token_id=getattr(config, "image_token_id", None),
            video_token_id=getattr(config, "video_token_id", None),
            pad_token_id=tokenizer.pad_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        log(f"vLLM generate after sync done: seconds={time.time() - start:.3f}")
        decode_first_completion(
            tokenizer=tokenizer,
            sequences=after_sequences,
            prompt_width=prompt_width,
            label="vLLM after sync",
        )
        log("RESULT=OK")
    finally:
        log("cleanup start")
        try_cleanup_vllm(rollout)
        del probe
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log("cleanup done")


if __name__ == "__main__":
    main()
