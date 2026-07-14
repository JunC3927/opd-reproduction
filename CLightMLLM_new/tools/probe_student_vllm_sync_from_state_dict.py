#!/usr/bin/env python3
"""Single-process probe: load an exported FSDP full state dict into student vLLM.

Use this after ``probe_fsdp_student_vllm_sync.py --stop-after-export``. It is
intentionally not launched by torchrun, so vLLM EngineCore cannot inherit the
training c10d rendezvous.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import fields
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
    parse_torch_dtype,
)
from src.method.vllm_student import VLLMStudentRollout  # noqa: E402
from src.model import load_processor_and_tokenizer  # noqa: E402


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

TORCHRUN_ENV_KEYS = {
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync exported FSDP full state into student vLLM.")
    parser.add_argument(
        "--config",
        default="config/opd/qwen3_vl_geo3k_hf.yaml",
        help="CLight YAML config used for model/tokenizer paths.",
    )
    parser.add_argument(
        "--state-dict-path",
        default="experiments/probes/fsdp_student_full_state.pt",
        help="Path written by the FSDP export probe.",
    )
    parser.add_argument("--vllm-device", default="cuda:0")
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--vllm-max-model-len", type=int, default=1536)
    parser.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ipc-bucket-size-mb", type=int, default=512)
    parser.add_argument("--ipc-use-shm", action="store_true")
    parser.add_argument("--ipc-timeout-sec", type=float, default=900.0)
    parser.add_argument(
        "--sync-dtype",
        default="bfloat16",
        help="Floating dtype used while syncing weights; use 'none' to keep exported dtype.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[state-dict-vllm-probe] {message}", flush=True)


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


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def scrub_torchrun_env() -> None:
    for key in TORCHRUN_ENV_KEYS:
        os.environ.pop(key, None)


def load_exported_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    log(f"state load start: path={path}")
    start = time.time()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    elif isinstance(payload, dict):
        state_dict = payload
        metadata = {"format": "raw_state_dict"}
    else:
        raise TypeError(f"Unexpected state payload type: {type(payload)}")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected state_dict type: {type(state_dict)}")
    tensor_count = sum(1 for value in state_dict.values() if torch.is_tensor(value))
    total_numel = sum(value.numel() for value in state_dict.values() if torch.is_tensor(value))
    log(
        "state load done: "
        f"seconds={time.time() - start:.3f}, tensors={tensor_count}, total_numel={total_numel:,}, "
        f"metadata={metadata}"
    )
    return state_dict, metadata


def state_dict_to_weight_items(state_dict: dict[str, torch.Tensor]) -> list[tuple[str, torch.Tensor]]:
    return [(name, tensor.detach()) for name, tensor in state_dict.items() if torch.is_tensor(tensor)]


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
    scrub_torchrun_env()
    os.chdir(ROOT)

    config_path = resolve_path(args.config)
    state_path = resolve_path(args.state_dict_path)
    log("single-process exported-state -> student vLLM sync probe")
    log(f"config={config_path}")
    log(f"state_dict_path={state_path}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    log(f"VLLM_WORKER_MULTIPROC_METHOD={os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')}")

    (
        _cl_sft_args,
        data_args,
        _loader_args,
        _method_args,
        model_args,
        _optimizer_args,
        _trainer_args,
        _tuning_args,
    ) = parse_yaml_args(str(config_path))

    state_dict, _metadata = load_exported_state(state_path)
    weights = state_dict_to_weight_items(state_dict)

    log("tokenizer load start")
    start = time.time()
    _processor, tokenizer, _common_kwargs = load_processor_and_tokenizer(model_args)
    log(f"tokenizer load done: seconds={time.time() - start:.3f}")

    rollout: VLLMStudentRollout | None = None
    try:
        log(
            "vLLM init start: "
            f"device={args.vllm_device}, dtype={args.vllm_dtype}, "
            f"gpu_memory_utilization={args.vllm_gpu_memory_utilization}, enforce_eager={args.vllm_enforce_eager}"
        )
        start = time.time()
        rollout = VLLMStudentRollout(
            model_path=str(model_args.model_name_or_path),
            tokenizer=tokenizer,
            torch_dtype=args.vllm_dtype,
            trust_remote_code=model_args.trust_remote_code,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            enforce_eager=args.vllm_enforce_eager,
            device=args.vllm_device,
        )
        log(f"vLLM init done: seconds={time.time() - start:.3f}")

        sync_dtype = None if args.sync_dtype.lower() == "none" else parse_torch_dtype(args.sync_dtype)
        log(
            "weight sync start: "
            f"tensors={len(weights)}, sync_dtype={sync_dtype}, "
            f"bucket_size_mb={args.ipc_bucket_size_mb}, use_shm={args.ipc_use_shm}"
        )
        start = time.time()
        summary = rollout.sync_from_weight_items_ipc(
            weights,
            bucket_size_mb=args.ipc_bucket_size_mb,
            use_shm=args.ipc_use_shm,
            timeout_sec=args.ipc_timeout_sec,
            sync_dtype=sync_dtype,
        )
        log(f"weight sync done: seconds={time.time() - start:.3f}, summary={summary}")
        log("RESULT=OK")
    finally:
        log("cleanup start")
        try_cleanup_vllm(rollout)
        del state_dict
        del weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log("cleanup done")


if __name__ == "__main__":
    main()
