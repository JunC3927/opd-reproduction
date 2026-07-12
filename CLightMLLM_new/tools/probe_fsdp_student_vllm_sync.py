#!/usr/bin/env python3
"""Stage-2 probe: export FSDP student weights and sync them into student vLLM.

Run this with torchrun. It intentionally does not train and does not enter the
Lightning Trainer. The goal is to validate the fragile boundary before wiring
student vLLM rollout into OPD:

1. initialize an NCCL training process group;
2. load and FSDP-wrap the HF student on all training ranks;
3. start one student vLLM engine on rank 0, preferably on a GPU outside the
   training ranks;
4. gather a full FSDP state dict with all ranks participating;
5. send rank-0 weights to vLLM with the bucketed IPC path;
6. barrier and clean up without c10d/NCCL timeout.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import fields, replace
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.distributed.fsdp import (
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe FSDP full-state export into student vLLM.")
    parser.add_argument(
        "--config",
        default="config/continual_sft/qwen3_vl_opd_geo3k.yaml",
        help="CLight YAML config to reuse for model/FSDP/vLLM settings.",
    )
    parser.add_argument("--dist-timeout-sec", type=int, default=1800)
    parser.add_argument(
        "--fsdp-min-num-params",
        type=int,
        default=None,
        help="Override trainer.fsdp_min_num_params for this probe.",
    )
    parser.add_argument(
        "--keep-gradient-checkpointing",
        action="store_true",
        help="Keep YAML gradient checkpointing. Default disables it because this probe does not train.",
    )
    parser.add_argument("--skip-vllm", action="store_true", help="Only test FSDP full-state export.")
    parser.add_argument("--vllm-device", default="cuda:4", help="Rank-0 vLLM device, e.g. cuda:4.")
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--vllm-max-model-len", type=int, default=1536)
    parser.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ipc-bucket-size-mb", type=int, default=512)
    parser.add_argument("--ipc-use-shm", action="store_true")
    parser.add_argument("--ipc-timeout-sec", type=float, default=900.0)
    parser.add_argument(
        "--sync-dtype",
        default="bfloat16",
        help="Floating dtype used while syncing weights; use 'none' to keep FSDP full-state dtype.",
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


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_rank0() -> bool:
    return rank() == 0


def log(message: str, *, all_ranks: bool = False) -> None:
    if all_ranks or is_rank0():
        print(f"[fsdp-vllm-probe rank={rank()}] {message}", flush=True)


def init_distributed(timeout_sec: int) -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP student vLLM sync probe requires CUDA.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Run this script with torchrun so RANK/WORLD_SIZE/LOCAL_RANK are set.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=int(timeout_sec)))
    return rank(), world_size(), local_rank, torch.device("cuda", local_rank)


def cuda_memory_line(device: torch.device | str | int | None = None) -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    dev = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    index = torch.cuda.current_device() if dev.index is None else dev.index
    free, total = torch.cuda.mem_get_info(index)
    allocated = torch.cuda.memory_allocated(index)
    reserved = torch.cuda.memory_reserved(index)
    return (
        f"cuda:{index} free={free / 1024**3:.2f}GiB total={total / 1024**3:.2f}GiB "
        f"allocated={allocated / 1024**3:.2f}GiB reserved={reserved / 1024**3:.2f}GiB"
    )


def sync_cuda(device: torch.device, label: str) -> None:
    torch.cuda.synchronize(device)
    log(f"{label}: {cuda_memory_line(device)}", all_ranks=True)


def report_model_parameters(model: torch.nn.Module) -> None:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    log(
        f"student pre-FSDP params: trainable={trainable:,} total={total:,} "
        f"trainable_pct={(100.0 * trainable / total if total else 0.0):.2f}%"
    )


def collect_fsdp_ignored_modules(trainer_args: TrainerArguments, model: torch.nn.Module) -> list[torch.nn.Module]:
    if not trainer_args.fsdp_ignore_lm_head:
        return []
    ignored: list[torch.nn.Module] = []
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, torch.nn.Module):
        ignored.append(lm_head)
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        if isinstance(embeddings, torch.nn.Module) and not any(embeddings is item for item in ignored):
            ignored.append(embeddings)
    if ignored:
        log("FSDP ignored modules: " + ", ".join(type(module).__name__ for module in ignored))
    return ignored


def build_fsdp_model(
    base_model: torch.nn.Module,
    trainer_args: TrainerArguments,
    *,
    device: torch.device,
    fsdp_min_num_params: int | None,
) -> FSDP:
    min_num_params = int(fsdp_min_num_params or trainer_args.fsdp_min_num_params)
    mixed_precision = MixedPrecision(
        param_dtype=None if trainer_args.fsdp_param_dtype is None else parse_torch_dtype(trainer_args.fsdp_param_dtype),
        reduce_dtype=None if trainer_args.fsdp_reduce_dtype is None else parse_torch_dtype(trainer_args.fsdp_reduce_dtype),
        buffer_dtype=None if trainer_args.fsdp_buffer_dtype is None else parse_torch_dtype(trainer_args.fsdp_buffer_dtype),
    )
    auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
    kwargs: dict[str, Any] = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "cpu_offload": CPUOffload(offload_params=trainer_args.fsdp_cpu_offload),
        "use_orig_params": trainer_args.fsdp_use_orig_params,
        "forward_prefetch": trainer_args.fsdp_forward_prefetch,
        "limit_all_gathers": True,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "device_id": device,
    }
    ignored = collect_fsdp_ignored_modules(trainer_args, base_model)
    if ignored:
        kwargs["ignored_modules"] = ignored
    log(
        "FSDP wrap kwargs: "
        f"min_num_params={min_num_params}, mixed_precision={mixed_precision}, "
        f"use_orig_params={trainer_args.fsdp_use_orig_params}, cpu_offload={trainer_args.fsdp_cpu_offload}"
    )
    return FSDP(base_model, **kwargs)


def state_dict_to_weight_items(state_dict: dict[str, torch.Tensor]) -> list[tuple[str, torch.Tensor]]:
    return [(name, tensor.detach()) for name, tensor in state_dict.items() if torch.is_tensor(tensor)]


def collect_full_state_dict(fsdp_model: FSDP) -> dict[str, torch.Tensor]:
    state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    log("FSDP full_state_dict start", all_ranks=True)
    start = time.time()
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, state_cfg):
        state_dict = fsdp_model.state_dict()
    elapsed = time.time() - start
    if is_rank0():
        tensor_count = sum(1 for value in state_dict.values() if torch.is_tensor(value))
        total_numel = sum(value.numel() for value in state_dict.values() if torch.is_tensor(value))
        log(
            "FSDP full_state_dict done: "
            f"seconds={elapsed:.3f}, tensors={tensor_count}, total_numel={total_numel:,}"
        )
    else:
        log(f"FSDP full_state_dict done: seconds={elapsed:.3f}, rank0_only_empty={len(state_dict) == 0}", all_ranks=True)
    return state_dict


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
    os.chdir(ROOT)

    r, ws, local_rank, device = init_distributed(args.dist_timeout_sec)
    rollout: VLLMStudentRollout | None = None
    fsdp_model: FSDP | None = None
    base_model: torch.nn.Module | None = None

    try:
        config_path = resolve_path(args.config)
        log("stage 2 FSDP -> student vLLM sync probe")
        log(f"config={config_path}")
        log(f"rank={r} world_size={ws} local_rank={local_rank} device={device}", all_ranks=True)
        log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        log(f"VLLM_WORKER_MULTIPROC_METHOD={os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')}")

        (
            _cl_sft_args,
            data_args,
            _loader_args,
            _method_args,
            model_args,
            _optimizer_args,
            trainer_args,
            tuning_args,
        ) = parse_yaml_args(config_path)
        if not args.keep_gradient_checkpointing:
            model_args = replace(model_args, gradient_checkpointing=False)
        model_args = replace(model_args, use_cache=False)

        log(
            "HF student load start: "
            f"model={model_args.model_name_or_path}, dtype={model_args.torch_dtype}, "
            f"gradient_checkpointing={model_args.gradient_checkpointing}",
            all_ranks=True,
        )
        start = time.time()
        base_model, _processor, tokenizer = load_vision_language_model(model_args, data_args.template)
        base_model = ModelTuner(tuning_args).apply(base_model)
        base_model.eval()
        if is_rank0():
            report_model_parameters(base_model)
        log(f"HF student load done: seconds={time.time() - start:.3f}", all_ranks=True)

        fsdp_model = build_fsdp_model(
            base_model,
            trainer_args,
            device=device,
            fsdp_min_num_params=args.fsdp_min_num_params,
        )
        fsdp_model.eval()
        sync_cuda(device, "FSDP wrap done")

        if not args.skip_vllm and is_rank0():
            log(
                "vLLM init start: "
                f"device={args.vllm_device}, dtype={args.vllm_dtype}, "
                f"gpu_memory_utilization={args.vllm_gpu_memory_utilization}"
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

        log("post-vLLM-init barrier start", all_ranks=True)
        dist.barrier()
        log("post-vLLM-init barrier done", all_ranks=True)

        state_dict = collect_full_state_dict(fsdp_model)
        dist.barrier()
        log("post-full-state barrier done", all_ranks=True)

        if not args.skip_vllm and is_rank0():
            if rollout is None:
                raise RuntimeError("Rank 0 rollout is unexpectedly None.")
            sync_dtype = None if args.sync_dtype.lower() == "none" else parse_torch_dtype(args.sync_dtype)
            weights = state_dict_to_weight_items(state_dict)
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

        log("post-sync barrier start", all_ranks=True)
        dist.barrier()
        log("post-sync barrier done", all_ranks=True)
        if is_rank0():
            log("RESULT=OK")
    finally:
        if is_rank0():
            log("cleanup start")
            try_cleanup_vllm(rollout)
        del fsdp_model
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
        if is_rank0():
            log("cleanup done")


if __name__ == "__main__":
    main()
