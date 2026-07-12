#!/usr/bin/env python3
"""Stage-2 probe: export FSDP student weights and sync them into student vLLM.

Run this with torchrun. It intentionally does not train and does not enter the
Lightning Trainer. The goal is to validate the fragile boundary before wiring
student vLLM rollout into OPD:

1. initialize an NCCL training process group;
2. load and FSDP-wrap the HF student on all training ranks;
3. gather a full FSDP state dict with all ranks participating;
4. optionally save the full state dict and stop, so vLLM can be launched from a
   separate non-torchrun process;
5. optionally send the gathered state directly to a standalone student vLLM
   server over bucketed IPC;
6. otherwise destroy the training process group and try in-process rank-0 vLLM
   sync. Prefer standalone-server modes when debugging c10d issues.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import sys
import time
import traceback
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
from src.method.vllm_student_client import RemoteStudentRollout  # noqa: E402
from src.method.vllm_student import describe_weight_items_for_ipc  # noqa: E402
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
    parser.add_argument(
        "--export-state-path",
        default="experiments/probes/fsdp_student_full_state.pt",
        help="Path where rank 0 saves the gathered full state dict.",
    )
    parser.add_argument(
        "--skip-save-state",
        action="store_true",
        help="Do not write the gathered FSDP full state dict to disk.",
    )
    parser.add_argument(
        "--full-state-offload-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FSDP FullStateDictConfig(offload_to_cpu=...). Disable to keep rank0 full state on GPU.",
    )
    parser.add_argument(
        "--sync-mode",
        choices=("local_vllm", "remote_ipc", "remote_ipc_summon", "export_only"),
        default="local_vllm",
        help=(
            "local_vllm preserves the original in-process vLLM smoke path; "
            "remote_ipc sends rank0 full-state weights to a standalone student vLLM server; "
            "remote_ipc_summon sends rank0 summoned FSDP parameter views without building a CPU full-state dict; "
            "export_only only gathers/saves the full state."
        ),
    )
    parser.add_argument(
        "--stop-after-export",
        action="store_true",
        help="Stop after saving the FSDP full state dict. Use the single-process sync script next.",
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
    parser.add_argument("--remote-student-host", default="127.0.0.1")
    parser.add_argument("--remote-student-port", type=int, default=29588)
    parser.add_argument(
        "--remote-sync-device",
        default=None,
        help="Sender device for remote IPC. Leave unset to use current rank0 CUDA device or SHM CPU path.",
    )
    parser.add_argument(
        "--summon-rank0-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FSDP.summon_full_params(rank0_only=...) for remote_ipc_summon.",
    )
    parser.add_argument(
        "--summon-offload-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use FSDP.summon_full_params(offload_to_cpu=...) for remote_ipc_summon.",
    )
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


def scrub_torchrun_env_for_vllm() -> None:
    for key in TORCHRUN_ENV_KEYS:
        os.environ.pop(key, None)


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


def distributed_barrier(label: str, *, local_rank: int | None = None) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    log(f"{label} barrier start", all_ranks=True)
    if local_rank is not None:
        try:
            dist.barrier(device_ids=[int(local_rank)])
        except TypeError:
            dist.barrier()
    else:
        dist.barrier()
    log(f"{label} barrier done", all_ranks=True)


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


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


def normalize_summoned_param_name(name: str) -> str:
    for prefix in ("_fsdp_wrapped_module.", "module."):
        while name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def save_exported_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    path: str,
    config_path: str,
    model_path: str | None,
    ws: int,
) -> None:
    if not is_rank0():
        return
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"saving FSDP full_state_dict: path={output_path}")
    start = time.time()
    torch.save(
        {
            "format": "clight_fsdp_full_state_probe_v1",
            "config_path": str(config_path),
            "model_path": str(model_path),
            "world_size": int(ws),
            "state_dict": state_dict,
        },
        output_path,
    )
    size_gib = output_path.stat().st_size / 1024**3
    log(f"saved FSDP full_state_dict: seconds={time.time() - start:.3f}, size={size_gib:.2f}GiB")


def collect_full_state_dict(fsdp_model: FSDP, *, offload_to_cpu: bool) -> dict[str, torch.Tensor]:
    state_cfg = FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=True)
    log(f"FSDP full_state_dict start: offload_to_cpu={offload_to_cpu}", all_ranks=True)
    start = time.time()
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, state_cfg):
        state_dict = fsdp_model.state_dict()
    elapsed = time.time() - start
    if is_rank0():
        tensor_count = sum(1 for value in state_dict.values() if torch.is_tensor(value))
        total_numel = sum(value.numel() for value in state_dict.values() if torch.is_tensor(value))
        devices = sorted({str(value.device) for value in state_dict.values() if torch.is_tensor(value)})
        log(
            "FSDP full_state_dict done: "
            f"seconds={elapsed:.3f}, tensors={tensor_count}, total_numel={total_numel:,}, devices={devices}"
        )
    else:
        log(f"FSDP full_state_dict done: seconds={elapsed:.3f}, rank0_only_empty={len(state_dict) == 0}", all_ranks=True)
    return state_dict


def sync_remote_student_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    host: str,
    port: int,
    timeout_sec: float,
    bucket_size_mb: int,
    use_shm: bool,
    sync_device: str | None,
    sync_dtype: torch.dtype | None,
) -> dict[str, Any]:
    return sync_remote_student_from_weight_items(
        state_dict_to_weight_items(state_dict),
        host=host,
        port=port,
        timeout_sec=timeout_sec,
        bucket_size_mb=bucket_size_mb,
        use_shm=use_shm,
        sync_device=sync_device,
        sync_dtype=sync_dtype,
        source_label="full_state_dict",
    )


def sync_remote_student_from_weight_items(
    weights: list[tuple[str, torch.Tensor]],
    *,
    host: str,
    port: int,
    timeout_sec: float,
    bucket_size_mb: int,
    use_shm: bool,
    sync_device: str | None,
    sync_dtype: torch.dtype | None,
    source_label: str,
) -> dict[str, Any]:
    weight_stats = describe_weight_items_for_ipc(weights, sync_dtype=sync_dtype)
    log(
        "remote student IPC sync start: "
        f"source={source_label}, server={host}:{port}, tensors={len(weights)}, sync_device={sync_device}, "
        f"sync_dtype={sync_dtype}, bucket_size_mb={bucket_size_mb}, use_shm={use_shm}, "
        f"weight_stats={weight_stats}"
    )
    largest_nbytes = weight_stats.get("largest_nbytes")
    if use_shm and largest_nbytes is not None and int(largest_nbytes) > (int(bucket_size_mb) << 20):
        raise ValueError(
            "Remote SHM sync cannot start because the bucket is smaller than the largest tensor: "
            f"largest={weight_stats.get('largest_name')} "
            f"estimated={int(largest_nbytes) / 1024**2:.1f}MiB, "
            f"bucket_size_mb={bucket_size_mb}. Increase --ipc-bucket-size-mb."
        )
    client = RemoteStudentRollout(host=host, port=port, timeout=timeout_sec)
    ping = client.ping()
    log(f"remote student ping: {ping}")
    start = time.time()
    try:
        response = client.sync_weight_items_ipc(
            weights,
            bucket_size_mb=bucket_size_mb,
            use_shm=use_shm,
            device=sync_device,
            sync_dtype=sync_dtype,
        )
    except Exception:
        log("remote student IPC sync failed:\n" + traceback.format_exc())
        raise
    response["client_total_sec"] = time.time() - start
    log(f"remote student IPC sync done: seconds={response['client_total_sec']:.3f}, response={response}")
    return response


@contextlib.contextmanager
def summon_full_params_compat(
    fsdp_model: FSDP,
    *,
    rank0_only: bool,
    offload_to_cpu: bool,
):
    kwargs = {
        "writeback": False,
        "recurse": True,
        "rank0_only": rank0_only,
        "offload_to_cpu": offload_to_cpu,
    }
    try:
        ctx = FSDP.summon_full_params(fsdp_model, **kwargs)
    except TypeError:
        log(
            "FSDP.summon_full_params does not accept rank0_only/offload_to_cpu in this torch build; "
            "falling back to writeback=False,recurse=True",
            all_ranks=True,
        )
        ctx = FSDP.summon_full_params(fsdp_model, writeback=False, recurse=True)
    with ctx:
        yield


def sync_remote_student_from_summoned_fsdp(
    fsdp_model: FSDP,
    *,
    host: str,
    port: int,
    timeout_sec: float,
    bucket_size_mb: int,
    use_shm: bool,
    sync_device: str | None,
    sync_dtype: torch.dtype | None,
    rank0_only: bool,
    offload_to_cpu: bool,
    local_rank: int,
) -> dict[str, Any] | None:
    log(
        "FSDP summon_full_params start: "
        f"rank0_only={rank0_only}, offload_to_cpu={offload_to_cpu}",
        all_ranks=True,
    )
    start = time.time()
    response = None
    with summon_full_params_compat(fsdp_model, rank0_only=rank0_only, offload_to_cpu=offload_to_cpu):
        log(f"FSDP summon_full_params entered: seconds={time.time() - start:.3f}", all_ranks=True)
        if is_rank0():
            weights: list[tuple[str, torch.Tensor]] = []
            seen: set[str] = set()
            for name, param in fsdp_model.named_parameters():
                if not torch.is_tensor(param):
                    continue
                normalized = normalize_summoned_param_name(name)
                if normalized in seen:
                    continue
                seen.add(normalized)
                weights.append((normalized, param.detach()))

            devices = sorted({str(tensor.device) for _, tensor in weights})
            dtypes = sorted({str(tensor.dtype) for _, tensor in weights})
            total_numel = sum(tensor.numel() for _, tensor in weights)
            log(
                "FSDP summoned parameter views ready: "
                f"tensors={len(weights)}, total_numel={total_numel:,}, devices={devices}, dtypes={dtypes}"
            )
            response = sync_remote_student_from_weight_items(
                weights,
                host=host,
                port=port,
                timeout_sec=timeout_sec,
                bucket_size_mb=bucket_size_mb,
                use_shm=use_shm,
                sync_device=sync_device,
                sync_dtype=sync_dtype,
                source_label="summon_full_params",
            )
        distributed_barrier("inside-summon-remote-sync", local_rank=local_rank)
    return response


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
    dist_destroyed = False

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

        distributed_barrier("post-FSDP-wrap", local_rank=local_rank)
        if args.sync_mode == "remote_ipc_summon":
            sync_dtype = None if args.sync_dtype.lower() == "none" else parse_torch_dtype(args.sync_dtype)
            sync_remote_student_from_summoned_fsdp(
                fsdp_model,
                host=args.remote_student_host,
                port=args.remote_student_port,
                timeout_sec=args.ipc_timeout_sec,
                bucket_size_mb=args.ipc_bucket_size_mb,
                use_shm=args.ipc_use_shm,
                sync_device=args.remote_sync_device,
                sync_dtype=sync_dtype,
                rank0_only=args.summon_rank0_only,
                offload_to_cpu=args.summon_offload_to_cpu,
                local_rank=local_rank,
            )
            if is_rank0():
                log("RESULT=OK")
            distributed_barrier("post-remote-ipc-summon-sync", local_rank=local_rank)
            return

        state_dict = collect_full_state_dict(fsdp_model, offload_to_cpu=args.full_state_offload_to_cpu)
        if args.skip_save_state:
            log("skip_save_state requested; not writing FSDP full_state_dict to disk")
        else:
            save_exported_state_dict(
                state_dict,
                path=args.export_state_path,
                config_path=config_path,
                model_path=model_args.model_name_or_path,
                ws=ws,
            )
        distributed_barrier("post-state-export", local_rank=local_rank)
        distributed_barrier("post-full-state", local_rank=local_rank)

        if args.sync_mode == "remote_ipc":
            sync_dtype = None if args.sync_dtype.lower() == "none" else parse_torch_dtype(args.sync_dtype)
            if is_rank0():
                sync_remote_student_from_state_dict(
                    state_dict,
                    host=args.remote_student_host,
                    port=args.remote_student_port,
                    timeout_sec=args.ipc_timeout_sec,
                    bucket_size_mb=args.ipc_bucket_size_mb,
                    use_shm=args.ipc_use_shm,
                    sync_device=args.remote_sync_device,
                    sync_dtype=sync_dtype,
                )
                log("RESULT=OK")
            distributed_barrier("post-remote-ipc-sync", local_rank=local_rank)
            return

        if args.stop_after_export or args.sync_mode == "export_only" or args.skip_vllm:
            log("export-only requested; skipping vLLM in torchrun process", all_ranks=True)
            if is_rank0():
                log("RESULT=OK")
            return

        log("destroying training process group before vLLM init", all_ranks=True)
        destroy_distributed()
        dist_destroyed = True
        log("training process group destroyed", all_ranks=True)

        del fsdp_model
        del base_model
        fsdp_model = None
        base_model = None
        gc.collect()
        torch.cuda.empty_cache()

        if is_rank0():
            # Keep CUDA_VISIBLE_DEVICES for device remapping, but remove torchrun
            # rendezvous variables so vLLM EngineCore cannot reconnect to the
            # training TCPStore.
            scrub_torchrun_env_for_vllm()
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
        if not dist_destroyed and dist.is_available() and dist.is_initialized():
            destroy_distributed()
        if is_rank0():
            log("cleanup done")


if __name__ == "__main__":
    main()
