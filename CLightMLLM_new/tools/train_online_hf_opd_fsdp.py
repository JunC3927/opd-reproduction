import argparse
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm may not be installed in lean envs.
    tqdm = None

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from replay_verl_opd_trace import (  # noqa: E402
    build_mm_kwargs,
    build_mm_token_type_ids,
    build_optimizer,
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
from replay_verl_opd_trace_fsdp import (  # noqa: E402
    compute_fsdp_update_stats,
    format_update_probe_names,
    init_distributed,
    reduce_sum,
    save_fsdp_hf_model,
    select_fsdp_update_probes,
    split_contiguous_rows,
)
from src.method.vllm_teacher_client import RemoteTeacherScorer  # noqa: E402
from src.method.vllm_student import VLLMStudentRollout  # noqa: E402
from src.model import ModelTuner, load_vision_language_model  # noqa: E402


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def rank_print(*args: Any, **kwargs: Any) -> None:
    if is_rank0():
        print(*args, **kwargs)


def progress_write(progress: Any, message: str) -> None:
    if progress is not None:
        progress.write(message)
    else:
        print(message, flush=True)


def estimate_initial_total_updates(paths: list[str], args: argparse.Namespace) -> int:
    if args.max_updates > 0:
        return int(args.max_updates)
    # In the VERL trace dump layout used here each chunk file is normally one update.
    # If a file contains multiple update groups, the tqdm total is adjusted after loading it.
    return int(len(paths) * args.epochs)


def trace_sort_key(path: str) -> tuple[int, int, int, str]:
    name = Path(path).name
    match = re.search(r"dump(\d+)_step(\d+)_chunk(\d+)", name)
    if match:
        dump_idx, step_idx, chunk_idx = (int(value) for value in match.groups())
        return dump_idx, step_idx, chunk_idx, name
    return 10**12, 10**12, 10**12, name


def row_value(values: Any, row: int) -> Any:
    if values is None:
        return None
    item = values[row]
    if hasattr(item, "item"):
        try:
            return item.item()
        except ValueError:
            return item
    return item


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return [value]


def normalize_multi_modal_data(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        out = dict(value)
        if "images" in out and "image" not in out:
            out["image"] = out["images"]
        if "image" in out and "images" not in out:
            out["images"] = out["image"]
        return out
    images = to_list(value)
    if not images:
        return None
    return {"images": images, "image": images}


def mm_data_from_non_tensor(non_tensor: dict[str, Any], row: int) -> dict[str, Any] | None:
    return normalize_multi_modal_data(row_value(non_tensor.get("multi_modal_data"), row))


def images_from_mm_data(mm_data: dict[str, Any] | None) -> list[Any]:
    if not mm_data:
        return []
    return to_list(mm_data.get("images", mm_data.get("image")))


def mm_kwargs_from_non_tensor(non_tensor: dict[str, Any], row: int) -> dict[str, Any] | None:
    value = row_value(non_tensor.get("mm_processor_kwargs"), row)
    return value if isinstance(value, dict) else None


def vllm_images_from_non_tensor(non_tensor: dict[str, Any], row: int) -> list[Any]:
    value = row_value(non_tensor.get("vllm_images"), row)
    mm_data = normalize_multi_modal_data(value)
    images = images_from_mm_data(mm_data)
    if images:
        return images
    return images_from_mm_data(mm_data_from_non_tensor(non_tensor, row))


def rollout_method_args(args: argparse.Namespace) -> SimpleNamespace:
    top_k = int(args.rollout_top_k)
    return SimpleNamespace(
        rollout_max_new_tokens=int(args.response_width),
        rollout_do_sample=bool(args.rollout_do_sample),
        rollout_temperature=float(args.rollout_temperature),
        rollout_top_p=float(args.rollout_top_p),
        rollout_top_k=None if top_k <= 0 else top_k,
    )


def autocast_context(dtype: str):
    if dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def parse_sync_dtype(dtype: str) -> torch.dtype | None:
    normalized = str(dtype).lower().replace("torch.", "")
    if normalized in {"none", "fp32", "float32"}:
        return None if normalized == "none" else torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported sync dtype {dtype!r}.")


def sample_next_token(logits: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    logits = logits.float()
    if not args.rollout_do_sample:
        return logits.argmax(dim=-1)

    temperature = max(float(args.rollout_temperature), 1e-6)
    logits = logits / temperature

    top_k = int(args.rollout_top_k) if args.rollout_top_k is not None else -1
    if top_k > 0 and top_k < logits.shape[-1]:
        values, _indices = torch.topk(logits, k=top_k, dim=-1)
        cutoff = values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    top_p = float(args.rollout_top_p)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def completion_mask(responses: torch.Tensor, *, pad_token_id: int, eos_token_id: int | None) -> torch.Tensor:
    mask = responses.ne(pad_token_id)
    if eos_token_id is not None:
        eos_seen = responses.eq(eos_token_id).cumsum(dim=1).bool()
        before_or_at_first_eos = torch.cat(
            [torch.ones_like(mask[:, :1], dtype=torch.bool), ~eos_seen[:, :-1]],
            dim=1,
        )
        mask = mask & before_or_at_first_eos
    return mask.long()


def pad_or_truncate_responses(
    responses: torch.Tensor,
    *,
    response_width: int,
    pad_token_id: int,
) -> torch.Tensor:
    if responses.shape[1] > response_width:
        return responses[:, :response_width]
    if responses.shape[1] == response_width:
        return responses
    pad = torch.full(
        (responses.shape[0], response_width - responses.shape[1]),
        pad_token_id,
        dtype=responses.dtype,
        device=responses.device,
    )
    return torch.cat([responses, pad], dim=1)


def generate_local_sequences(
    *,
    model: FSDP,
    base_model: torch.nn.Module,
    student_rollout: VLLMStudentRollout | None,
    batch: dict[str, Any],
    non_tensor: dict[str, Any],
    mm_inputs: list[Any],
    position_ids_cpu: torch.Tensor | None,
    row_start: int,
    row_end: int,
    prompt_width: int,
    response_width: int,
    tokenizer: Any,
    device: torch.device,
    image_token_id: int | None,
    video_token_id: int | None,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = batch["prompts"][row_start:row_end].to(device).long()
    attention_mask = batch["attention_mask"][row_start:row_end, :prompt_width].to(device).long()
    if position_ids_cpu is not None and args.generate_position_ids_mode != "none":
        raise NotImplementedError(
            "Non-default generate_position_ids_mode is intentionally unsupported for now. "
            "Use official HF position handling for online generation."
        )

    was_training = model.training
    model.eval()
    responses = torch.full(
        (row_end - row_start, response_width),
        int(tokenizer.pad_token_id),
        dtype=torch.long,
        device=device,
    )
    finished = torch.zeros(row_end - row_start, dtype=torch.bool, device=device)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = int(tokenizer.pad_token_id)
    use_kv_cache = args.rollout_backend in {"manual_cache", "hf_generate"}
    past_key_values = None

    with torch.no_grad():
        if args.rollout_backend in {"vllm_single", "vllm_ipc"}:
            if student_rollout is None:
                raise RuntimeError(f"rollout_backend={args.rollout_backend} requires a student_rollout instance.")
            rollout_batch = {
                "prompt_input_ids": sequences,
                "prompt_attention_mask": attention_mask,
                "vllm_images": [
                    vllm_images_from_non_tensor(non_tensor, row)
                    for row in range(row_start, row_end)
                ],
            }
            generated = student_rollout.generate(
                batch=rollout_batch,
                method_args=rollout_method_args(args),
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                pad_token_id=pad_token_id,
            )
            responses = pad_or_truncate_responses(
                generated[:, prompt_width:],
                response_width=response_width,
                pad_token_id=pad_token_id,
            )
            if was_training:
                model.train()
            responses_cpu = responses.detach().cpu().long()
            response_mask = completion_mask(
                responses_cpu,
                pad_token_id=pad_token_id,
                eos_token_id=None if eos_token_id is None else int(eos_token_id),
            )
            input_ids = torch.cat([batch["prompts"][row_start:row_end].cpu().long(), responses_cpu], dim=1)
            prompt_attention_mask = batch["attention_mask"][row_start:row_end, :prompt_width].cpu().long()
            attention_mask_cpu = torch.cat([prompt_attention_mask, response_mask], dim=1)
            return input_ids, attention_mask_cpu, response_mask

        for step in range(response_width):
            if use_kv_cache and past_key_values is not None:
                input_ids = sequences[:, -1:]
                forward_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "cache_position": torch.arange(
                        attention_mask.shape[1] - input_ids.shape[1],
                        attention_mask.shape[1],
                        dtype=torch.long,
                        device=device,
                    ),
                }
            else:
                input_ids = sequences
                forward_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": bool(use_kv_cache),
                }
                forward_kwargs.update(build_mm_kwargs(mm_inputs, row_start, row_end, device))
                mm_token_type_ids = build_mm_token_type_ids(base_model, input_ids)
                if mm_token_type_ids is not None and "image_grid_thw" in forward_kwargs:
                    forward_kwargs["mm_token_type_ids"] = mm_token_type_ids

            with autocast_context(args.generate_amp_dtype):
                outputs = model(**forward_kwargs)
            if use_kv_cache:
                past_key_values = getattr(outputs, "past_key_values", None)
            next_token = sample_next_token(outputs.logits[:, -1, :], args).long()
            next_token = torch.where(finished, torch.full_like(next_token, pad_token_id), next_token)
            responses[:, step] = next_token

            token_attention = (~finished).long()
            sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([attention_mask, token_attention.unsqueeze(1)], dim=1)
            if eos_token_id is not None:
                finished = finished | next_token.eq(int(eos_token_id))

            # FSDP forward contains collectives, so all ranks must run the same
            # number of generation iterations. Only stop early when every rank
            # has finished all of its local samples.
            local_done = torch.tensor(
                1 if bool(finished.all().item()) else 0,
                dtype=torch.int32,
                device=device,
            )
            dist.all_reduce(local_done, op=dist.ReduceOp.MIN)
            if int(local_done.item()) == 1:
                break

    if was_training:
        model.train()
    sync_cuda(device, f"generate rows {row_start}:{row_end}")

    responses_cpu = responses.detach().cpu().long()
    response_mask = completion_mask(
        responses_cpu,
        pad_token_id=pad_token_id,
        eos_token_id=None if eos_token_id is None else int(eos_token_id),
    )
    input_ids = torch.cat([batch["prompts"][row_start:row_end].cpu().long(), responses_cpu], dim=1)
    prompt_attention_mask = batch["attention_mask"][row_start:row_end, :prompt_width].cpu().long()
    attention_mask = torch.cat([prompt_attention_mask, response_mask], dim=1)
    return input_ids, attention_mask, response_mask


def collect_rank0_fsdp_weights(model: FSDP) -> list[tuple[str, torch.Tensor]] | None:
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state = model.state_dict()
    if not is_rank0():
        return None
    return [(name, tensor.detach()) for name, tensor in state.items() if torch.is_tensor(tensor)]


def scatter_tensor_from_rank0(
    tensor: torch.Tensor | None,
    *,
    local_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    world_size: int,
) -> torch.Tensor:
    output = torch.empty(local_shape, dtype=dtype, device=device)
    scatter_list = None
    if is_rank0():
        if tensor is None:
            raise RuntimeError("rank0 scatter source tensor is missing.")
        scatter_list = [chunk.contiguous().to(device) for chunk in torch.chunk(tensor, world_size, dim=0)]
    dist.scatter(output, scatter_list=scatter_list, src=0)
    return output.detach().cpu()


def generate_vllm_ipc_global_sequences(
    *,
    model: FSDP,
    base_model: torch.nn.Module,
    student_rollout: VLLMStudentRollout,
    batch: dict[str, Any],
    non_tensor: dict[str, Any],
    mm_inputs: list[Any],
    position_ids_cpu: torch.Tensor | None,
    group_start: int,
    group_end: int,
    local_rows: int,
    prompt_width: int,
    response_width: int,
    tokenizer: Any,
    device: torch.device,
    image_token_id: int | None,
    video_token_id: int | None,
    args: argparse.Namespace,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    sync_start = time.time()
    weights = collect_rank0_fsdp_weights(model)
    sync_info: dict[str, Any] = {}
    if is_rank0():
        sync_info = student_rollout.sync_from_weight_items_ipc(
            weights or [],
            bucket_size_mb=args.student_vllm_ipc_bucket_mb,
            use_shm=args.student_vllm_ipc_use_shm,
            timeout_sec=args.student_vllm_ipc_timeout_sec,
            sync_dtype=parse_sync_dtype(args.student_vllm_sync_dtype),
        )
    dist.barrier()
    sync_sec = time.time() - sync_start

    rollout_start = time.time()
    full_sequences = None
    full_attention = None
    full_response_mask = None
    if is_rank0():
        full_sequences, full_attention, full_response_mask = generate_local_sequences(
            model=model,
            base_model=base_model,
            student_rollout=student_rollout,
            batch=batch,
            non_tensor=non_tensor,
            mm_inputs=mm_inputs,
            position_ids_cpu=position_ids_cpu,
            row_start=group_start,
            row_end=group_end,
            prompt_width=prompt_width,
            response_width=response_width,
            tokenizer=tokenizer,
            device=device,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            args=args,
        )
    seq_len = prompt_width + response_width
    local_sequences_cpu = scatter_tensor_from_rank0(
        full_sequences,
        local_shape=(local_rows, seq_len),
        dtype=torch.long,
        device=device,
        world_size=world_size,
    )
    local_attention_cpu = scatter_tensor_from_rank0(
        full_attention,
        local_shape=(local_rows, seq_len),
        dtype=torch.long,
        device=device,
        world_size=world_size,
    )
    local_response_mask_cpu = scatter_tensor_from_rank0(
        full_response_mask,
        local_shape=(local_rows, response_width),
        dtype=torch.long,
        device=device,
        world_size=world_size,
    )
    rollout_sec = time.time() - rollout_start
    metrics = {
        "student_vllm_sync_sec": float(sync_sec),
        "student_vllm_rollout_sec": float(rollout_sec),
        "student_vllm_weight_count": int(sync_info.get("weight_count", 0)) if is_rank0() else 0,
        "student_vllm_sync_sender_sec": float(sync_info.get("sender_sec", 0.0)) if is_rank0() else 0.0,
        "student_vllm_sync_path": str(sync_info.get("path", "")) if is_rank0() else "",
    }
    return local_sequences_cpu, local_attention_cpu, local_response_mask_cpu, metrics


def gather_local_mm(
    *,
    non_tensor: dict[str, Any],
    row_start: int,
    row_end: int,
) -> tuple[list[list[Any]], list[dict[str, Any] | None], list[dict[str, Any] | None]]:
    images_per_sample = []
    mm_data_per_sample = []
    mm_kwargs_per_sample = []
    for row in range(row_start, row_end):
        mm_data = mm_data_from_non_tensor(non_tensor, row)
        mm_data_per_sample.append(mm_data)
        images_per_sample.append(images_from_mm_data(mm_data))
        mm_kwargs_per_sample.append(mm_kwargs_from_non_tensor(non_tensor, row))
    return images_per_sample, mm_data_per_sample, mm_kwargs_per_sample


def flatten_gathered(items: list[Any], key: str) -> list[Any]:
    out = []
    for item in items:
        out.extend(item[key])
    return out


def batched_teacher_score(
    *,
    scorer: RemoteTeacherScorer,
    local_sequences: torch.Tensor,
    local_attention_mask: torch.Tensor,
    local_images: list[list[Any]],
    local_mm_data: list[dict[str, Any] | None],
    local_mm_kwargs: list[dict[str, Any] | None],
    image_token_id: int | None,
    video_token_id: int | None,
    pad_token_id: int,
    topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_shape = tuple(local_sequences.shape)

    seq_chunks = [torch.empty_like(local_sequences) for _ in range(world_size)]
    mask_chunks = [torch.empty_like(local_attention_mask) for _ in range(world_size)]
    dist.all_gather(seq_chunks, local_sequences.contiguous())
    dist.all_gather(mask_chunks, local_attention_mask.contiguous())

    local_objects = {
        "images": local_images,
        "mm_data": local_mm_data,
        "mm_kwargs": local_mm_kwargs,
    }
    gathered_objects: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_objects, local_objects)

    local_bsz = local_shape[0]
    seq_len = local_shape[1]
    local_logps = torch.empty((local_bsz, seq_len - 1, topk), dtype=torch.float32, device=device)
    local_ids = torch.empty((local_bsz, seq_len - 1, topk), dtype=torch.long, device=device)

    if rank == 0:
        global_sequences = torch.cat(seq_chunks, dim=0)
        global_attention_mask = torch.cat(mask_chunks, dim=0)
        teacher_logps, teacher_ids = scorer.score(
            sequences=global_sequences,
            attention_mask=global_attention_mask,
            images_per_sample=flatten_gathered(gathered_objects, "images"),
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            pad_token_id=pad_token_id,
            mm_processor_kwargs_per_sample=flatten_gathered(gathered_objects, "mm_kwargs"),
            multi_modal_data_per_sample=flatten_gathered(gathered_objects, "mm_data"),
        )
        teacher_logps = teacher_logps.to(device=device, dtype=torch.float32).contiguous()
        teacher_ids = teacher_ids.to(device=device, dtype=torch.long).contiguous()
        logp_scatter = list(torch.chunk(teacher_logps, world_size, dim=0))
        id_scatter = list(torch.chunk(teacher_ids, world_size, dim=0))
    else:
        logp_scatter = None
        id_scatter = None

    dist.scatter(local_logps, scatter_list=logp_scatter, src=0)
    dist.scatter(local_ids, scatter_list=id_scatter, src=0)
    return local_logps, local_ids


def maybe_dump_online_trace(
    *,
    args: argparse.Namespace,
    update_step: int,
    epoch: int,
    source_path: str,
    row_start: int,
    row_end: int,
    local_sequences: torch.Tensor,
    local_attention_mask: torch.Tensor,
    local_response_mask: torch.Tensor,
    local_teacher_logps: torch.Tensor,
    local_teacher_ids: torch.Tensor,
) -> None:
    if not args.dump_traces:
        return
    rank = dist.get_rank()
    dump_dir = Path(args.dump_trace_dir or Path(args.metrics_output or ".").parent / "online_traces")
    dump_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "clight_online_hf_opd_fsdp_trace_rank_v1",
            "update_step": update_step,
            "epoch": epoch,
            "rank": rank,
            "source_path": source_path,
            "source_row_start": row_start,
            "source_row_end": row_end,
            "batch": {
                "input_ids": local_sequences.detach().cpu(),
                "attention_mask": local_attention_mask.detach().cpu(),
                "response_mask": local_response_mask.detach().cpu(),
                "teacher_ids": local_teacher_ids.detach().cpu(),
                "teacher_logprobs": local_teacher_logps.detach().cpu(),
            },
        },
        dump_dir / f"online_step{update_step:06d}_rank{rank:02d}.pt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run online OPD with current HF/FSDP student rollout and remote vLLM "
            "teacher top-k scoring, using VERL trace prompts/images as the data source."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("traces", nargs="+", help="VERL trace dump file(s) or glob pattern(s).")
    parser.add_argument("--teacher-host", default="127.0.0.1")
    parser.add_argument("--teacher-port", type=int, default=29577)
    parser.add_argument("--teacher-timeout", type=float, default=1800.0)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--samples-per-update", type=int, default=12)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--response-width", type=int, default=512)
    parser.add_argument(
        "--rollout-backend",
        choices=("manual", "manual_cache", "hf_generate", "vllm_single", "vllm_ipc"),
        default="manual",
        help=(
            "manual recomputes the full sequence every token; manual_cache uses FSDP forward with KV cache; "
            "hf_generate is kept as a compatibility alias for manual_cache; "
            "vllm_single uses a single-process student vLLM rollout and syncs from the HF model before each update; "
            "vllm_ipc uses rank0 student vLLM rollout with VERL bucketed IPC weight sync."
        ),
    )
    parser.add_argument("--rollout-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--rollout-top-k", type=int, default=-1)
    parser.add_argument("--student-vllm-dtype", default="bfloat16")
    parser.add_argument("--student-vllm-gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--student-vllm-max-model-len", type=int, default=1537)
    parser.add_argument("--student-vllm-max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--student-vllm-max-num-seqs", type=int, default=None)
    parser.add_argument("--student-vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--student-vllm-ipc-bucket-mb", type=int, default=512)
    parser.add_argument("--student-vllm-ipc-use-shm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--student-vllm-ipc-timeout-sec", type=float, default=900.0)
    parser.add_argument("--student-vllm-sync-dtype", choices=("none", "fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--generate-amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--train-amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--generate-position-ids-mode", choices=("none",), default="none")
    parser.add_argument("--position-ids-mode", choices=("none", "trace3", "trace4", "trace_batch4"), default="none")
    parser.add_argument("--teacher-shift-offset", type=int, default=-1)
    parser.add_argument("--log-prob-min-clamp", type=float, default=-10.0)
    parser.add_argument("--loss-max-clamp", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--save-model-dir", default=None)
    parser.add_argument("--swanlab-project", default=None)
    parser.add_argument("--swanlab-experiment-name", default=None)
    parser.add_argument("--swanlab-workspace", default=None)
    parser.add_argument("--swanlab-mode", default=None)
    parser.add_argument("--swanlab-logdir", default=None)
    parser.add_argument("--fsdp-min-num-params", type=int, default=10_000_000)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--debug-dtypes", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable the rank-0 tqdm progress bar.")
    parser.add_argument("--disable-update-probe", action="store_true")
    parser.add_argument("--update-probe-samples", type=int, default=64)
    parser.add_argument("--update-probe-max-params", type=int, default=1)
    parser.add_argument("--dump-traces", action="store_true")
    parser.add_argument("--dump-trace-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rank, world_size, local_rank, device = init_distributed()

    if args.samples_per_update % world_size != 0:
        raise ValueError(
            f"samples_per_update={args.samples_per_update} must be divisible by world_size={world_size}."
        )
    if args.rollout_backend == "vllm_single" and world_size != 1:
        raise ValueError("rollout_backend=vllm_single is only for a single training process. Use nproc_per_node=1.")
    if args.rollout_backend == "vllm_ipc" and world_size < 2:
        raise ValueError("rollout_backend=vllm_ipc is intended for multi-rank FSDP. Use vllm_single for world_size=1.")

    (
        _cl_sft_args,
        data_args,
        _loader_args,
        _method_args,
        model_args,
        optimizer_args,
        _trainer_args,
        tuning_args,
    ) = parse_yaml_args(args.config)
    if args.learning_rate is not None:
        optimizer_args = replace(optimizer_args, learning_rate=args.learning_rate)
    model_args = replace(model_args, use_cache=args.rollout_backend in {"manual_cache", "hf_generate"})
    if args.gradient_checkpointing and hasattr(model_args, "gradient_checkpointing"):
        model_args = replace(model_args, gradient_checkpointing=True)

    paths = sorted(expand_paths(args.traces), key=trace_sort_key)
    if not paths:
        raise FileNotFoundError(f"No trace files matched: {args.traces}")

    base_model, processor, tokenizer = load_vision_language_model(model_args, data_args.template)
    base_model = ModelTuner(tuning_args).apply(base_model)
    base_model.float()
    base_model.train()
    trainable, total = trainable_parameter_summary(base_model)
    vocab_size = int(
        getattr(getattr(base_model, "config", None), "vocab_size", 0)
        or base_model.get_input_embeddings().weight.shape[0]
    )
    image_token_id = getattr(getattr(base_model, "config", None), "image_token_id", None)
    video_token_id = getattr(getattr(base_model, "config", None), "video_token_id", None)

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
    optimizer = build_optimizer(model, optimizer_args)
    scorer = RemoteTeacherScorer(
        host=args.teacher_host,
        port=args.teacher_port,
        timeout=args.teacher_timeout,
        topk=args.topk,
    )
    student_rollout = None
    if args.rollout_backend in {"vllm_single", "vllm_ipc"} and is_rank0():
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        if model_args.model_name_or_path is None:
            raise ValueError(f"rollout_backend={args.rollout_backend} requires model.model_name_or_path in the config.")
        student_rollout = VLLMStudentRollout(
            model_path=model_args.model_name_or_path,
            tokenizer=tokenizer,
            torch_dtype=args.student_vllm_dtype,
            trust_remote_code=model_args.trust_remote_code,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.student_vllm_gpu_memory_utilization,
            max_model_len=args.student_vllm_max_model_len,
            max_num_batched_tokens=args.student_vllm_max_num_batched_tokens,
            max_num_seqs=args.student_vllm_max_num_seqs,
            enforce_eager=args.student_vllm_enforce_eager,
            device=f"cuda:{local_rank}",
            disable_log_stats=True,
            seed=0,
            limit_mm_per_prompt={"image": 1, "video": 0},
        )

    rank_print("=== online hf opd fsdp ===")
    rank_print(f"config={args.config}")
    rank_print(f"trace_count={len(paths)}")
    rank_print(f"world_size={world_size}")
    rank_print(f"local_rank={local_rank}")
    rank_print(f"samples_per_update={args.samples_per_update}")
    rank_print(f"local_samples_per_rank={args.samples_per_update // world_size}")
    rank_print(f"micro_batch_size={args.micro_batch_size}")
    rank_print(f"epochs={args.epochs}")
    rank_print(f"response_width={args.response_width}")
    rank_print(f"rollout_backend={args.rollout_backend}")
    if args.rollout_backend in {"vllm_single", "vllm_ipc"}:
        rank_print(f"student_vllm_dtype={args.student_vllm_dtype}")
        rank_print(f"student_vllm_gpu_memory_utilization={args.student_vllm_gpu_memory_utilization}")
        rank_print(f"student_vllm_max_model_len={args.student_vllm_max_model_len}")
        if args.rollout_backend == "vllm_ipc":
            rank_print("student_vllm_rank0_colocated=True")
            rank_print("student_vllm_rank0_colocated_warning=A100_40GB_may_OOM; use smoke max-updates first")
            rank_print(f"student_vllm_ipc_bucket_mb={args.student_vllm_ipc_bucket_mb}")
            rank_print(f"student_vllm_ipc_use_shm={args.student_vllm_ipc_use_shm}")
            rank_print(f"student_vllm_sync_dtype={args.student_vllm_sync_dtype}")
    rank_print(f"teacher={args.teacher_host}:{args.teacher_port}")
    rank_print(f"generate_amp_dtype={args.generate_amp_dtype}")
    rank_print(f"train_amp_dtype={args.train_amp_dtype}")
    rank_print(f"learning_rate={optimizer_args.learning_rate}")
    rank_print(f"fsdp_min_num_params={args.fsdp_min_num_params}")
    rank_print(f"trainable_params={trainable} total_params={total}")
    rank_print(f"student_vocab_size={vocab_size}")
    if args.debug_dtypes and is_rank0():
        print(f"[dtype] fsdp_param_dtypes={model_param_dtype_counts(model)}", flush=True)
        print(f"[dtype] fsdp_trainable_param_dtypes={model_param_dtype_counts(model, trainable_only=True)}", flush=True)
        print(f"[dtype] optimizer_defaults={optimizer.defaults}", flush=True)

    metrics_output = None
    if is_rank0():
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_output = open(metrics_path, "w", encoding="utf-8")

    swanlab_run = None
    if is_rank0():
        swanlab_run = init_swanlab(
            args,
            {
                "format": "clight_online_hf_opd_fsdp_config_v1",
                "config": args.config,
                "trace_count": len(paths),
                "world_size": world_size,
                "samples_per_update": args.samples_per_update,
                "epochs": args.epochs,
                "response_width": args.response_width,
                "generate_amp_dtype": args.generate_amp_dtype,
                "train_amp_dtype": args.train_amp_dtype,
                "learning_rate": optimizer_args.learning_rate,
                "grad_clip": args.grad_clip,
                "fsdp_min_num_params": args.fsdp_min_num_params,
                "trainable_params": trainable,
                "total_params": total,
            },
        )

    update_step = 0
    progress = None
    if is_rank0() and not args.no_progress:
        if tqdm is None:
            rank_print("progress_bar=disabled reason=tqdm_not_installed")
        else:
            progress = tqdm(
                total=estimate_initial_total_updates(paths, args),
                desc="online updates",
                unit="upd",
                dynamic_ncols=True,
                leave=True,
            )

    try:
        for epoch in range(args.epochs):
            for path in paths:
                payload = load_trace(path)
                batch = payload["batch"]
                non_tensor = payload.get("non_tensor_batch", {})
                mm_inputs = normalize_mm_inputs(non_tensor.get("multi_modal_inputs"))
                position_ids_cpu = batch.get("position_ids")
                prompts_cpu = batch["prompts"]
                prompt_width = int(prompts_cpu.shape[1])
                source_rows = int(prompts_cpu.shape[0])
                usable_rows = source_rows - (source_rows % args.samples_per_update)
                if usable_rows <= 0:
                    raise ValueError(
                        f"{path} has {source_rows} rows, fewer than samples_per_update={args.samples_per_update}."
                    )
                groups_in_file = usable_rows // args.samples_per_update
                if (
                    is_rank0()
                    and progress is not None
                    and args.max_updates <= 0
                    and groups_in_file > 1
                ):
                    progress.total = int(progress.total or 0) + groups_in_file - 1
                    progress.refresh()

                for group_start in range(0, usable_rows, args.samples_per_update):
                    if args.max_updates > 0 and update_step >= args.max_updates:
                        break
                    group_end = group_start + args.samples_per_update
                    local_offset_start, local_offset_end = split_contiguous_rows(
                        args.samples_per_update,
                        rank,
                        world_size,
                    )
                    row_start = group_start + local_offset_start
                    row_end = group_start + local_offset_end
                    update_step += 1

                    rollout_metrics: dict[str, float | int | str] = {}
                    if args.rollout_backend == "vllm_single" and student_rollout is not None:
                        student_rollout.sync_from_hf_model(base_model)
                        sync_cuda(device, f"student vLLM sync step {update_step}")

                    if args.rollout_backend == "vllm_ipc":
                        if is_rank0() and student_rollout is None:
                            raise RuntimeError("rank0 student_rollout is required for rollout_backend=vllm_ipc.")
                        local_sequences_cpu, local_attention_cpu, local_response_mask_cpu, rollout_metrics = (
                            generate_vllm_ipc_global_sequences(
                                model=model,
                                base_model=base_model,
                                student_rollout=student_rollout,  # type: ignore[arg-type]
                                batch=batch,
                                non_tensor=non_tensor,
                                mm_inputs=mm_inputs,
                                position_ids_cpu=position_ids_cpu,
                                group_start=group_start,
                                group_end=group_end,
                                local_rows=row_end - row_start,
                                prompt_width=prompt_width,
                                response_width=args.response_width,
                                tokenizer=tokenizer,
                                device=device,
                                image_token_id=image_token_id,
                                video_token_id=video_token_id,
                                args=args,
                                world_size=world_size,
                            )
                        )
                    else:
                        local_sequences_cpu, local_attention_cpu, local_response_mask_cpu = generate_local_sequences(
                            model=model,
                            base_model=base_model,
                            student_rollout=student_rollout,
                            batch=batch,
                            non_tensor=non_tensor,
                            mm_inputs=mm_inputs,
                            position_ids_cpu=position_ids_cpu,
                            row_start=row_start,
                            row_end=row_end,
                            prompt_width=prompt_width,
                            response_width=args.response_width,
                            tokenizer=tokenizer,
                            device=device,
                            image_token_id=image_token_id,
                            video_token_id=video_token_id,
                            args=args,
                        )
                    seq_len = int(local_sequences_cpu.shape[1])
                    local_sequences = local_sequences_cpu.to(device)
                    local_attention = local_attention_cpu.to(device)
                    local_response_mask = local_response_mask_cpu.to(device=device, dtype=torch.float32)
                    local_images, local_mm_data, local_mm_kwargs = gather_local_mm(
                        non_tensor=non_tensor,
                        row_start=row_start,
                        row_end=row_end,
                    )

                    local_teacher_logps, local_teacher_ids = batched_teacher_score(
                        scorer=scorer,
                        local_sequences=local_sequences,
                        local_attention_mask=local_attention,
                        local_images=local_images,
                        local_mm_data=local_mm_data,
                        local_mm_kwargs=local_mm_kwargs,
                        image_token_id=image_token_id,
                        video_token_id=video_token_id,
                        pad_token_id=int(tokenizer.pad_token_id),
                        topk=args.topk,
                        device=device,
                    )
                    validate_token_ids(
                        f"online step {update_step} generated input_ids local",
                        local_sequences,
                        vocab_size,
                    )

                    local_token_count = local_response_mask.sum()
                    global_token_count = reduce_sum(local_token_count.detach().clone()).clamp_min(1.0)
                    loss_num_metric = torch.tensor(0.0, device=device)
                    teacher_mass_num = torch.tensor(0.0, device=device)
                    student_mass_num = torch.tensor(0.0, device=device)
                    overlap_num = torch.tensor(0.0, device=device)
                    actual_token_count = torch.tensor(0.0, device=device)

                    optimizer.zero_grad(set_to_none=True)
                    if args.disable_update_probe:
                        update_probes = []
                    else:
                        update_probes = select_fsdp_update_probes(
                            model,
                            samples_per_param=args.update_probe_samples,
                            max_params=args.update_probe_max_params,
                        )

                    response_start = prompt_width + int(args.teacher_shift_offset)
                    if response_start < 0:
                        raise ValueError(f"Invalid response_start={response_start}.")

                    for local_start in range(0, local_sequences.shape[0], args.micro_batch_size):
                        local_end = min(local_start + args.micro_batch_size, local_sequences.shape[0])
                        input_ids = local_sequences[local_start:local_end]
                        attention_mask = local_attention[local_start:local_end]
                        response_mask = local_response_mask[local_start:local_end]

                        abs_start = row_start + local_start
                        abs_end = row_start + local_end
                        forward_kwargs: dict[str, Any] = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "use_cache": False,
                        }
                        forward_kwargs.update(build_mm_kwargs(mm_inputs, abs_start, abs_end, device))
                        mm_token_type_ids = build_mm_token_type_ids(base_model, input_ids)
                        if mm_token_type_ids is not None and "image_grid_thw" in forward_kwargs:
                            forward_kwargs["mm_token_type_ids"] = mm_token_type_ids

                        if position_ids_cpu is not None and args.position_ids_mode != "none":
                            raise NotImplementedError(
                                "Online generated responses currently use position_ids_mode=none. "
                                "Let Qwen3-VL compute positions from input_ids/image_grid_thw."
                            )

                        with autocast_context(args.train_amp_dtype):
                            outputs = model(**forward_kwargs)
                        sync_cuda(device, f"online step {update_step} rows {abs_start}:{abs_end} forward")
                        shifted_logits = outputs.logits[:, :-1, :]

                        max_student_len = shifted_logits.shape[1] - response_start
                        max_teacher_len = local_teacher_ids.shape[1] - response_start
                        current_response_len = min(args.response_width, max_student_len, max_teacher_len)
                        if current_response_len <= 0:
                            raise RuntimeError(
                                f"Empty response slice: response_start={response_start}, "
                                f"shifted_logits={tuple(shifted_logits.shape)}, "
                                f"teacher_ids={tuple(local_teacher_ids.shape)}"
                            )

                        student_logits = shifted_logits[
                            :,
                            response_start : response_start + current_response_len,
                            :,
                        ]
                        teacher_ids_slice = local_teacher_ids[
                            local_start:local_end,
                            response_start : response_start + current_response_len,
                            :,
                        ]
                        teacher_logps_slice = local_teacher_logps[
                            local_start:local_end,
                            response_start : response_start + current_response_len,
                            :,
                        ]
                        response_mask = response_mask[:, :current_response_len]
                        teacher_ids_slice = sanitize_teacher_ids(
                            teacher_ids=teacher_ids_slice.detach().cpu().long(),
                            response_mask=response_mask.detach().cpu(),
                            vocab_size=vocab_size,
                            file_idx=update_step,
                            row_start=abs_start,
                            response_start=response_start,
                        ).to(device)

                        loss_outputs = compute_topk_loss_from_logits(
                            student_logits=student_logits,
                            teacher_logps=teacher_logps_slice,
                            teacher_ids=teacher_ids_slice,
                            response_mask=response_mask,
                            log_prob_min_clamp=args.log_prob_min_clamp,
                            loss_max_clamp=args.loss_max_clamp,
                        )
                        loss_num = loss_outputs["loss_num"]
                        loss_num_metric += loss_num.detach()
                        micro_loss = loss_num * world_size / global_token_count

                        if args.debug_dtypes and update_step == 1 and local_start == 0 and is_rank0():
                            print(
                                "[dtype] first_forward="
                                + json.dumps(
                                    {
                                        "fsdp_param_dtypes": model_param_dtype_counts(model),
                                        "logits": str(outputs.logits.dtype),
                                        "student_logits": str(student_logits.dtype),
                                        "teacher_logps": str(teacher_logps_slice.dtype),
                                        "loss_num": str(loss_num.dtype),
                                        "micro_loss": str(micro_loss.dtype),
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )

                        micro_loss.backward()
                        sync_cuda(device, f"online step {update_step} rows {abs_start}:{abs_end} backward")

                        with torch.no_grad():
                            token_count = response_mask.sum()
                            actual_token_count += token_count
                            teacher_mass_num += (loss_outputs["teacher_mass"] * response_mask).sum()
                            student_mass_num += (loss_outputs["student_mass"] * response_mask).sum()
                            overlap_num += (
                                (loss_outputs["overlap_count"] / teacher_ids_slice.shape[-1]) * response_mask
                            ).sum()

                        del outputs, shifted_logits, student_logits, teacher_ids_slice, teacher_logps_slice

                    global_loss_num = reduce_sum(loss_num_metric.detach().clone())
                    global_teacher_mass_num = reduce_sum(teacher_mass_num.detach().clone())
                    global_student_mass_num = reduce_sum(student_mass_num.detach().clone())
                    global_overlap_num = reduce_sum(overlap_num.detach().clone())
                    global_actual_token_count = reduce_sum(actual_token_count.detach().clone()).clamp_min(1.0)
                    loss_value = global_loss_num / global_token_count

                    if args.debug_dtypes and update_step == 1 and is_rank0():
                        print(f"[dtype] grad_dtypes_before_step={model_grad_dtype_counts(model)}", flush=True)
                    grad_value = torch.tensor(0.0, device=device)
                    if args.grad_clip is not None and args.grad_clip > 0:
                        grad_value = model.clip_grad_norm_(args.grad_clip).detach()
                    optimizer.step()
                    if args.debug_dtypes and update_step == 1 and is_rank0():
                        print(
                            f"[dtype] optimizer_state_dtypes_after_step={optimizer_state_dtype_counts(optimizer)}",
                            flush=True,
                        )
                    update_stats = compute_fsdp_update_stats(model, update_probes)

                    maybe_dump_online_trace(
                        args=args,
                        update_step=update_step,
                        epoch=epoch,
                        source_path=path,
                        row_start=row_start,
                        row_end=row_end,
                        local_sequences=local_sequences,
                        local_attention_mask=local_attention,
                        local_response_mask=local_response_mask,
                        local_teacher_logps=local_teacher_logps,
                        local_teacher_ids=local_teacher_ids,
                    )

                    if is_rank0():
                        response_lengths = local_response_mask.sum(dim=1).detach().cpu()
                        gathered_lengths: list[Any] = [None for _ in range(world_size)]
                    else:
                        response_lengths = local_response_mask.sum(dim=1).detach().cpu()
                        gathered_lengths = [None for _ in range(world_size)]
                    dist.all_gather_object(gathered_lengths, response_lengths.tolist())
                    if is_rank0():
                        flat_lengths = [float(value) for group in gathered_lengths for value in group]
                        record = {
                            "format": "clight_online_hf_opd_fsdp_metrics_v1",
                            "online_update_step": update_step,
                            "replay_update_step": update_step,
                            "epoch": epoch,
                            "source_path": path,
                            "source_row_start": group_start,
                            "source_row_end": group_end,
                            "samples": args.samples_per_update,
                            "local_samples_per_rank": args.samples_per_update // world_size,
                            "world_size": world_size,
                            "tokens": int(global_actual_token_count.item()),
                            "loss": float(loss_value.detach().cpu().item()),
                            "teacher_mass": float((global_teacher_mass_num / global_actual_token_count).detach().cpu().item()),
                            "student_mass": float((global_student_mass_num / global_actual_token_count).detach().cpu().item()),
                            "topk_overlap": float((global_overlap_num / global_actual_token_count).detach().cpu().item()),
                            "grad_norm": float(grad_value.detach().cpu().item()),
                            "update_param": format_update_probe_names(update_probes),
                            "response_len_mean": float(sum(flat_lengths) / max(len(flat_lengths), 1)),
                            "response_len_max": float(max(flat_lengths) if flat_lengths else 0.0),
                            "rollout_backend": args.rollout_backend,
                            "generate_amp_dtype": args.generate_amp_dtype,
                            "train_amp_dtype": args.train_amp_dtype,
                        }
                        record.update(rollout_metrics)
                        record.update(update_stats)
                        message = (
                            " | ".join(
                                [
                                    f"step={update_step}",
                                    f"epoch={epoch}",
                                    f"path={path}",
                                    f"rows={group_start}:{group_end}",
                                    f"samples={args.samples_per_update}",
                                    f"tokens={record['tokens']}",
                                    f"loss={record['loss']:.8f}",
                                    f"teacher_mass={record['teacher_mass']:.8f}",
                                    f"student_mass={record['student_mass']:.8f}",
                                    f"topk_overlap={record['topk_overlap']:.8f}",
                                    f"grad_norm={record['grad_norm']:.8f}",
                                    f"resp_len_mean={record['response_len_mean']:.2f}",
                                ]
                            )
                        )
                        progress_write(progress, message)
                        if progress is not None:
                            progress.set_postfix(
                                loss=f"{record['loss']:.4f}",
                                grad=f"{record['grad_norm']:.2f}",
                                tokens=record["tokens"],
                                resp=f"{record['response_len_mean']:.1f}",
                                refresh=False,
                            )
                            progress.update(1)
                        if metrics_output is not None:
                            metrics_output.write(json.dumps(record, ensure_ascii=False) + "\n")
                            metrics_output.flush()
                        if swanlab_run is not None:
                            log_swanlab_metrics(record, int(record["online_update_step"]))

                    dist.barrier()
                if args.max_updates > 0 and update_step >= args.max_updates:
                    break
            if args.max_updates > 0 and update_step >= args.max_updates:
                break

        if args.save_model_dir:
            save_fsdp_hf_model(model, base_model, processor, args.save_model_dir)
    finally:
        if progress is not None:
            progress.close()
        if metrics_output is not None:
            metrics_output.close()
        if swanlab_run is not None:
            finish_swanlab()
        dist.barrier()
        dist.destroy_process_group()

    rank_print("train_online_hf_opd_fsdp_ok=True")


if __name__ == "__main__":
    main()
