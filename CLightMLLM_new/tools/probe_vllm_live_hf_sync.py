#!/usr/bin/env python3
"""Probe live HF-model to vLLM weight sync.

This is closer to VERL's hot-update path than probe_vllm_update_weight.py:

    live HF student state_dict -> (name, tensor) updates -> vLLM load_weights()

It does not train the model. Instead, it optionally applies a small in-memory
delta to one HF tensor to simulate an optimizer update, then sends either that
tensor or the full HF state_dict to the live vLLM model.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


AUTO_MODEL_CLASSES = (
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
    "AutoModelForCausalLM",
)


def safe_getattr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return None


def get_path(root: Any, path: str) -> Any:
    current = root
    for attr in path.split("."):
        current = safe_getattr(current, attr)
        if current is None:
            return None
    return current


def interesting_attrs(obj: Any) -> list[str]:
    terms = ("weight", "load", "model", "worker", "engine", "executor", "rpc", "collective")
    try:
        names = dir(obj)
    except Exception:
        return []
    return [name for name in names if any(term in name.lower() for term in terms)]


def find_load_weights_target(llm: Any, max_depth: int) -> tuple[Any | None, str | None]:
    explicit_paths = [
        "llm_engine.model_executor.driver_worker.model_runner.model",
        "llm_engine.model_executor.driver_worker.worker.model_runner.model",
        "llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model",
        "engine_core.engine_core.model_executor.driver_worker.model_runner.model",
        "llm_engine.engine_core.model_executor.driver_worker.model_runner.model",
        "llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model",
    ]
    for path in explicit_paths:
        target = get_path(llm, path)
        if target is not None and callable(safe_getattr(target, "load_weights")):
            return target, path

    seen: set[int] = set()
    queue: deque[tuple[Any, str, int]] = deque([(llm, "llm", 0)])
    while queue:
        obj, path, depth = queue.popleft()
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if callable(safe_getattr(obj, "load_weights")):
            return obj, path
        if depth >= max_depth:
            continue

        try:
            names = dir(obj)
        except Exception:
            continue
        for name in names:
            if name.startswith("__"):
                continue
            if name.startswith("_") and name not in {"_model_executor", "_engine_core"}:
                continue
            if not any(term in name.lower() for term in ("model", "worker", "engine", "executor", "runner", "core")):
                continue
            child = safe_getattr(obj, name)
            if child is None or isinstance(child, (str, bytes, int, float, bool, tuple, list, dict, Path)):
                continue
            queue.append((child, f"{path}.{name}", depth + 1))

    return None, None


def _load_weights_on_worker(model: Any, weights: list[tuple[str, torch.Tensor]]) -> str:
    result = model.load_weights(weights)
    return repr(result)


def _apply_model_load_weights(owner: Any, owner_name: str, weights: list[tuple[str, torch.Tensor]]) -> tuple[bool, Any]:
    apply_model = safe_getattr(owner, "apply_model")
    if not callable(apply_model):
        print(f"{owner_name}.apply_model = MISSING", flush=True)
        return False, None

    try:
        print(f"{owner_name}.apply_model signature =", inspect.signature(apply_model), flush=True)
    except Exception as exc:
        print(f"{owner_name}.apply_model signature = <unavailable: {type(exc).__name__}: {exc}>", flush=True)

    attempts = [
        ("partial_func_only", lambda: apply_model(functools.partial(_load_weights_on_worker, weights=weights))),
        ("func_plus_weights_positional", lambda: apply_model(_load_weights_on_worker, weights)),
        ("func_plus_args_tuple", lambda: apply_model(_load_weights_on_worker, args=(weights,))),
        ("func_plus_kwargs", lambda: apply_model(_load_weights_on_worker, kwargs={"weights": weights})),
    ]
    for attempt_name, attempt in attempts:
        try:
            print(f"trying {owner_name}.apply_model attempt={attempt_name}", flush=True)
            result = attempt()
            print(f"{owner_name}.apply_model attempt={attempt_name} return =", repr(result), flush=True)
            return True, result
        except Exception as exc:
            print(f"{owner_name}.apply_model attempt={attempt_name} failed: {type(exc).__name__}: {exc}", flush=True)
    return False, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe syncing a live HF model state_dict into vLLM.")
    parser.add_argument("--model", required=True, help="HF model path used for both HF and vLLM initialization.")
    parser.add_argument("--param-name", default=None, help="HF state_dict tensor name to update.")
    parser.add_argument("--all-weights", action="store_true", help="Send the whole HF state_dict instead of one tensor.")
    parser.add_argument("--delta", type=float, default=0.0, help="Optional add to one slice before syncing.")
    parser.add_argument("--delta-row", type=int, default=0)
    parser.add_argument("--delta-width", type=int, default=8)
    parser.add_argument("--hf-torch-dtype", default="bfloat16")
    parser.add_argument("--hf-device", default="cpu", help="Use cpu for low GPU memory, or cuda for a GPU live model.")
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.15)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recursive-depth", type=int, default=5)
    return parser.parse_args()


def parse_torch_dtype(value: str) -> torch.dtype:
    mapping = {
        "auto": torch.bfloat16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(value).lower().replace("torch.", "")
    if key not in mapping:
        raise ValueError(f"Unsupported dtype {value!r}. Choose from {sorted(mapping)}.")
    return mapping[key]


def load_hf_model(args: argparse.Namespace) -> torch.nn.Module:
    import transformers

    kwargs = {
        "torch_dtype": parse_torch_dtype(args.hf_torch_dtype),
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    errors = []
    for class_name in AUTO_MODEL_CLASSES:
        model_cls = getattr(transformers, class_name, None)
        if model_cls is None:
            continue
        try:
            model = model_cls.from_pretrained(args.model, **kwargs)
            break
        except Exception as exc:
            errors.append(f"{class_name}: {type(exc).__name__}: {exc}")
    else:
        raise RuntimeError("Could not load HF model. Tried:\n" + "\n".join(errors))

    if args.hf_device != "cpu":
        model.to(args.hf_device)
    model.eval()
    return model


def choose_live_param_name(state: dict[str, torch.Tensor], requested: str | None) -> str:
    if requested:
        if requested not in state:
            matches = [name for name in state if name.endswith(requested)]
            if len(matches) == 1:
                return matches[0]
            raise KeyError(f"Requested param {requested!r} not found. suffix_matches={matches[:8]}")
        return requested

    preferred_suffixes = (
        "model.language_model.layers.0.input_layernorm.weight",
        "language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.post_attention_layernorm.weight",
        "language_model.layers.0.post_attention_layernorm.weight",
        "model.language_model.norm.weight",
        "language_model.norm.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    )
    for suffix in preferred_suffixes:
        for name, tensor in state.items():
            if torch.is_tensor(tensor) and (name == suffix or name.endswith("." + suffix) or name.endswith(suffix)):
                return name
    for name, tensor in state.items():
        if torch.is_tensor(tensor):
            return name
    raise RuntimeError("HF model state_dict has no tensor entries.")


@torch.no_grad()
def apply_delta_to_live_state(state: dict[str, torch.Tensor], name: str, delta: float, row: int, width: int) -> None:
    if delta == 0.0:
        return
    tensor = state[name]
    if tensor.ndim >= 2:
        row = min(max(row, 0), tensor.shape[0] - 1)
        tensor[row, : min(width, tensor.shape[1])] += delta
    else:
        tensor[: min(width, tensor.numel())] += delta


def iter_live_weights(
    state: dict[str, torch.Tensor],
    *,
    selected_name: str,
    all_weights: bool,
) -> Iterable[tuple[str, torch.Tensor]]:
    if all_weights:
        for name, tensor in state.items():
            if torch.is_tensor(tensor):
                yield name, tensor.detach().cpu()
    else:
        yield selected_name, state[selected_name].detach().cpu()


def split_buffer_updates(
    model: torch.nn.Module,
    weights: list[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]], dict[str, torch.Tensor]]:
    named_buffers = dict(model.named_buffers())
    param_updates = []
    buffer_updates = []
    for name, tensor in weights:
        if name in named_buffers:
            buffer_updates.append((name, tensor))
        else:
            param_updates.append((name, tensor))
    return param_updates, buffer_updates, named_buffers


@torch.no_grad()
def apply_buffer_updates(
    buffer_updates: list[tuple[str, torch.Tensor]],
    named_buffers: dict[str, torch.Tensor],
) -> int:
    loaded = 0
    for name, tensor in buffer_updates:
        if name not in named_buffers:
            continue
        target = named_buffers[name]
        if tuple(target.shape) != tuple(tensor.shape):
            raise ValueError(f"Buffer shape mismatch for {name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}")
        target.copy_(tensor.to(device=target.device, dtype=target.dtype), non_blocking=False)
        loaded += 1
    return loaded


def load_vllm(args: argparse.Namespace) -> Any:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.vllm_dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": True,
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    print("vllm_kwargs =", kwargs, flush=True)
    return LLM(**kwargs)


def try_load_weights(llm: Any, weights: list[tuple[str, torch.Tensor]], recursive_depth: int) -> bool:
    target, target_path = find_load_weights_target(llm, recursive_depth)
    if target is not None:
        print("load_weights_target_path =", target_path, flush=True)
        print("load_weights_target_type =", type(target), flush=True)
        param_updates, buffer_updates, named_buffers = split_buffer_updates(target, weights)
        print("param_update_count =", len(param_updates), flush=True)
        print("buffer_update_count =", len(buffer_updates), flush=True)
        result = target.load_weights(param_updates)
        loaded_buffers = apply_buffer_updates(buffer_updates, named_buffers)
        print("load_weights_return =", repr(result), flush=True)
        print("loaded_buffers =", loaded_buffers, flush=True)
        print("weight_update_path = direct_load_weights", flush=True)
        return True

    print("direct_load_weights_target = NOT_FOUND", flush=True)
    print("llm interesting attrs =", interesting_attrs(llm), flush=True)
    engine = safe_getattr(llm, "llm_engine")
    if engine is not None:
        print("llm.llm_engine type =", type(engine), flush=True)
        print("llm.llm_engine interesting attrs =", interesting_attrs(engine), flush=True)

    owners = [("llm", llm)]
    if engine is not None:
        owners.append(("llm.llm_engine", engine))
    for owner_name, owner in owners:
        ok, _ = _apply_model_load_weights(owner, owner_name, weights)
        if ok:
            print("weight_update_path =", f"{owner_name}.apply_model", flush=True)
            return True
    return False


def main() -> None:
    args = parse_args()
    print("=== live HF -> vLLM sync probe ===", flush=True)
    print("model =", args.model, flush=True)
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    print("hf_device =", args.hf_device, flush=True)
    print("hf_torch_dtype =", args.hf_torch_dtype, flush=True)
    print("vllm_dtype =", args.vllm_dtype, flush=True)
    print("all_weights =", args.all_weights, flush=True)

    hf_model = load_hf_model(args)
    state = hf_model.state_dict()
    selected_name = choose_live_param_name(state, args.param_name)
    before = state[selected_name].detach().float().flatten()[: min(args.delta_width, state[selected_name].numel())].cpu()
    apply_delta_to_live_state(state, selected_name, args.delta, args.delta_row, args.delta_width)
    after = state[selected_name].detach().float().flatten()[: min(args.delta_width, state[selected_name].numel())].cpu()

    print("selected_param =", selected_name, flush=True)
    print("selected_shape =", tuple(state[selected_name].shape), flush=True)
    print("selected_dtype =", state[selected_name].dtype, flush=True)
    print("selected_device =", state[selected_name].device, flush=True)
    print("selected_before_head =", before.tolist(), flush=True)
    print("selected_after_head =", after.tolist(), flush=True)

    llm = load_vllm(args)
    weights = list(iter_live_weights(state, selected_name=selected_name, all_weights=args.all_weights))
    print("sync_weight_count =", len(weights), flush=True)
    print("sync_weight_source = live_hf_model_state_dict", flush=True)

    if try_load_weights(llm, weights, args.recursive_depth):
        print("RESULT=OK", flush=True)
    else:
        print("RESULT=NO_WEIGHT_UPDATE_PATH", flush=True)


if __name__ == "__main__":
    main()
