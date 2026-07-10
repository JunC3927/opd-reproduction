#!/usr/bin/env python3
"""Probe whether a local vLLM engine can accept in-place weight updates.

This is intentionally small: it launches one vLLM engine, finds an object that
exposes ``load_weights()``, loads one tensor from the HF checkpoint, and sends
that tensor into the live engine without restarting it.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe vLLM in-place load_weights support.")
    parser.add_argument("--model", required=True, help="HF model path used to initialize vLLM.")
    parser.add_argument("--param-name", default=None, help="Checkpoint tensor name to send to vLLM.")
    parser.add_argument("--dtype", default="bfloat16", help="vLLM dtype, e.g. bfloat16.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--delta", type=float, default=0.0, help="Optional tiny add to one row before loading.")
    parser.add_argument("--delta-row", type=int, default=0)
    parser.add_argument("--recursive-depth", type=int, default=5)
    return parser.parse_args()


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


def safetensor_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        seen = []
        for filename in index.get("weight_map", {}).values():
            path = model_dir / filename
            if path not in seen:
                seen.append(path)
        return seen

    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    return sorted(model_dir.glob("*.safetensors"))


def index_weight_map(model_dir: Path) -> dict[str, Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    return {name: model_dir / filename for name, filename in index.get("weight_map", {}).items()}


def choose_param_name(model_dir: Path, requested: str | None) -> str:
    if requested:
        return requested

    from safetensors import safe_open

    preferred_suffixes = (
        "model.language_model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    )
    fallback = None
    for path in safetensor_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if fallback is None and keys:
                fallback = keys[0]
            for suffix in preferred_suffixes:
                for key in keys:
                    if key == suffix or key.endswith("." + suffix) or key.endswith(suffix):
                        return key
    if fallback is None:
        raise FileNotFoundError(f"No safetensors weights found under {model_dir}")
    return fallback


def load_checkpoint_tensor(model_dir: Path, name: str) -> torch.Tensor:
    from safetensors.torch import load_file

    weight_map = index_weight_map(model_dir)
    candidate_files = [weight_map[name]] if name in weight_map else safetensor_files(model_dir)
    for path in candidate_files:
        if not path.exists():
            continue
        tensors = load_file(path, device="cpu")
        if name in tensors:
            return tensors[name]
    raise KeyError(f"Tensor {name!r} was not found in {model_dir}")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model)

    print("=== vLLM update weight probe ===")
    print("model =", args.model)
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("VLLM_USE_V1 =", os.environ.get("VLLM_USE_V1"))
    print("torch =", torch.__version__)
    print("cuda =", torch.cuda.is_available(), torch.cuda.device_count())

    from vllm import LLM

    import vllm

    print("vllm =", vllm.__version__)
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": True,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs

    print("llm_kwargs =", llm_kwargs)
    llm = LLM(**llm_kwargs)

    target, target_path = find_load_weights_target(llm, args.recursive_depth)
    if target is None:
        print("RESULT=NO_LOAD_WEIGHTS_TARGET")
        print("llm interesting attrs =", interesting_attrs(llm))
        engine = safe_getattr(llm, "llm_engine")
        if engine is not None:
            print("llm.llm_engine type =", type(engine))
            print("llm.llm_engine interesting attrs =", interesting_attrs(engine))
        return

    print("load_weights_target_path =", target_path)
    print("load_weights_target_type =", type(target))

    param_name = choose_param_name(model_dir, args.param_name)
    tensor = load_checkpoint_tensor(model_dir, param_name)
    if args.delta != 0.0:
        tensor = tensor.clone()
        if tensor.ndim >= 2:
            row = min(max(args.delta_row, 0), tensor.shape[0] - 1)
            tensor[row, : min(8, tensor.shape[1])] += args.delta
        else:
            tensor += args.delta

    print("selected_param =", param_name)
    print("selected_shape =", tuple(tensor.shape))
    print("selected_dtype =", tensor.dtype)
    print("selected_device =", tensor.device)

    result = target.load_weights([(param_name, tensor)])
    print("load_weights_return =", repr(result))
    print("RESULT=OK")


if __name__ == "__main__":
    main()
