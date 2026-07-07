import argparse
import glob
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


def torch_load(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def shape_of(value: Any) -> str:
    if torch.is_tensor(value):
        return f"{tuple(value.shape)} {value.dtype}"
    if hasattr(value, "shape"):
        return f"{tuple(value.shape)} {getattr(value, 'dtype', type(value))}"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, tuple):
        return f"tuple[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    return type(value).__name__


def expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)
    return sorted(dict.fromkeys(paths))


def tensor_sum(value: Any) -> float | None:
    if not torch.is_tensor(value):
        return None
    if value.numel() == 0:
        return 0.0
    return float(value.float().sum().item())


def describe_mm_item(value: Any, max_depth: int = 2) -> Any:
    if torch.is_tensor(value):
        return f"tensor{tuple(value.shape)} {value.dtype}"
    if hasattr(value, "shape"):
        return f"array{tuple(value.shape)} {getattr(value, 'dtype', type(value))}"
    if isinstance(value, dict):
        if max_depth <= 0:
            return f"dict[{len(value)}]"
        return {key: describe_mm_item(val, max_depth=max_depth - 1) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        if not value:
            return f"{type(value).__name__}[0]"
        if max_depth <= 0:
            return f"{type(value).__name__}[{len(value)}]"
        return {
            "type": type(value).__name__,
            "len": len(value),
            "first": describe_mm_item(value[0], max_depth=max_depth - 1),
        }
    return repr(type(value))


def collect_mm_keys(value: Any, prefix: str = "") -> Counter[str]:
    keys: Counter[str] = Counter()
    if isinstance(value, dict):
        for key, val in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            keys[name] += 1
            keys.update(collect_mm_keys(val, name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(collect_mm_keys(item, prefix))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect verl OPD trace dump files.")
    parser.add_argument("paths", nargs="+", help="Trace dump file(s) or glob pattern(s).")
    parser.add_argument("--show-keys", action="store_true", help="Print full key lists for every file.")
    parser.add_argument("--topk-rows", type=int, default=12, help="Max candidate teacher/topk keys to print.")
    args = parser.parse_args()

    paths = expand_paths(args.paths)
    if not paths:
        raise FileNotFoundError(f"No trace dump files matched: {args.paths}")

    total_samples = 0
    total_response_tokens = 0.0
    key_counter: Counter[str] = Counter()
    shape_by_key: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_keys: Counter[str] = Counter()
    non_tensor_key_counter: Counter[str] = Counter()
    mm_key_counter: Counter[str] = Counter()
    mm_summary_counter: Counter[str] = Counter()

    print("=== verl opd trace inspect ===")
    print(f"file_count={len(paths)}")

    for idx, path in enumerate(paths):
        payload = torch_load(path)
        batch = payload.get("batch", {})
        non_tensor = payload.get("non_tensor_batch", {})
        meta = payload.get("meta_info", {})
        sample_count = int(payload.get("sample_count", 0))
        total_samples += sample_count

        for key, value in batch.items():
            key_counter[key] += 1
            shape_by_key[key][shape_of(value)] += 1
            lowered = key.lower()
            if any(part in lowered for part in ("teacher", "topk", "logprob", "log_prob", "logps", "log_probs")):
                candidate_keys[key] += 1
        for key in non_tensor:
            non_tensor_key_counter[key] += 1

        response_tokens = None
        for mask_key in ("response_mask", "responses_mask", "loss_mask"):
            if mask_key in batch:
                response_tokens = tensor_sum(batch[mask_key])
                break
        if response_tokens is not None:
            total_response_tokens += response_tokens

        print(f"\n--- file {idx} ---")
        print(f"path={path}")
        print(f"format={payload.get('format')}")
        print(f"dump_index={payload.get('dump_index')} global_steps={payload.get('global_steps')}")
        print(f"sample_count={sample_count}")
        if response_tokens is not None:
            print(f"response_tokens={response_tokens:g}")

        for key in (
            "prompts",
            "input_ids",
            "attention_mask",
            "responses",
            "response_mask",
            "pixel_values",
            "image_grid_thw",
            "position_ids",
            "teacher_topk_ids",
            "teacher_topk_logps",
            "teacher_topk_log_probs",
        ):
            if key in batch:
                print(f"{key}: {shape_of(batch[key])}")

        if args.show_keys:
            print(f"batch_keys={list(batch.keys())}")
            print(f"non_tensor_keys={list(non_tensor.keys())}")
            print(f"meta_keys={list(meta.keys())}")

        for key in ("multi_modal_data", "vllm_images", "mm_processor_kwargs"):
            if key in non_tensor:
                print(f"{key}: {shape_of(non_tensor[key])}")
        if "multi_modal_inputs" in non_tensor:
            mm_inputs = non_tensor["multi_modal_inputs"]
            mm_key_counter.update(collect_mm_keys(mm_inputs))
            if len(mm_inputs) > 0:
                summary = describe_mm_item(mm_inputs[0])
                mm_summary_counter[repr(summary)] += 1
                print(f"multi_modal_inputs: {shape_of(mm_inputs)}")
                print(f"multi_modal_inputs[0]: {summary}")

    print("\n=== aggregate ===")
    print(f"total_samples={total_samples}")
    print(f"total_response_tokens={total_response_tokens:g}")
    print(f"all_batch_keys={sorted(key_counter)}")
    print(f"all_non_tensor_keys={sorted(non_tensor_key_counter)}")

    print("\n=== key shapes ===")
    for key in sorted(key_counter):
        shapes = ", ".join(f"{shape} x{count}" for shape, count in shape_by_key[key].most_common())
        print(f"{key}: {shapes}")

    print("\n=== teacher/topk/logprob candidate keys ===")
    for key, count in candidate_keys.most_common(args.topk_rows):
        shapes = ", ".join(f"{shape} x{shape_count}" for shape, shape_count in shape_by_key[key].most_common())
        print(f"{key}: files={count}; {shapes}")

    required_any = {
        "prompt": ("prompts", "input_ids"),
        "responses": ("responses",),
        "response_mask": ("response_mask",),
        "pixels": ("pixel_values",),
        "image_grid": ("image_grid_thw",),
    }
    print("\n=== required checks ===")
    for name, keys in required_any.items():
        present = any(key in key_counter for key in keys)
        print(f"{name}: {present} candidates={keys}")
    has_teacher_topk_ids = any(
        "teacher" in key.lower() and ("id" in key.lower()) for key in key_counter
    )
    has_teacher_topk_logps = any(
        "teacher" in key.lower()
        and any(part in key.lower() for part in ("logp", "log_prob", "logps"))
        for key in key_counter
    )
    print(f"teacher_topk_ids_like: {has_teacher_topk_ids}")
    print(f"teacher_topk_logps_like: {has_teacher_topk_logps}")
    print(f"teacher_ids: {'teacher_ids' in key_counter}")
    print(f"teacher_logprobs: {'teacher_logprobs' in key_counter}")
    print(f"layer_b_vllm_images: {'vllm_images' in non_tensor_key_counter or 'multi_modal_data' in non_tensor_key_counter}")
    print(f"layer_b_mm_processor_kwargs: {'mm_processor_kwargs' in non_tensor_key_counter}")

    print("\n=== multi_modal_inputs ===")
    print(f"multi_modal_key_counts={dict(mm_key_counter.most_common(40))}")
    if mm_summary_counter:
        print("first_item_summaries:")
        for summary, count in mm_summary_counter.most_common(5):
            print(f"  x{count}: {summary}")


if __name__ == "__main__":
    main()
