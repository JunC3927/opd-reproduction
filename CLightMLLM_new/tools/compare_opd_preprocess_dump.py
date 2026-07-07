import argparse
from typing import Any

import torch


KEY_ALIASES = {
    "prompt_input_ids": "prompts",
    "prompt_attention_mask": "prompt_attention_mask",
}
PADDED_SEQUENCE_KEYS = {
    "attention_mask",
    "input_ids",
    "position_ids",
    "prompt_attention_mask",
    "prompt_input_ids",
    "prompts",
}
SEMANTIC_TENSOR_KEYS = {"image_grid_thw", "pixel_values", "pixel_values_videos"}


def torch_load(path: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def compare_tensor(name: str, left: torch.Tensor, right: torch.Tensor, *, ignore_float_dtype: bool = False, atol: float = 0.0) -> bool:
    if tuple(left.shape) != tuple(right.shape):
        print(f"{name}: shape_mismatch left={tuple(left.shape)} right={tuple(right.shape)}")
        return False
    if left.dtype != right.dtype and not (ignore_float_dtype and torch.is_floating_point(left) and torch.is_floating_point(right)):
        print(f"{name}: dtype_mismatch left={left.dtype} right={right.dtype}")
        return False
    if torch.is_floating_point(left):
        diff = (left.float() - right.float()).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
        ok = bool(torch.allclose(left.float(), right.float(), atol=atol, rtol=0.0))
        print(
            f"{name}: same={ok} shape={tuple(left.shape)} "
            f"left_dtype={left.dtype} right_dtype={right.dtype} max_abs={max_abs} mean_abs={mean_abs} atol={atol}"
        )
        return ok
    same = bool(torch.equal(left, right))
    mismatch = int((left != right).sum().item()) if left.numel() else 0
    print(f"{name}: same={same} shape={tuple(left.shape)} mismatch={mismatch}")
    return same


def strip_left_padding(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[list[int]]:
    stripped = []
    for ids, mask in zip(input_ids, attention_mask, strict=False):
        stripped.append(ids[mask.bool()].tolist())
    return stripped


def compare_effective_prompts(left_tensors: dict[str, torch.Tensor], right_tensors: dict[str, torch.Tensor]) -> bool:
    left_ids = left_tensors.get("prompt_input_ids")
    left_mask = left_tensors.get("prompt_attention_mask")
    right_ids = right_tensors.get("prompts", right_tensors.get("prompt_input_ids"))
    right_mask = right_tensors.get("prompt_attention_mask")
    if left_ids is None or left_mask is None or right_ids is None or right_mask is None:
        print("effective_prompt_ids: skipped (missing prompt ids or mask)")
        return True

    left_effective = strip_left_padding(left_ids, left_mask)
    right_effective = strip_left_padding(right_ids, right_mask)
    same = left_effective == right_effective
    print(f"effective_prompt_ids: same={same}")
    if not same:
        for row, (left_row, right_row) in enumerate(zip(left_effective, right_effective, strict=False)):
            if left_row != right_row:
                print(f"  row={row} left_len={len(left_row)} right_len={len(right_row)}")
                limit = min(len(left_row), len(right_row))
                first_diff = next((idx for idx in range(limit) if left_row[idx] != right_row[idx]), None)
                print(f"  first_diff={first_diff}")
                if first_diff is not None:
                    lo = max(first_diff - 8, 0)
                    hi = min(first_diff + 8, limit)
                    print(f"  left_slice={left_row[lo:hi]}")
                    print(f"  right_slice={right_row[lo:hi]}")
                break
    return same


def compare_prompt_lengths(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_lengths = left.get("prompt_lengths")
    right_lengths = right.get("prompt_lengths")
    same = left_lengths == right_lengths
    print(f"prompt_lengths: same={same} left={left_lengths} right={right_lengths}")
    return same


def compare_sample_debug_values(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_debug = left.get("sample_debug") or []
    right_debug = right.get("sample_debug") or []
    if len(left_debug) != len(right_debug):
        print(f"sample_debug_values: same=False left_rows={len(left_debug)} right_rows={len(right_debug)}")
        return False

    ok = True
    for row, (left_item, right_item) in enumerate(zip(left_debug, right_debug, strict=False)):
        fields = ("prompt_len", "image_pad_count", "image_grid_thw")
        mismatches = [
            field
            for field in fields
            if left_item.get(field) != right_item.get(field)
        ]
        if mismatches:
            ok = False
            print(f"sample_debug_values row={row}: same=False mismatches={mismatches}")
    print(f"sample_debug_values: same={ok}")
    return ok


def compare_semantic_tensors(
    left_tensors: dict[str, torch.Tensor],
    right_tensors: dict[str, torch.Tensor],
    *,
    pixel_atol: float,
) -> bool:
    ok = True
    for key in sorted(SEMANTIC_TENSOR_KEYS):
        if key not in left_tensors or key not in right_tensors:
            print(f"{key}: skipped left_present={key in left_tensors} right_present={key in right_tensors}")
            continue
        ignore_float_dtype = key.startswith("pixel_values")
        ok = compare_tensor(
            key,
            left_tensors[key],
            right_tensors[key],
            ignore_float_dtype=ignore_float_dtype,
            atol=pixel_atol if ignore_float_dtype else 0.0,
        ) and ok
    return ok


def print_sample_debug(left: dict[str, Any], right: dict[str, Any]) -> None:
    left_debug = left.get("sample_debug") or []
    right_debug = right.get("sample_debug") or []
    if not left_debug and not right_debug:
        print("sample_debug: skipped (missing from both dumps)")
        return

    rows = max(len(left_debug), len(right_debug))
    for row in range(rows):
        left_item = left_debug[row] if row < len(left_debug) else {}
        right_item = right_debug[row] if row < len(right_debug) else {}
        print(f"row={row}")
        print(
            "  left:  prompt_len={prompt_len} image_pad_count={image_pad_count} "
            "image_count={image_count} image_grid_thw={image_grid_thw}".format(
                prompt_len=left_item.get("prompt_len"),
                image_pad_count=left_item.get("image_pad_count"),
                image_count=left_item.get("image_count"),
                image_grid_thw=left_item.get("image_grid_thw"),
            )
        )
        print(
            "  right: prompt_len={prompt_len} image_pad_count={image_pad_count} "
            "dataset_index={dataset_index} image_grid_thw={image_grid_thw}".format(
                prompt_len=right_item.get("prompt_len"),
                image_pad_count=right_item.get("image_pad_count"),
                dataset_index=right_item.get("dataset_index"),
                image_grid_thw=right_item.get("image_grid_thw"),
            )
        )
        if left_item.get("images") is not None:
            print(f"  left_images={left_item.get('images')}")
        if right_item.get("raw_message_images") is not None:
            print(f"  right_raw_message_images={right_item.get('raw_message_images')}")
        if right_item.get("processed_images") is not None:
            print(f"  right_processed_images={right_item.get('processed_images')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two OPD preprocessing dump files.")
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument(
        "--pixel-atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for pixel_values after casting both sides to float32.",
    )
    args = parser.parse_args()

    left = torch_load(args.left)
    right = torch_load(args.right)
    left_tensors = left.get("batch_tensors", {})
    right_tensors = right.get("batch_tensors", {})

    print("=== metadata ===")
    for key in ("format", "template", "sample_count", "prompt_lengths"):
        print(f"{key}: left={left.get(key)} right={right.get(key)}")

    print("=== tensor keys ===")
    left_keys = set(left_tensors)
    right_keys = set(right_tensors)
    print(f"left_only={sorted(left_keys - right_keys)}")
    print(f"right_only={sorted(right_keys - left_keys)}")

    print("=== semantic checks ===")
    ok = True
    ok = compare_prompt_lengths(left, right) and ok
    ok = compare_effective_prompts(left_tensors, right_tensors) and ok
    ok = compare_sample_debug_values(left, right) and ok
    ok = compare_semantic_tensors(left_tensors, right_tensors, pixel_atol=args.pixel_atol) and ok

    print("=== sample debug ===")
    print_sample_debug(left, right)

    print("=== alias tensors ===")
    for left_key, right_key in KEY_ALIASES.items():
        if left_key in left_tensors and right_key in right_tensors:
            print(
                f"{left_key} vs {right_key}: skipped raw padded tensor compare "
                f"left_shape={tuple(left_tensors[left_key].shape)} right_shape={tuple(right_tensors[right_key].shape)}"
            )

    print("=== raw tensor diagnostics ===")
    for key in sorted(left_keys & right_keys):
        if key in SEMANTIC_TENSOR_KEYS:
            continue
        if key in PADDED_SEQUENCE_KEYS:
            print(
                f"{key}: skipped raw padded tensor compare "
                f"left_shape={tuple(left_tensors[key].shape)} right_shape={tuple(right_tensors[key].shape)}"
            )
            continue
        compare_tensor(key, left_tensors[key], right_tensors[key])

    print("=== text ===")
    for key in ("prompt_text", "input_text", "batch_reference_text"):
        if key not in left or key not in right:
            print(f"{key}: skipped left_present={key in left} right_present={key in right}")
            continue
        same = left.get(key) == right.get(key)
        print(f"{key}: same={same} (diagnostic only; padded text may differ)")

    print(f"preprocess_compare_ok={ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
