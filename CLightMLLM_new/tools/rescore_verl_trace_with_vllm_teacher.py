import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from replay_verl_opd_trace import expand_paths, load_trace  # noqa: E402
from src.method.vllm_teacher_client import RemoteTeacherScorer  # noqa: E402


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def item_at(value: Any, index: int) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
        return value[index]
    try:
        return value[index]
    except Exception:
        return None


def is_image_like(value: Any) -> bool:
    if value is None or torch.is_tensor(value):
        return False
    module = type(value).__module__
    name = type(value).__name__
    if module.startswith("PIL.") or name.endswith("ImageFile") or name == "Image":
        return True
    if isinstance(value, (bytes, bytearray)):
        return True
    if isinstance(value, str):
        suffix = Path(value).suffix.lower()
        return suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return False


def collect_images(obj: Any, depth: int = 0) -> list[Any]:
    if obj is None or depth > 8:
        return []
    if is_image_like(obj):
        return [obj]
    if torch.is_tensor(obj):
        return []
    if isinstance(obj, dict):
        images: list[Any] = []
        for key in ("vllm_images", "raw_images", "images", "image"):
            if key in obj:
                images.extend(collect_images(obj[key], depth + 1))
        if "multi_modal_data" in obj:
            images.extend(collect_images(obj["multi_modal_data"], depth + 1))
        if obj.get("type") == "image" and "image" in obj:
            images.extend(collect_images(obj["image"], depth + 1))
        if "bytes" in obj and isinstance(obj["bytes"], (bytes, bytearray)):
            images.append(obj["bytes"])
        if "path" in obj and isinstance(obj["path"], str):
            images.append(obj["path"])
        return images
    if isinstance(obj, (list, tuple)):
        images = []
        for item in obj:
            images.extend(collect_images(item, depth + 1))
        return images
    return []


def images_from_trace_row(non_tensor: dict[str, Any], row: int) -> list[Any]:
    images: list[Any] = []
    for key in (
        "multi_modal_data",
        "vllm_images",
        "raw_images",
        "images",
        "multi_modal_inputs",
        "raw_prompt",
        "extra_info",
        "extras",
        "reward_model",
    ):
        if key not in non_tensor:
            continue
        images.extend(collect_images(item_at(non_tensor[key], row)))

    deduped = []
    seen = set()
    for image in images:
        marker = id(image) if not isinstance(image, (str, bytes, bytearray)) else image
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(image)
    return deduped


def mm_processor_kwargs_from_trace_row(non_tensor: dict[str, Any], row: int) -> dict[str, Any] | None:
    for key in ("mm_processor_kwargs", "vllm_mm_processor_kwargs"):
        if key not in non_tensor:
            continue
        value = item_at(non_tensor[key], row)
        if value is None:
            continue
        if isinstance(value, dict):
            return value
        if hasattr(value, "items"):
            return dict(value.items())
        raise TypeError(f"Unsupported {key} type for row {row}: {type(value)}")
    return None


def image_counts_from_input(input_ids: torch.Tensor, image_token_id: int | None) -> list[int]:
    if image_token_id is None:
        return [0 for _ in range(input_ids.shape[0])]
    return input_ids.eq(int(image_token_id)).sum(dim=1).tolist()


def fit_teacher_shape(
    *,
    new_logps: torch.Tensor,
    new_ids: torch.Tensor,
    old_logps: torch.Tensor,
    old_ids: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if old_logps.ndim != 3 or old_ids.ndim != 3:
        raise ValueError(f"Expected old teacher tensors to be rank-3, got {old_logps.shape}, {old_ids.shape}")
    if new_logps.shape[:1] != old_logps.shape[:1] or new_ids.shape[:1] != old_ids.shape[:1]:
        raise ValueError(f"Batch mismatch: new={new_logps.shape}, old={old_logps.shape}")
    if new_logps.shape[-1] != old_logps.shape[-1] or new_ids.shape[-1] != old_ids.shape[-1]:
        raise ValueError(f"Top-k mismatch: new={new_logps.shape}, old={old_logps.shape}")

    fitted_logps = torch.zeros_like(old_logps, dtype=torch.float32)
    fitted_ids = torch.full_like(old_ids, fill_value=int(pad_token_id))
    copy_len = min(old_logps.shape[1], new_logps.shape[1])
    fitted_logps[:, :copy_len, :] = new_logps[:, :copy_len, :].to(dtype=torch.float32)
    fitted_ids[:, :copy_len, :] = new_ids[:, :copy_len, :].to(dtype=old_ids.dtype)
    return fitted_logps, fitted_ids


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.to(dtype=value.dtype)
    denom = mask.sum().clamp_min(1.0)
    return float((value * mask).sum().item() / denom.item())


def compare_teacher_tensors(
    *,
    old_logps: torch.Tensor,
    old_ids: torch.Tensor,
    new_logps: torch.Tensor,
    new_ids: torch.Tensor,
    response_mask: torch.Tensor,
    response_start: int,
) -> dict[str, float]:
    response_len = int(response_mask.shape[1])
    max_len = min(
        response_len,
        old_logps.shape[1] - response_start,
        new_logps.shape[1] - response_start,
    )
    if max_len <= 0:
        raise ValueError(
            f"Invalid response slice: response_start={response_start}, "
            f"old_shape={tuple(old_logps.shape)}, new_shape={tuple(new_logps.shape)}"
        )

    old_l = old_logps[:, response_start : response_start + max_len, :].float()
    new_l = new_logps[:, response_start : response_start + max_len, :].float()
    old_i = old_ids[:, response_start : response_start + max_len, :].long()
    new_i = new_ids[:, response_start : response_start + max_len, :].long()
    mask = response_mask[:, :max_len].bool()
    mask3 = mask.unsqueeze(-1).expand_as(old_i)
    mask2 = mask.float()

    logp_delta = (old_l - new_l).abs()
    ids_equal = old_i.eq(new_i)
    old_mass = old_l.exp().sum(dim=-1)
    new_mass = new_l.exp().sum(dim=-1)
    return {
        "active_tokens": float(mask.sum().item()),
        "teacher_ids_same_ratio": float(ids_equal[mask3].float().mean().item()) if mask3.any() else 0.0,
        "teacher_logps_mean_abs": float(logp_delta[mask3].mean().item()) if mask3.any() else 0.0,
        "teacher_logps_max_abs": float(logp_delta[mask3].max().item()) if mask3.any() else 0.0,
        "old_teacher_mass": masked_mean(old_mass, mask2),
        "new_teacher_mass": masked_mean(new_mass, mask2),
    }


def output_path(output_dir: Path, prefix: str, payload: dict[str, Any], file_index: int) -> Path:
    dump_index = payload.get("dump_index")
    global_step = payload.get("global_steps")
    if dump_index is None:
        dump_index = file_index
    if global_step is None:
        global_step = file_index + 1
    return output_dir / f"{prefix}_dump{int(dump_index):03d}_step{int(global_step):06d}.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score verl OPD trace teacher top-k with a CLight vLLM teacher server.")
    parser.add_argument("traces", nargs="+", help="verl trace .pt file(s) or glob pattern(s).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="vllm_teacher_rescored")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29577)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=24)
    parser.add_argument("--teacher-shift-offset", type=int, default=-1)
    parser.add_argument("--image-token-id", type=int, default=151655)
    parser.add_argument("--video-token-id", type=int, default=151656)
    parser.add_argument("--pad-token-id", type=int, default=151643)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--keep-old-teacher", action="store_true")
    parser.add_argument("--metrics-output", default=None)
    args = parser.parse_args()

    paths = expand_paths(args.traces)
    if args.max_files is not None:
        paths = paths[: args.max_files]
    if not paths:
        raise FileNotFoundError(f"No trace files matched: {args.traces}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer = RemoteTeacherScorer(host=args.host, port=args.port, timeout=args.timeout, topk=args.topk)
    metrics_file = open(args.metrics_output, "w", encoding="utf-8") if args.metrics_output else None

    print("=== rescore verl trace with vllm teacher ===", flush=True)
    print(f"trace_count={len(paths)}", flush=True)
    print(f"teacher={args.host}:{args.port}", flush=True)
    print(f"topk={args.topk}", flush=True)
    print(f"output_dir={output_dir}", flush=True)

    try:
        for file_index, path in enumerate(paths):
            payload = load_trace(path)
            batch = payload["batch"]
            non_tensor = payload.get("non_tensor_batch", {})
            input_ids = batch["input_ids"].cpu()
            attention_mask = batch["attention_mask"].cpu()
            response_mask = batch["response_mask"].cpu()
            old_logps = batch["teacher_logprobs"].cpu()
            old_ids = batch["teacher_ids"].cpu()
            prompts = batch.get("prompts")
            prompt_width = int(prompts.shape[1]) if torch.is_tensor(prompts) else int(input_ids.shape[1] - response_mask.shape[1])
            response_start = prompt_width + int(args.teacher_shift_offset)

            batch_size = int(input_ids.shape[0])
            image_counts = image_counts_from_input(input_ids, args.image_token_id)
            images_per_sample = [images_from_trace_row(non_tensor, row) for row in range(batch_size)]
            mm_processor_kwargs_per_sample = [
                mm_processor_kwargs_from_trace_row(non_tensor, row) for row in range(batch_size)
            ]
            missing = [
                row
                for row, (token_count, images) in enumerate(zip(image_counts, images_per_sample, strict=True))
                if int(token_count) > 0 and not images
            ]
            if missing and not args.allow_missing_images:
                raise RuntimeError(
                    f"{path} has image tokens but no raw images for rows {missing[:8]}. "
                    "The old trace may only contain processed pixel_values. Re-dump the trace with raw/PIL images, "
                    "or pass --allow-missing-images only for text-only debugging."
                )

            chunks_logps = []
            chunks_ids = []
            for start in range(0, batch_size, args.micro_batch_size):
                end = min(start + args.micro_batch_size, batch_size)
                logps, ids = scorer.score(
                    sequences=input_ids[start:end],
                    attention_mask=attention_mask[start:end],
                    images_per_sample=images_per_sample[start:end],
                    image_token_id=args.image_token_id,
                    video_token_id=args.video_token_id,
                    pad_token_id=args.pad_token_id,
                    mm_processor_kwargs_per_sample=mm_processor_kwargs_per_sample[start:end],
                )
                chunks_logps.append(logps.cpu())
                chunks_ids.append(ids.cpu())

            new_logps_raw = torch.cat(chunks_logps, dim=0)
            new_ids_raw = torch.cat(chunks_ids, dim=0)
            new_logps, new_ids = fit_teacher_shape(
                new_logps=new_logps_raw,
                new_ids=new_ids_raw,
                old_logps=old_logps,
                old_ids=old_ids,
                pad_token_id=args.pad_token_id,
            )
            stats = compare_teacher_tensors(
                old_logps=old_logps,
                old_ids=old_ids,
                new_logps=new_logps,
                new_ids=new_ids,
                response_mask=response_mask,
                response_start=response_start,
            )

            out_payload = copy.copy(payload)
            out_payload["format"] = "clight_vllm_teacher_rescored_verl_trace_v1"
            out_payload["source_format"] = payload.get("format")
            out_payload["source_path"] = str(path)
            out_payload["teacher_rescore"] = {
                "backend": "clight_remote_vllm_teacher",
                "host": args.host,
                "port": args.port,
                "topk": args.topk,
                "teacher_shift_offset": args.teacher_shift_offset,
                "image_counts": image_counts,
                "image_list_lengths": [len(images) for images in images_per_sample],
                "has_mm_processor_kwargs": [kwargs is not None for kwargs in mm_processor_kwargs_per_sample],
                **stats,
            }
            out_batch = dict(batch)
            if args.keep_old_teacher:
                out_batch["old_teacher_logprobs"] = old_logps
                out_batch["old_teacher_ids"] = old_ids
            out_batch["teacher_logprobs"] = new_logps
            out_batch["teacher_ids"] = new_ids
            out_payload["batch"] = out_batch

            out_path = output_path(output_dir, args.output_prefix, payload, file_index)
            torch.save(out_payload, out_path)

            record = {
                "format": "clight_vllm_teacher_rescore_metrics_v1",
                "file_index": file_index,
                "path": str(path),
                "output": str(out_path),
                "dump_index": payload.get("dump_index"),
                "global_step": payload.get("global_steps"),
                "samples": batch_size,
                "prompt_width": prompt_width,
                "response_start": response_start,
                **stats,
            }
            if metrics_file is not None:
                metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_file.flush()

            print(
                " | ".join(
                    [
                        f"file={file_index}",
                        f"path={path}",
                        f"output={out_path}",
                        f"samples={batch_size}",
                        f"tokens={int(stats['active_tokens'])}",
                        f"ids_same={stats['teacher_ids_same_ratio']:.6f}",
                        f"logps_mean_abs={stats['teacher_logps_mean_abs']:.6e}",
                        f"logps_max_abs={stats['teacher_logps_max_abs']:.6e}",
                        f"old_mass={stats['old_teacher_mass']:.8f}",
                        f"new_mass={stats['new_teacher_mass']:.8f}",
                    ]
                ),
                flush=True,
            )
    finally:
        if metrics_file is not None:
            metrics_file.close()

    print("rescore_verl_trace_with_vllm_teacher_ok=True", flush=True)


if __name__ == "__main__":
    main()
