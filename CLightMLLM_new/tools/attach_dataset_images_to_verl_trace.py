import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from replay_verl_opd_trace import expand_paths, load_trace  # noqa: E402
from src.data.module import DatasetBuilder, TemplateFactory, VLCollator  # noqa: E402
from src.hparams import parse_torch_dtype  # noqa: E402
from train import TrainingApp  # noqa: E402


class _NoopStrategy:
    def barrier(self) -> None:
        return None


class _DummyTrainer:
    local_rank = 0
    global_rank = 0
    is_global_zero = True
    strategy = _NoopStrategy()


def load_processor_and_tokenizer(model_args: Any) -> tuple[Any, Any]:
    common_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "token": model_args.hf_hub_token,
        "local_files_only": model_args.local_files_only,
    }
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **common_kwargs)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("AutoProcessor must expose processor.tokenizer.")
    tokenizer.padding_side = model_args.padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model_args.image_min_pixels is not None:
        processor.image_min_pixels = int(model_args.image_min_pixels)
        if hasattr(processor, "image_processor"):
            processor.image_processor.min_pixels = int(model_args.image_min_pixels)
    if model_args.image_max_pixels is not None:
        processor.image_max_pixels = int(model_args.image_max_pixels)
        if hasattr(processor, "image_processor"):
            processor.image_processor.max_pixels = int(model_args.image_max_pixels)
    return processor, tokenizer


def effective_prompt_ids(prompts: torch.Tensor, pad_token_id: int) -> list[list[int]]:
    rows = []
    for row in prompts.detach().cpu():
        ids = row.tolist()
        start = 0
        while start < len(ids) and int(ids[start]) == int(pad_token_id):
            start += 1
        rows.append([int(token_id) for token_id in ids[start:]])
    return rows


def prompt_hash(ids: list[int]) -> str:
    joined = ",".join(str(int(token_id)) for token_id in ids)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def collapse_repeated_token(ids: list[int], token_id: int) -> list[int]:
    collapsed = []
    previous_is_target = False
    for value in ids:
        is_target = int(value) == int(token_id)
        if is_target and previous_is_target:
            continue
        collapsed.append(int(value))
        previous_is_target = is_target
    return collapsed


def normalized_prompt_hash(ids: list[int], image_token_id: int) -> str:
    return prompt_hash(collapse_repeated_token(ids, image_token_id))


def tensor_fingerprint(value: Any) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    if torch.is_floating_point(tensor):
        tensor = tensor.float()
    else:
        tensor = tensor.long()
    h = hashlib.sha1()
    h.update(str(tuple(tensor.shape)).encode("utf-8"))
    h.update(str(tensor.dtype).encode("utf-8"))
    h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def visual_fingerprint(pixel_values: Any | None, image_grid_thw: Any | None) -> str | None:
    if pixel_values is None or image_grid_thw is None:
        return None
    h = hashlib.sha1()
    h.update(tensor_fingerprint(pixel_values).encode("utf-8"))
    h.update(tensor_fingerprint(image_grid_thw).encode("utf-8"))
    return h.hexdigest()


def normalize_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def split_collated_mm_inputs(batch: dict[str, Any], images_per_sample: list[list[Any]]) -> list[dict[str, torch.Tensor]]:
    pixels = batch.get("pixel_values")
    grids = batch.get("image_grid_thw")
    if not torch.is_tensor(pixels) or not torch.is_tensor(grids):
        return [{} for _ in images_per_sample]

    rows = []
    grid_cursor = 0
    pixel_cursor = 0
    for images in images_per_sample:
        sample_grids = []
        sample_pixels = []
        for _ in images:
            if grid_cursor >= grids.shape[0]:
                break
            grid = grids[grid_cursor].detach().cpu().long()
            patch_count = int(grid.prod().item())
            sample_grids.append(grid)
            sample_pixels.append(pixels[pixel_cursor : pixel_cursor + patch_count].detach().cpu().float())
            grid_cursor += 1
            pixel_cursor += patch_count
        item: dict[str, torch.Tensor] = {}
        if sample_grids:
            item["image_grid_thw"] = torch.stack(sample_grids, dim=0)
        if sample_pixels:
            item["pixel_values"] = torch.cat(sample_pixels, dim=0)
        rows.append(item)
    return rows


def image_summary(images: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for image in images:
        item = {"type": type(image).__name__}
        for attr in ("size", "mode"):
            if hasattr(image, attr):
                value = getattr(image, attr)
                item[attr] = tuple(value) if attr == "size" else value
        rows.append(item)
    return rows


def build_image_index(args: argparse.Namespace) -> tuple[dict[str, dict[str, deque[dict[str, Any]]]], dict[str, Any]]:
    (
        cl_sft_args,
        data_args,
        _loader_args,
        _method_args,
        model_args,
        _optimizer_args,
        _trainer_args,
        _tuning_args,
    ) = TrainingApp.parse_yaml_args(args.config)
    if not cl_sft_args.stages:
        raise ValueError("cl_sft.stages is empty.")

    stage = cl_sft_args.stages[0]
    data_args = replace(
        data_args,
        dataset=stage.dataset,
        max_samples=args.dataset_max_samples if args.dataset_max_samples is not None else data_args.max_samples,
        preprocessing_num_workers=args.num_workers,
        overwrite_cache=not args.use_cache,
        log_first_sample=False,
    )

    processor, tokenizer = load_processor_and_tokenizer(model_args)
    template = TemplateFactory.from_args(tokenizer, data_args)
    dataset = DatasetBuilder(
        template=template,
        model_args=model_args,
        data_args=data_args,
        tokenizer=tokenizer,
        processor=processor,
        trainer=_DummyTrainer(),
    ).build()
    if len(dataset) == 0:
        raise RuntimeError("No examples survived preprocessing.")

    collator = VLCollator(
        template=template,
        model=None,
        tokenizer=tokenizer,
        processor=processor,
        pad_to_multiple_of=8,
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        torch_dtype=parse_torch_dtype(model_args.torch_dtype),
    )

    exact_index: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    collapsed_image_index: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    visual_index: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    exact_collision_keys = 0
    collapsed_collision_keys = 0
    visual_collision_keys = 0
    for start in range(0, len(dataset), args.index_batch_size):
        end = min(start + args.index_batch_size, len(dataset))
        samples = [dataset[i] for i in range(start, end)]
        batch = collator(samples)
        images_per_sample = batch.get("vllm_images") or [[] for _ in samples]
        mm_per_sample = split_collated_mm_inputs(batch, images_per_sample)
        for offset, sample in enumerate(samples):
            prompt_ids = [int(token_id) for token_id in sample["prompt_input_ids"]]
            exact_key = prompt_hash(prompt_ids)
            collapsed_key = normalized_prompt_hash(prompt_ids, args.image_token_id)
            visual_key = visual_fingerprint(
                mm_per_sample[offset].get("pixel_values"),
                mm_per_sample[offset].get("image_grid_thw"),
            )
            item = {
                "dataset_index": start + offset,
                "prompt_len": len(prompt_ids),
                "image_pad_count": sum(1 for token_id in prompt_ids if token_id == int(args.image_token_id)),
                "exact_prompt_hash": exact_key,
                "collapsed_prompt_hash": collapsed_key,
                "visual_hash": visual_key,
                "vllm_images": images_per_sample[offset],
                "image_summary": image_summary(images_per_sample[offset]),
            }
            if exact_key in exact_index:
                exact_collision_keys += 1
            exact_index[exact_key].append(item)

            if collapsed_key in collapsed_image_index:
                collapsed_collision_keys += 1
            collapsed_image_index[collapsed_key].append(item)
            if visual_key is not None:
                if visual_key in visual_index:
                    visual_collision_keys += 1
                visual_index[visual_key].append(item)

    meta = {
        "dataset": data_args.dataset,
        "dataset_size": len(dataset),
        "index_size": len(exact_index),
        "collision_entries": exact_collision_keys,
        "exact_index_size": len(exact_index),
        "exact_collision_entries": exact_collision_keys,
        "collapsed_image_index_size": len(collapsed_image_index),
        "collapsed_image_collision_entries": collapsed_collision_keys,
        "visual_index_size": len(visual_index),
        "visual_collision_entries": visual_collision_keys,
        "pad_token_id": tokenizer.pad_token_id,
        "image_token_id": int(args.image_token_id),
    }
    return {"exact": exact_index, "collapsed_image": collapsed_image_index, "visual": visual_index}, meta


def choose_candidate(candidates: deque[dict[str, Any]], prompt_len: int) -> dict[str, Any]:
    return min(
        candidates,
        key=lambda item: (
            abs(int(item.get("prompt_len", 0)) - int(prompt_len)),
            0 if item.get("vllm_images") else 1,
            int(item.get("dataset_index", 0)),
        ),
    )


def choose_visual_candidate(
    candidates: deque[dict[str, Any]],
    prompt_len: int,
    exact_prompt_key: str,
    collapsed_prompt_key: str,
) -> tuple[dict[str, Any], str]:
    exact_matches = [item for item in candidates if item.get("exact_prompt_hash") == exact_prompt_key]
    if exact_matches:
        return choose_candidate(deque(exact_matches), prompt_len), "visual_exact_prompt"
    collapsed_matches = [item for item in candidates if item.get("collapsed_prompt_hash") == collapsed_prompt_key]
    if collapsed_matches:
        return choose_candidate(deque(collapsed_matches), prompt_len), "visual_collapsed_prompt"
    return choose_candidate(candidates, prompt_len), "visual_only"


def output_path(output_dir: Path, prefix: str, payload: dict[str, Any], file_index: int) -> Path:
    dump_index = payload.get("dump_index")
    global_step = payload.get("global_steps")
    if dump_index is None:
        dump_index = file_index
    if global_step is None:
        global_step = file_index + 1
    return output_dir / f"{prefix}_dump{int(dump_index):03d}_step{int(global_step):06d}.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach raw/regularized dataset images to old verl OPD trace dumps by prompt-token hash.")
    parser.add_argument("--config", required=True)
    parser.add_argument("traces", nargs="+", help="Old verl trace .pt file(s) or glob pattern(s).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="with_images")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--dataset-max-samples", type=int, default=None)
    parser.add_argument("--index-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--image-token-id", type=int, default=151655)
    parser.add_argument("--metrics-output", default=None)
    args = parser.parse_args()

    os.chdir(ROOT)
    paths = expand_paths(args.traces)
    if args.max_files is not None:
        paths = paths[: args.max_files]
    if not paths:
        raise FileNotFoundError(f"No trace files matched: {args.traces}")

    print("=== attach dataset images to verl trace ===", flush=True)
    print(f"config={args.config}", flush=True)
    print(f"trace_count={len(paths)}", flush=True)
    print("building_prompt_image_index=True", flush=True)
    image_index, index_meta = build_image_index(args)
    print(
        " ".join(
            [
                f"dataset_size={index_meta['dataset_size']}",
                f"exact_index_size={index_meta['exact_index_size']}",
                f"exact_collisions={index_meta['exact_collision_entries']}",
                f"collapsed_index_size={index_meta['collapsed_image_index_size']}",
                f"collapsed_collisions={index_meta['collapsed_image_collision_entries']}",
                f"visual_index_size={index_meta['visual_index_size']}",
                f"visual_collisions={index_meta['visual_collision_entries']}",
            ]
        ),
        flush=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = open(args.metrics_output, "w", encoding="utf-8") if args.metrics_output else None

    try:
        for file_index, path in enumerate(paths):
            payload = load_trace(path)
            batch = payload["batch"]
            non_tensor = dict(payload.get("non_tensor_batch", {}))
            prompts = batch.get("prompts")
            if not torch.is_tensor(prompts):
                raise KeyError(f"{path} is missing tensor batch['prompts']")
            pad_token_id = int(index_meta["pad_token_id"])
            prompt_rows = effective_prompt_ids(prompts, pad_token_id)
            mm_inputs = normalize_sequence(non_tensor.get("multi_modal_inputs"))

            matched = 0
            matched_by: dict[str, int] = defaultdict(int)
            missing_rows = []
            images_per_row = []
            dataset_indices = []
            image_summaries = []
            match_methods = []
            for row_idx, ids in enumerate(prompt_rows):
                exact_key = prompt_hash(ids)
                collapsed_key = normalized_prompt_hash(ids, int(index_meta["image_token_id"]))
                trace_visual_key = None
                if row_idx < len(mm_inputs) and isinstance(mm_inputs[row_idx], dict):
                    trace_visual_key = visual_fingerprint(
                        mm_inputs[row_idx].get("pixel_values"),
                        mm_inputs[row_idx].get("image_grid_thw"),
                    )

                item = None
                method = "missing"
                if trace_visual_key is not None:
                    visual_candidates = image_index["visual"].get(trace_visual_key)
                    if visual_candidates:
                        item, method = choose_visual_candidate(visual_candidates, len(ids), exact_key, collapsed_key)

                if item is None:
                    exact_candidates = image_index["exact"].get(exact_key)
                    if exact_candidates:
                        item = choose_candidate(exact_candidates, len(ids))
                        method = "exact_prompt"

                if item is None:
                    collapsed_candidates = image_index["collapsed_image"].get(collapsed_key)
                    if collapsed_candidates:
                        item = choose_candidate(collapsed_candidates, len(ids))
                        method = "collapsed_prompt"

                if item is not None:
                    matched += 1
                    matched_by[method] += 1
                    images_per_row.append(item["vllm_images"])
                    dataset_indices.append(item["dataset_index"])
                    image_summaries.append(item["image_summary"])
                    match_methods.append(method)
                else:
                    missing_rows.append(row_idx)
                    images_per_row.append([])
                    dataset_indices.append(None)
                    image_summaries.append([])
                    match_methods.append(method)

            if missing_rows and not args.allow_missing:
                raise RuntimeError(
                    f"{path} has {len(missing_rows)} rows that could not be matched by prompt hash. "
                    f"First missing rows: {missing_rows[:10]}. Use --allow-missing only for debugging."
                )

            non_tensor["vllm_images"] = images_per_row
            non_tensor["attached_dataset_indices"] = dataset_indices
            non_tensor["attached_image_summaries"] = image_summaries
            non_tensor["attached_match_methods"] = match_methods

            out_payload = dict(payload)
            out_payload["format"] = "verl_opd_trace_batch_with_dataset_images_v1"
            out_payload["source_format"] = payload.get("format")
            out_payload["source_path"] = str(path)
            out_payload["image_attach"] = {
                "method": "clight_dataset_visual_hash_then_prompt_hash",
                "config": args.config,
                "matched": matched,
                "matched_by": dict(matched_by),
                "missing": len(missing_rows),
                "missing_rows": missing_rows[:100],
                "index_meta": index_meta,
            }
            out_payload["non_tensor_batch"] = non_tensor
            out_path = output_path(output_dir, args.output_prefix, payload, file_index)
            torch.save(out_payload, out_path)

            record = {
                "format": "clight_trace_image_attach_metrics_v1",
                "file_index": file_index,
                "path": str(path),
                "output": str(out_path),
                "dump_index": payload.get("dump_index"),
                "global_step": payload.get("global_steps"),
                "sample_count": int(prompts.shape[0]),
                "matched": matched,
                "matched_by": dict(matched_by),
                "missing": len(missing_rows),
                "missing_rows": missing_rows[:100],
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
                        f"samples={int(prompts.shape[0])}",
                        f"matched={matched}",
                        f"matched_by={dict(matched_by)}",
                        f"missing={len(missing_rows)}",
                    ]
                ),
                flush=True,
            )
    finally:
        if metrics_file is not None:
            metrics_file.close()

    print("attach_dataset_images_to_verl_trace_ok=True", flush=True)


if __name__ == "__main__":
    main()
