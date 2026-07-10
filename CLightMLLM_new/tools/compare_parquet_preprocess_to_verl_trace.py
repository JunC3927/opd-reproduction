import argparse
import glob
import json
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from PIL.Image import Image as ImageObject

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from check_opd_preprocess import _DummyTrainer, _load_processor_and_tokenizer  # noqa: E402
from src.data.module import DatasetBuilder, TemplateFactory, VLCollator  # noqa: E402
from src.hparams import parse_torch_dtype  # noqa: E402
from train import TrainingApp  # noqa: E402


def expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)
        else:
            raise FileNotFoundError(f"No files matched: {pattern}")
    return sorted(dict.fromkeys(paths))


def load_trace(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "batch" not in payload:
        raise ValueError(f"Not a VERL OPD trace file: {path}")
    return payload


def active_ids(ids: Any, mask: Any) -> list[int]:
    ids_t = torch.as_tensor(ids).detach().cpu().long()
    mask_t = torch.as_tensor(mask).detach().cpu().bool()
    return ids_t[mask_t].tolist()


def trace_prompt_rows(trace: dict[str, Any]) -> list[list[int]]:
    batch = trace["batch"]
    prompts = batch["prompts"].detach().cpu().long()
    attention_mask = batch["attention_mask"].detach().cpu().long()
    prompt_width = prompts.shape[1]
    prompt_mask = attention_mask[:, :prompt_width]
    return [active_ids(prompts[row], prompt_mask[row]) for row in range(prompts.shape[0])]


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
    if hasattr(value, "tolist") and not isinstance(value, ImageObject):
        return value.tolist()
    return [value]


def image_summary(image: Any) -> dict[str, Any]:
    summary = {"type": type(image).__name__}
    try:
        if isinstance(image, ImageObject):
            summary.update({"size": list(image.size), "mode": image.mode})
        elif isinstance(image, dict):
            summary["keys"] = sorted(str(key) for key in image)
            if image.get("path") is not None:
                summary["path"] = str(image.get("path"))
            if image.get("bytes") is not None:
                data = image["bytes"]
                summary["bytes_len"] = len(data)
                with Image.open(BytesIO(data)) as img:
                    summary.update({"size": list(img.size), "mode": img.mode})
        elif isinstance(image, (str, os.PathLike)):
            path = os.path.expanduser(os.fspath(image))
            summary["path"] = path
            with Image.open(path) as img:
                summary.update({"size": list(img.size), "mode": img.mode})
        elif isinstance(image, bytes):
            summary["bytes_len"] = len(image)
            with Image.open(BytesIO(image)) as img:
                summary.update({"size": list(img.size), "mode": img.mode})
    except Exception as exc:
        summary["inspect_error"] = repr(exc)
    return summary


def trace_images(trace: dict[str, Any], row: int) -> list[Any]:
    non_tensor = trace.get("non_tensor_batch", {})
    value = row_value(non_tensor.get("vllm_images"), row)
    images = to_list(value)
    if images:
        return images
    mm_data = row_value(non_tensor.get("multi_modal_data"), row)
    if isinstance(mm_data, dict):
        return to_list(mm_data.get("images", mm_data.get("image")))
    return to_list(mm_data)


def trace_mm_kwargs(trace: dict[str, Any], row: int) -> dict[str, Any] | None:
    value = row_value(trace.get("non_tensor_batch", {}).get("mm_processor_kwargs"), row)
    return value if isinstance(value, dict) else None


def sample_images(sample: dict[str, Any]) -> list[Any]:
    return to_list(sample.get("images"))


def build_dataset(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any]:
    os.chdir(ROOT)
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
    dataset_config = data_args.dataset_config
    if args.dataset_path:
        with open(dataset_config, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        dataset_names = stage.dataset if isinstance(stage.dataset, list) else [stage.dataset]
        for name in dataset_names:
            if name not in catalog:
                raise KeyError(f"Dataset {name!r} not found in {dataset_config}")
            catalog[name] = dict(catalog[name])
            catalog[name]["file_name_or_path"] = args.dataset_path
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False)
        with tmp:
            json.dump(catalog, tmp, ensure_ascii=False, indent=2)
        dataset_config = tmp.name

    max_samples = None if args.max_samples <= 0 else args.max_samples
    data_args = replace(
        data_args,
        dataset=stage.dataset,
        dataset_config=dataset_config,
        max_samples=max_samples,
        preprocessing_num_workers=args.num_workers,
        overwrite_cache=not args.use_cache,
        log_first_sample=False,
    )
    if args.model_path:
        model_args = replace(model_args, model_name_or_path=args.model_path)

    processor, tokenizer = _load_processor_and_tokenizer(model_args)
    template = TemplateFactory.from_args(tokenizer, data_args)
    dataset = DatasetBuilder(
        template=template,
        model_args=model_args,
        data_args=data_args,
        tokenizer=tokenizer,
        processor=processor,
        trainer=_DummyTrainer(),
    ).build()
    return dataset, template, tokenizer, processor, model_args


def collate_rows(
    *,
    dataset: Any,
    indices: list[int],
    template: Any,
    tokenizer: Any,
    processor: Any,
    model_args: Any,
) -> dict[str, Any]:
    samples = [dataset[index] for index in indices]
    collator = VLCollator(
        template=template,
        model=None,
        tokenizer=tokenizer,
        processor=processor,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        torch_dtype=parse_torch_dtype(model_args.torch_dtype),
    )
    return collator(samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match raw parquet preprocessing output to VERL OPD trace prompts."
    )
    parser.add_argument("--config", required=True, help="CLight OPD config that points to the parquet dataset.")
    parser.add_argument("--traces", nargs="+", required=True, help="VERL OPD trace files or glob patterns.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means scan the whole configured dataset.")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--model-path", default=None, help="Override model.model_name_or_path in the config.")
    parser.add_argument("--dataset-path", default=None, help="Override file_name_or_path for the selected dataset.")
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    parser.add_argument("--print-first", type=int, default=12)
    args = parser.parse_args()

    trace_paths = expand_paths(args.traces)
    dataset, template, tokenizer, processor, model_args = build_dataset(args)

    by_prompt: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        by_prompt[tuple(int(x) for x in dataset[index]["prompt_input_ids"])].append(index)

    records: list[dict[str, Any]] = []
    matched_indices: list[int] = []
    missing = 0

    for trace_i, trace_path in enumerate(trace_paths):
        trace = load_trace(trace_path)
        prompt_rows = trace_prompt_rows(trace)
        for row, prompt_ids in enumerate(prompt_rows):
            candidates = by_prompt.get(tuple(prompt_ids), [])
            dataset_index = candidates.pop(0) if candidates else None
            if dataset_index is None:
                missing += 1
            else:
                matched_indices.append(dataset_index)

            trace_imgs = trace_images(trace, row)
            sample_imgs = sample_images(dataset[dataset_index]) if dataset_index is not None else []
            record = {
                "trace": trace_path,
                "trace_index": trace_i,
                "row": row,
                "matched": dataset_index is not None,
                "dataset_index": dataset_index,
                "prompt_len": len(prompt_ids),
                "trace_image_count": len(trace_imgs),
                "dataset_image_count": len(sample_imgs),
                "image_count_same": len(trace_imgs) == len(sample_imgs),
                "trace_mm_processor_kwargs": trace_mm_kwargs(trace, row),
                "trace_image_summary": [image_summary(image) for image in trace_imgs[:2]],
                "dataset_image_summary": [image_summary(image) for image in sample_imgs[:2]],
            }
            records.append(record)

    unique_matched = sorted(set(index for index in matched_indices if index is not None))
    collator_prompt_same = None
    if unique_matched:
        collated = collate_rows(
            dataset=dataset,
            indices=unique_matched,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            model_args=model_args,
        )
        collated_prompts = [
            tuple(active_ids(ids, mask))
            for ids, mask in zip(
                collated["prompt_input_ids"],
                collated["prompt_attention_mask"],
                strict=False,
            )
        ]
        original_prompts = [tuple(int(x) for x in dataset[index]["prompt_input_ids"]) for index in unique_matched]
        collator_prompt_same = sum(a == b for a, b in zip(collated_prompts, original_prompts, strict=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(records)
    prompt_matched = sum(1 for record in records if record["matched"])
    image_count_same = sum(1 for record in records if record["matched"] and record["image_count_same"])
    print("=== compare parquet preprocess to VERL trace ===")
    print(f"config={args.config}")
    print(f"dataset_count={len(dataset)}")
    print(f"trace_count={len(trace_paths)}")
    print(f"trace_rows={total}")
    print(f"prompt_matched={prompt_matched}/{total}")
    print(f"image_count_same={image_count_same}/{prompt_matched}")
    if collator_prompt_same is not None:
        print(f"collator_prompt_same={collator_prompt_same}/{len(unique_matched)}")
    if args.output:
        print(f"output={args.output}")
    print("first records:")
    for record in records[: args.print_first]:
        print(
            "trace_i=",
            record["trace_index"],
            "row=",
            record["row"],
            "matched=",
            record["matched"],
            "dataset_index=",
            record["dataset_index"],
            "prompt_len=",
            record["prompt_len"],
            "image_count_same=",
            record["image_count_same"],
        )
    ok = prompt_matched == total and image_count_same == prompt_matched
    print(f"RESULT={'OK' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
