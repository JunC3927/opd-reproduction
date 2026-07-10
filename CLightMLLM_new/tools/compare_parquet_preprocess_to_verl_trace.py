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
from src.data.module import DatasetBuilder, SupervisedPreprocessor, TemplateFactory, VLCollator  # noqa: E402
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


def common_prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        count += 1
    return count


def same_ratio(left: list[int], right: list[int]) -> float:
    width = min(len(left), len(right))
    if width == 0:
        return 1.0 if len(left) == len(right) else 0.0
    return sum(a == b for a, b in zip(left[:width], right[:width], strict=False)) / width


def infer_mm_token_ids(tokenizer: Any, image_token_id: int | None = None) -> set[int]:
    tokens = ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
    token_ids: set[int] = set()
    for token in tokens:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            token_id = None
        if isinstance(token_id, int) and token_id >= 0:
            token_ids.add(token_id)
    if image_token_id is not None:
        token_ids.add(image_token_id)
    return token_ids


def strip_token_ids(ids: list[int], remove_ids: set[int]) -> list[int]:
    if not remove_ids:
        return ids
    return [token_id for token_id in ids if token_id not in remove_ids]


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


def closest_prompt_match(
    *,
    prompt_ids: list[int],
    dataset_prompts: list[list[int]],
    dataset_image_counts: list[int],
    trace_image_count: int,
    mm_token_ids: set[int],
) -> dict[str, Any] | None:
    stripped_prompt = strip_token_ids(prompt_ids, mm_token_ids)
    best: dict[str, Any] | None = None
    for index, candidate in enumerate(dataset_prompts):
        stripped_candidate = strip_token_ids(candidate, mm_token_ids)
        strip_mm_same = stripped_prompt == stripped_candidate
        score = (
            1 if dataset_image_counts[index] == trace_image_count else 0,
            1 if strip_mm_same else 0,
            common_prefix_len(prompt_ids, candidate),
            same_ratio(prompt_ids, candidate),
        )
        if best is None or score > best["_score"]:
            best = {
                "_score": score,
                "closest_dataset_index": index,
                "closest_prompt_len": len(candidate),
                "closest_image_count": dataset_image_counts[index],
                "closest_common_prefix": score[2],
                "closest_same_ratio": score[3],
                "strip_mm_same": strip_mm_same,
                "stripped_trace_len": len(stripped_prompt),
                "stripped_dataset_len": len(stripped_candidate),
            }
    if best is not None:
        best.pop("_score", None)
    return best


def _as_python(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, ImageObject):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def normalize_verl_prompt(value: Any) -> list[dict[str, str]]:
    value = _as_python(value)
    if value is None:
        raise ValueError("VERL parquet row is missing the prompt column.")

    turns: list[dict[str, str]] = []
    for turn in list(value):
        turn = _as_python(turn)
        if not isinstance(turn, dict):
            raise TypeError(f"Expected prompt turn to be a dict, got {type(turn).__name__}: {turn!r}")
        role = turn.get("role", turn.get("from"))
        content = turn.get("content", turn.get("value"))
        if role is None or content is None:
            raise KeyError(f"Prompt turn must contain role/content or from/value keys: {turn!r}")
        turns.append({"role": str(role), "content": str(content)})
    return turns


def detect_dataset_format(dataset_path: str | None) -> str | None:
    if not dataset_path:
        return None
    try:
        import pyarrow.parquet as pq

        path = Path(dataset_path)
        parquet_files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
        columns = set(pq.read_schema(parquet_files[0]).names)
    except Exception:
        return None
    if "prompt" in columns and "conversations" not in columns:
        return "verl"
    return None


def build_verl_parquet_dataset(
    *,
    args: argparse.Namespace,
    data_args: Any,
    template: Any,
    tokenizer: Any,
    processor: Any,
) -> list[dict[str, Any]]:
    if not args.dataset_path:
        raise ValueError("--dataset-path is required when --parquet-format=verl.")

    from datasets import load_dataset

    raw_dataset = load_dataset("parquet", data_files=args.dataset_path, split="train")
    if args.max_samples > 0:
        raw_dataset = raw_dataset.select(range(min(args.max_samples, len(raw_dataset))))

    preprocessor = SupervisedPreprocessor(
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
    )
    samples: list[dict[str, Any]] = []
    for row in raw_dataset:
        prompt = normalize_verl_prompt(row.get("prompt"))
        images = to_list(row.get("images", row.get("image")))
        prompt_ids = preprocessor.encode_prompt_example(prompt=prompt, system=row.get("system"), images=images)
        attention_mask = [1] * len(prompt_ids)
        samples.append(
            {
                "input_ids": prompt_ids,
                "attention_mask": attention_mask,
                "labels": [-100] * len(prompt_ids),
                "images": images,
                "prompt_input_ids": prompt_ids,
                "prompt_attention_mask": attention_mask,
                "reference_text": str(row.get("answer", "")),
            }
        )
    return samples


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
    parquet_format = args.parquet_format
    if parquet_format == "auto":
        parquet_format = detect_dataset_format(args.dataset_path) or "sharegpt"
    if parquet_format == "verl":
        dataset = build_verl_parquet_dataset(
            args=args,
            data_args=data_args,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
        )
        return dataset, template, tokenizer, processor, model_args

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
    parser.add_argument(
        "--parquet-format",
        choices=["auto", "sharegpt", "verl"],
        default="auto",
        help="Use 'verl' for VERL-format parquet with prompt/images columns.",
    )
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    parser.add_argument("--print-first", type=int, default=12)
    parser.add_argument("--image-token-id", type=int, default=151655)
    args = parser.parse_args()

    trace_paths = expand_paths(args.traces)
    dataset, template, tokenizer, processor, model_args = build_dataset(args)

    dataset_prompts = [[int(x) for x in dataset[index]["prompt_input_ids"]] for index in range(len(dataset))]
    dataset_image_counts = [len(sample_images(dataset[index])) for index in range(len(dataset))]
    mm_token_ids = infer_mm_token_ids(tokenizer, args.image_token_id)

    by_prompt: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, prompt_ids in enumerate(dataset_prompts):
        by_prompt[tuple(prompt_ids)].append(index)

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
            if dataset_index is None:
                record.update(
                    closest_prompt_match(
                        prompt_ids=prompt_ids,
                        dataset_prompts=dataset_prompts,
                        dataset_image_counts=dataset_image_counts,
                        trace_image_count=len(trace_imgs),
                        mm_token_ids=mm_token_ids,
                    )
                    or {}
                )
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
            "closest_dataset_index=",
            record.get("closest_dataset_index"),
            "closest_prompt_len=",
            record.get("closest_prompt_len"),
            "strip_mm_same=",
            record.get("strip_mm_same"),
            "closest_common_prefix=",
            record.get("closest_common_prefix"),
            "closest_same_ratio=",
            record.get("closest_same_ratio"),
        )
    ok = prompt_matched == total and image_count_same == prompt_matched
    print(f"RESULT={'OK' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
