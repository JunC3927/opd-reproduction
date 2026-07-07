import argparse
import os
import sys
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from PIL.Image import Image as ImageObject
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.module import DatasetBuilder, TemplateFactory, VLCollator  # noqa: E402
from src.hparams import parse_torch_dtype  # noqa: E402
from src.model import load_vision_language_model  # noqa: E402
from train import TrainingApp  # noqa: E402


class _NoopStrategy:
    def barrier(self) -> None:
        return None


class _DummyTrainer:
    local_rank = 0
    global_rank = 0
    is_global_zero = True
    strategy = _NoopStrategy()


def _load_processor_and_tokenizer(model_args: Any):
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


def _shape(value: Any) -> str:
    if torch.is_tensor(value):
        return f"{tuple(value.shape)} {value.dtype}"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return type(value).__name__


def _cpu_tensor_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tensor_tree(item) for item in value]
    return value


def _tensor_only(value: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {key: item.detach().cpu() for key, item in value.items() if torch.is_tensor(item)}


def _is_left_padded(mask: torch.Tensor) -> bool:
    for row in mask.detach().cpu():
        seen_one = False
        for value in row.tolist():
            if value:
                seen_one = True
            elif seen_one:
                return False
    return True


def _image_token_count(
    batch: dict[str, Any],
    tokenizer: Any,
    processor: Any,
    model: Any | None,
) -> tuple[int | None, int | None]:
    image_token_id = None
    if model is not None:
        image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    if image_token_id is None:
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None or image_token_id < 0:
        return None, None

    actual = int((batch["prompt_input_ids"] == image_token_id).sum().item())
    expected = None
    if "image_grid_thw" in batch:
        merge_size = getattr(getattr(processor, "image_processor", None), "merge_size", 1)
        expected = 0
        grids = batch["image_grid_thw"]
        for grid in grids:
            expected += int(grid.prod().item() // (merge_size**2))
    return actual, expected


def _effective_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[list[int]]:
    return [ids[mask.bool()].tolist() for ids, mask in zip(input_ids.detach().cpu(), attention_mask.detach().cpu(), strict=False)]


def _image_summary(image: Any) -> dict[str, Any]:
    summary = {"type": type(image).__name__}
    try:
        if isinstance(image, ImageObject):
            summary.update({"size": tuple(image.size), "mode": image.mode})
        elif isinstance(image, dict):
            summary["keys"] = sorted(str(key) for key in image)
            if image.get("path") is not None:
                summary["path"] = str(image.get("path"))
            if image.get("bytes") is not None:
                data = image["bytes"]
                summary["bytes_len"] = len(data)
                with Image.open(BytesIO(data)) as img:
                    summary.update({"size": tuple(img.size), "mode": img.mode})
        elif isinstance(image, (str, os.PathLike)):
            path = os.path.expanduser(os.fspath(image))
            summary["path"] = path
            with Image.open(path) as img:
                summary.update({"size": tuple(img.size), "mode": img.mode})
        elif isinstance(image, bytes):
            summary["bytes_len"] = len(image)
            with Image.open(BytesIO(image)) as img:
                summary.update({"size": tuple(img.size), "mode": img.mode})
    except Exception as exc:
        summary["inspect_error"] = repr(exc)
    return summary


def _sample_debug(
    samples: list[dict[str, Any]],
    batch: dict[str, Any],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    effective_prompts = _effective_ids(batch["prompt_input_ids"], batch["prompt_attention_mask"])
    image_grid_thw = batch.get("image_grid_thw")
    grid_cursor = 0
    rows = []
    for index, sample in enumerate(samples):
        images = sample.get("images") or []
        if not isinstance(images, (list, tuple)):
            images = [images]
        grids = []
        if torch.is_tensor(image_grid_thw):
            for _ in images:
                if grid_cursor < image_grid_thw.shape[0]:
                    grids.append(image_grid_thw[grid_cursor].detach().cpu().tolist())
                grid_cursor += 1
        prompt_ids = effective_prompts[index]
        rows.append(
            {
                "row": index,
                "prompt_len": len(prompt_ids),
                "image_pad_count": int(sum(token_id == image_token_id for token_id in prompt_ids)),
                "image_count": len(images),
                "image_grid_thw": grids,
                "images": [_image_summary(image) for image in images],
                "prompt_head": tokenizer.decode(prompt_ids[:80], skip_special_tokens=False),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OPD preprocessing without running training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--use-cache", action="store_true", help="Reuse existing HF datasets map cache.")
    parser.add_argument("--load-model", action="store_true", help="Load the student model to validate RoPE fields.")
    parser.add_argument("--output", default=None, help="Optional .pt path to save the preprocessed samples and batch.")
    args = parser.parse_args()

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
    data_args = replace(
        data_args,
        dataset=stage.dataset,
        max_samples=args.max_samples,
        preprocessing_num_workers=args.num_workers,
        overwrite_cache=not args.use_cache,
        log_first_sample=True,
    )

    if args.load_model:
        model, processor, tokenizer = load_vision_language_model(model_args, data_args.template)
        model.eval()
    else:
        model = None
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
    if len(dataset) == 0:
        raise RuntimeError("No examples survived preprocessing.")

    sample_count = min(args.batch_size, len(dataset))
    samples = [dataset[i] for i in range(sample_count)]
    prompt_lengths = [len(sample["prompt_input_ids"]) for sample in samples]
    if any(length > data_args.max_prompt_length for length in prompt_lengths):
        raise AssertionError(f"Found prompt longer than max_prompt_length={data_args.max_prompt_length}: {prompt_lengths}")

    collator = VLCollator(
        template=template,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        pad_to_multiple_of=8,
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        torch_dtype=parse_torch_dtype(model_args.torch_dtype),
    )
    batch = collator(samples)

    print("=== dataset ===")
    print(f"num_examples={len(dataset)} checked_batch={sample_count}")
    print(f"prompt_lengths={prompt_lengths}")
    print(f"max_prompt_length={data_args.max_prompt_length}")
    print("first_prompt:")
    print(tokenizer.decode(samples[0]["prompt_input_ids"], skip_special_tokens=False))

    print("=== batch keys ===")
    for key in sorted(batch):
        print(f"{key}: {_shape(batch[key])}")

    prompt_mask = batch["prompt_attention_mask"]
    if not _is_left_padded(prompt_mask):
        raise AssertionError("prompt_attention_mask is not left-padded.")
    print("prompt_attention_mask_left_padded=True")

    image_actual, image_expected = _image_token_count(batch, tokenizer, processor, model)
    if image_actual is not None:
        print(f"prompt_image_token_count={image_actual}")
    if image_expected is not None:
        print(f"image_grid_token_count={image_expected}")

    if args.load_model:
        if "position_ids" not in batch:
            raise AssertionError("--load-model was set, but collator did not produce position_ids.")
        print("position_ids_present=True")
    else:
        print("position_ids_check=skipped (use --load-model for full Qwen3-VL check)")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "clight_opd_preprocess_batch_v1",
            "config": str(Path(args.config).resolve()),
            "dataset": data_args.dataset,
            "template": data_args.template,
            "max_samples": args.max_samples,
            "sample_count": sample_count,
            "num_examples": len(dataset),
            "prompt_lengths": prompt_lengths,
            "tokenizer": {
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "image_token_id": tokenizer.convert_tokens_to_ids("<|image_pad|>"),
                "padding_side": tokenizer.padding_side,
            },
            "image_token_count": {
                "prompt_actual": image_actual,
                "grid_expected": image_expected,
            },
            "sample_debug": _sample_debug(samples, batch, tokenizer),
            "samples": _cpu_tensor_tree(samples),
            "batch_tensors": _tensor_only(batch),
            "batch_reference_text": batch.get("reference_text"),
            "prompt_text": tokenizer.batch_decode(
                batch["prompt_input_ids"].detach().cpu(),
                skip_special_tokens=False,
            ),
            "input_text": tokenizer.batch_decode(
                batch["input_ids"].detach().cpu(),
                skip_special_tokens=False,
            ),
        }
        torch.save(payload, output_path)
        print(f"saved_preprocess_batch={output_path}")

    print("=== sample debug ===")
    for item in _sample_debug(samples, batch, tokenizer):
        print(
            "row={row} prompt_len={prompt_len} image_pad_count={image_pad_count} "
            "image_count={image_count} image_grid_thw={image_grid_thw} images={images}".format(**item)
        )


if __name__ == "__main__":
    main()
