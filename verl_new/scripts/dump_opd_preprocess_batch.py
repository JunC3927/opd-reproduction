import argparse
import asyncio
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from PIL import Image
from PIL.Image import Image as ImageObject

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl.utils.dataset.rl_dataset import RLHFDataset  # noqa: E402
from verl.utils.tokenizer import (  # noqa: E402
    build_multimodal_processor_inputs,
    get_processor_token_id,
    hf_processor,
    hf_tokenizer,
    normalize_token_ids,
)


def move_tensors_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: move_tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors_to_cpu(item) for item in value]
    return value


def pad_token_ids(
    tokenizer: Any,
    tokens: list[int],
    *,
    max_length: int,
    padding_side: str,
    return_attention_mask: bool,
) -> dict[str, torch.Tensor]:
    if not tokens:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        result = {"input_ids": torch.full((1, max_length), pad_id, dtype=torch.long)}
        if return_attention_mask:
            result["attention_mask"] = torch.zeros((1, max_length), dtype=torch.long)
        return result

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        padded = tokenizer.pad(
            {"input_ids": tokens},
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=return_attention_mask,
        )
    finally:
        tokenizer.padding_side = old_padding_side
    if padded["input_ids"].dim() == 1:
        padded["input_ids"] = padded["input_ids"].unsqueeze(0)
        if return_attention_mask:
            padded["attention_mask"] = padded["attention_mask"].unsqueeze(0)
    return padded


def compute_position_ids(
    processor: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    multi_modal_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    if processor is None or not hasattr(processor, "get_rope_index"):
        return (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)

    multi_modal_kwargs = {
        "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
        "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
    }
    if "mm_token_type_ids" in multi_modal_inputs:
        mm_token_type_ids = torch.zeros_like(input_ids)
        image_token_id = get_processor_token_id(processor, "image")
        video_token_id = get_processor_token_id(processor, "video")
        if image_token_id is not None:
            mm_token_type_ids[input_ids == image_token_id] = 1
        if video_token_id is not None:
            mm_token_type_ids[input_ids == video_token_id] = 2
        multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

    vision_position_ids, _ = processor.get_rope_index(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **multi_modal_kwargs,
    )
    vision_position_ids = vision_position_ids.transpose(0, 1)

    text_position_ids = torch.ones((1, input_ids.shape[-1]), dtype=torch.long)
    valid_mask = attention_mask[0].bool()
    text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
    return torch.cat((text_position_ids.unsqueeze(0), vision_position_ids), dim=1)


def stack_or_concat(tensors: list[torch.Tensor], key: str) -> torch.Tensor:
    if key in {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "images_seqlens"}:
        return torch.cat(tensors, dim=0)
    return torch.cat(tensors, dim=0)


def can_cat_dim0(tensors: list[torch.Tensor]) -> bool:
    if not tensors:
        return False
    reference_shape = tuple(tensors[0].shape[1:])
    return all(tuple(tensor.shape[1:]) == reference_shape for tensor in tensors)


def image_summary(image: Any) -> dict[str, Any]:
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
            if image.get("image") is not None:
                nested = image["image"]
                nested_summary = image_summary(nested)
                summary["image"] = nested_summary
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


def raw_message_image_summary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                kept = {key: item[key] for key in sorted(item) if key not in {"image", "bytes"}}
                if item.get("image") is not None:
                    kept["image"] = image_summary(item["image"])
                if item.get("bytes") is not None:
                    kept["bytes_len"] = len(item["bytes"])
                rows.append(kept)
    return rows


async def build_one(
    *,
    dataset: RLHFDataset,
    tokenizer: Any,
    processor: Any,
    row: dict[str, Any],
    prompt_length: int,
    response_length: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[int], str, dict[str, Any]]:
    messages = list(row["raw_prompt"])
    image_patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", 14)
    images, videos, audios = await dataset.process_multi_modal_info(messages, image_patch_size, dataset.config)

    apply_kwargs = dict(dataset.apply_chat_template_kwargs or {})
    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **apply_kwargs)
    model_inputs = build_multimodal_processor_inputs(
        processor,
        text=[raw_prompt],
        images=images,
        videos=videos,
        audio=audios,
        mm_processor_kwargs=dataset.mm_processor_kwargs,
    )
    prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
    if len(prompt_ids) > prompt_length:
        raise ValueError(f"Prompt length {len(prompt_ids)} exceeds prompt_length={prompt_length}.")

    prompt_output = pad_token_ids(
        tokenizer,
        prompt_ids,
        max_length=prompt_length,
        padding_side="left",
        return_attention_mask=True,
    )
    response_output = pad_token_ids(
        tokenizer,
        [],
        max_length=response_length,
        padding_side="right",
        return_attention_mask=True,
    )
    response_mask_output = pad_token_ids(
        tokenizer,
        [],
        max_length=response_length,
        padding_side="right",
        return_attention_mask=False,
    )
    response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
    input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)
    attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)

    current_text = tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
    multi_modal_inputs = build_multimodal_processor_inputs(
        processor,
        text=[current_text],
        images=images,
        videos=videos,
        audio=audios,
        mm_processor_kwargs=dataset.mm_processor_kwargs,
    )
    multi_modal_inputs.pop("input_ids", None)
    multi_modal_inputs.pop("attention_mask", None)
    multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
    image_grid_thw = multi_modal_inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        multi_modal_inputs["images_seqlens"] = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2],
            image_grid_thw[:, 0],
        )

    position_ids = compute_position_ids(processor, input_ids, attention_mask, dict(multi_modal_inputs))
    tensors = {
        "prompts": prompt_output["input_ids"],
        "prompt_attention_mask": prompt_output["attention_mask"],
        "responses": response_output["input_ids"],
        "response_mask": response_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    image_token_id = get_processor_token_id(processor, "image")
    debug = {
        "row": int(row.get("index", len(prompt_ids))) if isinstance(row.get("index", None), int) else row.get("index"),
        "dataset_index": row.get("index"),
        "prompt_len": len(prompt_ids),
        "image_pad_count": int(sum(token_id == image_token_id for token_id in prompt_ids)),
        "raw_message_images": raw_message_image_summary(messages),
        "processed_images": [image_summary(image) for image in (images or [])],
        "image_grid_thw": image_grid_thw.detach().cpu().tolist() if torch.is_tensor(image_grid_thw) else None,
        "prompt_head": raw_prompt[:1000],
    }
    return tensors, multi_modal_inputs, prompt_ids, raw_prompt, debug


async def async_main(args: argparse.Namespace) -> None:
    os.chdir(ROOT)
    common_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    tokenizer = hf_tokenizer(args.model, **common_kwargs)
    processor = hf_processor(args.model, **common_kwargs)
    if processor is None:
        raise RuntimeError("This dump script requires a multimodal processor.")
    processor.image_min_pixels = args.image_min_pixels
    processor.image_max_pixels = args.image_max_pixels
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = args.image_min_pixels
        processor.image_processor.max_pixels = args.image_max_pixels

    data_config = OmegaConf.create(
        {
            "prompt_key": args.prompt_key,
            "image_key": args.image_key,
            "max_prompt_length": args.max_prompt_length,
            "filter_overlong_prompts": True,
            "truncation": "error",
            "shuffle": False,
            "return_multi_modal_inputs": True,
            "cache_dir": args.cache_dir,
            "mm_processor_kwargs": {},
            "apply_chat_template_kwargs": {},
        }
    )
    dataset = RLHFDataset(
        data_files=args.train_file,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=args.max_samples,
    )

    rows = [dataset[index] for index in range(min(args.batch_size, len(dataset)))]
    per_sample_tensors = []
    per_sample_mm = []
    prompt_ids = []
    raw_prompts = []
    sample_debug = []
    for row in rows:
        tensors, mm_inputs, ids, raw_prompt, debug = await build_one(
            dataset=dataset,
            tokenizer=tokenizer,
            processor=processor,
            row=row,
            prompt_length=args.max_prompt_length,
            response_length=args.max_response_length,
        )
        per_sample_tensors.append(tensors)
        per_sample_mm.append(mm_inputs)
        prompt_ids.append(ids)
        raw_prompts.append(raw_prompt)
        sample_debug.append(debug)

    batch_tensors = {
        key: torch.cat([sample[key] for sample in per_sample_tensors], dim=0)
        for key in per_sample_tensors[0]
    }
    mm_keys = sorted(set().union(*(mm.keys() for mm in per_sample_mm)))
    skipped_mm_shapes = {}
    for key in mm_keys:
        values = [mm[key] for mm in per_sample_mm if key in mm]
        if values and torch.is_tensor(values[0]):
            if can_cat_dim0(values):
                batch_tensors[key] = stack_or_concat(values, key)
            else:
                skipped_mm_shapes[key] = [tuple(value.shape) for value in values]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "verl_opd_preprocess_batch_v1",
        "model_name_or_path": args.model,
        "train_file": args.train_file,
        "sample_count": len(rows),
        "prompt_lengths": [len(ids) for ids in prompt_ids],
        "batch_tensors": move_tensors_to_cpu(batch_tensors),
        "skipped_mm_shapes": skipped_mm_shapes,
        "raw_prompt_text": raw_prompts,
        "prompt_text": tokenizer.batch_decode(batch_tensors["prompts"], skip_special_tokens=False),
        "effective_prompt_ids": prompt_ids,
        "sample_debug": sample_debug,
        "tokenizer": {
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "image_token_id": get_processor_token_id(processor, "image"),
            "padding_side": tokenizer.padding_side,
        },
    }
    torch.save(payload, output_path)

    print("=== dumped verl opd preprocess batch ===")
    print(f"output={output_path}")
    print(f"sample_count={len(rows)}")
    print(f"prompt_lengths={payload['prompt_lengths']}")
    for key, value in sorted(batch_tensors.items()):
        print(f"{key}: {tuple(value.shape)} {value.dtype}")
    if skipped_mm_shapes:
        print(f"skipped_mm_shapes={skipped_mm_shapes}")
    print("=== sample debug ===")
    for item in sample_debug:
        print(
            "row={row} dataset_index={dataset_index} prompt_len={prompt_len} "
            "image_pad_count={image_pad_count} image_grid_thw={image_grid_thw} "
            "raw_message_images={raw_message_images} processed_images={processed_images}".format(**item)
        )
    print("first_prompt:")
    print(payload["prompt_text"][0])
    print("dump_ok=True")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump verl OPD prompt preprocessing tensors.")
    parser.add_argument("--model", default="/raid/lwz/lzh/models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--train-file", default="/home/user01/data/geo3k/train.parquet")
    parser.add_argument("--output", default="/tmp/verl_opd_preprocess_batch.pt")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=512)
    parser.add_argument("--image-min-pixels", type=int, default=1024)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--image-key", default="images")
    parser.add_argument("--cache-dir", default="~/.cache/verl/rlhf")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
