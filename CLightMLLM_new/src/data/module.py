import inspect
import json
import logging
import math
import os
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, fields
from io import BytesIO
from typing import Any, Literal
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import numpy as np
import torch
from datasets import Image as DatasetImage
from datasets import concatenate_datasets, load_dataset
from peft import PeftModel
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from transformers.image_utils import get_image_size, make_flat_list_of_images, to_numpy_array

from ..hparams import parse_torch_dtype

IGNORE_INDEX = -100
IMAGE_PLACEHOLDER = "<image>"
MROPE_MODELS = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s|%(asctime)s] %(name)s:%(lineno)s >> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
for logger_name in ("httpx", "httpcore"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)

LOGGER = logging.getLogger(__name__)
rank_zero_info = rank_zero_only(LOGGER.info)
rank_zero_warning = rank_zero_only(LOGGER.warning)


@dataclass
class BasePlugin:
    image_token: str | None = None
    expand_mm_tokens: bool = True
    vision_bos_token: str = ""
    vision_eos_token: str = ""

    def validate_input(self, processor: Any, images: list[Any]) -> None:
        if images and self.image_token is None:
            raise ValueError("This template does not support image input.")
        if self.image_token is not None and images and processor is None:
            raise ValueError("Processor is required for image training.")
        if self.image_token is not None and images and getattr(processor, "image_processor", None) is None:
            raise ValueError("Image processor is required for image training.")

    def validate_messages(self, messages: list[dict[str, str]], images: list[Any]) -> None:
        placeholders = sum(message["content"].count(IMAGE_PLACEHOLDER) for message in messages)
        if len(images) != placeholders:
            raise ValueError(f"images={len(images)} but {IMAGE_PLACEHOLDER} placeholders={placeholders}: {messages}")

    def open_image(self, image: Any) -> ImageObject:
        if isinstance(image, ImageObject):
            return image
        if isinstance(image, str):
            with Image.open(os.path.expanduser(image)) as img:
                return img.copy()
        if isinstance(image, bytes):
            with Image.open(BytesIO(image)) as img:
                return img.copy()
        if isinstance(image, dict):
            if image.get("bytes") is not None:
                with Image.open(BytesIO(image["bytes"])) as img:
                    return img.copy()
            if image.get("path") is not None:
                with Image.open(os.path.expanduser(image["path"])) as img:
                    return img.copy()
        if hasattr(image, "read"):
            with Image.open(image) as img:
                return img.copy()
        raise TypeError(f"Unsupported image input type: {type(image)}")

    def preprocess_image(self, image: ImageObject, image_max_pixels: int, image_min_pixels: int, **kwargs) -> ImageObject:
        pixels = image.width * image.height
        if pixels > image_max_pixels:
            scale = math.sqrt(image_max_pixels / pixels)
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))

        pixels = image.width * image.height
        if pixels < image_min_pixels:
            scale = math.sqrt(image_min_pixels / pixels)
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))

        return image.convert("RGB") if image.mode != "RGB" else image

    def regularize_images(self, images: list[Any], processor: Any) -> list[ImageObject]:
        return [
            self.preprocess_image(
                self.open_image(image),
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )
            for image in images
        ]

    def build_mm_inputs(self, images: list[Any], processor: Any) -> dict[str, Any]:
        if not images:
            return {}
        kwargs = {}
        if getattr(processor, "image_do_pan_and_scan", False):
            kwargs.update(
                do_pan_and_scan=True,
                pan_and_scan_min_crop_size=256,
                pan_and_scan_max_num_crops=4,
                pan_and_scan_min_ratio_to_activate=1.2,
            )
        return processor.image_processor(self.regularize_images(images, processor), return_tensors="pt", **kwargs)

    def process_messages(self, messages: list[dict[str, str]], images: list[Any], processor: Any) -> list[dict[str, str]]:
        self.validate_input(processor, images)
        return messages

    def get_mm_inputs(self, images: list[Any], processor: Any) -> dict[str, Any]:
        self.validate_input(processor, images)
        return {} if processor is None or not images else self.build_mm_inputs(images, processor)


@dataclass
class LlavaPlugin(BasePlugin):
    def process_messages(self, messages: list[dict[str, str]], images: list[Any], processor: Any) -> list[dict[str, str]]:
        self.validate_input(processor, images)
        self.validate_messages(messages, images)
        messages = deepcopy(messages)
        image_seqlen = 1

        # LLaVA expands each placeholder to one token per visual patch.
        if self.expand_mm_tokens and images:
            mm_inputs = self.build_mm_inputs(images, processor)
            if "pixel_values" in mm_inputs:
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0]))
                image_seqlen = (height // processor.patch_size) * (width // processor.patch_size)
                image_seqlen += getattr(processor, "num_additional_image_tokens", 0)
                if getattr(processor, "vision_feature_select_strategy", None) == "default":
                    image_seqlen -= 1

        for message in messages:
            message["content"] = message["content"].replace(IMAGE_PLACEHOLDER, (self.image_token or "") * image_seqlen)
        return messages


@dataclass
class InternVLPlugin(BasePlugin):
    def build_mm_inputs(self, images: list[Any], processor: Any) -> dict[str, Any]:
        if not images:
            return {}

        kwargs = {}
        if getattr(processor, "crop_to_patches", False):
            kwargs.update(crop_to_patches=True, max_patches=12, min_patches=1)

        images = [
            self.preprocess_image(
                self.open_image(image),
                image_max_pixels=getattr(processor, "image_max_pixels", 1024 * 1024),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )
            for image in images
        ]
        image_inputs = processor.image_processor(images=make_flat_list_of_images(images), return_tensors="pt", **kwargs)
        num_patches = image_inputs.pop("num_patches")
        pixel_values = image_inputs.pop("pixel_values")
        patch_ends = np.cumsum(num_patches)
        patches = []

        for i in range(len(images)):
            start = patch_ends[i - 1] if i > 0 else 0
            patches.append(pixel_values[start : patch_ends[i]])

        return {"pixel_values": torch.cat(patches, dim=0), "image_num_patches": num_patches}

    def process_messages(self, messages: list[dict[str, str]], images: list[Any], processor: Any) -> list[dict[str, str]]:
        self.validate_input(processor, images)
        self.validate_messages(messages, images)
        messages = deepcopy(messages)
        mm_inputs = self.build_mm_inputs(images, processor)
        image_num_patches = mm_inputs.get("image_num_patches", [])
        image_seqlen = getattr(processor, "image_seq_length", 1) if self.expand_mm_tokens else 1
        image_idx = 0

        # InternVL wraps each dynamic patch group with image tags.
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                patches = int(image_num_patches[image_idx]) if self.expand_mm_tokens and len(image_num_patches) else 1
                content = content.replace(IMAGE_PLACEHOLDER, f"<img>{'<IMG_CONTEXT>' * image_seqlen * patches}</img>", 1)
                image_idx += 1
            message["content"] = content
        return messages

    def get_mm_inputs(self, images: list[Any], processor: Any) -> dict[str, Any]:
        self.validate_input(processor, images)
        if processor is None or not images:
            return {}
        mm_inputs = self.build_mm_inputs(images, processor)
        mm_inputs.pop("image_num_patches", None)
        return mm_inputs


@dataclass
class Qwen3VLPlugin(BasePlugin):
    vision_bos_token: str = "<|vision_start|>"
    vision_eos_token: str = "<|vision_end|>"

    def build_mm_inputs(self, images: list[Any], processor: Any) -> dict[str, Any]:
        if not images:
            return {}

        images = self.regularize_images(images, processor)
        image_text = "".join(
            f"{self.vision_bos_token}{self.image_token or ''}{self.vision_eos_token}" for _ in images
        )
        mm_inputs = processor(
            text=[image_text],
            images=images,
            return_tensors="pt",
        )
        mm_inputs = dict(mm_inputs)
        mm_inputs.pop("input_ids", None)
        mm_inputs.pop("attention_mask", None)
        return mm_inputs

    def regularize_images(self, images: list[Any], processor: Any) -> list[ImageObject]:
        if not images:
            return []
        try:
            from qwen_vl_utils import process_vision_info
        except Exception:
            return super().regularize_images(images, processor)

        content = []
        for image in images:
            if isinstance(image, ImageObject):
                content.append({"type": "image", "image": image.convert("RGB")})
            elif isinstance(image, dict):
                item = dict(image)
                if "bytes" in item and "image" not in item:
                    item["image"] = Image.open(BytesIO(item["bytes"]))
                elif "path" in item and "image" not in item:
                    item["image"] = os.path.expanduser(item["path"])
                item["type"] = "image"
                content.append(item)
            elif isinstance(image, bytes):
                content.append({"type": "image", "image": Image.open(BytesIO(image)).convert("RGB")})
            elif isinstance(image, str | os.PathLike):
                content.append({"type": "image", "image": os.path.expanduser(os.fspath(image))})
            elif hasattr(image, "read"):
                content.append({"type": "image", "image": Image.open(image).convert("RGB")})
            else:
                raise TypeError(f"Unsupported image input type: {type(image)}")

        processed_images, _processed_videos = process_vision_info(
            [{"role": "user", "content": content}],
            image_patch_size=getattr(getattr(processor, "image_processor", None), "patch_size", 14),
            return_video_metadata=True,
        )
        return [
            image.convert("RGB") if isinstance(image, ImageObject) and image.mode != "RGB" else image
            for image in (processed_images or [])
        ]

    def preprocess_image(self, image: ImageObject, **kwargs) -> ImageObject:
        image = super().preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            image = image.resize((max(image.width, 28), max(image.height, 28)))
        if image.width / image.height > 200:
            image = image.resize((image.height * 180, image.height))
        if image.height / image.width > 200:
            image = image.resize((image.width, image.width * 180))
        return image

    def process_messages(self, messages: list[dict[str, str]], images: list[Any], processor: Any) -> list[dict[str, str]]:
        self.validate_input(processor, images)
        self.validate_messages(messages, images)
        messages = deepcopy(messages)
        merge_length = getattr(processor.image_processor, "merge_size", 1) ** 2
        image_grid_thw = self.build_mm_inputs(images, processor).get("image_grid_thw", []) if self.expand_mm_tokens and images else []
        image_idx = 0

        # Qwen VL counts visual tokens after spatial merge.
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = int(image_grid_thw[image_idx].prod().item() // merge_length) if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{(self.image_token or '') * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                image_idx += 1
            message["content"] = content
        return messages


class Qwen2VLPlugin(Qwen3VLPlugin):
    pass


class PluginFactory:
    PLUGINS = {
        "base": BasePlugin,
        "llava": LlavaPlugin,
        "intern_vl": InternVLPlugin,
        "qwen2_vl": Qwen2VLPlugin,
        "qwen3_vl": Qwen3VLPlugin,
    }
    ALLOWED_KWARGS = {"expand_mm_tokens", "vision_bos_token", "vision_eos_token"}

    @classmethod
    def create(cls, name: str, image_token: str | None = None, **kwargs) -> BasePlugin:
        if name not in cls.PLUGINS:
            raise ValueError(f"Unknown multimodal plugin: {name}")
        kept = {key: value for key, value in kwargs.items() if key in cls.ALLOWED_KWARGS}
        return cls.PLUGINS[name](image_token=image_token, **kept)


TemplateElement = str | set[str] | dict[str, str]


@dataclass
class Template:
    user: list[TemplateElement] = field(default_factory=lambda: ["{{content}}"])
    assistant: list[TemplateElement] = field(default_factory=lambda: ["{{content}}", {"eos_token"}])
    system: list[TemplateElement] = field(default_factory=lambda: ["{{content}}"])
    prefix: list[TemplateElement] = field(default_factory=list)
    default_system: str = ""
    stop_words: list[str] = field(default_factory=list)
    replace_eos: bool = False
    efficient_eos: bool = False
    mm_plugin: BasePlugin = field(default_factory=lambda: PluginFactory.create("base"))

    def encode_multiturn(self, tokenizer: Any, messages: list[dict[str, str]], system: str | None = None) -> list[tuple[list[int], list[int]]]:
        encoded = self.encode_messages(tokenizer, messages, system)
        return [(encoded[i], encoded[i + 1]) for i in range(0, len(encoded), 2)]

    def format_elements(self, elements: list[TemplateElement], content: str = "") -> list[TemplateElement]:
        return [element.replace("{{content}}", content) if isinstance(element, str) else element for element in elements]

    def encode_elements(self, tokenizer: Any, elements: list[TemplateElement]) -> list[int]:
        token_ids = []
        for element in elements:
            if isinstance(element, str) and element:
                token_ids += tokenizer.encode(element, add_special_tokens=False)
            elif isinstance(element, set):
                if "bos_token" in element and tokenizer.bos_token_id is not None:
                    token_ids.append(tokenizer.bos_token_id)
                if "eos_token" in element and tokenizer.eos_token_id is not None:
                    token_ids.append(tokenizer.eos_token_id)
            elif isinstance(element, dict):
                token_ids.append(tokenizer.convert_tokens_to_ids(element["token"]))
        return token_ids

    def encode_messages(self, tokenizer: Any, messages: list[dict[str, str]], system: str | None = None) -> list[list[int]]:
        system = system or self.default_system
        encoded = []

        for idx, message in enumerate(messages):
            elements = []
            if idx == 0:
                elements += self.prefix
                if system:
                    elements += self.format_elements(self.system, system)

            if message["role"] == "user":
                elements += self.format_elements(self.user, message["content"])
            elif message["role"] == "assistant":
                elements += self.format_elements(self.assistant, message["content"])
            else:
                raise ValueError(f"Unexpected role: {message['role']}")

            encoded.append(self.encode_elements(tokenizer, elements))
        return encoded

    def fix_special_tokens(self, tokenizer: Any) -> None:
        stop_words = list(self.stop_words)
        if self.replace_eos and stop_words:
            eos_token = stop_words.pop(0)
            if tokenizer.eos_token != eos_token:
                tokenizer.add_special_tokens({"eos_token": eos_token})
                rank_zero_info(f"Set eos token: {tokenizer.eos_token}")

        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if stop_words:
            try:
                num_added_tokens = tokenizer.add_special_tokens(
                    {"additional_special_tokens": stop_words}, replace_additional_special_tokens=False
                )
            except TypeError:
                num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": stop_words})
            if num_added_tokens > 0:
                rank_zero_warning("New tokens have been added, make sure `resize_vocab` is True.")


class TemplateFactory:
    TEMPLATES = {
        "llava": Template(
            user=["USER: {{content}} ASSISTANT:"],
            default_system="A chat between a curious user and an artificial intelligence assistant.",
            mm_plugin=PluginFactory.create("llava", image_token="<image>"),
        ),
        "intern_vl": Template(
            user=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            assistant=["{{content}}<|im_end|>\n"],
            system=["<|im_start|>system\n{{content}}<|im_end|>\n"],
            prefix=[{"bos_token"}],
            default_system="你是书生·万象，英文名是InternVL。",
            stop_words=["<|im_end|>"],
            mm_plugin=PluginFactory.create("intern_vl", image_token="<image>"),
        ),
        "qwen2_vl": Template(
            user=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            assistant=["{{content}}<|im_end|>\n"],
            system=["<|im_start|>system\n{{content}}<|im_end|>\n"],
            default_system="You are a helpful assistant.",
            stop_words=["<|im_end|>"],
            replace_eos=True,
            mm_plugin=PluginFactory.create("qwen2_vl", image_token="<|image_pad|>"),
        ),
        "qwen3_vl": Template(
            user=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            assistant=["{{content}}<|im_end|>\n"],
            system=["<|im_start|>system\n{{content}}<|im_end|>\n"],
            stop_words=["<|im_end|>"],
            replace_eos=True,
            mm_plugin=PluginFactory.create("qwen3_vl", image_token="<|image_pad|>"),
        ),
    }

    @classmethod
    def from_args(cls, tokenizer: Any, data_args: Any) -> Template:
        if data_args.template not in cls.TEMPLATES:
            raise ValueError(f"Unknown template `{data_args.template}`. Available templates: {', '.join(cls.TEMPLATES)}")

        template = deepcopy(cls.TEMPLATES[data_args.template])
        if data_args.default_system is not None:
            template.default_system = data_args.default_system
        template.fix_special_tokens(tokenizer)
        return template


@dataclass
class SupervisedPreprocessor:
    template: Template
    tokenizer: Any
    processor: Any
    data_args: Any

    @staticmethod
    def infer_lengths(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
        if target_len * 2 < cutoff_len:
            max_target_len = cutoff_len
        elif source_len * 2 < cutoff_len:
            max_target_len = cutoff_len - source_len
        else:
            max_target_len = int(cutoff_len * target_len / (source_len + target_len))

        new_target_len = min(max_target_len, target_len)
        new_source_len = min(max(cutoff_len - new_target_len, 0), source_len)
        return new_source_len, new_target_len

    def encode_sft_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: str | None,
        images: list[Any],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, self.processor)
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system)
        input_ids: list[int] = []
        labels: list[int] = []
        total_length = 1 if self.template.efficient_eos else 0

        for source_ids, target_ids in encoded_pairs:
            if total_length >= self.data_args.cutoff_len:
                break
            source_len, target_len = self.infer_lengths(
                len(source_ids),
                len(target_ids),
                self.data_args.cutoff_len - total_length,
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len
            input_ids += source_ids + target_ids
            labels += [IGNORE_INDEX] * source_len + target_ids

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]
        return input_ids, labels

    def encode_prompt_example(
        self,
        prompt: list[dict[str, str]],
        system: str | None,
        images: list[Any],
    ) -> list[int]:
        messages = self.template.mm_plugin.process_messages(prompt, images, self.processor)
        encoded = self.template.encode_messages(self.tokenizer, messages, system)
        input_ids = [token_id for turn_ids in encoded for token_id in turn_ids]
        return input_ids

    def preprocess_batch(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "images": [],
            "prompt_input_ids": [],
            "prompt_attention_mask": [],
            "reference_text": [],
        }
        batches = zip(examples["_prompt"], examples["_response"], examples["_system"], examples["_images"])

        dropped_overlong = 0
        for prompt, response, system, images in batches:
            if len(prompt) % 2 != 1 or len(response) != 1:
                rank_zero_warning("Dropped invalid example: {}".format(prompt + response))
                continue

            prompt_ids = self.encode_prompt_example(prompt=prompt, system=system, images=images or [])
            if self.data_args.filter_overlong_prompts and len(prompt_ids) > self.data_args.max_prompt_length:
                dropped_overlong += 1
                continue

            input_ids, labels = self.encode_sft_example(
                prompt=prompt,
                response=response,
                system=system,
                images=images or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(images)
            model_inputs["prompt_input_ids"].append(prompt_ids)
            model_inputs["prompt_attention_mask"].append([1] * len(prompt_ids))
            model_inputs["reference_text"].append(response[0]["content"])

        if dropped_overlong:
            rank_zero_warning(f"Dropped {dropped_overlong} examples with prompt longer than {self.data_args.max_prompt_length}.")
        return model_inputs

    def log_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = [token_id for token_id in example["labels"] if token_id != IGNORE_INDEX]
        prompt_ids = example.get("prompt_input_ids", [])
        rank_zero_info(
            "input_ids:\n%s\ninputs:\n%s\nlabel_ids:\n%s\nlabels:\n%s\nprompt_len=%s\nprompt_input_ids:\n%s\nprompt:\n%s",
            example["input_ids"],
            self.tokenizer.decode(example["input_ids"], skip_special_tokens=False),
            example["labels"],
            self.tokenizer.decode(valid_labels, skip_special_tokens=False),
            len(prompt_ids),
            prompt_ids,
            self.tokenizer.decode(prompt_ids, skip_special_tokens=False),
        )


@dataclass
class DatasetAttr:
    data_path: str
    load_from: Literal["parquet", "json", "hf_hub"] = "parquet"
    formatting: Literal["sharegpt", "verl_prompt"] = "sharegpt"
    subset: str | None = None
    # Root directory for relative image paths in local JSON datasets.
    json_image_root: str | None = None
    split: str = "train"
    num_samples: int | None = None
    messages: str = "conversations"
    images: str = "image"
    system: str | None = None
    role_tag: str = "from"
    content_tag: str = "value"
    user_tag: str = "human"
    assistant_tag: str = "gpt"
    system_tag: str = "system"


@dataclass
class ShareGPTConverter:
    dataset_attr: DatasetAttr

    def normalize_images(self, images: Any) -> list[Any] | None:
        if images is None:
            return None

        images = images[:] if isinstance(images, list) else [images]
        if self.dataset_attr.json_image_root:
            image_root = os.path.expanduser(self.dataset_attr.json_image_root)
            # Keep absolute paths and non-path image objects unchanged.
            images = [
                os.path.join(image_root, image)
                if isinstance(image, str) and not os.path.isabs(os.path.expanduser(image))
                else image
                for image in images
            ]
        return images or None

    def split_system(self, example: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        attr = self.dataset_attr
        raw_turns = list(example[attr.messages])

        if attr.system_tag and raw_turns and raw_turns[0][attr.role_tag] == attr.system_tag:
            return raw_turns[0][attr.content_tag], raw_turns[1:]
        if attr.system:
            return example[attr.system], raw_turns
        return "", raw_turns

    def align_turns(self, raw_turns: list[dict[str, Any]]) -> list[dict[str, str]]:
        attr = self.dataset_attr
        dialogue = []

        for turn_idx, raw_turn in enumerate(raw_turns):
            is_user_turn = turn_idx % 2 == 0
            expected_tag = attr.user_tag if is_user_turn else attr.assistant_tag
            if raw_turn[attr.role_tag] != expected_tag:
                rank_zero_warning(f"Invalid role tag in {raw_turns}.")
                return []

            role = "user" if is_user_turn else "assistant"
            content = raw_turn[attr.content_tag]
            dialogue.append({"role": role, "content": content})

        return dialogue if len(dialogue) % 2 == 0 else []

    @staticmethod
    def split_prompt_response(dialogue: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        if not dialogue:
            return [], []
        return dialogue[:-1], dialogue[-1:]

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        attr = self.dataset_attr
        system_prompt, raw_turns = self.split_system(example)
        dialogue = self.align_turns(raw_turns)
        prompt_turns, response_turn = self.split_prompt_response(dialogue)
        image_inputs = self.normalize_images(example[attr.images]) if attr.images else None

        return {
            "_prompt": prompt_turns,
            "_response": response_turn,
            "_system": system_prompt,
            "_images": image_inputs,
        }


@dataclass
class VERLPromptConverter:
    dataset_attr: DatasetAttr

    def normalize_images(self, images: Any) -> list[Any] | None:
        if images is None:
            return None
        return images[:] if isinstance(images, list) else [images]

    def normalize_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces = []
            for item in content:
                if isinstance(item, str):
                    pieces.append(item)
                elif isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        pieces.append(str(item.get("text", "")))
                    elif item_type == "image":
                        pieces.append(IMAGE_PLACEHOLDER)
                    else:
                        pieces.append(str(item.get("text", item.get("content", ""))))
                else:
                    pieces.append(str(item))
            return "".join(pieces)
        return str(content)

    def normalize_prompt(self, raw_prompt: Any) -> tuple[str, list[dict[str, str]]]:
        if raw_prompt is None:
            return "", []
        turns = raw_prompt[:] if isinstance(raw_prompt, list) else [raw_prompt]
        system_prompt = ""
        prompt: list[dict[str, str]] = []
        for turn in turns:
            if isinstance(turn, dict):
                role = str(turn.get("role", "user"))
                content = self.normalize_content(turn.get("content", ""))
            else:
                role = "user"
                content = self.normalize_content(turn)
            if role == "system":
                system_prompt = content
            elif role in {"user", "assistant"}:
                prompt.append({"role": role, "content": content})
        return system_prompt, prompt

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        attr = self.dataset_attr
        system_prompt, prompt = self.normalize_prompt(example[attr.messages])
        images = self.normalize_images(example[attr.images]) if attr.images else None
        return {
            "_prompt": prompt,
            # OPD uses prompt-only on-policy rollouts; keep a dummy assistant turn so
            # the existing SFT preprocessor/collator can produce standard fields.
            "_response": [{"role": "assistant", "content": ""}],
            "_system": system_prompt,
            "_images": images,
        }


def setup_dataset_map_workers(data_args: Any) -> None:
    """Configure compact safeguards for HF Datasets `.map(num_proc=...)`."""
    workers = data_args.preprocessing_num_workers
    if workers and workers > 1:
        if str(os.environ.get("TOKENIZERS_PARALLELISM", "")).strip().lower() not in {"0", "false", "f", "no", "n", "off"}:
            rank_zero_warning(
                "Forcing TOKENIZERS_PARALLELISM=false because preprocessing_num_workers > 1. "
                "Dataset.map(num_proc=...) provides the parallelism here."
            )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        threads = data_args.preprocessing_omp_num_threads or 0
        if threads > 0:
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ.setdefault(name, str(threads))

        method = data_args.preprocessing_mp_start_method
        if method is not None:
            try:
                import multiprocess as mp
                mp.set_start_method(method, force=True)
                rank_zero_info(f"Using multiprocess start method {method!r} for HF Datasets map workers.")
            except Exception as exc:
                rank_zero_warning(f"Could not set datasets multiprocessing start method to {method!r}: {exc}")


@dataclass
class DatasetCatalog:
    config_path: str

    def select(self, dataset_names: list[str] | None) -> list[DatasetAttr]:
        if not dataset_names:
            return []

        config_path = os.path.expanduser(self.config_path)
        with open(config_path, encoding="utf-8") as f:
            dataset_info = json.load(f)

        attrs: list[DatasetAttr] = []
        attr_keys = {item.name for item in fields(DatasetAttr)} - {"data_path", "load_from"}
        for dataset_name in dataset_names:
            if dataset_name not in dataset_info:
                raise ValueError(f"Undefined dataset {dataset_name!r} in {config_path}.")

            info = dataset_info[dataset_name]
            load_from = info["load_from"]
            data_path = os.path.expanduser(info["file_name_or_path"])

            kwargs = {
                key: value
                for source in (info, info.get("columns", {}), info.get("tags", {}))
                for key, value in source.items()
                if key in attr_keys
            }
            attrs.append(DatasetAttr(data_path=data_path, load_from=load_from, **kwargs))
        return attrs


@dataclass
class DatasetBuilder:
    template: Template
    model_args: Any
    data_args: Any
    tokenizer: Any
    processor: Any = None
    trainer: Any | None = None

    @contextmanager
    def rank_zero_first(self, local: bool = True):
        rank = self.trainer.local_rank if local else self.trainer.global_rank

        # Let one rank create HF Datasets cache before others read it.
        if rank != 0:
            self.trainer.strategy.barrier()
        try:
            yield
        finally:
            if rank == 0:
                self.trainer.strategy.barrier()

    def build(self) -> Any:
        with self.rank_zero_first(local=True):
            raw_dataset = self.load_all()
        with self.rank_zero_first(local=True):
            return self.preprocess(raw_dataset)

    @staticmethod
    def parquet_files(path: str) -> list[str]:
        if os.path.isdir(path):
            files = [os.path.join(path, name) for name in sorted(os.listdir(path)) if name.endswith(".parquet")]
        elif os.path.isfile(path) and path.endswith(".parquet"):
            files = [path]
        else:
            files = []
        if not files:
            raise ValueError(f"No parquet files found under: {path}")
        return files

    def load_from_cache(self) -> bool:
        # Non-zero ranks always reuse the cache built by rank zero.
        return (not self.data_args.overwrite_cache) or self.trainer.local_rank != 0

    def align(self, dataset: Any, dataset_attr: DatasetAttr) -> Any:
        if dataset_attr.formatting == "sharegpt":
            converter = ShareGPTConverter(dataset_attr)
        elif dataset_attr.formatting == "verl_prompt":
            converter = VERLPromptConverter(dataset_attr)
        else:
            raise ValueError(f"Unsupported dataset formatting: {dataset_attr.formatting}")
        column_names = list(next(iter(dataset)).keys())
        return dataset.map(
            converter,
            remove_columns=column_names,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=self.load_from_cache(),
            desc="Converting sharegpt rows",
        )

    def load_one(self, dataset_attr: DatasetAttr) -> Any:
        if dataset_attr.load_from == "hf_hub":
            dataset = load_dataset(
                dataset_attr.data_path,
                name=dataset_attr.subset,
                split=dataset_attr.split,
                cache_dir=self.model_args.cache_dir,
                token=self.model_args.hf_hub_token,
                num_proc=self.data_args.preprocessing_num_workers,
            )
            # Avoid eager PIL decoding for HF Image features during map/cache.
            if dataset_attr.images in getattr(dataset, "column_names", []):
                image_feature = getattr(dataset, "features", {}).get(dataset_attr.images)
                if isinstance(image_feature, DatasetImage):
                    dataset = dataset.cast_column(dataset_attr.images, DatasetImage(decode=False))
        elif dataset_attr.load_from == "parquet":
            dataset = load_dataset(
                "parquet",
                data_files=self.parquet_files(dataset_attr.data_path),
                split=dataset_attr.split,
                cache_dir=self.model_args.cache_dir,
                token=self.model_args.hf_hub_token,
                num_proc=self.data_args.preprocessing_num_workers,
            )
        elif dataset_attr.load_from == "json":
            dataset = load_dataset(
                "json",
                data_files=dataset_attr.data_path,
                split=dataset_attr.split,
                cache_dir=self.model_args.cache_dir,
                token=self.model_args.hf_hub_token,
                num_proc=self.data_args.preprocessing_num_workers,
            )
        else:
            raise NotImplementedError(f"Unknown dataset load type: {dataset_attr.load_from}.")

        if dataset_attr.num_samples is not None:
            target = dataset_attr.num_samples
            indexes = np.random.permutation(len(dataset))[:target]
            if target > len(indexes):
                indexes = np.concatenate((indexes, np.random.choice(len(dataset), target - len(indexes))), axis=0)
            dataset = dataset.select(indexes)

        if self.data_args.max_samples is not None:
            dataset = dataset.select(range(min(self.data_args.max_samples, len(dataset))))

        return self.align(dataset, dataset_attr)

    def load_all(self) -> Any:
        attrs = DatasetCatalog(self.data_args.dataset_config).select(self.data_args.dataset)
        if not attrs:
            raise ValueError("data.dataset is required.")
        datasets = [self.load_one(attr) for attr in attrs]
        return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    def preprocess(self, dataset: Any) -> Any:
        preprocessor = SupervisedPreprocessor(
            template=self.template,
            tokenizer=self.tokenizer,
            processor=self.processor,
            data_args=self.data_args,
        )
        column_names = list(next(iter(dataset)).keys())
        dataset = dataset.map(
            preprocessor.preprocess_batch,
            batched=True,
            batch_size=self.data_args.preprocessing_batch_size,
            remove_columns=column_names,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=self.load_from_cache(),
            desc="Tokenizing sharegpt dataset",
        )

        if self.data_args.log_first_sample and self.trainer.is_global_zero:
            try:
                rank_zero_info("Training example:")
                preprocessor.log_data_example(next(iter(dataset)))
            except StopIteration as exc:
                raise RuntimeError("No valid examples found after preprocessing.") from exc
        return dataset


class VLCollator(DataCollatorForSeq2Seq):
    template: Template
    processor: Any
    torch_dtype: torch.dtype

    def __init__(self, *args, template: Template, processor: Any = None, torch_dtype: torch.dtype = torch.float32, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.template = template
        self.processor = processor
        self.torch_dtype = torch_dtype
        deprecation_warnings = getattr(self.tokenizer, "deprecation_warnings", None)
        if isinstance(deprecation_warnings, dict):
            deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        if isinstance(self.model, PeftModel):
            self.model = self.model.base_model.model

        # Resolve RoPE helper from the model or its inner model.
        self.get_rope_func = (
            getattr(self.model, "get_rope_index", None)
            or getattr(getattr(self.model, "model", None), "get_rope_index", None)
        )

    def inject_dummy_image(self, features: list[dict[str, Any]], images_per_sample: list[list[Any]]) -> bool:
        if self.processor is None or self.template.mm_plugin.image_token is None:
            return False
        if sum(len(images) for images in images_per_sample) != 0 or len(features) == 0:
            return False

        # Some VLM collators need image fields even for text-only batches.
        fake_image = Image.new("RGB", (64, 64), color=(255, 255, 255))
        fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
        fake_messages = self.template.mm_plugin.process_messages(fake_messages, [fake_image], self.processor)
        fake_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)

        features[0]["input_ids"] = list(features[0]["input_ids"]) + fake_ids
        features[0]["attention_mask"] = list(features[0]["attention_mask"]) + [0] * len(fake_ids)
        features[0]["labels"] = list(features[0]["labels"]) + [IGNORE_INDEX] * len(fake_ids)
        if "prompt_input_ids" in features[0]:
            features[0]["prompt_input_ids"] = list(features[0]["prompt_input_ids"]) + fake_ids
            features[0]["prompt_attention_mask"] = list(features[0]["prompt_attention_mask"]) + [0] * len(fake_ids)
        images_per_sample[0] = [fake_image]
        return True

    def pad_aux_sequences(
        self,
        features: list[dict[str, Any]],
        ids_key: str,
        mask_key: str,
        padding_side: str | None = None,
        return_tensors: Any = None,
    ) -> dict[str, torch.Tensor]:
        if not any(ids_key in feature for feature in features):
            return {}

        sequence_features = []
        for feature in features:
            input_ids = list(feature.get(ids_key) or [])
            attention_mask = list(feature.get(mask_key) or [1] * len(input_ids))
            sequence_features.append({"input_ids": input_ids, "attention_mask": attention_mask})

        original_padding_side = self.tokenizer.padding_side
        if padding_side is not None:
            self.tokenizer.padding_side = padding_side
        try:
            padded = self.tokenizer.pad(
                sequence_features,
                padding=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors or "pt",
            )
        finally:
            self.tokenizer.padding_side = original_padding_side
        return {ids_key: padded["input_ids"], mask_key: padded["attention_mask"]}

    def compute_rope_position_ids(self, features: dict[str, torch.Tensor], mm_inputs: dict[str, Any]) -> None:
        if self.get_rope_func is None:
            return

        kwargs = {
            "input_ids": features["input_ids"],
            "image_grid_thw": mm_inputs.get("image_grid_thw"),
            "attention_mask": (features["attention_mask"] >= 1).float(),
        }
        # Keep only arguments supported by the model-specific RoPE helper.
        params = inspect.signature(self.get_rope_func).parameters

        if "mm_token_type_ids" in params:
            # Mark image tokens for RoPE helpers that distinguish token types.
            image_token_id = getattr(self.model.config, "image_token_id", None)
            if image_token_id is None and self.template.mm_plugin.image_token is not None:
                image_token_id = self.tokenizer.convert_tokens_to_ids(self.template.mm_plugin.image_token)
            if image_token_id is not None:
                kwargs["mm_token_type_ids"] = torch.zeros_like(features["input_ids"])
                kwargs["mm_token_type_ids"][features["input_ids"] == image_token_id] = 1

        features["position_ids"], features["rope_deltas"] = self.get_rope_func(
            **{key: value for key, value in kwargs.items() if key in params}
        )

    def __call__(self, features: list[dict[str, Any]], return_tensors: Any = None) -> dict[str, Any]:
        text_keys = {"input_ids", "attention_mask", "labels", "position_ids", "token_type_ids"}
        raw_features, cleaned, images_per_sample = [], [], []

        for feature in features:
            feature = dict(feature)
            raw_images = feature.pop("images", None)
            raw_features.append(feature)

            if raw_images is None:
                images_per_sample.append([])
            elif isinstance(raw_images, (list, tuple)):
                images_per_sample.append([image for image in raw_images if image is not None])
            else:
                images_per_sample.append([raw_images])

        self.inject_dummy_image(raw_features, images_per_sample)
        reference_texts = [str(feature.get("reference_text", "")) for feature in raw_features]
        cleaned = [{key: value for key, value in feature.items() if key in text_keys} for feature in raw_features]
        batch_images = [image for images in images_per_sample for image in images]
        mm_inputs = self.template.mm_plugin.get_mm_inputs(batch_images, self.processor)

        padded = super().__call__(cleaned, return_tensors=return_tensors)
        padded.update(
            self.pad_aux_sequences(
                raw_features,
                "prompt_input_ids",
                "prompt_attention_mask",
                padding_side="left",
                return_tensors=return_tensors,
            )
        )
        padded["reference_text"] = reference_texts
        padded["vllm_images"] = [
            self.template.mm_plugin.regularize_images(images, self.processor)
            if self.processor is not None and images
            else []
            for images in images_per_sample
        ]
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        if self.get_rope_func is not None or model_type in MROPE_MODELS:
            self.compute_rope_position_ids(padded, mm_inputs)

        padded.update(mm_inputs)
        keep_float32_keys = {"pixel_values", "pixel_values_videos"}
        for key, value in list(padded.items()):
            if key not in keep_float32_keys and torch.is_tensor(value) and torch.is_floating_point(value):
                padded[key] = value.to(self.torch_dtype)
        return padded



class VLSFTDataModule(L.LightningDataModule):
    def __init__(
        self,
        template: Template,
        model_args: Any,
        data_args: Any,
        loader_args: Any,
        tokenizer: Any,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        super().__init__()
        self.template = template
        self.model_args = model_args
        self.data_args = data_args
        self.loader_args = loader_args
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.train_dataset = None
        self.data_collator = None

    def setup(self, stage: str | None = None) -> None:
        self.data_args.preprocessing_batch_size = int(self.data_args.preprocessing_batch_size)
        setup_dataset_map_workers(self.data_args)
        rank_zero_info(
            "Preprocessing uses batch_size=%s, num_workers=%s.",
            self.data_args.preprocessing_batch_size,
            self.data_args.preprocessing_num_workers,
        )

        self.train_dataset = DatasetBuilder(
            template=self.template,
            model_args=self.model_args,
            data_args=self.data_args,
            tokenizer=self.tokenizer,
            processor=self.processor,
            trainer=self.trainer,
        ).build()
        self.data_collator = VLCollator(
            template=self.template,
            model=self.model,
            tokenizer=self.tokenizer,
            processor=self.processor,
            # Pad to Tensor Core-friendly sequence lengths.
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id,
            torch_dtype=parse_torch_dtype(self.model_args.torch_dtype),
        )

    def train_dataloader(self) -> DataLoader:
        num_workers = self.loader_args.num_workers
        kwargs: dict[str, Any] = {
            "dataset": self.train_dataset,
            "batch_size": self.loader_args.per_device_train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": num_workers,
            "pin_memory": self.loader_args.pin_memory,
            "drop_last": self.loader_args.drop_last,
            "shuffle": self.loader_args.shuffle,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = self.loader_args.persistent_workers
            prefetch_factor = self.loader_args.prefetch_factor
            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = prefetch_factor

        return DataLoader(**kwargs)
