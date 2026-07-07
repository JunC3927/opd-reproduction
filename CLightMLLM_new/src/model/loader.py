from dataclasses import dataclass
import os
import sys
from typing import Any

import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor

from ..hparams import ModelArguments, TuningArguments, parse_torch_dtype

AUTO_MODEL_CLASSES = (
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
    "AutoModelForCausalLM",
)


def is_rank_zero_process() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def apply_verl_monkey_patch_if_requested(model: torch.nn.Module, model_args: ModelArguments) -> None:
    if not model_args.use_verl_monkey_patch:
        return

    if model_args.verl_repo_path and model_args.verl_repo_path not in sys.path:
        sys.path.insert(0, model_args.verl_repo_path)

    try:
        from verl.models.transformers.monkey_patch import apply_monkey_patch
    except Exception as exc:
        raise RuntimeError(
            "model.use_verl_monkey_patch=true requires verl to be importable. "
            "Set model.verl_repo_path or export PYTHONPATH=/path/to/verl_new:$PYTHONPATH."
        ) from exc

    apply_monkey_patch(
        model=model,
        ulysses_sp_size=model_args.verl_monkey_patch_ulysses_sp_size,
        use_remove_padding=model_args.verl_monkey_patch_use_remove_padding,
        use_fused_kernels=model_args.verl_monkey_patch_use_fused_kernels,
        fused_kernels_backend=model_args.verl_monkey_patch_fused_kernels_backend,
    )

    setattr(model, "_clight_verl_monkey_patched", True)
    if getattr(model, "config", None) is not None:
        setattr(model.config, "_clight_verl_monkey_patched", True)
    if is_rank_zero_process():
        print(
            "CLight applied verl monkey patch: "
            f"use_remove_padding={model_args.verl_monkey_patch_use_remove_padding}, "
            f"ulysses_sp_size={model_args.verl_monkey_patch_ulysses_sp_size}, "
            f"use_fused_kernels={model_args.verl_monkey_patch_use_fused_kernels}, "
            f"fused_kernels_backend={model_args.verl_monkey_patch_fused_kernels_backend}"
        )


def load_vision_language_model(model_args: ModelArguments, template_name: str):
    if not model_args.model_name_or_path:
        raise ValueError("model.model_name_or_path is required.")

    common_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "token": model_args.hf_hub_token,
        "local_files_only": model_args.local_files_only,
    }
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **common_kwargs)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("AutoProcessor must expose processor.tokenizer for SFT.")
    tokenizer.padding_side = model_args.padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {**common_kwargs, "torch_dtype": parse_torch_dtype(model_args.torch_dtype)}
    if model_args.device_map is not None:
        model_kwargs["device_map"] = model_args.device_map
    if model_args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = model_args.attn_implementation
    if model_args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    if model_args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    # Transformers moved VLM classes across releases; try compatible loaders.
    errors: list[str] = []
    for class_name in AUTO_MODEL_CLASSES:
        model_cls = getattr(transformers, class_name, None)
        if model_cls is not None:
            try:
                model = model_cls.from_pretrained(model_args.model_name_or_path, **model_kwargs)
                break
            except Exception as exc:
                errors.append(f"{class_name}: {exc}")
    else:
        raise RuntimeError("Could not load a vision-language model. Tried:\n" + "\n".join(errors))

    apply_verl_monkey_patch_if_requested(model, model_args)

    if model_args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": model_args.gradient_checkpointing_use_reentrant,
                }
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    if not model_args.use_cache:
        # Composite VLM configs can hide use_cache under nested configs.
        pending: list[Any] = [model]
        pending.extend(getattr(module, "config", None) for module in model.modules())
        seen: set[int] = set()
        while pending:
            obj = pending.pop()
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            if hasattr(obj, "use_cache"):
                obj.use_cache = False
            for attr in ("config", "generation_config", "text_config", "language_config"):
                pending.append(getattr(obj, attr, None))

    if template_name == "llava":
        # HF LLaVA processors may miss fields needed for image token expansion.
        model_config = getattr(model, "config", None)
        vision_config = getattr(model_config, "vision_config", None)
        processor.patch_size = getattr(processor, "patch_size", None) or getattr(vision_config, "patch_size", 14)
        processor.num_additional_image_tokens = getattr(processor, "num_additional_image_tokens", None) or 1
        processor.vision_feature_select_strategy = getattr(
            processor,
            "vision_feature_select_strategy",
            getattr(model_config, "vision_feature_select_strategy", "default"),
        )

    if model_args.image_min_pixels is not None:
        processor.image_min_pixels = int(model_args.image_min_pixels)
        if hasattr(processor, "image_processor"):
            processor.image_processor.min_pixels = int(model_args.image_min_pixels)
    if model_args.image_max_pixels is not None:
        processor.image_max_pixels = int(model_args.image_max_pixels)
        if hasattr(processor, "image_processor"):
            processor.image_processor.max_pixels = int(model_args.image_max_pixels)

    return model, processor, tokenizer


@dataclass
class ModelTuner:
    args: TuningArguments

    VISION_KEYWORDS = ("vision_tower", "vision_model", "visual")
    PROJECTOR_KEYWORDS = ("multi_modal_projector", "mm_projector", "projector")
    EXCLUDE_LORA_KEYWORDS = VISION_KEYWORDS + PROJECTOR_KEYWORDS + ("lm_head",)

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        freeze_keywords = ()
        if self.args.freeze_vision_tower:
            freeze_keywords += self.VISION_KEYWORDS
        if self.args.freeze_multi_modal_projector:
            freeze_keywords += self.PROJECTOR_KEYWORDS
        if freeze_keywords:
            for name, param in model.named_parameters():
                if any(keyword in name for keyword in freeze_keywords):
                    param.requires_grad = False

        lora = self.args.lora
        if lora.enable:
            if self.args.prepare_model_for_kbit_training:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(
                model,
                LoraConfig(
                    r=lora.r,
                    lora_alpha=lora.alpha,
                    lora_dropout=lora.dropout,
                    bias=lora.bias,
                    task_type=lora.task_type,
                    target_modules=self._lora_targets(lora.target_modules, model),
                ),
            )
            # Enable gradient flow for LoRA training with a frozen base model.
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

            model.print_trainable_parameters()
        return model

    @classmethod
    def _lora_targets(cls, target_modules: str | list[str], model: torch.nn.Module) -> list[str]:
        if isinstance(target_modules, str):
            if target_modules in {"all", "all-linear", "all_linear"}:
                return [
                    name
                    for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear)
                    and not any(keyword in name for keyword in cls.EXCLUDE_LORA_KEYWORDS)
                ]
            return [name.strip() for name in target_modules.split(",") if name.strip()]
        return list(target_modules)
