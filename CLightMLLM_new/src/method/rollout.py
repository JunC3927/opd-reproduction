import os
import inspect
import contextlib
from typing import Any

import torch
import torch.nn.functional as F
from transformers import GenerationConfig

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:  # pragma: no cover - FSDP can be unavailable in CPU-only installs.
    FSDP = None


class RolloutMixin:
    tokenizer: Any
    method_args: Any

    SEQUENCE_EXCLUDED_KEYS = {
        "input_ids",
        "attention_mask",
        "labels",
        "position_ids",
        "rope_deltas",
        "token_type_ids",
        "mm_token_type_ids",
        "vllm_images",
    }
    def prompt_width(self, batch: dict[str, Any]) -> int:
        """
        Return the padded prompt width used by generate().

        prompt_input_ids are left-padded to match verl's rollout layout, so
        completion tokens start after the full padded prompt, not after the
        per-sample number of non-pad tokens.
        """
        return int(batch["prompt_input_ids"].shape[1])

    def _build_mm_token_type_ids(
        self,
        batch: dict[str, Any],
        input_ids: torch.Tensor,
        prompt_width: int | None = None,
    ) -> torch.Tensor:
        """
        Qwen3-VL needs mm_token_type_ids when image_grid_thw/video_grid_thw exists.

        For OPD rollout:
        - prompt part keeps original mm token types
        - generated completion part is text, so type = 0
        """
        device = input_ids.device

        source = None

        if "prompt_mm_token_type_ids" in batch:
            source = batch["prompt_mm_token_type_ids"]
        elif "prompt_token_type_ids" in batch:
            source = batch["prompt_token_type_ids"]
        elif "mm_token_type_ids" in batch and batch["mm_token_type_ids"].shape == input_ids.shape:
            source = batch["mm_token_type_ids"]
        elif "token_type_ids" in batch and batch["token_type_ids"].shape == input_ids.shape:
            source = batch["token_type_ids"]

        if source is not None:
            source = source.to(device=device, dtype=input_ids.dtype)

            if source.shape[1] == input_ids.shape[1]:
                return source

            if prompt_width is None:
                prompt_width = min(source.shape[1], input_ids.shape[1])

            source = source[:, :prompt_width]

            if source.shape[1] < prompt_width:
                pad = torch.zeros(
                    source.shape[0],
                    prompt_width - source.shape[1],
                    dtype=source.dtype,
                    device=device,
                )
                source = torch.cat([source, pad], dim=1)

            if input_ids.shape[1] > prompt_width:
                completion_width = input_ids.shape[1] - prompt_width
                completion_types = torch.zeros(
                    input_ids.shape[0],
                    completion_width,
                    dtype=source.dtype,
                    device=device,
                )
                return torch.cat([source, completion_types], dim=1)

            return source[:, : input_ids.shape[1]]

        # Fallback:
        # text = 0, image = 1, video = 2
        # This works when model.config has image_token_id/video_token_id.
        mm_token_type_ids = torch.zeros_like(input_ids)

        config = getattr(self.model, "config", None)
        image_token_id = getattr(config, "image_token_id", None)
        video_token_id = getattr(config, "video_token_id", None)

        if image_token_id is not None:
            mm_token_type_ids = mm_token_type_ids.masked_fill(
                input_ids == image_token_id,
                1,
            )

        if video_token_id is not None:
            mm_token_type_ids = mm_token_type_ids.masked_fill(
                input_ids == video_token_id,
                2,
            )

        return mm_token_type_ids

    def _normalise_position_ids(self, position_ids: torch.Tensor) -> torch.Tensor:
        if position_ids.dim() == 3:
            if position_ids.shape[0] in (3, 4):
                return position_ids
            if position_ids.shape[1] in (3, 4):
                return position_ids.transpose(0, 1)
        if position_ids.dim() == 2:
            return position_ids.unsqueeze(0)
        raise ValueError(f"Unsupported position_ids shape for verl monkey patch: {tuple(position_ids.shape)}")

    def _text_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = attention_mask.bool()
        text_position_ids = torch.ones_like(attention_mask, dtype=torch.long)
        for row in range(attention_mask.shape[0]):
            valid_count = int(valid_mask[row].sum().item())
            if valid_count > 0:
                text_position_ids[row, valid_mask[row]] = torch.arange(
                    valid_count,
                    dtype=torch.long,
                    device=attention_mask.device,
                )
        return text_position_ids

    def _compute_vision_position_ids(
        self,
        batch: dict[str, Any],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        target_model = getattr(self.model, "model", None)
        compute_3d_position_ids = getattr(target_model, "compute_3d_position_ids", None)
        if compute_3d_position_ids is None:
            return None

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_grid_thw": batch.get("image_grid_thw"),
            "video_grid_thw": batch.get("video_grid_thw"),
            "mm_token_type_ids": self._build_mm_token_type_ids(batch, input_ids),
        }
        params = inspect.signature(compute_3d_position_ids).parameters
        if "inputs_embeds" in params:
            kwargs["inputs_embeds"] = self.model.get_input_embeddings()(input_ids)
        output = compute_3d_position_ids(**{key: value for key, value in kwargs.items() if key in params})
        if isinstance(output, tuple):
            output = output[0]
        if not torch.is_tensor(output):
            return None
        return self._normalise_position_ids(output.to(device=input_ids.device, dtype=torch.long))

    def _build_verl_position_ids(
        self,
        batch: dict[str, Any],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        source = self._compute_vision_position_ids(batch, input_ids, attention_mask)
        if source is None:
            source = batch.get("position_ids")
            if source is None:
                return None
            source = self._normalise_position_ids(source.to(device=input_ids.device, dtype=torch.long))
            seq_len = input_ids.shape[1]
            if source.shape[-1] < seq_len:
                pad_len = seq_len - source.shape[-1]
                last = source[..., -1:]
                delta = torch.arange(1, pad_len + 1, dtype=source.dtype, device=source.device)
                view_shape = (1,) * (source.dim() - 1) + (pad_len,)
                extension = last + delta.reshape(view_shape)
                source = torch.cat([source, extension.expand(*source.shape[:-1], pad_len)], dim=-1)
            source = source[..., :seq_len]

        if source.shape[-1] != input_ids.shape[1]:
            raise ValueError(
                "verl position_ids length mismatch: "
                f"position_ids={tuple(source.shape)}, input_ids={tuple(input_ids.shape)}"
            )

        if source.shape[0] == 4:
            return source
        if source.shape[0] == 3:
            text_position_ids = self._text_position_ids(attention_mask).unsqueeze(0)
            return torch.cat([text_position_ids, source], dim=0)
        return source

    def _is_verl_monkey_patched(self) -> bool:
        config = getattr(self.model, "config", None)
        return bool(
            getattr(self.model, "_clight_verl_monkey_patched", False)
            or getattr(config, "_clight_verl_monkey_patched", False)
        )

    def prompt_model_kwargs(self, batch: dict[str, Any]) -> dict[str, Any]:
        kwargs = {
            key: value
            for key, value in self.model_kwargs(batch, include_labels=False).items()
            if key not in self.SEQUENCE_EXCLUDED_KEYS
        }

        prompt_width = self.prompt_width(batch)

        input_ids = batch["prompt_input_ids"][:, :prompt_width]
        attention_mask = batch["prompt_attention_mask"][:, :prompt_width]

        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask

        is_verl_patched = self._is_verl_monkey_patched()

        if ("image_grid_thw" in kwargs or "video_grid_thw" in kwargs) and not is_verl_patched:
            kwargs["mm_token_type_ids"] = self._build_mm_token_type_ids(
                batch=batch,
                input_ids=input_ids,
                prompt_width=input_ids.shape[1],
            )

        if is_verl_patched:
            position_ids = self._build_verl_position_ids(batch, input_ids, attention_mask)
            if position_ids is not None:
                kwargs["position_ids"] = position_ids

        return kwargs

    def sequence_model_kwargs(
        self,
        batch: dict[str, Any],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, Any]:
        kwargs = {
            key: value
            for key, value in self.model_kwargs(batch, include_labels=False).items()
            if key not in self.SEQUENCE_EXCLUDED_KEYS
        }

        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        kwargs["use_cache"] = False

        is_verl_patched = self._is_verl_monkey_patched()

        if ("image_grid_thw" in kwargs or "video_grid_thw" in kwargs) and not is_verl_patched:
            kwargs["mm_token_type_ids"] = self._build_mm_token_type_ids(
                batch=batch,
                input_ids=input_ids,
                prompt_width=self.prompt_width(batch),
            )

        if is_verl_patched:
            position_ids = self._build_verl_position_ids(batch, input_ids, attention_mask)
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
        
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        if os.getenv("CLIGHT_OPD_MM_DEBUG") == "1" and rank == 0 and not hasattr(self, "_printed_mm_debug"):
            print("========== MM DEBUG ==========")
            print("batch keys:", list(batch.keys()))
            print("kwargs keys:", list(kwargs.keys()))

            if "mm_token_type_ids" in kwargs:
                print("input_ids shape:", kwargs["input_ids"].shape)
                print("mm_token_type_ids shape:", kwargs["mm_token_type_ids"].shape)
                print(
                    "unique mm_token_type_ids:",
                    torch.unique(kwargs["mm_token_type_ids"]).detach().cpu(),
                )

            print("image_token_id:", self._get_config_attr("image_token_id", None) if hasattr(self, "_get_config_attr") else getattr(getattr(self.model, "config", None), "image_token_id", None))
            print("video_token_id:", self._get_config_attr("video_token_id", None) if hasattr(self, "_get_config_attr") else getattr(getattr(self.model, "config", None), "video_token_id", None))
            print("==============================")
            self._printed_mm_debug = True

        return kwargs

    def generation_config(self) -> GenerationConfig:
        kwargs = {
            "max_new_tokens": self.method_args.rollout_max_new_tokens,
            "do_sample": self.method_args.rollout_do_sample,
            "temperature": self.method_args.rollout_temperature,
            "top_p": self.method_args.rollout_top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.method_args.rollout_top_k is not None:
            kwargs["top_k"] = self.method_args.rollout_top_k
        return GenerationConfig(**kwargs)

    def generate_rollout(self, batch: dict[str, Any]) -> torch.Tensor:
        if self.method_args.rollout_backend == "reference":
            return batch["input_ids"]

        student_rollout = getattr(self, "student_rollout", None)
        if student_rollout is not None:
            config = getattr(self.model, "config", None)
            return student_rollout.generate(
                batch=batch,
                method_args=self.method_args,
                image_token_id=getattr(config, "image_token_id", None),
                video_token_id=getattr(config, "video_token_id", None),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_inputs = self.prompt_model_kwargs(batch)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad(), self._summon_fsdp_for_generate():
            sequences = self.model.generate(
                **prompt_inputs,
                generation_config=self.generation_config(),
            )
        if was_training:
            self.model.train()
        return sequences

    def _summon_fsdp_for_generate(self):
        if FSDP is None:
            return contextlib.nullcontext()

        candidates = [self.model]
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            candidates.extend(
                candidate
                for candidate in (
                    getattr(trainer, "model", None),
                    getattr(trainer, "lightning_module", None),
                    getattr(getattr(trainer, "strategy", None), "model", None),
                )
                if candidate is not None
            )

        fsdp_modules = []
        seen: set[int] = set()
        for candidate in candidates:
            for module in candidate.modules():
                if isinstance(module, FSDP) and id(module) not in seen:
                    fsdp_modules.append(module)
                    seen.add(id(module))

        if not fsdp_modules:
            return contextlib.nullcontext()

        @contextlib.contextmanager
        def summon_all():
            with contextlib.ExitStack() as stack:
                for module in fsdp_modules:
                    # recurse=False follows PyTorch/verl guidance for generate with nested FSDP.
                    stack.enter_context(FSDP.summon_full_params(module, writeback=False, recurse=False))
                yield

        return summon_all()

    def completion_mask(self, sequences: torch.Tensor, prompt_width: int) -> torch.Tensor:
        completion_ids = sequences[:, prompt_width:]
        mask = completion_ids.ne(self.tokenizer.pad_token_id)

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            eos_seen = completion_ids.eq(eos_token_id).cumsum(dim=1).bool()
            before_or_at_first_eos = torch.cat(
                [torch.ones_like(mask[:, :1], dtype=torch.bool), ~eos_seen[:, :-1]],
                dim=1,
            )
            mask = mask & before_or_at_first_eos

        return mask

    def sequence_attention_mask(
        self,
        batch: dict[str, Any],
        sequences: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        prompt_width = sequences.shape[1] - completion_mask.shape[1]
        prompt_attention_mask = batch["prompt_attention_mask"][:, :prompt_width]

        return torch.cat(
            [prompt_attention_mask, completion_mask.to(prompt_attention_mask.dtype)],
            dim=1,
        )[:, : sequences.shape[1]]

    def sequence_token_logps(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        prompt_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = model(**self.sequence_model_kwargs(batch, sequences, attention_mask))
        token_logps = self.gather_token_logps(outputs.logits, sequences)
        shift_mask = self.shift_completion_mask(token_logps, completion_mask, prompt_width)
        return token_logps, shift_mask

    @staticmethod
    def gather_token_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1].float()
        shift_labels = labels[:, 1:]
        logps = F.log_softmax(shift_logits, dim=-1)
        return torch.gather(
            logps,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

    @staticmethod
    def shift_completion_mask(
        token_values: torch.Tensor,
        completion_mask: torch.Tensor,
        prompt_width: int,
    ) -> torch.Tensor:
        shift_mask = torch.zeros_like(token_values, dtype=torch.float32)
        start = max(prompt_width - 1, 0)
        available = max(token_values.shape[1] - start, 0)

        if available > 0:
            shift_mask[:, start:] = completion_mask[:, :available].float()

        return shift_mask
