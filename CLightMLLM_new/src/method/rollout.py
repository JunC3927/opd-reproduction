import inspect
import contextlib
from typing import Any

import torch
import torch.nn.functional as F

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

    def _model_accepts_kwarg(self, name: str) -> bool:
        for module in (self.model, getattr(self.model, "module", None)):
            if module is None:
                continue
            for method_name in ("forward", "prepare_inputs_for_generation"):
                method = getattr(module, method_name, None)
                if method is None:
                    continue
                try:
                    if name in inspect.signature(method).parameters:
                        return True
                except (TypeError, ValueError):
                    continue
        return False

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

        if (
            ("image_grid_thw" in kwargs or "video_grid_thw" in kwargs)
            and self._model_accepts_kwarg("mm_token_type_ids")
        ):
            kwargs["mm_token_type_ids"] = self._build_mm_token_type_ids(
                batch=batch,
                input_ids=input_ids,
                prompt_width=input_ids.shape[1],
            )

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

        if (
            ("image_grid_thw" in kwargs or "video_grid_thw" in kwargs)
            and self._model_accepts_kwarg("mm_token_type_ids")
        ):
            kwargs["mm_token_type_ids"] = self._build_mm_token_type_ids(
                batch=batch,
                input_ids=input_ids,
                prompt_width=self.prompt_width(batch),
            )

        return kwargs

    def generation_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "max_new_tokens": self.method_args.rollout_max_new_tokens,
            "do_sample": self.method_args.rollout_do_sample,
            "temperature": self.method_args.rollout_temperature,
            "top_p": self.method_args.rollout_top_p,
            "top_k": self.method_args.rollout_top_k if self.method_args.rollout_top_k is not None else 0,
            "use_cache": self.method_args.rollout_use_cache,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        return kwargs

    def generate_rollout(self, batch: dict[str, Any]) -> torch.Tensor:
        student_rollout = getattr(self, "student_rollout", None)
        if student_rollout is not None:
            config = getattr(self.model, "config", None)
            rollout_result = student_rollout.generate(
                batch=batch,
                method_args=self.method_args,
                image_token_id=getattr(config, "image_token_id", None),
                video_token_id=getattr(config, "video_token_id", None),
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if isinstance(rollout_result, tuple):
                sequences, weight_version = rollout_result
                setattr(self, "_last_student_rollout_weight_version", int(weight_version))
                return sequences
            return rollout_result

        prompt_inputs = self.prompt_model_kwargs(batch)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad(), self._summon_fsdp_for_generate():
            sequences = self.model.generate(
                **prompt_inputs,
                **self.generation_kwargs(),
            )
        if was_training:
            self.model.train()
        return sequences

    def _summon_fsdp_for_generate(self):
        if FSDP is None:
            return contextlib.nullcontext()

        trainer = getattr(self, "trainer", None)
        strategy = getattr(trainer, "strategy", None)
        if strategy is not None and "FSDP" in type(strategy).__name__:
            # In Lightning FSDP, training_step runs inside the wrapper forward.
            # Calling summon_full_params there raises "Cannot manually unshard
            # parameters during forward/backward"; regular FSDP forwards during
            # generate can unshard their own shards on demand.
            return contextlib.nullcontext()

        candidates = [self.model]
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
