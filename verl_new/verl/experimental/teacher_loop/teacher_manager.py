# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient


def _teacher_request_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _teacher_request_to_cpu(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return tuple(_teacher_request_to_cpu(val) for val in value)
    if isinstance(value, list):
        return [_teacher_request_to_cpu(val) for val in value]
    if hasattr(value, "items"):
        try:
            return {key: _teacher_request_to_cpu(val) for key, val in value.items()}
        except Exception:
            return repr(value)
    return value


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if teacher_model_config.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return {
        "max_tokens": 1,
        "temperature": teacher_model_config.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client
        self._teacher_request_dump_count = 0

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    def _maybe_dump_teacher_request(
        self,
        *,
        request_id: str,
        teacher_key: str,
        sequence_ids: list[int],
        sampling_params: dict[str, Any],
        multi_modal_data: dict[str, Any],
        mm_processor_kwargs: Optional[dict[str, Any]],
    ) -> None:
        dump_dir = os.getenv("VERL_OPD_TEACHER_REQUEST_DUMP_DIR")
        if not dump_dir:
            return

        limit = int(os.getenv("VERL_OPD_TEACHER_REQUEST_DUMP_LIMIT", "0") or "0")
        if limit <= 0 or self._teacher_request_dump_count >= limit:
            return

        dump_index = self._teacher_request_dump_count
        self._teacher_request_dump_count += 1
        os.makedirs(dump_dir, exist_ok=True)
        payload = {
            "format": "verl_teacher_vllm_request_v1",
            "request_index": dump_index,
            "request_id": request_id,
            "pid": os.getpid(),
            "teacher_key": teacher_key,
            "sequence_ids": [int(token_id) for token_id in sequence_ids],
            "sequence_len": len(sequence_ids),
            "sampling_params": dict(sampling_params),
            "multi_modal_data": _teacher_request_to_cpu(multi_modal_data),
            "mm_processor_kwargs": _teacher_request_to_cpu(mm_processor_kwargs),
            "image_count": len(multi_modal_data.get("images") or []),
            "video_count": len(multi_modal_data.get("videos") or []),
            "audio_count": len(multi_modal_data.get("audios") or []),
        }
        dump_path = os.path.join(dump_dir, f"teacher_request_pid{os.getpid()}_idx{dump_index:05d}.pt")
        torch.save(payload, dump_path)
        print(
            f"Saved VERL teacher vLLM request dump to {dump_path}; "
            f"request_index={dump_index}; sequence_len={len(sequence_ids)}; "
            f"image_count={payload['image_count']}; mm_processor_kwargs={mm_processor_kwargs is not None}",
            flush=True,
        )

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        request_id = uuid4().hex
        sampling_params = _get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config)
        self._maybe_dump_teacher_request(
            request_id=request_id,
            teacher_key=teacher_key,
            sequence_ids=sequence_ids,
            sampling_params=sampling_params,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        teacher_output = await client.generate(
            request_id=request_id,
            prompt_ids=sequence_ids,
            sampling_params=sampling_params,
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
        return teacher_ids, teacher_logprobs
