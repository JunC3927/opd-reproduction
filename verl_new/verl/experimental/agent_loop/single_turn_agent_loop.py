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
import logging
import os
from typing import Any
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _opd_rollout_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _opd_rollout_to_cpu(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return tuple(_opd_rollout_to_cpu(val) for val in value)
    if isinstance(value, list):
        return [_opd_rollout_to_cpu(val) for val in value]
    if hasattr(value, "items"):
        try:
            return {key: _opd_rollout_to_cpu(val) for key, val in value.items()}
        except Exception:
            return repr(value)
    return value


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self._opd_student_rollout_dump_count = 0

    def _maybe_dump_opd_student_rollout(
        self,
        *,
        request_id: str,
        messages: list[dict[str, Any]],
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        multi_modal_data: dict[str, Any],
        mm_processor_kwargs: dict[str, Any] | None,
        output: TokenOutput,
    ) -> None:
        dump_dir = os.getenv("VERL_OPD_STUDENT_ROLLOUT_DUMP_DIR")
        if not dump_dir:
            return

        limit = int(os.getenv("VERL_OPD_STUDENT_ROLLOUT_DUMP_LIMIT", "0") or "0")
        if limit <= 0 or self._opd_student_rollout_dump_count >= limit:
            return

        dump_index = self._opd_student_rollout_dump_count
        self._opd_student_rollout_dump_count += 1
        os.makedirs(dump_dir, exist_ok=True)

        payload = {
            "format": "verl_student_rollout_request_output_v1",
            "dump_index": dump_index,
            "request_id": request_id,
            "pid": os.getpid(),
            "prompt_length_config": self.prompt_length,
            "response_length_config": self.response_length,
            "raw_prompt": _opd_rollout_to_cpu(messages),
            "prompt_ids": [int(token_id) for token_id in prompt_ids],
            "prompt_len": len(prompt_ids),
            "sampling_params": dict(sampling_params),
            "multi_modal_data": _opd_rollout_to_cpu(multi_modal_data),
            "mm_processor_kwargs": _opd_rollout_to_cpu(mm_processor_kwargs),
            "image_count": len(multi_modal_data.get("images") or []),
            "video_count": len(multi_modal_data.get("videos") or []),
            "audio_count": len(multi_modal_data.get("audios") or []),
            "output": {
                "token_ids": [int(token_id) for token_id in output.token_ids],
                "truncated_response_ids": [int(token_id) for token_id in output.token_ids[: self.response_length]],
                "response_len": len(output.token_ids[: self.response_length]),
                "stop_reason": output.stop_reason,
                "num_preempted": output.num_preempted,
                "log_probs": _opd_rollout_to_cpu(output.log_probs),
                "extra_fields": _opd_rollout_to_cpu(output.extra_fields),
            },
        }
        dump_path = os.path.join(dump_dir, f"student_rollout_pid{os.getpid()}_idx{dump_index:05d}.pt")
        torch.save(payload, dump_path)
        print(
            f"Saved VERL student rollout dump to {dump_path}; "
            f"dump_index={dump_index}; prompt_len={payload['prompt_len']}; "
            f"response_len={payload['output']['response_len']}; image_count={payload['image_count']}",
            flush=True,
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        # 3. generate sequences
        metrics = {}
        request_id = uuid4().hex
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
                audio_data=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )
        self._maybe_dump_opd_student_rollout(
            request_id=request_id,
            messages=messages,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            output=output,
        )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_mask = [1] * len(output.token_ids)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
