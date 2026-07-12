from __future__ import annotations

from typing import Any

import torch

from .teacher_rpc import rpc_call


class RemoteStudentRollout:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout: float,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

    def ping(self) -> dict[str, Any]:
        response = rpc_call(self.host, self.port, {"op": "ping"}, self.timeout)
        return self._checked_response(response)

    @torch.no_grad()
    def generate(
        self,
        *,
        batch: dict[str, Any],
        method_args: Any,
        image_token_id: int | None,
        video_token_id: int | None,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, int]:
        device = batch["prompt_input_ids"].device
        request = {
            "op": "generate",
            "batch": self._cpu_batch(batch),
            "method_args": self._method_args_dict(method_args),
            "image_token_id": image_token_id,
            "video_token_id": video_token_id,
            "pad_token_id": int(pad_token_id),
        }
        response = self._checked_response(rpc_call(self.host, self.port, request, self.timeout))
        sequences = response["sequences"].to(device=device, dtype=batch["prompt_input_ids"].dtype)
        return sequences, int(response.get("weight_version", 0))

    def sync_state_dict(self, state_dict_path: str) -> dict[str, Any]:
        return self._checked_response(
            rpc_call(
                self.host,
                self.port,
                {
                    "op": "sync_state_dict",
                    "state_dict_path": state_dict_path,
                },
                self.timeout,
            )
        )

    def shutdown(self) -> dict[str, Any]:
        return self._checked_response(rpc_call(self.host, self.port, {"op": "shutdown"}, self.timeout))

    @staticmethod
    def _checked_response(response: Any) -> dict[str, Any]:
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected student server response type: {type(response)}")
        if response.get("ok") is not True:
            raise RuntimeError(response.get("error") or "Remote student rollout request failed.")
        return response

    @staticmethod
    def _cpu_batch(batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    @staticmethod
    def _method_args_dict(method_args: Any) -> dict[str, Any]:
        keys = (
            "rollout_max_new_tokens",
            "rollout_do_sample",
            "rollout_temperature",
            "rollout_top_p",
            "rollout_top_k",
        )
        return {key: getattr(method_args, key) for key in keys}
