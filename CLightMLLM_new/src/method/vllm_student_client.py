from __future__ import annotations

import os
import time
from typing import Any

import torch

from .vllm_student import send_weight_items_ipc
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
        self._rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

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
        started = time.time()
        print(
            f"[student-vllm-client rank={self._rank}] generate start: "
            f"server={self.host}:{self.port}, batch={int(batch['prompt_input_ids'].shape[0])}",
            flush=True,
        )
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
        weight_version = int(response.get("weight_version", 0))
        print(
            f"[student-vllm-client rank={self._rank}] generate done: "
            f"seconds={time.time() - started:.3f}, shape={tuple(sequences.shape)}, "
            f"weight_version={weight_version}",
            flush=True,
        )
        return sequences, weight_version

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

    def fingerprint_weight(self, name: str, *, numel: int = 256) -> dict[str, Any]:
        return self._checked_response(
            rpc_call(
                self.host,
                self.port,
                {
                    "op": "fingerprint_weight",
                    "name": name,
                    "numel": int(numel),
                },
                self.timeout,
            )
        )

    def sync_weight_items_ipc(
        self,
        weights: Any,
        *,
        bucket_size_mb: int,
        use_shm: bool | None = None,
        device: str | torch.device | None,
        sync_dtype: torch.dtype | None,
        zmq_handle: str | None = None,
    ) -> dict[str, Any]:
        start_response = self._checked_response(
            rpc_call(
                self.host,
                self.port,
                {
                    "op": "start_weight_sync",
                    "zmq_handle": zmq_handle,
                    "use_shm": use_shm,
                },
                self.timeout,
            )
        )
        resolved_use_shm = bool(start_response.get("use_shm", False))
        sender_summary = send_weight_items_ipc(
            weights,
            zmq_handle=start_response["zmq_handle"],
            bucket_size_mb=bucket_size_mb,
            use_shm=resolved_use_shm,
            device=device,
            sync_dtype=sync_dtype,
        )
        sender_summary["requested_use_shm"] = use_shm
        sender_summary["resolved_use_shm"] = resolved_use_shm
        finish_response = self._checked_response(
            rpc_call(
                self.host,
                self.port,
                {
                    "op": "finish_weight_sync",
                    "session_id": start_response["session_id"],
                    "sender_summary": sender_summary,
                },
                self.timeout,
            )
        )
        finish_response["sender_summary"] = sender_summary
        return finish_response

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
