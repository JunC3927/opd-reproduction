from typing import Any

import pickle
import socket
import struct
import torch

try:
    from .teacher_rpc import rpc_call
except ModuleNotFoundError:
    HEADER = struct.Struct("!Q")

    def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
        chunks = []
        remaining = nbytes
        while remaining:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ConnectionError("Socket closed while receiving RPC payload.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _send_message(sock: socket.socket, message: Any) -> None:
        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        sock.sendall(HEADER.pack(len(payload)))
        sock.sendall(payload)

    def _recv_message(sock: socket.socket) -> Any:
        header = _recv_exact(sock, HEADER.size)
        (payload_size,) = HEADER.unpack(header)
        payload = _recv_exact(sock, payload_size)
        return pickle.loads(payload)

    def rpc_call(host: str, port: int, message: Any, timeout: float) -> Any:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            _send_message(sock, message)
            return _recv_message(sock)


class RemoteTeacherScorer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout: float,
        topk: int,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.topk = int(topk)

    @torch.no_grad()
    def score(
        self,
        *,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        images_per_sample: list[list[Any]] | None,
        image_token_id: int | None,
        video_token_id: int | None,
        pad_token_id: int,
        model_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs_per_sample: list[dict[str, Any] | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = sequences.device
        cpu_model_kwargs = None
        if model_kwargs is not None:
            cpu_model_kwargs = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in model_kwargs.items()
            }
        request = {
            "op": "score",
            "sequences": sequences.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "images_per_sample": images_per_sample,
            "image_token_id": image_token_id,
            "video_token_id": video_token_id,
            "pad_token_id": int(pad_token_id),
            "topk": self.topk,
            "model_kwargs": cpu_model_kwargs,
            "mm_processor_kwargs_per_sample": mm_processor_kwargs_per_sample,
        }
        response = rpc_call(self.host, self.port, request, self.timeout)
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected teacher server response type: {type(response)}")
        if response.get("ok") is not True:
            raise RuntimeError(response.get("error") or "Remote teacher scoring failed.")

        logps = response["teacher_topk_logps"].to(device=device, dtype=torch.float32)
        ids = response["teacher_topk_ids"].to(device=device, dtype=torch.long)
        return logps, ids


RemoteVLLMTeacherScorer = RemoteTeacherScorer
