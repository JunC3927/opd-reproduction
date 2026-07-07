import argparse
import os
import pickle
import socketserver
import struct
import sys
import threading
import traceback
from typing import Any

import torch
import torch.nn.functional as F


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.hparams import ModelArguments  # noqa: E402
from src.model import load_vision_language_model  # noqa: E402

try:
    from src.method.teacher_rpc import recv_message, send_message  # noqa: E402
except ModuleNotFoundError:
    HEADER = struct.Struct("!Q")

    def _recv_exact(sock, nbytes: int) -> bytes:
        chunks = []
        remaining = nbytes
        while remaining:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ConnectionError("Socket closed while receiving RPC payload.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def send_message(sock, message: Any) -> None:
        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        sock.sendall(HEADER.pack(len(payload)))
        sock.sendall(payload)

    def recv_message(sock) -> Any:
        header = _recv_exact(sock, HEADER.size)
        (payload_size,) = HEADER.unpack(header)
        payload = _recv_exact(sock, payload_size)
        return pickle.loads(payload)


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


class HFTeacherScorer:
    def __init__(
        self,
        *,
        model_path: str,
        topk: int,
        torch_dtype: str,
        template: str,
        device: str,
        device_map: str | None,
        trust_remote_code: bool,
        local_files_only: bool,
        attn_implementation: str | None,
        image_min_pixels: int | None,
        image_max_pixels: int | None,
        temperature: float,
    ) -> None:
        self.topk = int(topk)
        self.temperature = float(temperature)
        model_args = ModelArguments(
            model_name_or_path=model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=local_files_only,
            attn_implementation=attn_implementation,
            gradient_checkpointing=False,
            use_cache=False,
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
        )
        model, _processor, _tokenizer = load_vision_language_model(model_args, template)
        if device_map is None:
            model.to(torch.device(device))
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.model = model

    @torch.no_grad()
    def score(self, request: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        model_kwargs = request.get("model_kwargs")
        if not isinstance(model_kwargs, dict):
            raise ValueError("HF teacher server requires request['model_kwargs'].")

        device = first_parameter_device(self.model)
        kwargs = move_to_device(model_kwargs, device)
        if "use_cache" not in kwargs:
            kwargs["use_cache"] = False

        topk = int(request.get("topk") or self.topk)
        outputs = self.model(**kwargs)
        logits = outputs.logits[:, :-1].float() / self.temperature
        logps = F.log_softmax(logits, dim=-1)
        topk_logps, topk_ids = torch.topk(logps, k=topk, dim=-1)
        return topk_logps.cpu(), topk_ids.cpu()


class TeacherState:
    def __init__(self, scorer: HFTeacherScorer) -> None:
        self.scorer = scorer
        self.lock = threading.Lock()


class TeacherTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass, state: TeacherState):
        super().__init__(server_address, RequestHandlerClass)
        self.state = state


class TeacherHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        try:
            request = recv_message(self.request)
            op = request.get("op") if isinstance(request, dict) else None
            if op == "ping":
                send_message(self.request, {"ok": True, "message": "pong"})
                return
            if op != "score":
                send_message(self.request, {"ok": False, "error": f"Unsupported op: {op!r}"})
                return

            server = self.server
            assert isinstance(server, TeacherTCPServer)
            with server.state.lock:
                logps, ids = server.state.scorer.score(request)
            send_message(
                self.request,
                {
                    "ok": True,
                    "teacher_topk_logps": logps,
                    "teacher_topk_ids": ids,
                },
            )
        except Exception:
            send_message(self.request, {"ok": False, "error": traceback.format_exc()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared HF teacher scorer for CLight OPD DDP training.")
    parser.add_argument("--model", required=True, help="Teacher model path.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29578)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--template", default="qwen3_vl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--image-min-pixels", type=int, default=None)
    parser.add_argument("--image-max-pixels", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "Starting CLight shared HF teacher:",
        f"host={args.host}",
        f"port={args.port}",
        f"model={args.model}",
        f"dtype={args.torch_dtype}",
        f"device={args.device}",
        f"device_map={args.device_map}",
        f"attn_implementation={args.attn_implementation}",
        f"image_min_pixels={args.image_min_pixels}",
        f"image_max_pixels={args.image_max_pixels}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    scorer = HFTeacherScorer(
        model_path=args.model,
        topk=args.topk,
        torch_dtype=args.torch_dtype,
        template=args.template,
        device=args.device,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        temperature=args.temperature,
    )
    state = TeacherState(scorer)
    with TeacherTCPServer((args.host, args.port), TeacherHandler, state) as server:
        print(f"CLight shared HF teacher listening on {args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
