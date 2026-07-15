#!/usr/bin/env python3
"""Shared vLLM student rollout server for CLight OPD training.

This server is intentionally separate from torchrun/Lightning. It owns one
student vLLM engine, serves rollout generation requests, and can hot-load
student weights through the existing bucketed IPC path.
"""

from __future__ import annotations

import argparse
import gc
import os
import socketserver
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method.rpc import recv_message, send_message  # noqa: E402
from src.method.vllm_student import VLLMStudentRollout  # noqa: E402
from src.model import load_processor_and_tokenizer  # noqa: E402
from src.hparams import ModelArguments, parse_torch_dtype  # noqa: E402


TORCHRUN_ENV_KEYS = {
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
}


def scrub_torchrun_env() -> None:
    for key in TORCHRUN_ENV_KEYS:
        os.environ.pop(key, None)


class StudentState:
    def __init__(
        self,
        rollout: VLLMStudentRollout,
        *,
        bucket_size_mb: int,
        use_shm: bool,
        ipc_timeout_sec: float,
        sync_dtype: torch.dtype | None,
    ) -> None:
        self.rollout = rollout
        self.bucket_size_mb = int(bucket_size_mb)
        self.use_shm = bool(use_shm)
        self.ipc_timeout_sec = float(ipc_timeout_sec)
        self.sync_dtype = sync_dtype
        self.weight_version = 0
        self.lock = threading.Lock()
        self.active_sync_session: dict[str, Any] | None = None


class StudentTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass, state: StudentState):
        super().__init__(server_address, RequestHandlerClass)
        self.state = state


class StudentHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        try:
            request = recv_message(self.request)
            op = request.get("op") if isinstance(request, dict) else None
            server = self.server
            assert isinstance(server, StudentTCPServer)

            if op == "ping":
                send_message(
                    self.request,
                    {
                        "ok": True,
                        "message": "pong",
                        "weight_version": server.state.weight_version,
                        "active_sync_session": server.state.active_sync_session is not None,
                    },
                )
                return

            if op == "shutdown":
                send_message(self.request, {"ok": True, "message": "shutdown"})
                threading.Thread(target=server.shutdown, daemon=True).start()
                return

            if op not in {"generate", "start_weight_sync", "finish_weight_sync", "abort_weight_sync"}:
                send_message(self.request, {"ok": False, "error": f"Unsupported op: {op!r}"})
                return

            with server.state.lock:
                if op == "generate":
                    if server.state.active_sync_session is not None:
                        raise RuntimeError("Cannot generate while a remote weight sync session is active.")
                    method_args = SimpleNamespace(**request["method_args"])
                    sequences = server.state.rollout.generate(
                        batch=request["batch"],
                        method_args=method_args,
                        image_token_id=request.get("image_token_id"),
                        video_token_id=request.get("video_token_id"),
                        pad_token_id=int(request["pad_token_id"]),
                    )
                    send_message(
                        self.request,
                        {
                            "ok": True,
                            "sequences": sequences.detach().cpu(),
                            "weight_version": server.state.weight_version,
                        },
                    )
                    return

                if op == "start_weight_sync":
                    if server.state.active_sync_session is not None:
                        raise RuntimeError("A remote weight sync session is already active.")
                    session_id = uuid.uuid4().hex
                    requested_use_shm = request.get("use_shm")
                    use_shm = server.state.use_shm if requested_use_shm is None else bool(requested_use_shm)
                    zmq_handle = (
                        request.get("zmq_handle")
                        or f"ipc:///tmp/clight-student-vllm-remote-{os.getpid()}-{session_id}.sock"
                    )
                    session = server.state.rollout.start_weight_sync_receiver(
                        zmq_handle=zmq_handle,
                        use_shm=use_shm,
                    )
                    session["session_id"] = session_id
                    session["requested_use_shm"] = requested_use_shm
                    server.state.active_sync_session = session
                    time.sleep(0.5)
                    if session["result_box"].get("ok") is False:
                        server.state.active_sync_session = None
                        raise RuntimeError(
                            "Student vLLM remote IPC receiver failed before sender started: "
                            f"{session['result_box'].get('error')}"
                        )
                    send_message(
                        self.request,
                        {
                            "ok": True,
                            "session_id": session_id,
                            "zmq_handle": zmq_handle,
                            "use_shm": use_shm,
                            "server_default_use_shm": server.state.use_shm,
                            "weight_version": server.state.weight_version,
                        },
                    )
                    return

                if op == "abort_weight_sync":
                    session = server.state.active_sync_session
                    if session is None:
                        send_message(
                            self.request,
                            {
                                "ok": True,
                                "message": "no active remote weight sync session",
                                "weight_version": server.state.weight_version,
                            },
                        )
                        return
                    if request.get("session_id") != session.get("session_id"):
                        raise RuntimeError(
                            f"Remote weight sync session mismatch: got {request.get('session_id')!r}, "
                            f"expected {session.get('session_id')!r}."
                        )
                    timeout_sec = request.get("timeout_sec")
                    if timeout_sec is None:
                        timeout_sec = min(server.state.ipc_timeout_sec, 30.0)
                    timeout_sec = max(0.0, min(float(timeout_sec), server.state.ipc_timeout_sec))
                    thread = session["thread"]
                    thread.join(timeout=timeout_sec)
                    if thread.is_alive():
                        raise TimeoutError(
                            "Student vLLM remote IPC receiver is still active after abort "
                            f"timeout={timeout_sec}s. Restart the student server before continuing."
                        )
                    result_box = dict(session["result_box"])
                    server.state.active_sync_session = None
                    summary = {
                        "path": f"{session['owner_name']}.apply_model_remote_ipc",
                        "receiver_result": result_box.get("result"),
                        "receiver_ok": result_box.get("ok"),
                        "receiver_error": result_box.get("error"),
                        "reason": request.get("reason"),
                        "wait_sec": time.time() - session["started_at"],
                    }
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    send_message(
                        self.request,
                        {
                            "ok": True,
                            "message": "remote weight sync session aborted",
                            "weight_version": server.state.weight_version,
                            "summary": summary,
                        },
                    )
                    return

                if op == "finish_weight_sync":
                    session = server.state.active_sync_session
                    if session is None:
                        raise RuntimeError("No remote weight sync session is active.")
                    if request.get("session_id") != session.get("session_id"):
                        raise RuntimeError(
                            f"Remote weight sync session mismatch: got {request.get('session_id')!r}, "
                            f"expected {session.get('session_id')!r}."
                        )
                    start = time.time()
                    thread = session["thread"]
                    thread.join(timeout=server.state.ipc_timeout_sec)
                    if thread.is_alive():
                        raise TimeoutError(
                            f"Student vLLM remote IPC receiver timed out after {server.state.ipc_timeout_sec}s."
                        )
                    result_box = session["result_box"]
                    if not result_box.get("ok"):
                        server.state.active_sync_session = None
                        raise RuntimeError(f"Student vLLM remote IPC receiver failed: {result_box.get('error')}")
                    server.state.active_sync_session = None
                    server.state.weight_version += 1
                    summary = {
                        "path": f"{session['owner_name']}.apply_model_remote_ipc",
                        "receiver_result": result_box.get("result"),
                        "receiver_wait_sec": time.time() - session["started_at"],
                        "finish_sec": time.time() - start,
                        "sender_summary": request.get("sender_summary"),
                    }
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    send_message(
                        self.request,
                        {
                            "ok": True,
                            "weight_version": server.state.weight_version,
                            "summary": summary,
                        },
                    )
                    return

        except Exception:
            print("[student request] failed:\n" + traceback.format_exc(), flush=True)
            send_message(self.request, {"ok": False, "error": traceback.format_exc()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared vLLM student rollout server.")
    parser.add_argument("--model", required=True, help="Student model path.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29588)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--max-model-len", type=int, default=1536)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--load-format", default=None)
    parser.add_argument("--distributed-executor-backend", default=None)
    parser.add_argument("--enable-chunked-prefill", dest="enable_chunked_prefill", action="store_true")
    parser.add_argument("--disable-chunked-prefill", dest="enable_chunked_prefill", action="store_false")
    parser.add_argument("--enable-prefix-caching", dest="enable_prefix_caching", action="store_true")
    parser.add_argument("--disable-prefix-caching", dest="enable_prefix_caching", action="store_false")
    parser.add_argument("--disable-log-stats", dest="disable_log_stats", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--image-min-pixels", type=int, default=None)
    parser.add_argument("--image-max-pixels", type=int, default=None)
    parser.add_argument("--ipc-bucket-size-mb", type=int, default=512)
    parser.add_argument("--ipc-use-shm", action="store_true")
    parser.add_argument("--ipc-timeout-sec", type=float, default=900.0)
    parser.add_argument("--sync-dtype", default="bfloat16")
    parser.set_defaults(
        enable_chunked_prefill=None,
        enable_prefix_caching=None,
        disable_log_stats=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    scrub_torchrun_env()

    print(
        "Starting CLight shared vLLM student:",
        f"host={args.host}",
        f"port={args.port}",
        f"model={args.model}",
        f"trust_remote_code={args.trust_remote_code}",
        f"tp={args.tensor_parallel_size}",
        f"dtype={args.torch_dtype}",
        f"max_model_len={args.max_model_len}",
        f"max_num_batched_tokens={args.max_num_batched_tokens}",
        f"max_num_seqs={args.max_num_seqs}",
        f"enable_chunked_prefill={args.enable_chunked_prefill}",
        f"enable_prefix_caching={args.enable_prefix_caching}",
        f"image_min_pixels={args.image_min_pixels}",
        f"image_max_pixels={args.image_max_pixels}",
        f"ipc_bucket_size_mb={args.ipc_bucket_size_mb}",
        f"sync_dtype={args.sync_dtype}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    processor_args = ModelArguments(
        model_name_or_path=args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
    )
    _processor, tokenizer, _common_kwargs = load_processor_and_tokenizer(processor_args)
    sync_dtype = None if args.sync_dtype.lower() == "none" else parse_torch_dtype(args.sync_dtype)
    limit_mm_per_prompt = None
    if args.limit_images is not None:
        limit_mm_per_prompt = {"image": int(args.limit_images), "video": 0}

    print("[student server] vLLM init start", flush=True)
    start = time.time()
    rollout = VLLMStudentRollout(
        model_path=args.model,
        tokenizer=tokenizer,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        load_format=args.load_format,
        distributed_executor_backend=args.distributed_executor_backend,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=args.enable_prefix_caching,
        disable_log_stats=args.disable_log_stats,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        device=args.device,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )
    print(f"[student server] vLLM init done seconds={time.time() - start:.3f}", flush=True)

    state = StudentState(
        rollout,
        bucket_size_mb=args.ipc_bucket_size_mb,
        use_shm=args.ipc_use_shm,
        ipc_timeout_sec=args.ipc_timeout_sec,
        sync_dtype=sync_dtype,
    )
    with StudentTCPServer((args.host, args.port), StudentHandler, state) as server:
        print(f"CLight shared vLLM student listening on {args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
