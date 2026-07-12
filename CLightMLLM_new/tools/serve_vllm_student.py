#!/usr/bin/env python3
"""Shared vLLM student rollout server for CLight OPD probes.

This server is intentionally separate from torchrun/Lightning. It owns one
student vLLM engine, serves rollout generation requests, and can hot-load
exported student weights through the existing bucketed IPC path.
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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method.teacher_rpc import recv_message, send_message  # noqa: E402
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


def tensor_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def image_count(images_per_sample: Any) -> int:
    if images_per_sample is None:
        return 0
    return sum(len(images) for images in images_per_sample)


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_exported_state(path: str) -> tuple[list[tuple[str, torch.Tensor]], dict[str, Any]]:
    state_path = resolve_path(path)
    print(f"[student server] state load start path={state_path}", flush=True)
    start = time.time()
    try:
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(state_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    elif isinstance(payload, dict):
        state_dict = payload
        metadata = {"format": "raw_state_dict"}
    else:
        raise TypeError(f"Unexpected exported state payload type: {type(payload)}")
    weights = [(name, tensor.detach()) for name, tensor in state_dict.items() if torch.is_tensor(tensor)]
    total_numel = sum(tensor.numel() for _, tensor in weights)
    print(
        "[student server] state load done "
        f"seconds={time.time() - start:.3f} tensors={len(weights)} total_numel={total_numel:,} "
        f"metadata={metadata}",
        flush=True,
    )
    return weights, metadata


class StudentState:
    def __init__(
        self,
        rollout: VLLMStudentRollout,
        *,
        log_requests: bool,
        bucket_size_mb: int,
        use_shm: bool,
        ipc_timeout_sec: float,
        sync_dtype: torch.dtype | None,
    ) -> None:
        self.rollout = rollout
        self.log_requests = log_requests
        self.bucket_size_mb = int(bucket_size_mb)
        self.use_shm = bool(use_shm)
        self.ipc_timeout_sec = float(ipc_timeout_sec)
        self.sync_dtype = sync_dtype
        self.weight_version = 0
        self.lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.request_count = 0

    def next_request_id(self) -> int:
        with self.request_lock:
            self.request_count += 1
            return self.request_count


class StudentTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

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
            request_id = server.state.next_request_id()

            if op == "ping":
                send_message(
                    self.request,
                    {
                        "ok": True,
                        "message": "pong",
                        "weight_version": server.state.weight_version,
                    },
                )
                return

            if op == "shutdown":
                send_message(self.request, {"ok": True, "message": "shutdown"})
                threading.Thread(target=server.shutdown, daemon=True).start()
                return

            if op not in {"generate", "sync_state_dict"}:
                send_message(self.request, {"ok": False, "error": f"Unsupported op: {op!r}"})
                return

            if server.state.log_requests:
                if op == "generate":
                    batch = request.get("batch") or {}
                    print(
                        f"[student request {request_id}] received op=generate "
                        f"prompt={tensor_shape(batch.get('prompt_input_ids'))} "
                        f"images={image_count(batch.get('vllm_images'))} "
                        f"weight_version={server.state.weight_version}",
                        flush=True,
                    )
                else:
                    print(
                        f"[student request {request_id}] received op=sync_state_dict "
                        f"path={request.get('state_dict_path')}",
                        flush=True,
                    )

            lock_wait_start = time.time()
            with server.state.lock:
                if server.state.log_requests:
                    print(
                        f"[student request {request_id}] op={op} lock_acquired "
                        f"wait_sec={time.time() - lock_wait_start:.3f}",
                        flush=True,
                    )

                if op == "generate":
                    method_args = SimpleNamespace(**request["method_args"])
                    start = time.time()
                    sequences = server.state.rollout.generate(
                        batch=request["batch"],
                        method_args=method_args,
                        image_token_id=request.get("image_token_id"),
                        video_token_id=request.get("video_token_id"),
                        pad_token_id=int(request["pad_token_id"]),
                    )
                    if server.state.log_requests:
                        print(
                            f"[student request {request_id}] generate_done "
                            f"sequences={tensor_shape(sequences)} seconds={time.time() - start:.3f} "
                            f"weight_version={server.state.weight_version}",
                            flush=True,
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

                weights, metadata = load_exported_state(str(request["state_dict_path"]))
                start = time.time()
                summary = server.state.rollout.sync_from_weight_items_ipc(
                    weights,
                    bucket_size_mb=server.state.bucket_size_mb,
                    use_shm=server.state.use_shm,
                    timeout_sec=server.state.ipc_timeout_sec,
                    sync_dtype=server.state.sync_dtype,
                )
                server.state.weight_version += 1
                del weights
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"[student request {request_id}] sync_done "
                    f"seconds={time.time() - start:.3f} weight_version={server.state.weight_version} "
                    f"summary={summary}",
                    flush=True,
                )
                send_message(
                    self.request,
                    {
                        "ok": True,
                        "weight_version": server.state.weight_version,
                        "summary": summary,
                        "metadata": metadata,
                    },
                )
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
    parser.add_argument(
        "--log-requests",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print one-line request lifecycle logs.",
    )
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
        f"log_requests={args.log_requests}",
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
        log_requests=args.log_requests,
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
