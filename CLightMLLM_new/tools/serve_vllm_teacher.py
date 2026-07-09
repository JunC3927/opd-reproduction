import argparse
import os
import socketserver
import sys
import threading
import traceback


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.method.teacher_rpc import recv_message, send_message  # noqa: E402
from src.method.vllm_teacher import VLLMTeacherScorer  # noqa: E402


class TeacherState:
    def __init__(self, scorer: VLLMTeacherScorer) -> None:
        self.scorer = scorer
        self.lock = threading.Lock()


class TeacherTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

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
            if op not in ("score", "score_prompt_requests"):
                send_message(self.request, {"ok": False, "error": f"Unsupported op: {op!r}"})
                return

            server = self.server
            assert isinstance(server, TeacherTCPServer)
            with server.state.lock:
                if op == "score_prompt_requests":
                    logps, ids, lengths = server.state.scorer.score_prompt_requests(
                        requests=request["requests"],
                        pad_token_id=request["pad_token_id"],
                    )
                    send_message(
                        self.request,
                        {
                            "ok": True,
                            "teacher_topk_logps": logps.cpu(),
                            "teacher_topk_ids": ids.cpu(),
                            "teacher_lengths": lengths.cpu(),
                        },
                    )
                    return

                logps, ids = server.state.scorer.score(
                    sequences=request["sequences"],
                    attention_mask=request["attention_mask"],
                    images_per_sample=request.get("images_per_sample"),
                    image_token_id=request.get("image_token_id"),
                    video_token_id=request.get("video_token_id"),
                    pad_token_id=request["pad_token_id"],
                    mm_processor_kwargs_per_sample=request.get("mm_processor_kwargs_per_sample"),
                    multi_modal_data_per_sample=request.get("multi_modal_data_per_sample"),
                )
                send_message(
                    self.request,
                    {
                        "ok": True,
                        "teacher_topk_logps": logps.cpu(),
                        "teacher_topk_ids": ids.cpu(),
                    },
                )
        except Exception:
            send_message(self.request, {"ok": False, "error": traceback.format_exc()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared vLLM teacher scorer for CLight OPD DDP training.")
    parser.add_argument("--model", required=True, help="Teacher model path.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29577)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-logprobs", type=int, default=None)
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
    parser.add_argument("--logprobs-mode", default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-min-pixels", type=int, default=None)
    parser.add_argument("--image-max-pixels", type=int, default=None)
    parser.add_argument("--dedup-mm-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(
        enable_chunked_prefill=None,
        enable_prefix_caching=None,
        disable_log_stats=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "Starting CLight shared vLLM teacher:",
        f"host={args.host}",
        f"port={args.port}",
        f"model={args.model}",
        f"tp={args.tensor_parallel_size}",
        f"dtype={args.torch_dtype}",
        f"max_model_len={args.max_model_len}",
        f"max_num_batched_tokens={args.max_num_batched_tokens}",
        f"max_num_seqs={args.max_num_seqs}",
        f"enable_chunked_prefill={args.enable_chunked_prefill}",
        f"enable_prefix_caching={args.enable_prefix_caching}",
        f"logprobs_mode={args.logprobs_mode}",
        f"image_min_pixels={args.image_min_pixels}",
        f"image_max_pixels={args.image_max_pixels}",
        f"dedup_mm_tokens={args.dedup_mm_tokens}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    limit_mm_per_prompt = None
    if args.limit_images is not None:
        limit_mm_per_prompt = {"image": int(args.limit_images), "video": 0}
    scorer = VLLMTeacherScorer(
        model_path=args.model,
        topk=args.topk,
        torch_dtype=args.torch_dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_logprobs=args.max_logprobs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        load_format=args.load_format,
        distributed_executor_backend=args.distributed_executor_backend,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=args.enable_prefix_caching,
        disable_log_stats=args.disable_log_stats,
        seed=args.seed,
        limit_mm_per_prompt=limit_mm_per_prompt,
        logprobs_mode=args.logprobs_mode,
        enforce_eager=args.enforce_eager,
        device=args.device,
        local_files_only=args.local_files_only,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        dedup_mm_tokens=args.dedup_mm_tokens,
    )
    state = TeacherState(scorer)
    with TeacherTCPServer((args.host, args.port), TeacherHandler, state) as server:
        print(f"CLight shared vLLM teacher listening on {args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
