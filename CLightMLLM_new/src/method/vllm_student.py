import os
import asyncio
from contextlib import contextmanager
import functools
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import torch

from .rpc import rpc_call

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
VERL_ROOT = REPO_ROOT / "verl_new"
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))


def is_rank_zero_process() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def resolve_cuda_device(device: str | None) -> str | None:
    if device is None:
        return None
    normalized = str(device).lower()
    if normalized in {"auto", "current", "local_rank", "same_as_rank"}:
        if not torch.cuda.is_available():
            return None
        return f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    return device


def resolve_visible_device_for_child(device: str | None) -> str | None:
    if device is None or not str(device).startswith("cuda"):
        return None
    index = torch.device(device).index
    if index is None:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return str(index)
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if index >= len(entries):
        raise ValueError(
            f"Requested student vLLM device {device}, but CUDA_VISIBLE_DEVICES={visible!r} "
            f"only exposes {len(entries)} device(s)."
        )
    return entries[index]


@contextmanager
def isolated_vllm_distributed_env(cuda_visible_devices: str | None = None):
    """Prevent vLLM worker subprocesses from inheriting torchrun ranks."""
    keys = [
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
    ]
    if cuda_visible_devices is not None:
        keys.append("CUDA_VISIBLE_DEVICES")
    saved = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_weights_into_model(model: Any, weights: list[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    named_buffers = dict(model.named_buffers())
    param_updates = []
    buffer_updates = []
    for name, tensor in weights:
        if name in named_buffers:
            buffer_updates.append((name, tensor))
        else:
            param_updates.append((name, tensor))

    result = model.load_weights(param_updates) if param_updates else []
    loaded_buffers = 0
    for name, tensor in buffer_updates:
        if name not in named_buffers:
            continue
        target = named_buffers[name]
        if tuple(target.shape) != tuple(tensor.shape):
            raise ValueError(f"Buffer shape mismatch for {name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}")
        target.copy_(tensor.to(device=target.device, dtype=target.dtype), non_blocking=False)
        loaded_buffers += 1

    return {
        "load_weights": result,
        "param_updates": len(param_updates),
        "buffer_updates": len(buffer_updates),
        "loaded_buffers": loaded_buffers,
    }


def _infer_model_device(model: torch.nn.Module) -> torch.device:
    for tensor in list(model.parameters()) + list(model.buffers()):
        return tensor.device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _ipc_load_weights_on_worker(model: Any, zmq_handle: str, use_shm: bool) -> str:
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

    device = _infer_model_device(model)
    bucket_summaries: list[dict[str, Any]] = []
    receiver = BucketedWeightReceiver(
        zmq_handle=zmq_handle,
        device=device,
        use_shm=use_shm,
    )

    def on_bucket_received(weights: list[tuple[str, torch.Tensor]]) -> None:
        summary = _load_weights_into_model(model, weights)
        bucket_summaries.append(
            {
                "bucket_idx": len(bucket_summaries),
                "tensor_count": len(weights),
                "param_updates": summary["param_updates"],
                "buffer_updates": summary["buffer_updates"],
                "loaded_buffers": summary["loaded_buffers"],
            }
        )

    receiver.receive_weights(on_bucket_received=on_bucket_received)
    return repr(
        {
            "device": str(device),
            "use_shm": use_shm,
            "bucket_count": len(bucket_summaries),
            "tensor_count": sum(item["tensor_count"] for item in bucket_summaries),
            "param_updates": sum(item["param_updates"] for item in bucket_summaries),
            "buffer_updates": sum(item["buffer_updates"] for item in bucket_summaries),
            "loaded_buffers": sum(item["loaded_buffers"] for item in bucket_summaries),
        }
    )


def _send_weights_via_ipc(
    weights: Any,
    *,
    zmq_handle: str,
    bucket_size_mb: int,
    use_shm: bool,
) -> None:
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    sender = BucketedWeightSender(
        zmq_handle=zmq_handle,
        bucket_size_mb=bucket_size_mb,
        use_shm=use_shm,
    )
    asyncio.run(sender.async_send_weights(iter(weights)))


def _iter_prepared_weight_items_for_ipc(
    weights: Any,
    *,
    device: torch.device | str | None,
    use_shm: bool,
    sync_dtype: torch.dtype | None,
):
    target_device = torch.device(device) if device is not None else None
    for name, tensor in weights:
        if not torch.is_tensor(tensor):
            continue
        value = tensor.detach()
        if sync_dtype is not None and value.is_floating_point():
            value = value.to(dtype=sync_dtype)
        if use_shm:
            value = value.cpu()
        elif target_device is not None:
            value = value.to(device=target_device, non_blocking=True)
        elif not value.is_cuda:
            value = value.cuda()
        yield name, value.contiguous()


def _estimated_ipc_nbytes(tensor: torch.Tensor, sync_dtype: torch.dtype | None) -> int:
    dtype = sync_dtype if sync_dtype is not None and tensor.is_floating_point() else tensor.dtype
    itemsize = torch.empty((), dtype=dtype).element_size()
    return int(tensor.numel()) * int(itemsize)


def describe_weight_items_for_ipc(weights: Any, *, sync_dtype: torch.dtype | None) -> dict[str, Any]:
    if not isinstance(weights, (list, tuple)):
        return {
            "weight_count": None,
            "total_nbytes": None,
            "largest_name": None,
            "largest_nbytes": None,
            "largest_shape": None,
            "largest_dtype": None,
        }
    total_nbytes = 0
    largest: tuple[str, torch.Tensor, int] | None = None
    count = 0
    for name, tensor in weights:
        if not torch.is_tensor(tensor):
            continue
        count += 1
        nbytes = _estimated_ipc_nbytes(tensor, sync_dtype)
        total_nbytes += nbytes
        if largest is None or nbytes > largest[2]:
            largest = (name, tensor, nbytes)
    if largest is None:
        return {
            "weight_count": count,
            "total_nbytes": total_nbytes,
            "largest_name": None,
            "largest_nbytes": None,
            "largest_shape": None,
            "largest_dtype": None,
        }
    largest_name, largest_tensor, largest_nbytes = largest
    return {
        "weight_count": count,
        "total_nbytes": total_nbytes,
        "largest_name": largest_name,
        "largest_nbytes": largest_nbytes,
        "largest_shape": tuple(int(dim) for dim in largest_tensor.shape),
        "largest_dtype": str(largest_tensor.dtype),
    }


def send_weight_items_ipc(
    weights: Any,
    *,
    zmq_handle: str,
    bucket_size_mb: int = 512,
    use_shm: bool = False,
    device: torch.device | str | None = None,
    sync_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Send weight tensors to a waiting vLLM receiver over VERL bucketed IPC.

    The weights are moved/cast lazily so the sender does not materialize the
    full bf16 model on one GPU before transfer.
    """
    previous_device = None
    target_device = torch.device(device) if device is not None else None
    if not use_shm and target_device is not None and target_device.type == "cuda" and torch.cuda.is_available():
        previous_device = torch.cuda.current_device()
        if target_device.index is not None:
            torch.cuda.set_device(target_device.index)

    counter = {"count": 0}
    weight_stats = describe_weight_items_for_ipc(weights, sync_dtype=sync_dtype)
    bucket_nbytes = int(bucket_size_mb) << 20
    largest_nbytes = weight_stats.get("largest_nbytes")
    if use_shm and largest_nbytes is not None and int(largest_nbytes) > bucket_nbytes:
        largest_mb = int(largest_nbytes) / 1024**2
        raise ValueError(
            "SHM bucket is smaller than the largest tensor after sync dtype conversion: "
            f"largest={weight_stats.get('largest_name')} "
            f"shape={weight_stats.get('largest_shape')} "
            f"dtype={weight_stats.get('largest_dtype')} "
            f"estimated={largest_mb:.1f}MiB, bucket_size_mb={bucket_size_mb}. "
            "Increase --ipc-bucket-size-mb or use CUDA IPC."
        )

    def prepared_iter():
        for item in _iter_prepared_weight_items_for_ipc(
            weights,
            device=target_device,
            use_shm=use_shm,
            sync_dtype=sync_dtype,
        ):
            counter["count"] += 1
            yield item

    try:
        start = time.time()
        _send_weights_via_ipc(
            prepared_iter(),
            zmq_handle=zmq_handle,
            bucket_size_mb=bucket_size_mb,
            use_shm=use_shm,
        )
        sender_sec = time.time() - start
    finally:
        if previous_device is not None:
            torch.cuda.set_device(previous_device)

    return {
        "weight_count": counter["count"],
        "sender_sec": sender_sec,
        "bucket_size_mb": int(bucket_size_mb),
        "use_shm": bool(use_shm),
        "zmq_handle": zmq_handle,
        "weight_stats": weight_stats,
    }


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


class VLLMStudentRollout:
    def __init__(
        self,
        *,
        model_path: str,
        tokenizer: Any,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int | None = None,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        load_format: str | None = None,
        distributed_executor_backend: str | None = None,
        enable_chunked_prefill: bool | None = None,
        enable_prefix_caching: bool | None = None,
        disable_log_stats: bool | None = None,
        seed: int | None = None,
        enforce_eager: bool = False,
        device: str | None = None,
        visible_devices: str | None = None,
        limit_mm_per_prompt: dict[str, int] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        device = resolve_cuda_device(device)
        self.device = device
        self.visible_devices = visible_devices
        self.vllm_worker_visible_device = resolve_visible_device_for_child(device)

        if visible_devices is not None:
            current_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if current_visible != visible_devices:
                warnings.warn(
                    "rollout_vllm_visible_devices cannot safely change CUDA_VISIBLE_DEVICES after "
                    "the Python process has started. Launch the script with "
                    f"CUDA_VISIBLE_DEVICES={visible_devices} instead.",
                    stacklevel=2,
                )
            if os.getenv("CLIGHT_OPD_VLLM_DEBUG") == "1" and is_rank_zero_process():
                print(f"OPD requested student rollout CUDA_VISIBLE_DEVICES={visible_devices}")

        try:
            os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError("method.rollout_backend='vllm' requires the vllm package.") from exc

        self._sampling_params_cls = SamplingParams
        self._tokens_prompt_cls = self._load_tokens_prompt_cls()

        previous_device = None
        if device is not None and device.startswith("cuda") and torch.cuda.is_available():
            previous_device = torch.cuda.current_device()
            index = torch.device(device).index
            if index is not None:
                if os.getenv("CLIGHT_OPD_VLLM_DEBUG") == "1" and is_rank_zero_process():
                    print(f"OPD student rollout vLLM set_device(cuda:{index})")
                torch.cuda.set_device(index)
                if os.getenv("CLIGHT_OPD_VLLM_DEBUG") == "1" and is_rank_zero_process():
                    free, total = torch.cuda.mem_get_info(index)
                    print(
                        "OPD student rollout vLLM memory before init:",
                        f"cuda:{index}",
                        f"free={free / 1024**3:.2f}GiB",
                        f"total={total / 1024**3:.2f}GiB",
                        f"gpu_memory_utilization={gpu_memory_utilization}",
                    )

        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": torch_dtype,
            "enforce_eager": enforce_eager,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs
        if load_format is not None:
            llm_kwargs["load_format"] = load_format
        if distributed_executor_backend is not None:
            llm_kwargs["distributed_executor_backend"] = distributed_executor_backend
        if enable_chunked_prefill is not None:
            llm_kwargs["enable_chunked_prefill"] = enable_chunked_prefill
        if enable_prefix_caching is not None:
            llm_kwargs["enable_prefix_caching"] = enable_prefix_caching
        if disable_log_stats is not None:
            llm_kwargs["disable_log_stats"] = disable_log_stats
        if seed is not None:
            llm_kwargs["seed"] = seed
        if limit_mm_per_prompt is not None:
            llm_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt

        try:
            with isolated_vllm_distributed_env(cuda_visible_devices=self.vllm_worker_visible_device):
                self.llm = LLM(**llm_kwargs)
        finally:
            if previous_device is not None:
                torch.cuda.set_device(previous_device)

    @staticmethod
    def _load_tokens_prompt_cls():
        try:
            from vllm.inputs import TokensPrompt

            return TokensPrompt
        except Exception:
            return None

    @staticmethod
    def _dedup_consecutive_mm_tokens(
        token_ids: list[int],
        image_token_id: int | None,
        video_token_id: int | None,
    ) -> list[int]:
        if image_token_id is None and video_token_id is None:
            return token_ids

        mm_ids = {token_id for token_id in (image_token_id, video_token_id) if token_id is not None}
        deduped = []
        previous_was_mm = False
        for token_id in token_ids:
            current_is_mm = token_id in mm_ids
            if current_is_mm and previous_was_mm:
                continue
            deduped.append(token_id)
            previous_was_mm = current_is_mm
        return deduped

    def _build_prompt(
        self,
        token_ids: list[int],
        images: list[Any],
        image_token_id: int | None,
        video_token_id: int | None,
    ) -> Any:
        prompt_token_ids = self._dedup_consecutive_mm_tokens(token_ids, image_token_id, video_token_id)
        prompt_kwargs: dict[str, Any] = {"prompt_token_ids": prompt_token_ids}
        prompt_kwargs["multi_modal_data"] = {"image": images} if images else {}

        if self._tokens_prompt_cls is not None:
            try:
                return self._tokens_prompt_cls(**prompt_kwargs)
            except TypeError:
                pass
        return prompt_kwargs

    @staticmethod
    def _first_active_index(attention_mask: torch.Tensor) -> int:
        active = torch.nonzero(attention_mask.bool(), as_tuple=False).flatten()
        if active.numel() == 0:
            return int(attention_mask.numel())
        return int(active[0].item())

    def _sampling_params(self, method_args: Any) -> Any:
        kwargs: dict[str, Any] = {
            "max_tokens": method_args.rollout_max_new_tokens,
            "temperature": method_args.rollout_temperature if method_args.rollout_do_sample else 0.0,
            "top_p": method_args.rollout_top_p,
        }
        if method_args.rollout_top_k is not None:
            kwargs["top_k"] = method_args.rollout_top_k
        if self.tokenizer.eos_token_id is not None:
            kwargs["stop_token_ids"] = [int(self.tokenizer.eos_token_id)]
        return self._sampling_params_cls(**kwargs)

    @torch.no_grad()
    def generate(
        self,
        *,
        batch: dict[str, Any],
        method_args: Any,
        image_token_id: int | None,
        video_token_id: int | None,
        pad_token_id: int,
    ) -> torch.Tensor:
        prompt_ids = batch["prompt_input_ids"]
        prompt_attention_mask = batch["prompt_attention_mask"]
        device = prompt_ids.device
        batch_size, prompt_width = prompt_ids.shape

        prompts = []
        images_per_sample = batch.get("vllm_images") or [[] for _ in range(batch_size)]
        for row_idx in range(batch_size):
            start = self._first_active_index(prompt_attention_mask[row_idx])
            active_prompt_ids = prompt_ids[row_idx, start:].detach().cpu().tolist()
            prompts.append(
                self._build_prompt(
                    token_ids=active_prompt_ids,
                    images=images_per_sample[row_idx],
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                )
            )

        outputs = self.llm.generate(prompts, self._sampling_params(method_args), use_tqdm=False)
        completions = []
        max_completion_len = 0
        for output in outputs:
            if not getattr(output, "outputs", None):
                token_ids = []
            else:
                token_ids = list(getattr(output.outputs[0], "token_ids", []) or [])
            max_completion_len = max(max_completion_len, len(token_ids))
            completions.append(token_ids)

        max_completion_len = min(max_completion_len, method_args.rollout_max_new_tokens)
        if max_completion_len == 0:
            return prompt_ids

        completion_tensor = torch.full(
            (batch_size, max_completion_len),
            fill_value=int(pad_token_id),
            dtype=prompt_ids.dtype,
            device=device,
        )
        for row_idx, token_ids in enumerate(completions):
            token_ids = token_ids[:max_completion_len]
            if token_ids:
                completion_tensor[row_idx, : len(token_ids)] = torch.tensor(token_ids, dtype=prompt_ids.dtype, device=device)

        if os.getenv("CLIGHT_OPD_VLLM_DEBUG") == "1" and is_rank_zero_process():
            print(
                "OPD student vLLM rollout:",
                f"batch={batch_size}",
                f"prompt_width={prompt_width}",
                f"completion_width={max_completion_len}",
            )

        return torch.cat([prompt_ids, completion_tensor], dim=1)

    def sync_from_hf_model(self, model: torch.nn.Module) -> None:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise NotImplementedError(
                "Student vLLM weight sync currently supports a single training process only. "
                "For true FSDP multi-rank sync, CLight needs a verl-style rollout worker and "
                "distributed weight transfer path."
            )

        vllm_model = self._find_vllm_model_with_load_weights()
        weights = []
        for name, tensor in model.state_dict().items():
            if not torch.is_tensor(tensor):
                continue
            weights.append((name, tensor.detach().cpu()))
        vllm_model.load_weights(weights)

        if os.getenv("CLIGHT_OPD_VLLM_DEBUG") == "1" and is_rank_zero_process():
            print(f"OPD student vLLM synced {len(weights)} tensors from HF student.")

    def _resolve_apply_model_for_ipc(self) -> tuple[Any, str]:
        apply_model = getattr(self.llm, "apply_model", None)
        owner_name = "llm"
        if not callable(apply_model):
            engine = getattr(self.llm, "llm_engine", None)
            apply_model = getattr(engine, "apply_model", None)
            owner_name = "llm.llm_engine"
        if not callable(apply_model):
            raise RuntimeError("Could not find llm.apply_model or llm.llm_engine.apply_model for IPC sync.")
        return apply_model, owner_name

    def start_weight_sync_receiver(
        self,
        *,
        zmq_handle: str,
        use_shm: bool = False,
    ) -> dict[str, Any]:
        apply_model, owner_name = self._resolve_apply_model_for_ipc()
        result_box: dict[str, Any] = {}

        def receiver_target() -> None:
            try:
                result_box["result"] = apply_model(
                    functools.partial(_ipc_load_weights_on_worker, zmq_handle=zmq_handle, use_shm=use_shm)
                )
                result_box["ok"] = True
            except Exception as exc:
                result_box["ok"] = False
                result_box["error"] = f"{type(exc).__name__}: {exc}"

        receiver_thread = threading.Thread(
            target=receiver_target,
            name="clight-student-vllm-remote-ipc-receiver",
            daemon=True,
        )
        receiver_thread.start()
        return {
            "thread": receiver_thread,
            "result_box": result_box,
            "owner_name": owner_name,
            "zmq_handle": zmq_handle,
            "use_shm": bool(use_shm),
            "started_at": time.time(),
        }

    def _find_vllm_model_with_load_weights(self) -> Any:
        candidates = [
            "llm_engine.model_executor.driver_worker.model_runner.model",
            "llm_engine.model_executor.driver_worker.worker.model_runner.model",
            "llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model",
            "engine_core.engine_core.model_executor.driver_worker.model_runner.model",
        ]
        for path in candidates:
            current: Any = self.llm
            for attr in path.split("."):
                current = getattr(current, attr, None)
                if current is None:
                    break
            if current is not None and hasattr(current, "load_weights"):
                return current

        raise RuntimeError(
            "Could not find a vLLM model object with load_weights(). "
            "This vLLM version may hide the in-process model runner; use HF rollout "
            "or add a version-specific vLLM weight sync adapter."
        )
