import os
import asyncio
import functools
import threading
import time
from typing import Any

import torch

from .bucketed_weight_transfer import BucketedWeightReceiver, BucketedWeightSender
from .rpc import rpc_call


def resolve_cuda_device(device: str | None) -> str | None:
    if device is None:
        return None
    normalized = str(device).lower()
    if normalized in {"auto", "current", "local_rank", "same_as_rank"}:
        if not torch.cuda.is_available():
            return None
        return f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    return device


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


def _largest_weight_item_for_ipc(weights: Any, *, sync_dtype: torch.dtype | None) -> dict[str, Any] | None:
    if not isinstance(weights, (list, tuple)):
        return None
    largest: tuple[str, torch.Tensor, int] | None = None
    for name, tensor in weights:
        if not torch.is_tensor(tensor):
            continue
        nbytes = _estimated_ipc_nbytes(tensor, sync_dtype)
        if largest is None or nbytes > largest[2]:
            largest = (name, tensor, nbytes)
    if largest is None:
        return None
    largest_name, largest_tensor, largest_nbytes = largest
    return {
        "name": largest_name,
        "nbytes": largest_nbytes,
        "shape": tuple(int(dim) for dim in largest_tensor.shape),
        "dtype": str(largest_tensor.dtype),
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
    largest_weight = _largest_weight_item_for_ipc(weights, sync_dtype=sync_dtype) if use_shm else None
    bucket_nbytes = int(bucket_size_mb) << 20
    if largest_weight is not None and int(largest_weight["nbytes"]) > bucket_nbytes:
        largest_mb = int(largest_weight["nbytes"]) / 1024**2
        raise ValueError(
            "SHM bucket is smaller than the largest tensor after sync dtype conversion: "
            f"largest={largest_weight['name']} "
            f"shape={largest_weight['shape']} "
            f"dtype={largest_weight['dtype']} "
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
        weight_version = int(response.get("weight_version", 0))
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
        try:
            sender_summary = send_weight_items_ipc(
                weights,
                zmq_handle=start_response["zmq_handle"],
                bucket_size_mb=bucket_size_mb,
                use_shm=resolved_use_shm,
                device=device,
                sync_dtype=sync_dtype,
            )
        except Exception as exc:
            abort_timeout_sec = min(float(self.timeout), 30.0)
            try:
                self.abort_weight_sync(
                    session_id=str(start_response["session_id"]),
                    reason=f"{type(exc).__name__}: {exc}",
                    timeout_sec=abort_timeout_sec,
                )
            except Exception as abort_exc:
                raise RuntimeError(
                    "Student vLLM IPC sender failed and the server sync session could not be aborted: "
                    f"sender_error={type(exc).__name__}: {exc}; "
                    f"abort_error={type(abort_exc).__name__}: {abort_exc}"
                ) from exc
            raise
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

    def abort_weight_sync(
        self,
        *,
        session_id: str,
        reason: str | None = None,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        return self._checked_response(
            rpc_call(
                self.host,
                self.port,
                {
                    "op": "abort_weight_sync",
                    "session_id": session_id,
                    "reason": reason,
                    "timeout_sec": timeout_sec,
                },
                self.timeout,
            )
        )

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
        limit_mm_per_prompt: dict[str, int] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        device = resolve_cuda_device(device)
        self.device = device

        try:
            os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError("The student vLLM server requires the vllm package.") from exc

        self._sampling_params_cls = SamplingParams
        self._tokens_prompt_cls = self._load_tokens_prompt_cls()

        previous_device = None
        if device is not None and device.startswith("cuda") and torch.cuda.is_available():
            previous_device = torch.cuda.current_device()
            index = torch.device(device).index
            if index is not None:
                torch.cuda.set_device(index)

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

        return torch.cat([prompt_ids, completion_tensor], dim=1)

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
