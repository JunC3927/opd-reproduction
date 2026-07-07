import os
import warnings
from typing import Any

import torch


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
