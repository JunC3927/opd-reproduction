import inspect
import os
from typing import Any

import torch

from .rpc import rpc_call


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
        multi_modal_data_per_sample: list[dict[str, Any] | None] | None = None,
        response_mask: torch.Tensor | None = None,
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
            "response_mask": None if response_mask is None else response_mask.detach().cpu(),
            "images_per_sample": images_per_sample,
            "image_token_id": image_token_id,
            "video_token_id": video_token_id,
            "pad_token_id": int(pad_token_id),
            "topk": self.topk,
            "model_kwargs": cpu_model_kwargs,
            "mm_processor_kwargs_per_sample": mm_processor_kwargs_per_sample,
            "multi_modal_data_per_sample": multi_modal_data_per_sample,
        }
        response = rpc_call(self.host, self.port, request, self.timeout)
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected teacher server response type: {type(response)}")
        if response.get("ok") is not True:
            raise RuntimeError(response.get("error") or "Remote teacher scoring failed.")

        logps = response["teacher_topk_logps"].to(device=device, dtype=torch.float32)
        ids = response["teacher_topk_ids"].to(device=device, dtype=torch.long)
        return logps, ids

    @torch.no_grad()
    def score_prompt_requests(
        self,
        *,
        requests: list[dict[str, Any]],
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        response = rpc_call(
            self.host,
            self.port,
            {
                "op": "score_prompt_requests",
                "requests": requests,
                "pad_token_id": int(pad_token_id),
                "topk": self.topk,
            },
            self.timeout,
        )
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected teacher server response type: {type(response)}")
        if response.get("ok") is not True:
            raise RuntimeError(response.get("error") or "Remote teacher scoring failed.")

        logps = response["teacher_topk_logps"].to(dtype=torch.float32)
        ids = response["teacher_topk_ids"].to(dtype=torch.long)
        lengths = response["teacher_lengths"].to(dtype=torch.long)
        return logps, ids, lengths


class VLLMTeacherScorer:
    def __init__(
        self,
        *,
        model_path: str,
        topk: int,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int | None = None,
        max_logprobs: int | None = None,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        load_format: str | None = None,
        distributed_executor_backend: str | None = None,
        enable_chunked_prefill: bool | None = None,
        enable_prefix_caching: bool | None = None,
        disable_log_stats: bool | None = None,
        seed: int | None = None,
        limit_mm_per_prompt: dict[str, int] | None = None,
        logprobs_mode: str | None = None,
        enforce_eager: bool = False,
        device: str | None = None,
        local_files_only: bool = False,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
        dedup_mm_tokens: bool = True,
    ) -> None:
        self.topk = topk
        self.dedup_mm_tokens = bool(dedup_mm_tokens)
        device = resolve_cuda_device(device)
        self.device = device

        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        previous_device = None
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError("The teacher vLLM server requires the vllm package.") from exc
        self._tokens_prompt_cls = self._load_tokens_prompt_cls()
        self._sampling_params_cls = SamplingParams

        if device is not None and device.startswith("cuda") and torch.cuda.is_available():
            previous_device = torch.cuda.current_device()
            index = torch.device(device).index
            if index is not None:
                if os.getenv("CLIGHT_OPD_VLLM_DEBUG") == "1" and is_rank_zero_process():
                    print(f"OPD vLLM set_device(cuda:{index})")
                torch.cuda.set_device(index)
                if os.getenv("CLIGHT_OPD_VLLM_DEBUG") == "1" and is_rank_zero_process():
                    free, total = torch.cuda.mem_get_info(index)
                    print(
                        "OPD teacher vLLM memory before init:",
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
            "max_logprobs": max_logprobs or topk,
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
        if logprobs_mode is not None:
            llm_kwargs["logprobs_mode"] = logprobs_mode
        mm_processor_kwargs = {}
        if image_min_pixels is not None:
            mm_processor_kwargs["min_pixels"] = int(image_min_pixels)
        if image_max_pixels is not None:
            mm_processor_kwargs["max_pixels"] = int(image_max_pixels)
        if mm_processor_kwargs:
            llm_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
        llm_kwargs = self._filter_llm_kwargs(LLM, llm_kwargs)
        try:
            self.llm = LLM(**llm_kwargs)
        finally:
            if previous_device is not None:
                torch.cuda.set_device(previous_device)

        self._sampling_params_keys = self._accepted_sampling_params_keys(SamplingParams)
        self.sampling_params = self._make_sampling_params(
            {
                "max_tokens": 1,
                "temperature": 1.0,
                "prompt_logprobs": topk,
            }
        )

    @staticmethod
    def _filter_llm_kwargs(llm_cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(llm_cls.__init__)
        except (TypeError, ValueError):
            return kwargs

        parameters = signature.parameters
        accepts_kwargs = any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values())
        if accepts_kwargs:
            return kwargs

        filtered = {key: value for key, value in kwargs.items() if key in parameters}
        skipped = sorted(set(kwargs) - set(filtered))
        if skipped and os.getenv("CLIGHT_OPD_VLLM_DEBUG") == "1" and is_rank_zero_process():
            print(f"OPD teacher vLLM skipped unsupported LLM kwargs: {skipped}")
        return filtered

    @staticmethod
    def _load_tokens_prompt_cls():
        try:
            from vllm.inputs import TokensPrompt

            return TokensPrompt
        except Exception:
            return None

    @staticmethod
    def _accepted_sampling_params_keys(sampling_params_cls: Any) -> set[str] | None:
        try:
            signature = inspect.signature(sampling_params_cls.__init__)
        except (TypeError, ValueError):
            return None
        parameters = signature.parameters
        if any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()):
            return None
        return {key for key in parameters if key != "self"}

    def _make_sampling_params(self, params: dict[str, Any]):
        params.setdefault("detokenize", False)
        if self._sampling_params_keys is not None:
            params = {key: value for key, value in params.items() if key in self._sampling_params_keys}
        return self._sampling_params_cls(**params)

    def _sampling_params_from_request(self, request: dict[str, Any]):
        params = {
            "max_tokens": 1,
            "temperature": 1.0,
            "prompt_logprobs": self.topk,
        }
        request_params = request.get("sampling_params")
        if isinstance(request_params, dict):
            params.update({key: value for key, value in request_params.items() if value is not None})
        params.setdefault("prompt_logprobs", self.topk)
        return self._make_sampling_params(params)

    @staticmethod
    def _dedup_consecutive_mm_tokens(
        token_ids: list[int],
        image_token_id: int | None,
        video_token_id: int | None,
    ) -> tuple[list[int], list[int]]:
        if image_token_id is None and video_token_id is None:
            return token_ids, list(range(len(token_ids)))

        mm_ids = {token_id for token_id in (image_token_id, video_token_id) if token_id is not None}
        deduped = []
        kept_indices = []
        previous_was_mm = False
        for idx, token_id in enumerate(token_ids):
            current_is_mm = token_id in mm_ids
            if current_is_mm and previous_was_mm:
                continue
            deduped.append(token_id)
            kept_indices.append(idx)
            previous_was_mm = current_is_mm
        return deduped, kept_indices

    def _build_prompt(
        self,
        token_ids: list[int],
        images: list[Any],
        image_token_id: int | None,
        video_token_id: int | None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        multi_modal_data: dict[str, Any] | None = None,
    ) -> tuple[Any, list[int]]:
        if self.dedup_mm_tokens:
            prompt_token_ids, kept_indices = self._dedup_consecutive_mm_tokens(
                token_ids,
                image_token_id,
                video_token_id,
            )
        else:
            prompt_token_ids = token_ids
            kept_indices = list(range(len(token_ids)))
        prompt_kwargs: dict[str, Any] = {"prompt_token_ids": prompt_token_ids}
        if multi_modal_data is not None:
            prompt_mm_data = {}
            if "image" in multi_modal_data:
                prompt_mm_data["image"] = multi_modal_data["image"]
            elif "images" in multi_modal_data:
                prompt_mm_data["image"] = multi_modal_data["images"]
            if "video" in multi_modal_data:
                prompt_mm_data["video"] = multi_modal_data["video"]
            elif "videos" in multi_modal_data:
                prompt_mm_data["video"] = multi_modal_data["videos"]
            if "audio" in multi_modal_data:
                prompt_mm_data["audio"] = multi_modal_data["audio"]
            elif "audios" in multi_modal_data:
                prompt_mm_data["audio"] = multi_modal_data["audios"]
            prompt_kwargs["multi_modal_data"] = prompt_mm_data
        elif images:
            prompt_kwargs["multi_modal_data"] = {"image": images}
        else:
            prompt_kwargs["multi_modal_data"] = {}
        if mm_processor_kwargs:
            prompt_kwargs["mm_processor_kwargs"] = mm_processor_kwargs

        if self._tokens_prompt_cls is not None:
            try:
                return self._tokens_prompt_cls(**prompt_kwargs), kept_indices
            except TypeError:
                pass
        return prompt_kwargs, kept_indices

    @staticmethod
    def _first_active_index(attention_mask: torch.Tensor) -> int:
        active = torch.nonzero(attention_mask.bool(), as_tuple=False).flatten()
        if active.numel() == 0:
            return int(attention_mask.numel())
        return int(active[0].item())

    @staticmethod
    def _active_span(attention_mask: torch.Tensor) -> tuple[int, int]:
        active = torch.nonzero(attention_mask.bool(), as_tuple=False).flatten()
        if active.numel() == 0:
            length = int(attention_mask.numel())
            return length, length
        return int(active[0].item()), int(active[-1].item()) + 1

    def _extract_topk(self, output: Any, expected_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_logprobs = getattr(output, "prompt_logprobs", None)
        if prompt_logprobs is None:
            raise RuntimeError("vLLM output did not include prompt_logprobs.")

        entries = prompt_logprobs[1:]
        if len(entries) != expected_len:
            raise RuntimeError(f"vLLM prompt_logprobs length {len(entries)} != expected {expected_len}.")

        ids_rows, logp_rows = [], []
        for position, logprobs_dict in enumerate(entries):
            if logprobs_dict is None:
                raise RuntimeError(f"Missing vLLM prompt_logprobs at shifted position {position}.")

            ids = [None] * self.topk
            logps = [None] * self.topk
            for token_id, token_logprob in logprobs_dict.items():
                rank = getattr(token_logprob, "rank", None)
                if rank is None:
                    continue
                if rank > self.topk:
                    continue
                ids[rank - 1] = int(token_id)
                logps[rank - 1] = float(getattr(token_logprob, "logprob"))

            if any(token_id is None for token_id in ids) or any(logp is None for logp in logps):
                raise RuntimeError(f"Incomplete top-{self.topk} vLLM prompt_logprobs at shifted position {position}.")

            ids_rows.append(ids)
            logp_rows.append(logps)

        return (
            torch.tensor(logp_rows, dtype=torch.float32),
            torch.tensor(ids_rows, dtype=torch.long),
        )

    def _request_to_prompt(self, request: dict[str, Any]) -> Any:
        prompt_kwargs = request.get("prompt_kwargs") or {}
        if not prompt_kwargs:
            if "prompt_token_ids" in request:
                prompt_kwargs = {"prompt_token_ids": request["prompt_token_ids"]}
            else:
                raise KeyError("Final prompt request has neither prompt_kwargs nor prompt_token_ids.")
            multi_modal_data = request.get("multi_modal_data")
            if multi_modal_data is not None:
                prompt_kwargs["multi_modal_data"] = multi_modal_data
            mm_processor_kwargs = request.get("mm_processor_kwargs")
            if mm_processor_kwargs is not None:
                prompt_kwargs["mm_processor_kwargs"] = mm_processor_kwargs

        if self._tokens_prompt_cls is not None:
            try:
                return self._tokens_prompt_cls(**prompt_kwargs)
            except TypeError:
                pass
        return prompt_kwargs

    @torch.no_grad()
    def score_prompt_requests(
        self,
        *,
        requests: list[dict[str, Any]],
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompts = [self._request_to_prompt(request) for request in requests]
        sampling_params = [self._sampling_params_from_request(request) for request in requests]
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        if len(outputs) != len(requests):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(requests)} prompts.")

        logps_rows = []
        ids_rows = []
        lengths = []
        for output in outputs:
            prompt_logprobs = getattr(output, "prompt_logprobs", None)
            if prompt_logprobs is None:
                raise RuntimeError("vLLM output did not include prompt_logprobs.")
            output_len = len(prompt_logprobs) - 1
            logps, ids = self._extract_topk(output, expected_len=output_len)
            logps_rows.append(logps)
            ids_rows.append(ids)
            lengths.append(output_len)

        max_len = max(lengths, default=0)
        full_logps = torch.zeros(len(requests), max_len, self.topk, dtype=torch.float32)
        full_ids = torch.full(
            (len(requests), max_len, self.topk),
            fill_value=int(pad_token_id),
            dtype=torch.long,
        )
        for row, (logps, ids, length) in enumerate(zip(logps_rows, ids_rows, lengths, strict=True)):
            if length > 0:
                full_logps[row, :length] = logps
                full_ids[row, :length] = ids
        return full_logps, full_ids, torch.tensor(lengths, dtype=torch.long)

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
        multi_modal_data_per_sample: list[dict[str, Any] | None] | None = None,
        response_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = sequences.device
        batch_size, seq_len = sequences.shape
        full_logps = torch.zeros(batch_size, max(seq_len - 1, 0), self.topk, dtype=torch.float32, device=device)
        full_ids = torch.full(
            (batch_size, max(seq_len - 1, 0), self.topk),
            fill_value=int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        if seq_len <= 1:
            return full_logps, full_ids

        response_shift_start = None
        response_mask_cpu = None
        if response_mask is not None:
            response_mask_cpu = response_mask.detach().cpu().bool()
            if response_mask_cpu.ndim != 2 or response_mask_cpu.shape[0] != batch_size:
                raise RuntimeError(
                    "Unexpected response_mask shape: "
                    f"got {tuple(response_mask_cpu.shape)}, expected batch {batch_size}."
                )
            response_width = int(response_mask_cpu.shape[1])
            response_shift_start = seq_len - response_width - 1
            if response_shift_start < 0:
                raise RuntimeError(
                    f"Invalid response_mask width {response_width} for sequence length {seq_len}."
                )

        prompts = []
        spans = []
        if images_per_sample is None:
            images_per_sample = [[] for _ in range(batch_size)]
        else:
            images_per_sample = list(images_per_sample)
        if len(images_per_sample) < batch_size:
            images_per_sample.extend([[] for _ in range(batch_size - len(images_per_sample))])

        if mm_processor_kwargs_per_sample is None:
            mm_processor_kwargs_per_sample = [None for _ in range(batch_size)]
        else:
            mm_processor_kwargs_per_sample = list(mm_processor_kwargs_per_sample)
        if len(mm_processor_kwargs_per_sample) < batch_size:
            mm_processor_kwargs_per_sample.extend([None for _ in range(batch_size - len(mm_processor_kwargs_per_sample))])

        if multi_modal_data_per_sample is None:
            multi_modal_data_per_sample = [None for _ in range(batch_size)]
        else:
            multi_modal_data_per_sample = list(multi_modal_data_per_sample)
        if len(multi_modal_data_per_sample) < batch_size:
            multi_modal_data_per_sample.extend([None for _ in range(batch_size - len(multi_modal_data_per_sample))])

        for row_idx in range(batch_size):
            start, end = self._active_span(attention_mask[row_idx])
            token_ids = sequences[row_idx, start:end].detach().cpu().tolist()
            prompt, kept_indices = self._build_prompt(
                token_ids=token_ids,
                images=images_per_sample[row_idx],
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                mm_processor_kwargs=mm_processor_kwargs_per_sample[row_idx],
                multi_modal_data=multi_modal_data_per_sample[row_idx],
            )
            response_shift_indices = None
            if response_mask_cpu is not None:
                assert response_shift_start is not None
                local_response_positions = torch.nonzero(
                    response_mask_cpu[row_idx],
                    as_tuple=False,
                ).flatten()
                response_shift_indices = [
                    int(response_shift_start + position.item())
                    for position in local_response_positions
                    if 0 <= int(response_shift_start + position.item()) < max(seq_len - 1, 0)
                ]
            prompts.append(prompt)
            spans.append((row_idx, start, len(token_ids), kept_indices, response_shift_indices))

        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        if len(outputs) != len(spans):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(spans)} prompts.")

        for output, (row_idx, start, active_len, kept_indices, response_shift_indices) in zip(outputs, spans):
            prompt_logprobs = getattr(output, "prompt_logprobs", None)
            if prompt_logprobs is None:
                raise RuntimeError("vLLM output did not include prompt_logprobs.")

            output_len = len(prompt_logprobs) - 1
            expanded_len = max(active_len - 1, 0)
            dedup_len = max(len(kept_indices) - 1, 0)
            if os.getenv("CLIGHT_OPD_VLLM_DEBUG") == "1" and is_rank_zero_process():
                print(
                    "OPD vLLM lengths:",
                    f"row={row_idx}",
                    f"start={start}",
                    f"active_len={active_len}",
                    f"kept_len={len(kept_indices)}",
                    f"output_len={output_len}",
                    f"expanded_shift_len={expanded_len}",
                    f"dedup_shift_len={dedup_len}",
                )
            logps, ids = self._extract_topk(output, expected_len=output_len)

            if response_shift_indices is not None:
                response_len = len(response_shift_indices)
                if response_len == 0:
                    continue
                if output_len < response_len:
                    raise RuntimeError(
                        "Unexpected vLLM prompt_logprobs length for response alignment: "
                        f"got {output_len}, need at least response tokens {response_len} "
                        f"(row={row_idx}, active_len={active_len}, kept_len={len(kept_indices)})."
                    )
                if os.getenv("CLIGHT_OPD_VLLM_DEBUG") == "1" and is_rank_zero_process():
                    print(
                        "OPD vLLM response suffix alignment:",
                        f"row={row_idx}",
                        f"output_len={output_len}",
                        f"response_len={response_len}",
                        f"first_shift={response_shift_indices[0]}",
                        f"last_shift={response_shift_indices[-1]}",
                    )
                response_logps = logps[-response_len:].to(device)
                response_ids = ids[-response_len:].to(device)
                full_logps[row_idx, response_shift_indices] = response_logps
                full_ids[row_idx, response_shift_indices] = response_ids
            elif output_len == expanded_len:
                full_slice = slice(start, start + active_len - 1)
                full_logps[row_idx, full_slice] = logps.to(device)
                full_ids[row_idx, full_slice] = ids.to(device)
            elif output_len == dedup_len:
                for dedup_shift_idx, original_token_idx in enumerate(kept_indices[1:]):
                    original_shift_idx = start + original_token_idx - 1
                    full_logps[row_idx, original_shift_idx] = logps[dedup_shift_idx].to(device)
                    full_ids[row_idx, original_shift_idx] = ids[dedup_shift_idx].to(device)
            else:
                raise RuntimeError(
                    "Unexpected vLLM prompt_logprobs length: "
                    f"got {output_len}, expected expanded {expanded_len} or deduped {dedup_len}."
                )

        return full_logps, full_ids
