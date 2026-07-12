import contextlib
import gc
import os
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ..hparams import parse_torch_dtype
from .base import BaseLearner
from .rollout import RolloutMixin
from .vllm_student import VLLMStudentRollout, describe_weight_items_for_ipc
from .vllm_student_client import RemoteStudentRollout
from .vllm_teacher import VLLMTeacherScorer
from .vllm_teacher_client import RemoteTeacherScorer, RemoteVLLMTeacherScorer


def is_rank_zero_process() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


class OPDLearner(RolloutMixin, BaseLearner):
    def __init__(
        self,
        *args,
        tokenizer: Any,
        method_args: Any,
        student_model_path: str | None = None,
        torch_dtype: str = "bfloat16",
        teacher_model: torch.nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.method_args = method_args
        object.__setattr__(self, "_teacher_model", teacher_model)
        object.__setattr__(self, "_teacher_scorer", None)
        object.__setattr__(self, "_student_rollout", None)
        self._last_student_rollout_sync_step = -1
        self._last_student_rollout_source_fingerprint: dict[str, Any] | None = None
        self._last_student_rollout_remote_fingerprint: dict[str, Any] | None = None
        if self.method_args.rollout_backend == "vllm":
            if not student_model_path:
                raise ValueError("method.rollout_backend='vllm' requires a student model path.")
            object.__setattr__(
                self,
                "_student_rollout",
                VLLMStudentRollout(
                    model_path=student_model_path,
                    tokenizer=tokenizer,
                    torch_dtype=torch_dtype,
                    tensor_parallel_size=self.method_args.rollout_vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.method_args.rollout_vllm_gpu_memory_utilization,
                    max_model_len=self.method_args.rollout_vllm_max_model_len,
                    enforce_eager=self.method_args.rollout_vllm_enforce_eager,
                    device=self.method_args.rollout_vllm_device,
                    visible_devices=self.method_args.rollout_vllm_visible_devices,
                ),
            )
        elif self.method_args.rollout_backend == "vllm_student_server":
            if (
                self.method_args.rollout_vllm_sync_after_optimizer_step
                and self.method_args.rollout_student_server_sync_backend != "remote_ipc_summon"
            ):
                raise ValueError(
                    "method.rollout_backend='vllm_student_server' with "
                    "method.rollout_vllm_sync_after_optimizer_step=true requires "
                    "method.rollout_student_server_sync_backend='remote_ipc_summon'."
                )
            object.__setattr__(
                self,
                "_student_rollout",
                RemoteStudentRollout(
                    host=self.method_args.rollout_student_server_host,
                    port=self.method_args.rollout_student_server_port,
                    timeout=self.method_args.rollout_student_server_timeout,
                ),
            )
            if is_rank_zero_process():
                print(
                    "[student-vllm-client] init done: "
                    f"server={self.method_args.rollout_student_server_host}:"
                    f"{self.method_args.rollout_student_server_port}",
                    flush=True,
                )
        if self.method_args.opd_teacher_backend == "vllm":
            if not self.method_args.opd_teacher_model_name_or_path:
                raise ValueError("OPD vLLM teacher requires method.opd_teacher_model_name_or_path.")
            object.__setattr__(
                self,
                "_teacher_scorer",
                VLLMTeacherScorer(
                    model_path=self.method_args.opd_teacher_model_name_or_path,
                    topk=self.method_args.opd_topk,
                    tensor_parallel_size=self.method_args.opd_vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.method_args.opd_vllm_gpu_memory_utilization,
                    max_model_len=self.method_args.opd_vllm_max_model_len,
                    max_logprobs=self.method_args.opd_vllm_max_logprobs,
                    max_num_batched_tokens=self.method_args.opd_vllm_max_num_batched_tokens,
                    max_num_seqs=self.method_args.opd_vllm_max_num_seqs,
                    load_format=self.method_args.opd_vllm_load_format,
                    distributed_executor_backend=self.method_args.opd_vllm_distributed_executor_backend,
                    enable_chunked_prefill=self.method_args.opd_vllm_enable_chunked_prefill,
                    enable_prefix_caching=self.method_args.opd_vllm_enable_prefix_caching,
                    disable_log_stats=self.method_args.opd_vllm_disable_log_stats,
                    seed=self.method_args.opd_vllm_seed,
                    limit_mm_per_prompt=(
                        {"image": self.method_args.opd_vllm_limit_images, "video": 0}
                        if self.method_args.opd_vllm_limit_images is not None
                        else None
                    ),
                    logprobs_mode=self.method_args.opd_vllm_logprobs_mode,
                    enforce_eager=self.method_args.opd_vllm_enforce_eager,
                    device=self.method_args.opd_vllm_device,
                    visible_devices=self.method_args.opd_vllm_visible_devices,
                ),
            )
        elif self.method_args.opd_teacher_backend in {"vllm_server", "hf_server"}:
            object.__setattr__(
                self,
                "_teacher_scorer",
                RemoteTeacherScorer(
                    host=self.method_args.opd_teacher_server_host,
                    port=self.method_args.opd_teacher_server_port,
                    timeout=self.method_args.opd_teacher_server_timeout,
                    topk=self.method_args.opd_topk,
                ),
            )
        if self.teacher_model is not None:
            if self.method_args.opd_teacher_device is not None and not self._is_auto_teacher_device():
                self.teacher_model.to(torch.device(self.method_args.opd_teacher_device))
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad_(False)

    @property
    def teacher_model(self) -> torch.nn.Module | None:
        return getattr(self, "_teacher_model", None)

    @property
    def teacher_scorer(self) -> VLLMTeacherScorer | RemoteTeacherScorer | RemoteVLLMTeacherScorer | None:
        return getattr(self, "_teacher_scorer", None)

    @property
    def student_rollout(self) -> VLLMStudentRollout | RemoteStudentRollout | None:
        return getattr(self, "_student_rollout", None)

    def _is_auto_teacher_device(self) -> bool:
        return str(self.method_args.opd_teacher_device).lower() in {"auto", "same_as_student", "current"}

    def on_fit_start(self) -> None:
        self.move_student_io_modules_to_device()
        if self.teacher_model is not None and self._is_auto_teacher_device():
            self.teacher_model.to(self.device)
            self.teacher_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher_model is not None:
            self.teacher_model.eval()
        return self

    def on_train_batch_end(self, outputs: Any, batch: dict[str, Any], batch_idx: int) -> None:
        if self.student_rollout is None:
            return
        if not self.method_args.rollout_vllm_sync_after_optimizer_step:
            return
        current_step = int(getattr(self.trainer, "global_step", 0))
        if current_step <= self._last_student_rollout_sync_step:
            return
        if isinstance(self.student_rollout, RemoteStudentRollout):
            self._sync_remote_student_rollout(current_step=current_step)
            self._last_student_rollout_sync_step = current_step
            return
        sync_from_hf_model = getattr(self.student_rollout, "sync_from_hf_model", None)
        if not callable(sync_from_hf_model):
            raise RuntimeError(
                f"rollout_backend={self.method_args.rollout_backend!r} does not support in-process "
                "sync_from_hf_model. Disable rollout_vllm_sync_after_optimizer_step for smoke, or use "
                "the remote FSDP IPC sync hook once it is enabled."
            )
        sync_from_hf_model(self.model)
        self._last_student_rollout_sync_step = current_step

    def _sync_remote_student_rollout(self, *, current_step: int) -> None:
        if self.method_args.rollout_student_server_sync_backend != "remote_ipc_summon":
            raise RuntimeError(
                "Remote student rollout sync only supports "
                "method.rollout_student_server_sync_backend='remote_ipc_summon'."
            )
        if not isinstance(self.student_rollout, RemoteStudentRollout):
            raise RuntimeError("Remote student rollout sync requires a RemoteStudentRollout client.")

        fsdp_model = self._find_fsdp_summon_module()
        if fsdp_model is None:
            raise RuntimeError(
                "Remote student rollout sync requires a Lightning FSDP-wrapped model. "
                "No FullyShardedDataParallel module was found."
            )

        rank = self._dist_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        sync_dtype = self._remote_sync_dtype()
        start = time.time()
        print(
            f"[student-vllm-sync rank={rank}] remote weight sync start: "
            f"step={current_step}, backend=remote_ipc_summon, "
            f"rank0_only={self.method_args.rollout_student_server_summon_rank0_only}, "
            f"offload_to_cpu={self.method_args.rollout_student_server_summon_offload_to_cpu}",
            flush=True,
        )

        response = None
        with self._summon_full_params_compat(
            fsdp_model,
            rank0_only=bool(self.method_args.rollout_student_server_summon_rank0_only),
            offload_to_cpu=bool(self.method_args.rollout_student_server_summon_offload_to_cpu),
        ):
            print(
                f"[student-vllm-sync rank={rank}] FSDP summon_full_params entered: "
                f"step={current_step}, seconds={time.time() - start:.3f}",
                flush=True,
            )
            if rank == 0:
                weights: list[tuple[str, torch.Tensor]] | None = None
                try:
                    weights = self._student_model_weight_items()
                    source_fingerprint = self._local_sync_fingerprint(weights)
                    weight_stats = describe_weight_items_for_ipc(weights, sync_dtype=sync_dtype)
                    print(
                        "[student-vllm-sync rank=0] weight views ready: "
                        f"step={current_step}, tensors={len(weights)}, sync_dtype={sync_dtype}, "
                        f"bucket_size_mb={self.method_args.rollout_student_server_sync_bucket_size_mb}, "
                        f"use_shm={self.method_args.rollout_student_server_sync_use_shm}, "
                        f"weight_stats={weight_stats}",
                        flush=True,
                    )
                    if source_fingerprint is not None:
                        print(
                            "[student-vllm-sync-verify rank=0] source fingerprint: "
                            f"step={current_step}, fingerprint={source_fingerprint}",
                            flush=True,
                        )
                    response = self.student_rollout.sync_weight_items_ipc(
                        weights,
                        bucket_size_mb=int(self.method_args.rollout_student_server_sync_bucket_size_mb),
                        use_shm=bool(self.method_args.rollout_student_server_sync_use_shm),
                        device=self.method_args.rollout_student_server_sync_device,
                        sync_dtype=sync_dtype,
                    )
                    print(
                        "[student-vllm-sync rank=0] remote weight sync done: "
                        f"step={current_step}, seconds={time.time() - start:.3f}, "
                        f"weight_version={response.get('weight_version')}, "
                        f"summary={response.get('summary')}",
                        flush=True,
                    )
                    if source_fingerprint is not None:
                        self._verify_remote_sync_fingerprint(
                            source_fingerprint,
                            current_step=current_step,
                            weight_version=response.get("weight_version"),
                        )
                finally:
                    if weights is not None:
                        weights.clear()
                    gc.collect()
            self._dist_barrier("inside-remote-student-sync", local_rank=local_rank)
        self._dist_barrier("post-remote-student-sync", local_rank=local_rank)
        if rank != 0:
            print(
                f"[student-vllm-sync rank={rank}] remote weight sync done: "
                f"step={current_step}, seconds={time.time() - start:.3f}",
                flush=True,
            )

    def _student_model_weight_items(self) -> list[tuple[str, torch.Tensor]]:
        weights: list[tuple[str, torch.Tensor]] = []
        seen: set[str] = set()
        for name, param in self.model.named_parameters():
            if not torch.is_tensor(param):
                continue
            normalized = self._normalize_summoned_param_name(name)
            if normalized in seen:
                continue
            seen.add(normalized)
            weights.append((normalized, param.detach()))
        return weights

    def _local_sync_fingerprint(self, weights: list[tuple[str, torch.Tensor]]) -> dict[str, Any] | None:
        if not self.method_args.rollout_student_server_verify_sync:
            return None
        if not weights:
            return None
        verify_name = self.method_args.rollout_student_server_verify_name
        if verify_name is None:
            verify_name = self._choose_sync_verify_weight_name(weights)
        tensor = next((value for name, value in weights if name == verify_name), None)
        if tensor is None:
            available = [name for name, _tensor in weights[:20]]
            raise RuntimeError(
                f"method.rollout_student_server_verify_name={verify_name!r} was not found in summoned weights. "
                f"Available head: {available}"
            )
        sample = tensor.detach().flatten()[: int(self.method_args.rollout_student_server_verify_numel)].float().cpu()
        sample_abs = sample.abs()
        return {
            "name": verify_name,
            "numel": int(sample.numel()),
            "shape": tuple(int(dim) for dim in tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "sum": float(sample.sum().item()),
            "mean": float(sample.mean().item()) if sample.numel() else 0.0,
            "abs_sum": float(sample_abs.sum().item()),
            "sq_sum": float((sample * sample).sum().item()),
            "max_abs": float(sample_abs.max().item()) if sample.numel() else 0.0,
        }

    @staticmethod
    def _choose_sync_verify_weight_name(weights: list[tuple[str, torch.Tensor]]) -> str:
        names = [name for name, _tensor in weights]
        name_set = set(names)
        preferred = (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.o_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.1.self_attn.q_proj.weight",
            "model.language_model.layers.1.mlp.down_proj.weight",
        )
        for name in preferred:
            if name in name_set:
                return name

        suffixes = (
            ".self_attn.q_proj.weight",
            ".self_attn.o_proj.weight",
            ".mlp.down_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
        )
        for name in names:
            if ".layers." in name and name.endswith(suffixes):
                return name

        fallback = (
            "lm_head.weight",
            "model.language_model.embed_tokens.weight",
        )
        for name in fallback:
            if name in name_set:
                return name
        return names[0]

    @staticmethod
    def _fingerprint_delta(
        current: dict[str, Any],
        previous: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if previous is None:
            return None
        if current.get("name") != previous.get("name") and current.get("requested_name") != previous.get("requested_name"):
            return {
                "changed": None,
                "reason": "different_weight_name",
                "previous_name": previous.get("name"),
                "current_name": current.get("name"),
            }
        deltas = {}
        changed = False
        for key in ("sum", "mean", "abs_sum", "sq_sum", "max_abs"):
            current_value = float(current.get(key, 0.0))
            previous_value = float(previous.get(key, 0.0))
            abs_delta = abs(current_value - previous_value)
            deltas[key] = {
                "previous": previous_value,
                "current": current_value,
                "abs_delta": abs_delta,
            }
            changed = changed or abs_delta > 1.0e-12
        return {"changed": changed, "deltas": deltas}

    def _verify_remote_sync_fingerprint(
        self,
        source_fingerprint: dict[str, Any],
        *,
        current_step: int,
        weight_version: Any,
    ) -> None:
        if not isinstance(self.student_rollout, RemoteStudentRollout):
            return
        response = self.student_rollout.fingerprint_weight(
            str(source_fingerprint["name"]),
            numel=int(source_fingerprint["numel"]),
        )
        remote_fingerprint = response.get("fingerprint") or {}
        source_delta = self._fingerprint_delta(
            source_fingerprint,
            self._last_student_rollout_source_fingerprint,
        )
        remote_delta = self._fingerprint_delta(
            remote_fingerprint,
            self._last_student_rollout_remote_fingerprint,
        )
        comparisons = {}
        for key in ("sum", "mean", "abs_sum", "sq_sum", "max_abs"):
            source_value = float(source_fingerprint.get(key, 0.0))
            remote_value = float(remote_fingerprint.get(key, 0.0))
            diff = abs(source_value - remote_value)
            allowed = (
                float(self.method_args.rollout_student_server_verify_atol)
                + float(self.method_args.rollout_student_server_verify_rtol) * abs(source_value)
            )
            comparisons[key] = {
                "source": source_value,
                "remote": remote_value,
                "abs_diff": diff,
                "allowed": allowed,
                "ok": diff <= allowed,
            }
        match = bool(remote_fingerprint.get("ok")) and all(item["ok"] for item in comparisons.values())
        print(
            "[student-vllm-sync-verify rank=0] remote fingerprint: "
            f"step={current_step}, weight_version={weight_version}, match={match}, "
            f"source_dtype={source_fingerprint.get('dtype')}, "
            f"remote_dtype={remote_fingerprint.get('dtype')}, "
            f"source_device={source_fingerprint.get('device')}, "
            f"remote_device={remote_fingerprint.get('device')}, "
            f"source_delta_from_prev={source_delta}, "
            f"remote_delta_from_prev={remote_delta}, "
            f"comparisons={comparisons}, "
            f"remote={remote_fingerprint}",
            flush=True,
        )
        self._last_student_rollout_source_fingerprint = dict(source_fingerprint)
        self._last_student_rollout_remote_fingerprint = dict(remote_fingerprint)

    def _remote_sync_dtype(self) -> torch.dtype | None:
        value = self.method_args.rollout_student_server_sync_dtype
        if value is None:
            return None
        if str(value).lower() in {"none", "null"}:
            return None
        return parse_torch_dtype(str(value))

    def _find_fsdp_summon_module(self) -> FSDP | None:
        candidates = [
            getattr(getattr(self, "trainer", None), "model", None),
            getattr(getattr(getattr(self, "trainer", None), "strategy", None), "model", None),
            self,
            self.model,
        ]
        seen: set[int] = set()
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            if isinstance(candidate, FSDP):
                return candidate
            modules = getattr(candidate, "modules", None)
            if callable(modules):
                for module in modules():
                    if isinstance(module, FSDP):
                        return module
        return None

    @staticmethod
    @contextlib.contextmanager
    def _summon_full_params_compat(
        fsdp_model: FSDP,
        *,
        rank0_only: bool,
        offload_to_cpu: bool,
    ):
        kwargs = {
            "writeback": False,
            "recurse": True,
            "rank0_only": rank0_only,
            "offload_to_cpu": offload_to_cpu,
        }
        try:
            ctx = FSDP.summon_full_params(fsdp_model, **kwargs)
        except TypeError:
            print(
                "[student-vllm-sync] FSDP.summon_full_params does not accept "
                "rank0_only/offload_to_cpu; falling back to writeback=False,recurse=True",
                flush=True,
            )
            ctx = FSDP.summon_full_params(fsdp_model, writeback=False, recurse=True)
        with ctx:
            yield

    @staticmethod
    def _normalize_summoned_param_name(name: str) -> str:
        prefixes = ("_fsdp_wrapped_module.", "module.")
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                while name.startswith(prefix):
                    name = name[len(prefix) :]
                    changed = True
        return name.replace("._fsdp_wrapped_module.", ".")

    @staticmethod
    def _dist_rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    @staticmethod
    def _dist_barrier(label: str, *, local_rank: int) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        print(f"[student-vllm-sync rank={dist.get_rank()}] {label} barrier start", flush=True)
        if str(dist.get_backend()).lower() == "nccl" and torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            dist.barrier(device_ids=[device_id])
        else:
            dist.barrier()
        print(f"[student-vllm-sync rank={dist.get_rank()}] {label} barrier done", flush=True)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"] = {
            key: value
            for key, value in checkpoint["state_dict"].items()
            if not key.startswith("teacher_model.")
        }

    @staticmethod
    def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

    @staticmethod
    def first_parameter_device(model: torch.nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        if self.teacher_model is None and self.teacher_scorer is None:
            raise ValueError("OPD requires an HF teacher_model or method.opd_teacher_backend='vllm'.")
        if self.method_args.opd_loss_type != "forward_kl_topk":
            raise NotImplementedError(
                "This OPD learner currently implements method.opd_loss_type='forward_kl_topk' only."
            )

        prompt_width = self.prompt_width(batch)
        rollout_losses = []
        distill_losses = []
        sft_losses = []
        teacher_masses = []
        student_masses = []
        overlap_ratios = []
        response_token_counts = []

        for _ in range(self.method_args.rollout_num_generations):
            sequences = self.generate_rollout(batch)
            if self.method_args.rollout_backend == "reference":
                attention_mask = batch["attention_mask"]
                labels = batch.get("labels")
                if labels is None:
                    raise ValueError("method.rollout_backend='reference' requires batch labels.")
                response_mask = labels[:, 1:].ne(-100) & attention_mask[:, 1:].bool()
            else:
                completion_mask = self.completion_mask(sequences, prompt_width)
                attention_mask = self.sequence_attention_mask(batch, sequences, completion_mask)
                response_mask = self.shift_completion_mask(
                    token_values=sequences[:, 1:],
                    completion_mask=completion_mask,
                    prompt_width=prompt_width,
                )

            if response_mask.sum() == 0:
                continue
            response_mask = response_mask.to(dtype=torch.float32)
            response_token_counts.append(response_mask.sum().detach())

            with torch.no_grad():
                teacher_topk_logps, teacher_topk_ids = self.compute_teacher_topk(
                    batch=batch,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    response_mask=response_mask,
                )

            student_outputs = self.model(**self.sequence_model_kwargs(batch, sequences, attention_mask))
            student_logits = student_outputs.logits[:, :-1].float() / self.method_args.opd_temperature
            loss_outputs = self.compute_forward_kl_topk_loss(
                student_logits=student_logits,
                teacher_topk_logps=teacher_topk_logps,
                teacher_topk_ids=teacher_topk_ids,
                response_mask=response_mask,
            )
            distill_loss = loss_outputs["loss"] * (self.method_args.opd_temperature**2)
            total_loss = self.method_args.opd_alpha * distill_loss

            if self.method_args.opd_sft_coef > 0:
                sft_loss = self.model(**self.model_kwargs(batch)).loss
                sft_losses.append(sft_loss.detach())
                total_loss = total_loss + self.method_args.opd_sft_coef * sft_loss

            rollout_losses.append(total_loss)
            distill_losses.append(distill_loss.detach())
            teacher_masses.append(self.masked_mean(loss_outputs["teacher_mass"].detach(), response_mask))
            student_masses.append(self.masked_mean(loss_outputs["student_mass"].detach(), response_mask))
            overlap_ratios.append(
                self.masked_mean(
                    loss_outputs["overlap_count"].detach().float() / self.method_args.opd_topk,
                    response_mask,
                )
            )

        if not rollout_losses:
            return self.model(**self.model_kwargs(batch)).loss * 0.0

        loss = torch.stack(rollout_losses).mean()
        self.log_metric("train/opd_loss", torch.stack(distill_losses).mean(), batch, prog_bar=True)
        self.log_metric("train/teacher_mass", torch.stack(teacher_masses).mean(), batch)
        self.log_metric("train/student_mass", torch.stack(student_masses).mean(), batch)
        self.log_metric("train/topk_overlap_ratio", torch.stack(overlap_ratios).mean(), batch)
        self.log_metric("train/response_tokens_per_rank", torch.stack(response_token_counts).mean(), batch)
        if sft_losses:
            self.log_metric("train/sft_loss", torch.stack(sft_losses).mean(), batch)
        return loss

    def compute_teacher_topk(
        self,
        *,
        batch: dict[str, Any],
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.teacher_scorer is not None:
            config = getattr(self.model, "config", None)
            model_kwargs = None
            if self.method_args.opd_teacher_backend == "hf_server":
                model_kwargs = self.sequence_model_kwargs(batch, sequences, attention_mask)
            return self.teacher_scorer.score(
                sequences=sequences,
                attention_mask=attention_mask,
                images_per_sample=None if self.method_args.opd_teacher_backend == "hf_server" else batch.get("vllm_images"),
                image_token_id=getattr(config, "image_token_id", None),
                video_token_id=getattr(config, "video_token_id", None),
                pad_token_id=self.tokenizer.pad_token_id,
                model_kwargs=model_kwargs,
                mm_processor_kwargs_per_sample=batch.get("mm_processor_kwargs"),
                multi_modal_data_per_sample=batch.get("multi_modal_data"),
                response_mask=response_mask,
            )

        assert self.teacher_model is not None
        teacher_device = self.first_parameter_device(self.teacher_model)
        teacher_batch = self.move_batch_to_device(batch, teacher_device)
        teacher_sequences = sequences.to(teacher_device)
        teacher_attention_mask = attention_mask.to(teacher_device)
        teacher_outputs = self.teacher_model(
            **self.sequence_model_kwargs(teacher_batch, teacher_sequences, teacher_attention_mask)
        )
        teacher_logits = teacher_outputs.logits[:, :-1].float() / self.method_args.opd_temperature
        teacher_logps = F.log_softmax(teacher_logits, dim=-1)
        teacher_topk_logps, teacher_topk_ids = torch.topk(
            teacher_logps,
            k=self.method_args.opd_topk,
            dim=-1,
        )
        return teacher_topk_logps.to(sequences.device), teacher_topk_ids.to(sequences.device)

    def compute_forward_kl_topk_loss(
        self,
        student_logits: torch.Tensor,
        teacher_topk_logps: torch.Tensor,
        teacher_topk_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        topk = teacher_topk_ids.shape[-1]
        student_logps = F.log_softmax(student_logits, dim=-1)
        student_topk_ids = torch.topk(student_logps, k=topk, dim=-1).indices
        student_at_teacher_topk = torch.gather(student_logps, dim=-1, index=teacher_topk_ids)

        student_mass = student_at_teacher_topk.exp().sum(dim=-1)
        teacher_mass = teacher_topk_logps.exp().sum(dim=-1)

        if self.method_args.opd_log_prob_min_clamp is not None:
            student_at_teacher_topk = student_at_teacher_topk.clamp_min(self.method_args.opd_log_prob_min_clamp)
            teacher_topk_logps = teacher_topk_logps.clamp_min(self.method_args.opd_log_prob_min_clamp)

        teacher_probs = teacher_topk_logps.exp()
        token_loss = (teacher_probs * (teacher_topk_logps - student_at_teacher_topk)).sum(dim=-1)
        token_loss = token_loss.clamp_min(0.0)
        if self.method_args.opd_loss_max_clamp is not None:
            token_loss = token_loss.clamp(
                min=-self.method_args.opd_loss_max_clamp,
                max=self.method_args.opd_loss_max_clamp,
            )

        loss_num = (token_loss * response_mask).sum()
        loss_den = response_mask.sum().detach()
        if dist.is_available() and dist.is_initialized():
            global_loss_den = loss_den.clone()
            dist.all_reduce(global_loss_den, op=dist.ReduceOp.SUM)
            loss = loss_num * dist.get_world_size() / global_loss_den.clamp_min(1.0)
        else:
            loss = loss_num / loss_den.clamp_min(1.0)
        overlap_count = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1).sum(dim=-1)

        return {
            "loss": loss,
            "token_loss": token_loss,
            "student_mass": student_mass,
            "teacher_mass": teacher_mass,
            "overlap_count": overlap_count,
        }

    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def move_student_io_modules_to_device(self) -> None:
        modules = []
        lm_head = getattr(self.model, "lm_head", None)
        if isinstance(lm_head, torch.nn.Module):
            modules.append(lm_head)

        get_input_embeddings = getattr(self.model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            input_embeddings = get_input_embeddings()
            if isinstance(input_embeddings, torch.nn.Module) and not any(input_embeddings is item for item in modules):
                modules.append(input_embeddings)

        for module in modules:
            module.to(self.device)

        if os.environ.get("CLIGHT_FSDP_DEBUG") == "1" and is_rank_zero_process():
            print(
                "CLight moved student IO modules to device: "
                + ", ".join(type(module).__name__ for module in modules)
                + f" -> {self.device}"
            )
