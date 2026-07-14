from dataclasses import dataclass, field
from typing import Any, Literal

import torch


TORCH_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def parse_torch_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    try:
        return TORCH_DTYPES[str(value).lower()]
    except KeyError as exc:
        supported = ", ".join(TORCH_DTYPES)
        raise ValueError(f"Unsupported torch_dtype={value!r}. Supported values: {supported}.") from exc


def csv(value: str | list[str] | None) -> list[str] | None:
    if value is None or isinstance(value, list):
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class CLStageArguments:
    name: str
    dataset: str | list[str]

    def __post_init__(self) -> None:
        self.dataset = csv(self.dataset)
        if not self.dataset:
            raise ValueError(f"cl_sft stage {self.name!r} requires a non-empty dataset.")


@dataclass
class CLSFTArguments:
    stages: list[str | CLStageArguments] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stages = [self.build_stage(stage) for stage in self.stages]

    @staticmethod
    def build_stage(stage: str | CLStageArguments) -> CLStageArguments:
        if isinstance(stage, CLStageArguments):
            return stage
        if isinstance(stage, str):
            return CLStageArguments(name=stage, dataset=stage)
        raise TypeError("cl_sft.stages must be a list of dataset names.")


@dataclass
class MethodArguments:
    name: Literal["base", "lwf", "grpo", "opd"] = "base"
    lwf_alpha: float = 1.0
    lwf_temperature: float = 2.0
    rollout_backend: Literal["hf", "vllm_student_server"] = "hf"
    rollout_max_new_tokens: int = 64
    rollout_num_generations: int = 1
    rollout_do_sample: bool = True
    rollout_temperature: float = 0.7
    rollout_top_p: float = 0.9
    rollout_top_k: int | None = None
    rollout_use_cache: bool = True
    rollout_vllm_sync_after_optimizer_step: bool = True
    rollout_student_server_host: str = "127.0.0.1"
    rollout_student_server_port: int = 29588
    rollout_student_server_timeout: float = 3600.0
    rollout_student_server_sync_backend: Literal["none", "remote_ipc_summon"] = "remote_ipc_summon"
    rollout_student_server_sync_bucket_size_mb: int = 2048
    rollout_student_server_sync_use_shm: bool = False
    rollout_student_server_sync_device: str | None = None
    rollout_student_server_sync_dtype: str | None = "none"
    rollout_student_server_summon_rank0_only: bool = True
    rollout_student_server_summon_offload_to_cpu: bool = False
    grpo_reward_type: Literal["reference_match", "length", "none"] = "reference_match"
    grpo_kl_coef: float = 0.0
    grpo_reference_model: bool = False
    opd_teacher_model_name_or_path: str | None = None
    opd_teacher_backend: Literal["hf", "vllm_server", "hf_server"] = "hf"
    opd_teacher_device: str | None = None
    opd_teacher_torch_dtype: str | None = None
    opd_teacher_server_host: str = "127.0.0.1"
    opd_teacher_server_port: int = 29577
    opd_teacher_server_timeout: float = 600.0
    opd_alpha: float = 1.0
    opd_temperature: float = 1.0
    opd_loss_type: Literal["kl", "direct","forward_kl_topk"] = "kl"
    opd_sft_coef: float = 0.0
    opd_topk: int = 32
    opd_topk_renorm :bool = True
    opd_log_prob_min_clamp: float | None = -10.0
    opd_loss_max_clamp: float | None = 10.0

    def __post_init__(self) -> None:
        if self.name not in {"base", "lwf", "grpo", "opd"}:
            raise ValueError("method.name must be 'base', 'lwf', 'grpo' or 'opd'.")
        if self.lwf_alpha < 0:
            raise ValueError("method.lwf_alpha must be non-negative.")
        if self.lwf_temperature <= 0:
            raise ValueError("method.lwf_temperature must be positive.")
        if self.rollout_max_new_tokens <= 0:
            raise ValueError("method.rollout_max_new_tokens must be positive.")
        if self.rollout_num_generations <= 0:
            raise ValueError("method.rollout_num_generations must be positive.")
        if self.rollout_temperature <= 0:
            raise ValueError("method.rollout_temperature must be positive.")
        if not 0 < self.rollout_top_p <= 1:
            raise ValueError("method.rollout_top_p must be in (0, 1].")
        if self.rollout_top_k is not None and self.rollout_top_k <= 0:
            raise ValueError("method.rollout_top_k must be positive when set.")
        if self.rollout_student_server_port <= 0:
            raise ValueError("method.rollout_student_server_port must be positive.")
        if self.rollout_student_server_timeout <= 0:
            raise ValueError("method.rollout_student_server_timeout must be positive.")
        if self.rollout_student_server_sync_bucket_size_mb <= 0:
            raise ValueError("method.rollout_student_server_sync_bucket_size_mb must be positive.")
        if self.rollout_student_server_sync_dtype is not None:
            sync_dtype = str(self.rollout_student_server_sync_dtype).lower()
            if sync_dtype not in {"none", "null"}:
                parse_torch_dtype(sync_dtype)
        if self.grpo_kl_coef < 0:
            raise ValueError("method.grpo_kl_coef must be non-negative.")
        if self.opd_alpha < 0:
            raise ValueError("method.opd_alpha must be non-negative.")
        if self.opd_temperature <= 0:
            raise ValueError("method.opd_temperature must be positive.")
        if self.opd_sft_coef < 0:
            raise ValueError("method.opd_sft_coef must be non-negative.")
        if self.opd_topk <= 0:
            raise ValueError("method.opd_topk must be positive.")
        if self.opd_loss_max_clamp is not None and self.opd_loss_max_clamp <= 0:
            raise ValueError("method.opd_loss_max_clamp must be positive when set.")
        if self.opd_teacher_server_port <= 0:
            raise ValueError("method.opd_teacher_server_port must be positive.")
        if self.opd_teacher_server_timeout <= 0:
            raise ValueError("method.opd_teacher_server_timeout must be positive.")


@dataclass
class DataArguments:
    template: Literal["llava", "intern_vl", "qwen2_vl", "qwen3_vl"]
    dataset: str | list[str] | None = None
    dataset_config: str = "config/dataset.json"
    cutoff_len: int = 2048
    max_prompt_length: int = 1024
    filter_overlong_prompts: bool = True
    max_samples: int | None = None
    preprocessing_batch_size: int = 2000
    preprocessing_num_workers: int | None = 16
    preprocessing_mp_start_method: Literal["spawn", "forkserver", "fork"] | None = "spawn"
    preprocessing_omp_num_threads: int | None = 1
    overwrite_cache: bool = False
    default_system: str | None = None
    ignore_pad_token_for_loss: bool = True
    log_first_sample: bool = True

    def __post_init__(self) -> None:
        self.dataset = csv(self.dataset)
        if self.max_prompt_length <= 0:
            raise ValueError("data.max_prompt_length must be positive.")


@dataclass
class LoaderArguments:
    per_device_train_batch_size: int = 1
    num_workers: int = 8
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = 2
    shuffle: bool = True


@dataclass
class ModelArguments:
    model_name_or_path: str | None = None
    cache_dir: str | None = None
    hf_hub_token: str | None = None
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    padding_side: str = "right"
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    attn_implementation: str | None = None
    device_map: str | dict[str, Any] | None = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_cache: bool = False
    local_files_only: bool = False
    image_min_pixels: int | None = None
    image_max_pixels: int | None = None


@dataclass
class OptimizerArguments:
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    warmup_steps: int | None = None


@dataclass
class TrainerArguments:
    seed: int = 42
    tf32: bool = True
    run_name: str = "lycllm_sft"
    wandb_project: str | None = None
    swanlab_project: str | None = None
    swanlab_workspace: str | None = None
    swanlab_mode: str | None = None
    log_model: bool = False
    export_hf_model_at_end: bool = True
    merge_lora_before_export: bool = False
    resume_from_checkpoint: str | None = None
    save_dir: str = "experiments/lightning_sft"
    accelerator: str = "auto"
    strategy: str = "auto"
    devices: int | list[int] | str = "auto"
    num_nodes: int = 1
    precision: str = "bf16-mixed"
    fsdp_min_num_params: int = 100_000_000
    fsdp_param_dtype: str | None = "bfloat16"
    fsdp_reduce_dtype: str | None = "float32"
    fsdp_buffer_dtype: str | None = "float32"
    fsdp_cpu_offload: bool = False
    fsdp_use_orig_params: bool = False
    fsdp_forward_prefetch: bool = False
    fsdp_ignore_lm_head: bool = False
    metrics_jsonl: str | None = None
    max_epochs: int = 1
    max_steps: int = -1
    accumulate_grad_batches: int = 1
    gradient_clip_val: float | None = 1.0
    log_every_n_steps: int = 1
    num_sanity_val_steps: int = 0
    enable_progress_bar: bool = True
    enable_checkpointing: bool = False

    def lightning_kwargs(self) -> dict[str, Any]:
        # Keep app-only fields out of Lightning Trainer(**kwargs).
        app_keys = {
            "seed",
            "tf32",
            "run_name",
            "wandb_project",
            "swanlab_project",
            "swanlab_workspace",
            "swanlab_mode",
            "log_model",
            "save_dir",
            "export_hf_model_at_end",
            "merge_lora_before_export",
            "resume_from_checkpoint",
            "fsdp_min_num_params",
            "fsdp_param_dtype",
            "fsdp_reduce_dtype",
            "fsdp_buffer_dtype",
            "fsdp_cpu_offload",
            "fsdp_use_orig_params",
            "fsdp_forward_prefetch",
            "fsdp_ignore_lm_head",
            "metrics_jsonl",
        }
        kwargs = {key: value for key, value in vars(self).items() if key not in app_keys}
        kwargs["default_root_dir"] = self.save_dir
        return kwargs


@dataclass
class LoraArguments:
    enable: bool = True
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: str | list[str] = "all"


@dataclass
class TuningArguments:
    lora: LoraArguments | dict[str, Any] = field(default_factory=LoraArguments)
    freeze_vision_tower: bool = True
    freeze_multi_modal_projector: bool = True
    prepare_model_for_kbit_training: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.lora, dict):
            self.lora = LoraArguments(**self.lora)
