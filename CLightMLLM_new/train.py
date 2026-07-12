import argparse
import gc
import inspect
import logging
import os

from dataclasses import fields, replace
from functools import partial
from typing import Any
import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from transformers import set_seed

from src.data import TemplateFactory, VLSFTDataModule
from src.hparams import (
    CLSFTArguments,
    DataArguments,
    LoaderArguments,
    MethodArguments,
    ModelArguments,
    OptimizerArguments,
    TrainerArguments,
    TuningArguments,
    parse_torch_dtype,
)
from src.method import create_learner
from src.model import (
    HFModelExportCallback,
    JSONLMetricsCallback,
    ModelTuner,
    RankZeroWandbFinishCallback,
    SwanLabLogger,
    load_vision_language_model,
)


LOGGER = logging.getLogger(__name__)
rank_zero_info = rank_zero_only(LOGGER.info)


def is_rank_zero_process() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


class TrainingApp:
    # YAML top-level groups and their dataclass schemas.
    ARG_GROUPS = {
        "cl_sft": CLSFTArguments,
        "data": DataArguments,
        "loader": LoaderArguments,
        "method": MethodArguments,
        "model": ModelArguments,
        "optimizer": OptimizerArguments,
        "trainer": TrainerArguments,
        "tuning": TuningArguments,
    }

    def run(self) -> None:
        cli_args = self.parse_args()
        (
            cl_sft_args,
            data_args,
            loader_args,
            method_args,
            model_args,
            optimizer_args,
            trainer_args,
            tuning_args,
        ) = self.parse_yaml_args(cli_args.config)
        stage_count = len(cl_sft_args.stages)
        if stage_count == 0:
            raise ValueError("cl_sft.stages is required. For regular SFT, define exactly one CL SFT stage.")
        if stage_count > 1 and not trainer_args.export_hf_model_at_end:
            raise ValueError("Multi-stage cl_sft requires trainer.export_hf_model_at_end=true.")
        if stage_count > 1 and tuning_args.lora.enable and not trainer_args.merge_lora_before_export:
            raise ValueError("Multi-stage LoRA cl_sft requires trainer.merge_lora_before_export=true.")

        self.configure_torch_backend(trainer_args)
        set_seed(trainer_args.seed)
        L.seed_everything(trainer_args.seed, workers=True)

        current_model_path = model_args.model_name_or_path
        for stage_idx, stage in enumerate(cl_sft_args.stages, start=1):
            teacher_model_path, reference_model_path = self.resolve_aux_model_paths(
                method_args=method_args,
                current_model_path=current_model_path,
                stage_idx=stage_idx,
            )
            stage_data_args = replace(data_args, dataset=stage.dataset)
            stage_model_args = replace(model_args, model_name_or_path=current_model_path)
            stage_trainer_args = replace(
                trainer_args,
                save_dir=os.path.join(trainer_args.save_dir, stage.name),
                run_name=f"{trainer_args.run_name}_{stage.name}" if trainer_args.run_name else stage.name,
            )
            stage_model_output_dir = os.path.join(stage_trainer_args.save_dir, "model")

            rank_zero_info(
                "[CL SFT] Stage %s/%s: %s | dataset=%s | model=%s | experiment=%s | output=%s",
                stage_idx,
                stage_count,
                stage.name,
                ",".join(stage.dataset),
                stage_model_args.model_name_or_path,
                stage_trainer_args.save_dir,
                stage_model_output_dir,
            )
            try:
                self.run_stage(
                    data_args=stage_data_args,
                    loader_args=loader_args,
                    method_args=method_args,
                    model_args=stage_model_args,
                    optimizer_args=optimizer_args,
                    trainer_args=stage_trainer_args,
                    tuning_args=tuning_args,
                    teacher_model_path=teacher_model_path,
                    reference_model_path=reference_model_path,
                )
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            current_model_path = stage_model_output_dir

    def run_stage(
        self,
        data_args: DataArguments,
        loader_args: LoaderArguments,
        method_args: MethodArguments,
        model_args: ModelArguments,
        optimizer_args: OptimizerArguments,
        trainer_args: TrainerArguments,
        tuning_args: TuningArguments,
        teacher_model_path: str | None = None,
        reference_model_path: str | None = None,
    ) -> None:
        if model_args.use_verl_monkey_patch and (teacher_model_path is not None or reference_model_path is not None):
            raise ValueError(
                "model.use_verl_monkey_patch modifies Qwen3-VL forward methods globally. "
                "Use it with vLLM teacher/reference-free OPD, or disable HF teacher/reference models in this process."
            )
        model, processor, tokenizer = load_vision_language_model(model_args, data_args.template)
        teacher_model = self.load_teacher_model(
            model_args,
            data_args.template,
            teacher_model_path,
            torch_dtype=method_args.opd_teacher_torch_dtype,
        )
        reference_model = self.load_teacher_model(model_args, data_args.template, reference_model_path)
        model = ModelTuner(tuning_args).apply(model)
        self.report_model_parameters(model, prefix="student pre-FSDP")
        template = TemplateFactory.from_args(tokenizer, data_args)
        datamodule = VLSFTDataModule(
            template=template,
            model_args=model_args,
            data_args=data_args,
            loader_args=loader_args,
            tokenizer=tokenizer,
            processor=processor,
            model=model,
        )
        module = create_learner(
            method_args,
            model=model,
            optimizer_args=optimizer_args,
            tokenizer=tokenizer,
            teacher_model=teacher_model,
            reference_model=reference_model,
            student_model_path=model_args.model_name_or_path,
            torch_dtype=model_args.torch_dtype,
        )
        model_output_dir = os.path.join(trainer_args.save_dir, "model")
        trainer_kwargs = trainer_args.lightning_kwargs()
        trainer_kwargs["strategy"] = self.build_strategy(trainer_args, model=model)
        loggers = self.build_loggers(trainer_args)
        callbacks = [
            HFModelExportCallback(
                processor=processor,
                output_dir=model_output_dir,
                enabled=trainer_args.export_hf_model_at_end,
                merge_lora_before_export=trainer_args.merge_lora_before_export,
            ),
            RankZeroWandbFinishCallback(),
            JSONLMetricsCallback(trainer_args.metrics_jsonl),
        ]
        if loggers:
            callbacks.insert(0, LearningRateMonitor(logging_interval="step"))
        trainer = L.Trainer(
            **trainer_kwargs,
            logger=loggers,
            callbacks=callbacks,
        )
        trainer.fit(module, datamodule=datamodule, ckpt_path=trainer_args.resume_from_checkpoint)

    @staticmethod
    def report_model_parameters(model: torch.nn.Module, prefix: str) -> None:
        if not is_rank_zero_process():
            return

        total = sum(param.numel() for param in model.parameters())
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        non_trainable = total - trainable
        trainable_pct = 100.0 * trainable / total if total else 0.0
        print(
            f"CLight {prefix} params: "
            f"trainable={trainable:,} total={total:,} non_trainable={non_trainable:,} "
            f"trainable_pct={trainable_pct:.2f}%",
            flush=True,
        )

    @staticmethod
    def build_loggers(trainer_args: TrainerArguments) -> list[Any] | bool:
        loggers = []
        if trainer_args.wandb_project:
            loggers.append(
                WandbLogger(
                    project=trainer_args.wandb_project,
                    name=trainer_args.run_name,
                    save_dir=trainer_args.save_dir,
                    log_model=trainer_args.log_model,
                )
            )
        if trainer_args.swanlab_project:
            loggers.append(
                SwanLabLogger(
                    project=trainer_args.swanlab_project,
                    experiment_name=trainer_args.run_name,
                    workspace=trainer_args.swanlab_workspace,
                    mode=trainer_args.swanlab_mode,
                    save_dir=trainer_args.save_dir,
                    config={
                        "run_name": trainer_args.run_name,
                        "save_dir": trainer_args.save_dir,
                    },
                )
            )
        return loggers or False

    @staticmethod
    def resolve_aux_model_paths(
        method_args: MethodArguments,
        current_model_path: str | None,
        stage_idx: int,
    ) -> tuple[str | None, str | None]:
        teacher_model_path = None
        reference_model_path = None
        if method_args.name == "lwf" and stage_idx > 1:
            teacher_model_path = current_model_path
        elif method_args.name == "opd" and method_args.opd_teacher_backend == "hf":
            teacher_model_path = method_args.opd_teacher_model_name_or_path or current_model_path
        elif method_args.name == "grpo" and (method_args.grpo_reference_model or method_args.grpo_kl_coef > 0):
            reference_model_path = current_model_path
        return teacher_model_path, reference_model_path

    @staticmethod
    def load_teacher_model(
        model_args: ModelArguments,
        template_name: str,
        teacher_model_path: str | None,
        torch_dtype: str | None = None,
    ) -> torch.nn.Module | None:
        if teacher_model_path is None:
            return None
        teacher_args = replace(
            model_args,
            model_name_or_path=teacher_model_path,
            torch_dtype=torch_dtype or model_args.torch_dtype,
            gradient_checkpointing=False,
            use_cache=False,
            use_verl_monkey_patch=False,
        )
        teacher_model, _, _ = load_vision_language_model(teacher_args, template_name)
        return teacher_model

    @staticmethod
    def build_strategy(trainer_args: TrainerArguments, model: torch.nn.Module | None = None) -> Any:
        strategy = trainer_args.strategy
        if strategy != "fsdp_auto_wrap":
            return strategy

        from lightning.pytorch.strategies import FSDPStrategy
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=trainer_args.fsdp_min_num_params,
        )
        mixed_precision = MixedPrecision(
            param_dtype=(
                None if trainer_args.fsdp_param_dtype is None else parse_torch_dtype(trainer_args.fsdp_param_dtype)
            ),
            reduce_dtype=(
                None if trainer_args.fsdp_reduce_dtype is None else parse_torch_dtype(trainer_args.fsdp_reduce_dtype)
            ),
            buffer_dtype=(
                None if trainer_args.fsdp_buffer_dtype is None else parse_torch_dtype(trainer_args.fsdp_buffer_dtype)
            ),
        )
        kwargs = {
            "auto_wrap_policy": auto_wrap_policy,
            "mixed_precision": mixed_precision,
            "cpu_offload": CPUOffload(offload_params=trainer_args.fsdp_cpu_offload),
            "use_orig_params": trainer_args.fsdp_use_orig_params,
            "forward_prefetch": trainer_args.fsdp_forward_prefetch,
            "limit_all_gathers": True,
        }
        ignored_modules = TrainingApp.collect_fsdp_ignored_modules(trainer_args, model)
        if ignored_modules:
            kwargs["ignored_modules"] = ignored_modules
        signature = inspect.signature(FSDPStrategy)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
        supported_kwargs = kwargs if accepts_kwargs else {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
        if is_rank_zero_process():
            print(
                "CLight FSDPStrategy kwargs: "
                + ", ".join(f"{key}={value!r}" for key, value in supported_kwargs.items() if key != "auto_wrap_policy")
            )
        return FSDPStrategy(**supported_kwargs)

    @staticmethod
    def collect_fsdp_ignored_modules(
        trainer_args: TrainerArguments,
        model: torch.nn.Module | None,
    ) -> list[torch.nn.Module]:
        if not trainer_args.fsdp_ignore_lm_head or model is None:
            return []

        ignored: list[torch.nn.Module] = []
        lm_head = getattr(model, "lm_head", None)
        if isinstance(lm_head, torch.nn.Module):
            ignored.append(lm_head)

        get_input_embeddings = getattr(model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            input_embeddings = get_input_embeddings()
            if isinstance(input_embeddings, torch.nn.Module) and not any(input_embeddings is item for item in ignored):
                ignored.append(input_embeddings)

        if is_rank_zero_process() and ignored:
            print(
                "CLight FSDP ignored modules: "
                + ", ".join(type(module).__name__ for module in ignored)
            )
        return ignored

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True, help="Path to YAML config.")
        return parser.parse_args()

    @classmethod
    def parse_yaml_args(cls, path: str) -> tuple[Any, ...]:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        unknown = sorted(set(config) - set(cls.ARG_GROUPS))
        if unknown:
            raise KeyError(f"Unsupported config groups: {unknown}. Allowed groups: {sorted(cls.ARG_GROUPS)}")

        # Fail fast on misspelled YAML keys.
        hparams = []
        for group, group_cls in cls.ARG_GROUPS.items():
            group_config = config.get(group) or {}
            allowed = {field.name for field in fields(group_cls) if field.init}
            unknown = sorted(set(group_config) - allowed)
            if unknown:
                raise KeyError(f"Unsupported {group_cls.__name__} config keys: {unknown}")
            hparams.append(group_cls(**group_config))

        return tuple(hparams)

    @staticmethod
    def configure_torch_backend(args: TrainerArguments) -> None:
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = args.tf32
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = args.tf32


if __name__ == "__main__":
    TrainingApp().run()
