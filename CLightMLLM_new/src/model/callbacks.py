import logging
import json
import os
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from peft import PeftModel


LOGGER = logging.getLogger(__name__)
rank_zero_info = rank_zero_only(LOGGER.info)


class HFModelExportCallback(Callback):
    def __init__(
        self,
        processor: Any,
        output_dir: str,
        enabled: bool = True,
        merge_lora_before_export: bool = False,
    ) -> None:
        self.processor = processor
        self.output_dir = output_dir
        self.enabled = enabled
        self.merge_lora_before_export = merge_lora_before_export

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self.enabled:
            rank_zero_info("Training finished. HF model export is disabled by trainer.export_hf_model_at_end=false.")
            return

        if self.processor is None:
            raise ValueError("processor is required when export_hf_model_at_end=true.")

        rank_zero_info("Exporting HF model to %s.", self.output_dir)
        self.save_hf_pretrained(trainer, pl_module.model)
        rank_zero_info("HF model export finished.")

    def prepare_hf_model_for_export(self, hf_model: torch.nn.Module) -> torch.nn.Module:
        if self.merge_lora_before_export and isinstance(hf_model, PeftModel):
            rank_zero_info("Merging LoRA weights before HF export.")
            return hf_model.merge_and_unload()
        return hf_model

    def save_hf_pretrained(self, trainer: L.Trainer, hf_model: torch.nn.Module) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        trainer.strategy.barrier()
        is_zero3 = isinstance(trainer.strategy, DeepSpeedStrategy) and getattr(
            trainer.strategy,
            "zero_stage_3",
            False,
        )

        if is_zero3:
            import deepspeed
            # ZeRO-3 shards parameters; gather before rank-zero export.
            with deepspeed.zero.GatheredParameters(list(hf_model.parameters()), modifier_rank=0):
                if trainer.is_global_zero:
                    hf_model = self.prepare_hf_model_for_export(hf_model)
                    hf_model.save_pretrained(self.output_dir)
                    self.processor.save_pretrained(self.output_dir)
        elif trainer.is_global_zero:
            hf_model = self.prepare_hf_model_for_export(hf_model)
            hf_model.save_pretrained(self.output_dir)
            self.processor.save_pretrained(self.output_dir)

        trainer.strategy.barrier()


class RankZeroWandbFinishCallback(Callback):
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        trainer.strategy.barrier()
        try:
            if trainer.is_global_zero:
                import wandb

                if wandb.run is not None:
                    wandb.finish()
        finally:
            trainer.strategy.barrier()


class JSONLMetricsCallback(Callback):
    def __init__(self, output_path: str | None) -> None:
        self.output_path = output_path
        self._handle = None

    @staticmethod
    def to_jsonable(value: Any) -> Any:
        if torch.is_tensor(value):
            value = value.detach()
            if value.numel() == 1:
                return value.float().cpu().item()
            return value.float().cpu().tolist()
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self.output_path or not trainer.is_global_zero:
            return
        path = self.output_path
        if not os.path.isabs(path):
            path = os.path.join(str(trainer.default_root_dir), path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._handle = open(path, "w", encoding="utf-8")

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._handle is None or not trainer.is_global_zero:
            return
        record = {
            "format": "clight_lightning_metrics_v1",
            "global_step": int(trainer.global_step),
            "epoch": int(trainer.current_epoch),
            "batch_idx": int(batch_idx),
        }
        record.update(
            {
                key: self.to_jsonable(value)
                for key, value in trainer.callback_metrics.items()
                if key.startswith("train/")
            }
        )
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class ConsoleMetricsCallback(Callback):
    DEFAULT_KEYS = (
        "train/loss",
        "train/opd_loss",
        "train/grad_norm",
        "train/teacher_mass",
        "train/student_mass",
        "train/topk_overlap_ratio",
        "train/response_tokens_per_rank",
        "train/param_update_max_abs",
        "train/param_update_mean_abs",
        "train/param_update_rel_mean",
    )

    def __init__(self, every_n_steps: int = 1, keys: tuple[str, ...] | None = None) -> None:
        self.every_n_steps = max(1, int(every_n_steps))
        self.keys = keys or self.DEFAULT_KEYS

    @staticmethod
    def format_value(value: Any) -> str:
        if torch.is_tensor(value):
            value = value.detach()
            if value.numel() != 1:
                return f"tensor{tuple(value.shape)}"
            value = value.float().cpu().item()
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        if step <= 0 or step % self.every_n_steps != 0:
            return

        parts = [
            f"step={step}",
            f"epoch={int(trainer.current_epoch)}",
            f"batch={int(batch_idx)}",
        ]
        metrics = trainer.callback_metrics
        for key in self.keys:
            if key in metrics:
                parts.append(f"{key}={self.format_value(metrics[key])}")
        print("[metrics] " + " | ".join(parts), flush=True)
