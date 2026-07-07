import logging
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
