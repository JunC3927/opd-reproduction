import math
from typing import Any

import lightning as L
import torch
import torch.distributed as dist
from transformers import get_scheduler

from ..hparams import OptimizerArguments


class BaseLearner(L.LightningModule):
    AUX_BATCH_KEYS = {
        "prompt_input_ids",
        "prompt_attention_mask",
        "reference_text",
        "vllm_images",
    }

    def __init__(self, model: torch.nn.Module, optimizer_args: OptimizerArguments) -> None:
        super().__init__()
        self.model = model
        self.optimizer_args = optimizer_args

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log_metric("train/loss", loss, batch, prog_bar=True)
        return loss

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(**self.model_kwargs(batch)).loss

    def model_kwargs(self, batch: dict[str, Any], include_labels: bool = True) -> dict[str, Any]:
        kwargs = {key: value for key, value in batch.items() if key not in self.AUX_BATCH_KEYS}
        if not include_labels:
            kwargs.pop("labels", None)
        return kwargs

    def log_metric(
        self,
        name: str,
        value: torch.Tensor,
        batch: dict[str, torch.Tensor],
        prog_bar: bool = False,
    ) -> None:
        self.log(
            name,
            value,
            prog_bar=prog_bar,
            logger=True,
            batch_size=batch["input_ids"].shape[0],
            sync_dist=True,
        )

    def on_before_optimizer_step(self, optimizer) -> None:
        grad_sq = None
        for name, param in self.model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            local_sq = grad.detach().float().pow(2).sum()
            grad_sq = local_sq if grad_sq is None else grad_sq + local_sq

        if grad_sq is not None:
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(grad_sq, op=dist.ReduceOp.SUM)
            self.log(
                "train/grad_norm",
                grad_sq.sqrt(),
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | int | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        if gradient_clip_val is None or float(gradient_clip_val) <= 0:
            return

        algorithm = str(gradient_clip_algorithm or "norm").lower()
        strategy = getattr(self.trainer, "strategy", None)
        if strategy is not None and "FSDP" in type(strategy).__name__ and "norm" in algorithm:
            fsdp_model = self._find_fsdp_clip_module()
            if fsdp_model is None:
                raise RuntimeError("Lightning FSDP gradient clipping requested, but no FSDP clip_grad_norm_ module was found.")
            fsdp_model.clip_grad_norm_(float(gradient_clip_val))
            return

        self.clip_gradients(
            optimizer,
            gradient_clip_val=float(gradient_clip_val),
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def _find_fsdp_clip_module(self) -> torch.nn.Module | None:
        candidates = [
            getattr(getattr(self, "trainer", None), "model", None),
            getattr(getattr(getattr(self, "trainer", None), "strategy", None), "model", None),
            self,
            self.model,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            if callable(getattr(candidate, "clip_grad_norm_", None)):
                return candidate
            modules = getattr(candidate, "modules", None)
            if callable(modules):
                for module in modules():
                    if callable(getattr(module, "clip_grad_norm_", None)):
                        return module
        return None

    def configure_optimizers(self):
        args = self.optimizer_args
        decay, no_decay = [], []
        no_decay_marks = ("bias", "norm.weight", "layernorm.weight")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or any(mark in name.lower() for mark in no_decay_marks):
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer_kwargs: dict[str, Any] = {
            "lr": args.learning_rate,
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim.lower() == "adamw_torch_fused" and torch.cuda.is_available():
            optimizer_kwargs["fused"] = True

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": args.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            **optimizer_kwargs,
        )
        scheduler_name = args.lr_scheduler_type.lower()
        if scheduler_name in {"none", "no", "null"}:
            return optimizer

        estimated_steps = self.trainer.estimated_stepping_batches
        if isinstance(estimated_steps, float) and math.isinf(estimated_steps):
            return optimizer

        total_steps = max(1, int(estimated_steps))
        warmup_steps = (
            int(args.warmup_steps)
            if args.warmup_steps is not None
            else int(args.warmup_ratio * total_steps)
        )
        scheduler = get_scheduler(
            scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
