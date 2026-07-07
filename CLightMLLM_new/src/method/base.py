import math
import os
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
        self._tracked_param_name: str | None = None
        self._tracked_param_before: torch.Tensor | None = None
        self._tracked_param_before_abs_mean: torch.Tensor | None = None

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
        self.log_tracked_param_update()

        grad_sq = None
        tracked_candidates = []
        for name, param in self.model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            local_sq = grad.detach().float().pow(2).sum()
            grad_sq = local_sq if grad_sq is None else grad_sq + local_sq
            if param.requires_grad and param.numel() > 0 and param.ndim >= 2:
                tracked_candidates.append((name, param))

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

        tracked = self.select_tracked_param(tracked_candidates)
        if tracked is not None:
            name, param = tracked
            param_slice = param.detach().flatten()[:1024].float().cpu().clone()
            self._tracked_param_name = name
            self._tracked_param_before = param_slice
            self._tracked_param_before_abs_mean = param_slice.abs().mean()

    @staticmethod
    def select_tracked_param(candidates: list[tuple[str, torch.nn.Parameter]]) -> tuple[str, torch.nn.Parameter] | None:
        if not candidates:
            return None

        preferred_marks = (
            "lm_head.weight",
            "language_model",
            "model.layers",
            "model.language_model",
            "embed_tokens.weight",
        )
        for mark in preferred_marks:
            for name, param in candidates:
                if mark in name:
                    return name, param
        return candidates[-1]

    def log_tracked_param_update(self) -> None:
        before = self._tracked_param_before
        name = self._tracked_param_name
        if before is None or name is None:
            return

        current_param = None
        for param_name, param in self.model.named_parameters():
            if param_name == name:
                current_param = param
                break
        if current_param is None:
            return

        after = current_param.detach().flatten()[: before.numel()].float().cpu()
        delta = (after - before).abs()
        device = self.device if isinstance(self.device, torch.device) else torch.device("cpu")
        max_abs = delta.max().to(device)
        mean_abs = delta.mean().to(device)
        denom = (self._tracked_param_before_abs_mean or before.abs().mean()).clamp_min(1.0e-12).to(device)
        rel_mean = mean_abs / denom
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
            dist.all_reduce(mean_abs, op=dist.ReduceOp.SUM)
            dist.all_reduce(rel_mean, op=dist.ReduceOp.SUM)
            mean_abs = mean_abs / dist.get_world_size()
            rel_mean = rel_mean / dist.get_world_size()
        self.log("train/param_update_max_abs", max_abs, logger=True, prog_bar=False, sync_dist=False)
        self.log("train/param_update_mean_abs", mean_abs, logger=True, prog_bar=False, sync_dist=False)
        self.log("train/param_update_rel_mean", rel_mean, logger=True, prog_bar=False, sync_dist=False)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else int(os.environ.get("RANK", "0"))
        if os.environ.get("CLIGHT_UPDATE_DEBUG") == "1" and rank == 0:
            print(
                "CLight update debug:",
                f"param={name}",
                f"max_abs={float(max_abs.detach().cpu())}",
                f"mean_abs={float(mean_abs.detach().cpu())}",
                f"rel_mean={float(rel_mean.detach().cpu())}",
                flush=True,
            )

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
