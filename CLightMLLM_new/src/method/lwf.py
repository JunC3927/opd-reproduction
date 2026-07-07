import torch
import torch.nn.functional as F

from .base import BaseLearner


class LwFLearner(BaseLearner):
    def __init__(
        self,
        *args,
        teacher_model: torch.nn.Module | None = None,
        alpha: float = 1.0,
        temperature: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher_model is not None:
            self.teacher_model.eval()
        return self

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"] = {
            key: value
            for key, value in checkpoint["state_dict"].items()
            if not key.startswith("teacher_model.")
        }

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(**self.model_kwargs(batch))
        if self.teacher_model is None or self.alpha <= 0:
            return outputs.loss

        with torch.no_grad():
            teacher_logits = self.teacher_model(**self.model_kwargs(batch, include_labels=False)).logits

        lwf_loss = self.distill_loss(outputs.logits, teacher_logits, batch)
        self.log_metric("train/sft_loss", outputs.loss.detach(), batch)
        self.log_metric("train/lwf_loss", lwf_loss.detach(), batch)
        return outputs.loss + self.alpha * lwf_loss

    def distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        temperature = self.temperature
        student_logits = student_logits[:, :-1].float() / temperature
        teacher_logits = teacher_logits[:, :-1].float() / temperature
        token_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="none",
        ).sum(dim=-1)

        labels = batch.get("labels")
        if labels is None:
            mask = batch["attention_mask"][:, 1:].bool()
        else:
            mask = labels[:, 1:].ne(-100)
        return (token_loss * mask).sum() * (temperature**2) / mask.sum().clamp_min(1)

