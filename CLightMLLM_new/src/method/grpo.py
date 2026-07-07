import re
from typing import Any

import torch

from .base import BaseLearner
from .rollout import RolloutMixin


class GRPOLearner(RolloutMixin, BaseLearner):
    def __init__(
        self,
        *args,
        tokenizer: Any,
        method_args: Any,
        reference_model: torch.nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.method_args = method_args
        self.reference_model = reference_model
        if self.reference_model is not None:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.reference_model is not None:
            self.reference_model.eval()
        return self

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"] = {
            key: value
            for key, value in checkpoint["state_dict"].items()
            if not key.startswith("reference_model.")
        }

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        prompt_width = batch["prompt_input_ids"].shape[1]
        rollouts = [self.generate_rollout(batch) for _ in range(self.method_args.rollout_num_generations)]
        rewards = torch.stack([self.compute_rewards(sequences, prompt_width, batch) for sequences in rollouts], dim=0)
        advantages = self.group_advantages(rewards)

        losses = []
        sampled_kls = []
        for rollout_idx, sequences in enumerate(rollouts):
            completion_mask = self.completion_mask(sequences, prompt_width)
            attention_mask = self.sequence_attention_mask(batch, sequences, completion_mask)
            token_logps, token_mask = self.sequence_token_logps(
                self.model,
                batch,
                sequences,
                attention_mask,
                completion_mask,
                prompt_width,
            )

            old_token_logps = token_logps.detach()
            ratio = torch.exp((token_logps - old_token_logps).clamp(min=-20.0, max=20.0))
            per_token_loss = -ratio * advantages[rollout_idx].unsqueeze(1) * token_mask
            loss = per_token_loss.sum() / token_mask.sum().clamp_min(1.0)

            if self.reference_model is not None and self.method_args.grpo_kl_coef > 0:
                with torch.no_grad():
                    ref_token_logps, _ = self.sequence_token_logps(
                        self.reference_model,
                        batch,
                        sequences,
                        attention_mask,
                        completion_mask,
                        prompt_width,
                    )
                sampled_kl = ((token_logps - ref_token_logps) * token_mask).sum() / token_mask.sum().clamp_min(1.0)
                sampled_kls.append(sampled_kl.detach())
                loss = loss + self.method_args.grpo_kl_coef * sampled_kl

            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        self.log_metric("train/grpo_loss", total_loss.detach(), batch)
        self.log_metric("train/reward", rewards.mean().detach(), batch, prog_bar=True)
        self.log_metric("train/reward_std", rewards.std(unbiased=False).detach(), batch)
        if sampled_kls:
            self.log_metric("train/ref_kl", torch.stack(sampled_kls).mean(), batch)
        return total_loss

    def compute_rewards(
        self,
        sequences: torch.Tensor,
        prompt_width: int,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        reward_type = self.method_args.grpo_reward_type
        if reward_type == "none":
            return torch.zeros(sequences.shape[0], device=sequences.device)

        completion_mask = self.completion_mask(sequences, prompt_width)
        if reward_type == "length":
            lengths = completion_mask.sum(dim=1).float()
            return lengths / max(float(self.method_args.rollout_max_new_tokens), 1.0)

        completions = self.tokenizer.batch_decode(sequences[:, prompt_width:], skip_special_tokens=True)
        references = batch.get("reference_text") or [""] * len(completions)
        scores = [
            self.reference_match_reward(prediction, reference)
            for prediction, reference in zip(completions, references)
        ]
        return torch.tensor(scores, dtype=torch.float32, device=sequences.device)

    @staticmethod
    def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
        if rewards.shape[0] == 1:
            return rewards
        centered = rewards - rewards.mean(dim=0, keepdim=True)
        return centered / rewards.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)

    @staticmethod
    def reference_match_reward(prediction: str, reference: str) -> float:
        prediction = re.sub(r"\s+", " ", prediction.strip().lower())
        reference = re.sub(r"\s+", " ", reference.strip().lower())
        if not prediction or not reference:
            return 0.0
        if prediction == reference:
            return 1.0

        pred_tokens = prediction.split() or list(prediction)
        ref_tokens = reference.split() or list(reference)
        overlap = len(set(pred_tokens) & set(ref_tokens))
        if overlap == 0:
            return 0.0
        precision = overlap / max(len(set(pred_tokens)), 1)
        recall = overlap / max(len(set(ref_tokens)), 1)
        return 2 * precision * recall / max(precision + recall, 1.0e-12)
