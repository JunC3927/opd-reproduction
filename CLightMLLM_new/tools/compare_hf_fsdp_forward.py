import argparse
import importlib.util
import os
import sys
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hparams import (  # noqa: E402
    CLSFTArguments,
    DataArguments,
    LoaderArguments,
    MethodArguments,
    ModelArguments,
    OptimizerArguments,
    TrainerArguments,
    TuningArguments,
)
from src.model import load_vision_language_model  # noqa: E402


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


def load_rollout_mixin():
    rollout_path = ROOT / "src" / "method" / "rollout.py"
    spec = importlib.util.spec_from_file_location("_clight_rollout_compare", rollout_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load rollout module from {rollout_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RolloutMixin


RolloutMixin = load_rollout_mixin()


class _ForwardProbe(RolloutMixin):
    AUX_BATCH_KEYS = {
        "prompt_input_ids",
        "prompt_attention_mask",
        "reference_text",
        "vllm_images",
    }

    def __init__(self, model: torch.nn.Module, tokenizer: Any, method_args: MethodArguments) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.method_args = method_args

    def model_kwargs(self, batch: dict[str, Any], include_labels: bool = True) -> dict[str, Any]:
        kwargs = {key: value for key, value in batch.items() if key not in self.AUX_BATCH_KEYS}
        if not include_labels:
            kwargs.pop("labels", None)
        return kwargs


def parse_yaml_args(path: str) -> tuple[Any, ...]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    unknown = sorted(set(config) - set(ARG_GROUPS))
    if unknown:
        raise KeyError(f"Unsupported config groups: {unknown}. Allowed groups: {sorted(ARG_GROUPS)}")

    hparams = []
    for group, group_cls in ARG_GROUPS.items():
        group_config = config.get(group) or {}
        allowed = {field.name for field in fields(group_cls) if field.init}
        unknown = sorted(set(group_config) - allowed)
        if unknown:
            raise KeyError(f"Unsupported {group_cls.__name__} config keys: {unknown}")
        hparams.append(group_cls(**group_config))
    return tuple(hparams)


def setup_dist(require_process_group: bool = False) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if (world_size > 1 or require_process_group) and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
            dist.init_process_group(backend=backend)
        else:
            init_file = Path(tempfile.gettempdir()) / "clight_single_rank_fsdp_init"
            init_file.unlink(missing_ok=True)
            dist.init_process_group(
                backend=backend,
                init_method=f"file://{init_file}",
                rank=0,
                world_size=1,
            )
    return rank, world_size, local_rank


def cleanup_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def move_tensors_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors_to_device(item, device) for key, item in value.items()}
    return value


def response_token_logps(logits: torch.Tensor, sequences: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].float()
    shift_labels = sequences[:, 1:]
    logps = F.log_softmax(shift_logits, dim=-1)
    token_logps = torch.gather(logps, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_logps[response_mask.bool()].detach().cpu()


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.float() - b.float()).abs()
    return {
        f"{name}_max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        f"{name}_mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HF forward with FSDP-wrapped forward on one fixed OPD batch.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch", required=True, help="Path produced by tools/dump_opd_forward_batch.py.")
    parser.add_argument("--device", default=None, help="Override device. Defaults to cuda:LOCAL_RANK when available.")
    parser.add_argument("--use-fsdp", action="store_true", help="Wrap the second forward model with FSDP.")
    args = parser.parse_args()

    os.chdir(ROOT)
    rank, world_size, local_rank = setup_dist(require_process_group=args.use_fsdp)
    device = torch.device(args.device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    try:
        (
            _cl_sft_args,
            _data_args,
            _loader_args,
            method_args,
            model_args,
            _optimizer_args,
            _trainer_args,
            _tuning_args,
        ) = parse_yaml_args(args.config)

        payload = torch.load(args.batch, map_location="cpu")
        batch = move_tensors_to_device(payload["batch"], device)
        sequences = payload["sequences"].to(device)
        attention_mask = payload["attention_mask"].to(device)
        response_mask = payload["response_mask"].to(device)

        hf_model, _processor, tokenizer = load_vision_language_model(model_args, payload.get("template", "qwen3_vl"))
        hf_model.to(device)
        hf_model.eval()
        probe = _ForwardProbe(model=hf_model, tokenizer=tokenizer, method_args=method_args)
        forward_kwargs = probe.sequence_model_kwargs(batch, sequences, attention_mask)

        with torch.no_grad():
            hf_logits = hf_model(**forward_kwargs).logits.detach()
            hf_response_logps = response_token_logps(hf_logits, sequences, response_mask)
        del hf_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        fsdp_model, _processor, tokenizer = load_vision_language_model(model_args, payload.get("template", "qwen3_vl"))
        fsdp_model.to(device)
        fsdp_model.eval()

        if args.use_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            fsdp_model = FSDP(
                fsdp_model,
                device_id=device if device.type == "cuda" else None,
                use_orig_params=True,
            )

        with torch.no_grad():
            fsdp_logits = fsdp_model(**forward_kwargs).logits.detach()
            fsdp_response_logps = response_token_logps(fsdp_logits, sequences, response_mask)

        stats = {}
        stats.update(compare("response_logps", hf_response_logps, fsdp_response_logps))
        stats.update(compare("logits", hf_logits.detach().cpu(), fsdp_logits.detach().cpu()))

        ok = bool(torch.isfinite(hf_response_logps).all().item() and torch.isfinite(fsdp_response_logps).all().item())
        if rank == 0:
            print("=== hf vs fsdp forward ===")
            print(f"rank={rank} world_size={world_size} device={device}")
            print(f"use_fsdp={args.use_fsdp}")
            print(f"batch_file={args.batch}")
            print(f"sequences_shape={tuple(sequences.shape)}")
            print(f"response_tokens={int(response_mask.sum().item())}")
            for key, value in stats.items():
                print(f"{key}={value}")
            print(f"finite_response_logps={ok}")
            print("hf_fsdp_forward_compare_ok=True")
    finally:
        cleanup_dist()


if __name__ == "__main__":
    main()
