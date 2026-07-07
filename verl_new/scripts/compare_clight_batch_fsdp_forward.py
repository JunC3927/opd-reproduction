import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl.models.transformers.monkey_patch import apply_monkey_patch  # noqa: E402
from verl.models.transformers.qwen3_vl import get_rope_index as get_qwen3_vl_rope_index  # noqa: E402
from verl.utils.model import get_hf_auto_model_class  # noqa: E402


SEQUENCE_EXCLUDED_KEYS = {
    "input_ids",
    "attention_mask",
    "labels",
    "position_ids",
    "rope_deltas",
    "token_type_ids",
    "mm_token_type_ids",
    "prompt_input_ids",
    "prompt_attention_mask",
    "prompt_mm_token_type_ids",
    "prompt_token_type_ids",
    "vllm_images",
}


def parse_torch_dtype(value: str | None) -> torch.dtype | str:
    if value is None or value == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return mapping[value]


def setup_dist(require_process_group: bool) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if (world_size > 1 or require_process_group) and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
            dist.init_process_group(backend=backend)
        else:
            init_file = Path(tempfile.gettempdir()) / "verl_single_rank_fsdp_init"
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


def torch_load(path: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def move_tensors_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors_to_device(item, device) for key, item in value.items()}
    return value


def build_mm_token_type_ids(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    mm_token_type_ids = torch.zeros_like(input_ids)
    config = getattr(model, "config", None)
    image_token_id = getattr(config, "image_token_id", None)
    video_token_id = getattr(config, "video_token_id", None)

    if image_token_id is not None:
        mm_token_type_ids = mm_token_type_ids.masked_fill(input_ids == image_token_id, 1)
    if video_token_id is not None:
        mm_token_type_ids = mm_token_type_ids.masked_fill(input_ids == video_token_id, 2)
    return mm_token_type_ids


def sequence_model_kwargs(
    model: torch.nn.Module,
    batch: dict[str, Any],
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    pass_mm_token_type_ids: bool = False,
) -> dict[str, Any]:
    kwargs = {
        key: value
        for key, value in batch.items()
        if torch.is_tensor(value) and key not in SEQUENCE_EXCLUDED_KEYS
    }
    kwargs["input_ids"] = sequences
    kwargs["attention_mask"] = attention_mask
    kwargs["use_cache"] = False
    if position_ids is not None:
        kwargs["position_ids"] = position_ids

    if pass_mm_token_type_ids and ("image_grid_thw" in kwargs or "video_grid_thw" in kwargs):
        kwargs["mm_token_type_ids"] = build_mm_token_type_ids(model, sequences)
    return kwargs


def has_multimodal_inputs(batch: dict[str, Any]) -> bool:
    return "image_grid_thw" in batch or "video_grid_thw" in batch


def build_verl_qwen3_vl_position_ids(
    processor: Any,
    batch: dict[str, Any],
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if sequences.shape[0] != 1:
        raise NotImplementedError("This probe currently rebuilds Qwen3-VL position_ids for batch_size=1.")
    if "image_grid_thw" not in batch and "video_grid_thw" not in batch:
        return None

    vision_position_ids = get_qwen3_vl_rope_index(
        processor,
        input_ids=sequences[0],
        image_grid_thw=batch.get("image_grid_thw"),
        video_grid_thw=batch.get("video_grid_thw"),
        attention_mask=attention_mask[0],
    )  # (3, seq_len)

    if mode == "vision3":
        return vision_position_ids.unsqueeze(1)

    valid_mask = attention_mask[0].bool()
    text_position_ids = torch.ones(
        (1, sequences.shape[1]),
        dtype=torch.long,
        device=sequences.device,
    )
    text_position_ids[0, valid_mask] = torch.arange(
        valid_mask.sum().item(),
        dtype=torch.long,
        device=sequences.device,
    )

    if mode != "verl4":
        raise ValueError(f"Unsupported position_ids mode: {mode}")

    # verl FSDP's padded VLM path passes position_ids as (4, batch, seq_len).
    return torch.cat((text_position_ids, vision_position_ids), dim=0).unsqueeze(1)


def response_token_logps(logits: torch.Tensor, sequences: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].float()
    shift_labels = sequences[:, 1:]
    logps = F.log_softmax(shift_logits, dim=-1)
    token_logps = torch.gather(logps, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_logps[response_mask.bool()].detach().cpu()


def strip_left_padding(
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if sequences.shape[0] != 1:
        raise NotImplementedError("This probe currently strips left padding for batch_size=1.")

    valid_indices = attention_mask[0].bool().nonzero(as_tuple=False).flatten()
    stripped_sequences = sequences[:, valid_indices]
    stripped_attention_mask = torch.ones_like(stripped_sequences, dtype=attention_mask.dtype)
    stripped_response_mask = torch.zeros(
        (1, stripped_sequences.shape[1] - 1),
        dtype=response_mask.dtype,
        device=response_mask.device,
    )

    for stripped_token_idx in range(1, stripped_sequences.shape[1]):
        original_token_idx = int(valid_indices[stripped_token_idx].item())
        if original_token_idx > 0:
            stripped_response_mask[0, stripped_token_idx - 1] = response_mask[0, original_token_idx - 1]

    return stripped_sequences, stripped_attention_mask, stripped_response_mask, valid_indices


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"{name} shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    diff = (a.float() - b.float()).abs()
    return {
        f"{name}_max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        f"{name}_mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def print_stats(title: str, stats: dict[str, float]) -> None:
    print(title)
    for key, value in stats.items():
        print(f"{key}={value}")


def format_token(tokenizer: Any, token_id: int) -> str:
    try:
        token = tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        token = str(token_id)
    return repr(token)


def response_shift_indices(response_mask: torch.Tensor) -> list[int]:
    if response_mask.shape[0] != 1:
        raise NotImplementedError("Debug table currently supports batch_size=1.")
    return response_mask[0].bool().nonzero(as_tuple=False).flatten().tolist()


def topk_for_response(
    logits: torch.Tensor,
    response_mask: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if topk <= 0:
        return None, None
    shift_logits = logits[:, :-1].float()
    selected = shift_logits[response_mask.bool()]
    logps = F.log_softmax(selected, dim=-1)
    topk_logps, topk_ids = logps.topk(topk, dim=-1)
    return topk_ids.detach().cpu(), topk_logps.detach().cpu()


def format_topk(tokenizer: Any, ids: torch.Tensor | None, logps: torch.Tensor | None, row: int) -> str:
    if ids is None or logps is None:
        return "NA"
    values = []
    for token_id, logp in zip(ids[row].tolist(), logps[row].tolist(), strict=False):
        values.append(f"{token_id}:{format_token(tokenizer, int(token_id))}:{float(logp):.4f}")
    return " | ".join(values)


def install_forward_capture(model: torch.nn.Module) -> tuple[list[Any], dict[str, Any]]:
    base_model = getattr(model, "module", model)
    target_model = getattr(base_model, "model", None)
    language_model = getattr(target_model, "language_model", None) if target_model is not None else None
    captures: dict[str, Any] = {}
    handles = []

    class _RestoreAttr:
        def __init__(self, obj: Any, name: str, value: Any) -> None:
            self.obj = obj
            self.name = name
            self.value = value

        def remove(self) -> None:
            setattr(self.obj, self.name, self.value)

    def make_hook(name: str):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            captures[f"{name}_keys"] = sorted(kwargs.keys())
            position_ids = kwargs.get("position_ids")
            if torch.is_tensor(position_ids):
                captures[f"{name}_position_ids"] = position_ids.detach().cpu()
            input_ids = kwargs.get("input_ids")
            if torch.is_tensor(input_ids):
                captures[f"{name}_input_ids_shape"] = tuple(input_ids.shape)
            attention_mask = kwargs.get("attention_mask")
            if torch.is_tensor(attention_mask):
                captures[f"{name}_attention_mask_shape"] = tuple(attention_mask.shape)

        return hook

    if target_model is not None:
        handles.append(target_model.register_forward_pre_hook(make_hook("model"), with_kwargs=True))
        if hasattr(target_model, "compute_3d_position_ids"):
            original_compute_3d_position_ids = target_model.compute_3d_position_ids

            def wrapped_compute_3d_position_ids(*args: Any, **kwargs: Any) -> Any:
                output = original_compute_3d_position_ids(*args, **kwargs)
                if torch.is_tensor(output):
                    captures["model_computed_position_ids"] = output.detach().cpu()
                elif isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                    captures["model_computed_position_ids"] = output[0].detach().cpu()
                return output

            target_model.compute_3d_position_ids = wrapped_compute_3d_position_ids
            handles.append(_RestoreAttr(target_model, "compute_3d_position_ids", original_compute_3d_position_ids))
    if language_model is not None:
        handles.append(language_model.register_forward_pre_hook(make_hook("language_model"), with_kwargs=True))
    return handles, captures


def remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def captured_position_at(capture: dict[str, Any] | None, key: str, shift_idx: int) -> str:
    if capture is None:
        return "NA"
    position_ids = capture.get(key)
    if not torch.is_tensor(position_ids):
        return "None"
    if position_ids.dim() == 3:
        if position_ids.shape[0] in (3, 4):
            return str(position_ids[:, 0, shift_idx].tolist())
        return str(position_ids[0, :, shift_idx].tolist())
    if position_ids.dim() == 2:
        return str(position_ids[0, shift_idx].item())
    return f"shape={tuple(position_ids.shape)}"


def capture_summary(capture: dict[str, Any] | None) -> str:
    if capture is None:
        return "None"
    parts = []
    for key in sorted(capture):
        value = capture[key]
        if torch.is_tensor(value):
            parts.append(f"{key}:shape={tuple(value.shape)}")
        else:
            parts.append(f"{key}:{value}")
    return "; ".join(parts)


def print_response_debug_table(
    tokenizer: Any,
    sequences: torch.Tensor,
    response_mask: torch.Tensor,
    token_position_map: torch.Tensor,
    clight_logps: torch.Tensor | None,
    plain_debug: dict[str, torch.Tensor | None] | None,
    patched_debug: dict[str, torch.Tensor | None],
    plain_capture: dict[str, Any] | None,
    patched_capture: dict[str, Any] | None,
    clight_topk_ids: torch.Tensor | None,
    clight_topk_logps: torch.Tensor | None,
    max_rows: int,
) -> None:
    shift_indices = response_shift_indices(response_mask)
    plain_logps = None if plain_debug is None else plain_debug["response_logps"]
    patched_logps = patched_debug["response_logps"]
    plain_topk_ids = None if plain_debug is None else plain_debug.get("topk_ids")
    plain_topk_logps = None if plain_debug is None else plain_debug.get("topk_logps")
    patched_topk_ids = patched_debug.get("topk_ids")
    patched_topk_logps = patched_debug.get("topk_logps")

    print("=== response token debug ===")
    print(
        "row\tshift_idx\torig_shift_idx\ttoken_idx\torig_token_idx\ttoken_id\ttoken"
        "\tclight\tplain\tpatched\tplain-clight\tpatched-clight\tpatched-plain"
        "\tplain_model_pos\tplain_computed_pos\tpatched_model_pos\tpatched_lm_pos"
    )
    rows = min(len(shift_indices), max_rows)
    for row in range(rows):
        shift_idx = int(shift_indices[row])
        token_idx = shift_idx + 1
        orig_shift_idx = int(token_position_map[shift_idx].item())
        orig_token_idx = int(token_position_map[token_idx].item())
        token_id = int(sequences[0, token_idx].item())
        clight_value = None if clight_logps is None else float(clight_logps[row].item())
        plain_value = None if plain_logps is None else float(plain_logps[row].item())
        patched_value = float(patched_logps[row].item())
        plain_pos = captured_position_at(plain_capture, "model_position_ids", shift_idx)
        plain_computed_pos = captured_position_at(plain_capture, "model_computed_position_ids", shift_idx)
        patched_model_pos = captured_position_at(patched_capture, "model_position_ids", shift_idx)
        patched_lm_pos = captured_position_at(patched_capture, "language_model_position_ids", shift_idx)
        plain_minus_clight = "NA" if clight_value is None or plain_value is None else f"{plain_value - clight_value:.6f}"
        patched_minus_clight = "NA" if clight_value is None else f"{patched_value - clight_value:.6f}"
        patched_minus_plain = "NA" if plain_value is None else f"{patched_value - plain_value:.6f}"
        print(
            f"{row}\t{shift_idx}\t{orig_shift_idx}\t{token_idx}\t{orig_token_idx}\t{token_id}\t"
            f"{format_token(tokenizer, token_id)}\t"
            f"{'NA' if clight_value is None else f'{clight_value:.6f}'}\t"
            f"{'NA' if plain_value is None else f'{plain_value:.6f}'}\t"
            f"{patched_value:.6f}\t{plain_minus_clight}\t{patched_minus_clight}\t{patched_minus_plain}\t"
            f"{plain_pos}\t{plain_computed_pos}\t{patched_model_pos}\t{patched_lm_pos}"
        )
        print(f"  clight_topk: {format_topk(tokenizer, clight_topk_ids, clight_topk_logps, row)}")
        print(f"  plain_topk:  {format_topk(tokenizer, plain_topk_ids, plain_topk_logps, row)}")
        print(f"  patched_topk:{format_topk(tokenizer, patched_topk_ids, patched_topk_logps, row)}")
    if len(shift_indices) > rows:
        print(f"... skipped {len(shift_indices) - rows} response debug rows")


def load_model(
    model_path: str,
    torch_dtype: torch.dtype | str,
    trust_remote_code: bool,
    local_files_only: bool,
    attn_implementation: str | None,
) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    auto_class = get_hf_auto_model_class(config)
    kwargs = {
        "pretrained_model_name_or_path": model_path,
        "torch_dtype": torch_dtype,
        "config": config,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = auto_class.from_pretrained(**kwargs)
    model.config.use_cache = False
    return model


def forward_response_debug(
    model: torch.nn.Module,
    batch: dict[str, Any],
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    pass_mm_token_type_ids: bool = False,
    topk: int = 5,
) -> dict[str, torch.Tensor | None]:
    handles, capture = install_forward_capture(model)
    with torch.no_grad():
        try:
            outputs = model(
                **sequence_model_kwargs(
                    model,
                    batch,
                    sequences,
                    attention_mask,
                    position_ids,
                    pass_mm_token_type_ids=pass_mm_token_type_ids,
                )
            )
        finally:
            remove_hooks(handles)
        if not hasattr(outputs, "logits") or outputs.logits is None:
            raise RuntimeError("Model output does not contain logits; disable fused kernels for this probe.")
        topk_ids, topk_logps = topk_for_response(outputs.logits, response_mask, topk=topk)
        return {
            "response_logps": response_token_logps(outputs.logits, sequences, response_mask),
            "topk_ids": topk_ids,
            "topk_logps": topk_logps,
            "capture": capture,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a CLight OPD batch with verl-patched HF/FSDP forward.")
    parser.add_argument("--batch", required=True, help="Path produced by CLight tools/dump_opd_forward_batch.py.")
    parser.add_argument("--model", default=None, help="Override model path. Defaults to payload['model_name_or_path'].")
    parser.add_argument("--device", default=None, help="Defaults to cuda:LOCAL_RANK when CUDA is available.")
    parser.add_argument("--torch-dtype", default=None, help="Defaults to payload['torch_dtype'], e.g. bfloat16.")
    parser.add_argument("--attn-implementation", default=None, help="Optional HF attn_implementation override.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--use-fsdp", action="store_true")
    parser.add_argument("--skip-plain-hf", action="store_true", help="Useful for multi-card memory checks.")
    parser.add_argument("--use-remove-padding", action="store_true", help="Match verl rmpad patch path.")
    parser.add_argument("--no-verl-position-ids", action="store_true", help="Do not rebuild verl-style Qwen3-VL position_ids.")
    parser.add_argument(
        "--position-ids-mode",
        choices=["verl4", "vision3", "none"],
        default="verl4",
        help="Which position_ids layout to pass into verl-patched Qwen3-VL.",
    )
    parser.add_argument("--strip-left-padding", action="store_true", help="Strip dense left padding before verl-patched forward.")
    parser.add_argument("--pass-mm-token-type-ids", action="store_true", help="Pass mm_token_type_ids to the model forward.")
    parser.add_argument(
        "--restore-hf-vision-pos-embed",
        action="store_true",
        help="After verl monkey patch, restore HF Qwen3-VL vision fast_pos_embed_interpolate.",
    )
    parser.add_argument("--debug-response-topk", type=int, default=5)
    parser.add_argument("--debug-max-rows", type=int, default=32)
    parser.add_argument("--ulysses-sp-size", type=int, default=1)
    args = parser.parse_args()

    os.chdir(ROOT)
    rank, world_size, local_rank = setup_dist(require_process_group=args.use_fsdp)
    device = torch.device(args.device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    try:
        payload = torch_load(args.batch)
        model_path = args.model or payload["model_name_or_path"]
        torch_dtype = parse_torch_dtype(args.torch_dtype or payload.get("torch_dtype", "auto"))
        batch = move_tensors_to_device(payload["batch"], device)
        sequences = payload["sequences"].to(device)
        attention_mask = payload["attention_mask"].to(device)
        response_mask = payload["response_mask"].to(device)
        token_position_map = torch.arange(sequences.shape[1], device=device)
        if args.strip_left_padding:
            sequences, attention_mask, response_mask, token_position_map = strip_left_padding(
                sequences=sequences,
                attention_mask=attention_mask,
                response_mask=response_mask,
            )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        verl_position_ids = None
        position_ids_mode = "none" if args.no_verl_position_ids else args.position_ids_mode
        if position_ids_mode != "none":
            verl_position_ids = build_verl_qwen3_vl_position_ids(
                processor=processor,
                batch=batch,
                sequences=sequences,
                attention_mask=attention_mask,
                mode=position_ids_mode,
            )

        if rank == 0:
            print("=== verl clight batch forward compare ===")
            print(f"rank={rank} world_size={world_size} device={device}")
            print(f"model={model_path}")
            print(f"batch_file={args.batch}")
            print(f"payload_config={payload.get('config')}")
            print(f"payload_reused_batch={payload.get('reused_batch')}")
            print(f"payload_use_verl_monkey_patch={payload.get('use_verl_monkey_patch')}")
            print(f"payload_verl_monkey_patch_applied={payload.get('verl_monkey_patch_applied')}")
            print(f"payload_verl_repo_path={payload.get('verl_repo_path')}")
            print(f"sequences_shape={tuple(sequences.shape)}")
            print(f"response_tokens={int(response_mask.sum().item())}")
            print(f"attention_active_tokens={int(attention_mask.sum().item())}")
            print(f"token_position_map_head={token_position_map[:16].detach().cpu().tolist()}")
            print(f"token_position_map_tail={token_position_map[-16:].detach().cpu().tolist()}")
            print(f"batch_tensor_shapes={{{', '.join(f'{key}: {tuple(value.shape)}' for key, value in batch.items() if torch.is_tensor(value))}}}")
            print(f"use_fsdp={args.use_fsdp}")
            print(f"use_remove_padding={args.use_remove_padding}")
            print(f"strip_left_padding={args.strip_left_padding}")
            print(f"pass_mm_token_type_ids={args.pass_mm_token_type_ids}")
            print(f"ulysses_sp_size={args.ulysses_sp_size}")
            print(f"position_ids_mode={position_ids_mode}")
            print(f"position_ids_shape={tuple(verl_position_ids.shape) if verl_position_ids is not None else None}")

        if not args.skip_plain_hf:
            plain_model = load_model(
                model_path=model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
                attn_implementation=args.attn_implementation,
            ).to(device)
            plain_model.eval()
            plain_debug = forward_response_debug(
                plain_model,
                batch,
                sequences,
                attention_mask,
                response_mask,
                pass_mm_token_type_ids=args.pass_mm_token_type_ids or has_multimodal_inputs(batch),
                topk=args.debug_response_topk,
            )
            del plain_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            plain_debug = None

        patched_model = load_model(
            model_path=model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        ).to(device)
        patched_model.eval()
        original_fast_pos_embed = None
        qwen3_vl_vision_cls = None
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

            qwen3_vl_vision_cls = Qwen3VLVisionModel
            original_fast_pos_embed = Qwen3VLVisionModel.fast_pos_embed_interpolate
        except Exception:
            pass
        apply_monkey_patch(
            model=patched_model,
            ulysses_sp_size=args.ulysses_sp_size,
            use_remove_padding=args.use_remove_padding,
            use_fused_kernels=False,
            fused_kernels_backend=None,
        )
        if args.restore_hf_vision_pos_embed and qwen3_vl_vision_cls is not None and original_fast_pos_embed is not None:
            qwen3_vl_vision_cls.fast_pos_embed_interpolate = original_fast_pos_embed
            print("restored_hf_vision_fast_pos_embed=True")
        if torch_dtype != "auto":
            patched_model.to(torch_dtype)

        if args.use_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            patched_model = FSDP(
                patched_model,
                device_id=device if device.type == "cuda" else None,
                use_orig_params=True,
            )

        patched_debug = forward_response_debug(
            patched_model,
            batch,
            sequences,
            attention_mask,
            response_mask,
            position_ids=verl_position_ids,
            pass_mm_token_type_ids=args.pass_mm_token_type_ids,
            topk=args.debug_response_topk,
        )

        if rank == 0:
            patched_logps = patched_debug["response_logps"]
            plain_logps = None if plain_debug is None else plain_debug["response_logps"]
            ok = bool(torch.isfinite(patched_logps).all().item())
            print(f"finite_verl_response_logps={ok}")
            clight_logps = None
            if "clight_response_logps" in payload:
                clight_logps = payload["clight_response_logps"].detach().cpu()
                print_stats(
                    "=== clight_hf_vs_verl_patched_response_logps ===",
                    compare("clight_hf_vs_verl_patched_response_logps", clight_logps, patched_logps),
                )
            else:
                print("clight_response_logps_missing=True")
            if plain_logps is not None:
                if "clight_response_logps" in payload:
                    print_stats(
                        "=== clight_hf_vs_plain_hf_response_logps ===",
                        compare("clight_hf_vs_plain_hf_response_logps", clight_logps, plain_logps),
                    )
                print_stats(
                    "=== plain_hf_vs_verl_patched_response_logps ===",
                    compare("plain_hf_vs_verl_patched_response_logps", plain_logps, patched_logps),
                )
            print("=== forward capture summary ===")
            print(f"plain_capture={capture_summary(None if plain_debug is None else plain_debug.get('capture'))}")
            print(f"patched_capture={capture_summary(patched_debug.get('capture'))}")
            print_response_debug_table(
                tokenizer=tokenizer,
                sequences=sequences,
                response_mask=response_mask,
                token_position_map=token_position_map,
                clight_logps=clight_logps,
                plain_debug=plain_debug,
                patched_debug=patched_debug,
                plain_capture=None if plain_debug is None else plain_debug.get("capture"),
                patched_capture=patched_debug.get("capture"),
                clight_topk_ids=payload.get("clight_response_topk_ids"),
                clight_topk_logps=payload.get("clight_response_topk_logps"),
                max_rows=args.debug_max_rows,
            )
            print("verl_clight_batch_forward_compare_ok=True")
    finally:
        cleanup_dist()


if __name__ == "__main__":
    main()
