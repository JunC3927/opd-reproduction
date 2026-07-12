#!/usr/bin/env python3
"""Probe the standalone student vLLM rollout server.

Start ``tools/serve_vllm_student.py`` in a separate process first. This client
then builds one real OPD batch, asks the server to generate, optionally asks it
to hot-load an exported FSDP/HF state dict, and generates again.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from probe_student_vllm_rollout_sync import (  # noqa: E402
    decode_first_completion,
    load_probe_batch,
    log,
    move_batch_to_device,
    parse_yaml_args,
    resolve_device,
    resolve_path,
)
from src.method.vllm_student_client import RemoteStudentRollout  # noqa: E402
from src.model import ModelTuner, load_vision_language_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe standalone student vLLM rollout server.")
    parser.add_argument(
        "--config",
        default="config/continual_sft/qwen3_vl_opd_geo3k.yaml",
        help="CLight YAML config to reuse for model/data/method settings.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29588)
    parser.add_argument("--timeout-sec", type=float, default=1200.0)
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--client-device",
        default="cpu",
        help="Device used for the lightweight client-side HF model needed by VLCollator.",
    )
    parser.add_argument(
        "--state-dict-path",
        default=None,
        help="Optional exported student state dict. If set, the server syncs it before the second generate.",
    )
    parser.add_argument("--skip-before-generate", action="store_true")
    parser.add_argument("--shutdown-server", action="store_true")
    parser.add_argument(
        "--keep-gradient-checkpointing",
        action="store_true",
        help="Keep YAML gradient checkpointing. Default disables it because this probe does not train.",
    )
    return parser.parse_args()


def server_generate(
    *,
    client: RemoteStudentRollout,
    batch: dict[str, Any],
    method_args: Any,
    config: Any,
    pad_token_id: int,
    tokenizer: Any,
    prompt_width: int,
    label: str,
) -> tuple[torch.Tensor, int]:
    log(f"{label} start")
    start = time.time()
    sequences, weight_version = client.generate(
        batch=batch,
        method_args=method_args,
        image_token_id=getattr(config, "image_token_id", None),
        video_token_id=getattr(config, "video_token_id", None),
        pad_token_id=pad_token_id,
    )
    log(f"{label} done: seconds={time.time() - start:.3f}, weight_version={weight_version}")
    decode_first_completion(
        tokenizer=tokenizer,
        sequences=sequences,
        prompt_width=prompt_width,
        label=label,
    )
    return sequences, weight_version


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    log("standalone student vLLM server probe")
    log(f"config={config_path}")
    log(f"server={args.host}:{args.port}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    (
        cl_sft_args,
        data_args,
        loader_args,
        method_args,
        model_args,
        _optimizer_args,
        _trainer_args,
        tuning_args,
    ) = parse_yaml_args(config_path)

    if not cl_sft_args.stages:
        raise ValueError("cl_sft.stages is empty.")
    if args.stage_index < 0 or args.stage_index >= len(cl_sft_args.stages):
        raise IndexError(f"stage-index={args.stage_index} outside 0..{len(cl_sft_args.stages) - 1}")

    stage = cl_sft_args.stages[args.stage_index]
    data_args = replace(data_args, dataset=stage.dataset)
    loader_args = replace(
        loader_args,
        per_device_train_batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        shuffle=False,
    )
    method_args = replace(
        method_args,
        rollout_backend="hf",
        rollout_max_new_tokens=args.max_new_tokens,
    )
    if not args.keep_gradient_checkpointing:
        model_args = replace(model_args, gradient_checkpointing=False)

    client_device = resolve_device(args.client_device)
    log(
        "client HF load start: "
        f"model={model_args.model_name_or_path}, dtype={model_args.torch_dtype}, device={client_device}"
    )
    start = time.time()
    model, processor, tokenizer = load_vision_language_model(model_args, data_args.template)
    model = ModelTuner(tuning_args).apply(model)
    model.eval()
    model.to(client_device)
    log(f"client HF load done: seconds={time.time() - start:.3f}")

    batch = load_probe_batch(
        data_args=data_args,
        loader_args=loader_args,
        model_args=model_args,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        sample_index=args.sample_index,
        batch_size=args.batch_size,
    )
    batch = move_batch_to_device(batch, client_device)
    prompt_width = int(batch["prompt_input_ids"].shape[1])
    config = getattr(model, "config", None)

    client = RemoteStudentRollout(host=args.host, port=args.port, timeout=args.timeout_sec)
    log("ping server start")
    ping = client.ping()
    log(f"ping server done: {ping}")

    if not args.skip_before_generate:
        server_generate(
            client=client,
            batch=batch,
            method_args=method_args,
            config=config,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
            prompt_width=prompt_width,
            label="server generate before sync",
        )

    if args.state_dict_path:
        state_path = resolve_path(args.state_dict_path)
        log(f"server weight sync start: state_dict_path={state_path}")
        start = time.time()
        sync_response = client.sync_state_dict(state_path)
        log(f"server weight sync done: seconds={time.time() - start:.3f}, response={sync_response}")
    else:
        log("server weight sync skipped: no --state-dict-path")

    server_generate(
        client=client,
        batch=batch,
        method_args=method_args,
        config=config,
        pad_token_id=tokenizer.pad_token_id,
        tokenizer=tokenizer,
        prompt_width=prompt_width,
        label="server generate after sync",
    )

    if args.shutdown_server:
        log("shutdown server start")
        log(f"shutdown server done: {client.shutdown()}")
    log("RESULT=OK")


if __name__ == "__main__":
    main()
