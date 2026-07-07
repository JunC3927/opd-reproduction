import argparse
import importlib.util
import json
import os
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.module import DatasetBuilder, TemplateFactory, VLCollator  # noqa: E402
from src.hparams import (  # noqa: E402
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
from src.method.vllm_student import VLLMStudentRollout  # noqa: E402


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


class _NoopStrategy:
    def barrier(self) -> None:
        return None


class _DummyTrainer:
    local_rank = 0
    global_rank = 0
    is_global_zero = True
    strategy = _NoopStrategy()


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


def load_processor_tokenizer(model_args: ModelArguments, data_args: DataArguments) -> tuple[Any, Any]:
    kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "token": model_args.hf_hub_token,
        "local_files_only": model_args.local_files_only,
    }
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **kwargs)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("AutoProcessor must expose processor.tokenizer.")
    tokenizer.padding_side = model_args.padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.image_min_pixels is not None:
        processor.image_min_pixels = int(model_args.image_min_pixels)
        if hasattr(processor, "image_processor"):
            processor.image_processor.min_pixels = int(model_args.image_min_pixels)
    if model_args.image_max_pixels is not None:
        processor.image_max_pixels = int(model_args.image_max_pixels)
        if hasattr(processor, "image_processor"):
            processor.image_processor.max_pixels = int(model_args.image_max_pixels)
    return processor, tokenizer


def first_active_index(attention_mask: torch.Tensor) -> int:
    active = torch.nonzero(attention_mask.bool(), as_tuple=False).flatten()
    if active.numel() == 0:
        return int(attention_mask.numel())
    return int(active[0].item())


def build_vllm_prompts(
    rollout: VLLMStudentRollout,
    batch: dict[str, Any],
    *,
    image_token_id: int | None,
    video_token_id: int | None,
) -> list[Any]:
    prompt_ids = batch["prompt_input_ids"]
    prompt_attention_mask = batch["prompt_attention_mask"]
    images_per_sample = batch.get("vllm_images") or [[] for _ in range(prompt_ids.shape[0])]
    prompts = []
    for row_idx in range(prompt_ids.shape[0]):
        start = first_active_index(prompt_attention_mask[row_idx])
        active_prompt_ids = prompt_ids[row_idx, start:].detach().cpu().tolist()
        prompts.append(
            rollout._build_prompt(  # noqa: SLF001 - this script intentionally probes the rollout adapter.
                token_ids=active_prompt_ids,
                images=images_per_sample[row_idx],
                image_token_id=image_token_id,
                video_token_id=video_token_id,
            )
        )
    return prompts


def build_sampling_params(
    rollout: VLLMStudentRollout,
    method_args: MethodArguments,
    *,
    seed: int | None,
    max_new_tokens: int | None,
    do_sample: bool | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    eos_token_id: int | None,
) -> Any:
    sample = method_args.rollout_do_sample if do_sample is None else do_sample
    kwargs: dict[str, Any] = {
        "max_tokens": max_new_tokens or method_args.rollout_max_new_tokens,
        "temperature": (method_args.rollout_temperature if temperature is None else temperature) if sample else 0.0,
        "top_p": method_args.rollout_top_p if top_p is None else top_p,
    }
    resolved_top_k = method_args.rollout_top_k if top_k is None else top_k
    if resolved_top_k is not None:
        kwargs["top_k"] = resolved_top_k
    if eos_token_id is not None:
        kwargs["stop_token_ids"] = [int(eos_token_id)]
    if seed is not None:
        kwargs["seed"] = int(seed)
    return rollout._sampling_params_cls(**kwargs)  # noqa: SLF001


def generate_ids(rollout: VLLMStudentRollout, prompts: list[Any], sampling_params: Any) -> list[list[int]]:
    outputs = rollout.llm.generate(prompts, sampling_params, use_tqdm=False)
    completions: list[list[int]] = []
    for output in outputs:
        if not getattr(output, "outputs", None):
            completions.append([])
        else:
            completions.append(list(getattr(output.outputs[0], "token_ids", []) or []))
    return completions


def prefix_match_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        count += 1
    return count


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def decode(tokenizer: Any, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True).replace("\n", "\\n")


def compare_runs(left: list[list[int]], right: list[list[int]]) -> dict[str, float]:
    exact = [a == b for a, b in zip(left, right, strict=True)]
    first = [bool(a and b and a[0] == b[0]) for a, b in zip(left, right, strict=True)]
    prefix = [
        prefix_match_len(a, b) / max(min(len(a), len(b)), 1)
        for a, b in zip(left, right, strict=True)
    ]
    len_abs_diff = [abs(len(a) - len(b)) for a, b in zip(left, right, strict=True)]
    return {
        "exact_same_rate": mean([float(x) for x in exact]),
        "first_token_same_rate": mean([float(x) for x in first]),
        "prefix_match_ratio_mean": mean(prefix),
        "len_abs_diff_mean": mean([float(x) for x in len_abs_diff]),
    }


def dump_examples(tokenizer: Any, left: list[list[int]], right: list[list[int]], limit: int) -> None:
    print("=== examples ===")
    for row_idx, (a, b) in enumerate(zip(left, right, strict=True)):
        if row_idx >= limit:
            break
        common = prefix_match_len(a, b)
        print(
            f"row={row_idx} len_a={len(a)} len_b={len(b)} "
            f"first_same={bool(a and b and a[0] == b[0])} exact={a == b} prefix={common}"
        )
        print(f"  a: {decode(tokenizer, a[:120])}")
        print(f"  b: {decode(tokenizer, b[:120])}")


def write_jsonl(
    path: str,
    tokenizer: Any,
    left: list[list[int]],
    right: list[list[int]],
    *,
    seed_a: int | None,
    seed_b: int | None,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row_idx, (a, b) in enumerate(zip(left, right, strict=True)):
            row = {
                "row": row_idx,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "ids_a": a,
                "ids_b": b,
                "len_a": len(a),
                "len_b": len(b),
                "first_token_same": bool(a and b and a[0] == b[0]),
                "exact_same": a == b,
                "prefix_match_len": prefix_match_len(a, b),
                "text_a": decode(tokenizer, a),
                "text_b": decode(tokenizer, b),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_limit_mm_per_prompt(value: str | None) -> dict[str, int] | None:
    if value is None or value.strip().lower() in {"", "none", "null"}:
        return None
    result: dict[str, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --limit-mm-per-prompt item {item!r}; expected key=value.")
        key, raw_count = item.split("=", 1)
        result[key.strip()] = int(raw_count)
    return result


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether vLLM rollout is deterministic for fixed prompts and seed.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--second-seed", type=int, default=None, help="Defaults to --seed. Set a different value as a sanity check.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--distributed-executor-backend", default="mp")
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-log-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--limit-mm-per-prompt",
        default="image=1,video=0",
        help="vLLM multimodal profile limit, for example image=1,video=0. Use none to omit.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Optional vLLM CUDA device. Leave unset for this determinism probe so vLLM can "
            "initialize CUDA inside its own worker process from CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument("--visible-devices", default=None)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.chdir(ROOT)
    (
        cl_sft_args,
        data_args,
        _loader_args,
        method_args,
        model_args,
        _optimizer_args,
        _trainer_args,
        _tuning_args,
    ) = parse_yaml_args(args.config)
    if not cl_sft_args.stages:
        raise ValueError("cl_sft.stages is empty.")
    if args.model_path is not None:
        model_args = replace(model_args, model_name_or_path=args.model_path)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    stage = cl_sft_args.stages[0]
    data_args = replace(
        data_args,
        dataset=stage.dataset,
        max_samples=args.max_samples,
        preprocessing_num_workers=args.num_workers,
        overwrite_cache=not args.use_cache,
        log_first_sample=False,
    )
    processor, tokenizer = load_processor_tokenizer(model_args, data_args)
    template = TemplateFactory.from_args(tokenizer, data_args)
    dataset = DatasetBuilder(
        template=template,
        model_args=model_args,
        data_args=data_args,
        tokenizer=tokenizer,
        processor=processor,
        trainer=_DummyTrainer(),
    ).build()
    if len(dataset) == 0:
        raise RuntimeError("No examples survived preprocessing.")

    sample_count = min(args.batch_size, len(dataset))
    samples = [dataset[i] for i in range(sample_count)]
    collator = VLCollator(
        template=template,
        model=None,
        tokenizer=tokenizer,
        processor=processor,
        pad_to_multiple_of=8,
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        torch_dtype=parse_torch_dtype(model_args.torch_dtype),
    )
    batch = collator(samples)

    image_token_id = None
    if template.mm_plugin.image_token is not None:
        image_token_id = tokenizer.convert_tokens_to_ids(template.mm_plugin.image_token)
    video_token = getattr(template.mm_plugin, "video_token", None)
    video_token_id = tokenizer.convert_tokens_to_ids(video_token) if video_token is not None else None

    engine_dtype = args.dtype or model_args.torch_dtype
    engine_tp = args.tensor_parallel_size or method_args.rollout_vllm_tensor_parallel_size
    engine_gpu_memory = args.gpu_memory_utilization or method_args.rollout_vllm_gpu_memory_utilization
    engine_max_model_len = args.max_model_len or method_args.rollout_vllm_max_model_len

    print("=== vllm rollout determinism init ===")
    print(
        "backend_modules="
        f"vllm={module_available('vllm')} "
        f"xformers={module_available('xformers')} "
        f"flashinfer={module_available('flashinfer')}"
    )
    print(
        "engine_args="
        f"dtype={engine_dtype} "
        f"tp={engine_tp} "
        f"gpu_memory_utilization={engine_gpu_memory} "
        f"max_model_len={engine_max_model_len} "
        f"max_num_batched_tokens={args.max_num_batched_tokens} "
        f"max_num_seqs={args.max_num_seqs} "
        f"load_format={args.load_format} "
        f"distributed_executor_backend={args.distributed_executor_backend} "
        f"enable_chunked_prefill={args.enable_chunked_prefill} "
        f"enable_prefix_caching={args.enable_prefix_caching} "
        f"disable_log_stats={args.disable_log_stats}"
    )

    rollout = VLLMStudentRollout(
        model_path=model_args.model_name_or_path,
        tokenizer=tokenizer,
        torch_dtype=engine_dtype,
        trust_remote_code=model_args.trust_remote_code,
        tensor_parallel_size=engine_tp,
        gpu_memory_utilization=engine_gpu_memory,
        max_model_len=engine_max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        load_format=args.load_format,
        distributed_executor_backend=args.distributed_executor_backend,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=args.enable_prefix_caching,
        disable_log_stats=args.disable_log_stats,
        seed=args.seed,
        enforce_eager=method_args.rollout_vllm_enforce_eager if args.enforce_eager is None else args.enforce_eager,
        device=args.device,
        visible_devices=args.visible_devices or method_args.rollout_vllm_visible_devices,
        limit_mm_per_prompt=parse_limit_mm_per_prompt(args.limit_mm_per_prompt),
    )
    prompts = build_vllm_prompts(
        rollout,
        batch,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
    )

    seed_b = args.seed if args.second_seed is None else args.second_seed
    params_a = build_sampling_params(
        rollout,
        method_args,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
    )
    params_b = build_sampling_params(
        rollout,
        method_args,
        seed=seed_b,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
    )

    print("=== vllm rollout determinism ===")
    print(f"config={args.config}")
    print(f"model={model_args.model_name_or_path}")
    print(f"samples={sample_count}")
    print(f"prompt_width={tuple(batch['prompt_input_ids'].shape)}")
    print(f"image_counts={[len(images) for images in batch.get('vllm_images', [])]}")
    print(f"seed_a={args.seed} seed_b={seed_b}")
    print(f"sampling_a={params_a}")
    print(f"sampling_b={params_b}")
    print(f"limit_mm_per_prompt={parse_limit_mm_per_prompt(args.limit_mm_per_prompt)}")

    run_a = generate_ids(rollout, prompts, params_a)
    run_b = generate_ids(rollout, prompts, params_b)
    metrics = compare_runs(run_a, run_b)

    print("=== summary ===")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")
    print(f"deterministic_ok={metrics['exact_same_rate'] == 1.0}")
    dump_examples(tokenizer, run_a, run_b, args.examples)
    if args.output:
        write_jsonl(args.output, tokenizer, run_a, run_b, seed_a=args.seed, seed_b=seed_b)
        print(f"saved_output={args.output}")


if __name__ == "__main__":
    main()
