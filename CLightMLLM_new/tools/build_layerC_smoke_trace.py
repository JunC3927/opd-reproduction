import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from replay_verl_opd_trace import load_trace  # noqa: E402
from rescore_verl_teacher_requests import first_active_index  # noqa: E402
from src.method.vllm_teacher_client import RemoteTeacherScorer  # noqa: E402


def expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)
        else:
            raise FileNotFoundError(f"No files matched: {pattern}")
    return sorted(dict.fromkeys(paths))


def row_value(values: Any, row: int) -> Any:
    if values is None:
        return None
    item = values[row]
    if hasattr(item, "item"):
        try:
            return item.item()
        except ValueError:
            return item
    return item


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return [value]


def normalize_multi_modal_data(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        out = dict(value)
        if "images" in out and "image" not in out:
            out["image"] = out["images"]
        if "image" in out and "images" not in out:
            out["images"] = out["image"]
        return out
    images = to_list(value)
    if not images:
        return None
    return {"image": images, "images": images}


def images_from_non_tensor(non_tensor: dict[str, Any], row: int, source: str) -> list[Any]:
    if source == "vllm_images":
        value = row_value(non_tensor.get("vllm_images"), row)
        images = to_list(value)
        if images:
            return images

    mm_data = row_value(non_tensor.get("multi_modal_data"), row)
    if isinstance(mm_data, dict):
        images = mm_data.get("image", mm_data.get("images"))
        return to_list(images)
    return to_list(mm_data)


def mm_data_from_non_tensor(non_tensor: dict[str, Any], row: int) -> dict[str, Any] | None:
    return normalize_multi_modal_data(row_value(non_tensor.get("multi_modal_data"), row))


def mm_kwargs_from_non_tensor(non_tensor: dict[str, Any], row: int) -> dict[str, Any] | None:
    value = row_value(non_tensor.get("mm_processor_kwargs"), row)
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return None


def dedup_mm_tokens(ids: list[int], image_token_id: int | None, video_token_id: int | None) -> list[int]:
    mm_ids = {token for token in (image_token_id, video_token_id) if token is not None}
    if not mm_ids:
        return ids

    out: list[int] = []
    previous: int | None = None
    for token in ids:
        token = int(token)
        if token in mm_ids and token == previous:
            continue
        out.append(token)
        previous = token
    return out


def active_prompt_ids(
    prompts: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    row: int,
) -> list[int]:
    ids = prompts[row].cpu()
    mask = prompt_attention_mask[row].cpu().bool()
    return ids[mask].long().tolist()


def build_student_prompts(
    *,
    trace: dict[str, Any],
    rows: list[int],
    image_source: str,
    image_token_id: int | None,
    video_token_id: int | None,
) -> list[dict[str, Any]]:
    batch = trace["batch"]
    non_tensor = trace.get("non_tensor_batch", {})
    prompts = batch["prompts"]
    prompt_width = prompts.shape[1]
    prompt_attention_mask = batch["attention_mask"][:, :prompt_width]

    vllm_prompts: list[dict[str, Any]] = []
    for row in rows:
        prompt_ids = active_prompt_ids(prompts, prompt_attention_mask, row)
        final_prompt_ids = dedup_mm_tokens(prompt_ids, image_token_id, video_token_id)
        prompt: dict[str, Any] = {"prompt_token_ids": final_prompt_ids}
        images = images_from_non_tensor(non_tensor, row, image_source)
        if images:
            prompt["multi_modal_data"] = {"image": images}
        mm_kwargs = mm_kwargs_from_non_tensor(non_tensor, row)
        if mm_kwargs:
            prompt["mm_processor_kwargs"] = mm_kwargs
        vllm_prompts.append(prompt)
    return vllm_prompts


def pad_responses(
    outputs: list[Any],
    *,
    response_width: int,
    pad_token_id: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    responses = torch.full((len(outputs), response_width), pad_token_id, dtype=dtype)
    response_mask = torch.zeros((len(outputs), response_width), dtype=torch.long)
    lengths: list[int] = []
    for row, output in enumerate(outputs):
        token_ids = list(output.outputs[0].token_ids)
        token_ids = token_ids[:response_width]
        length = len(token_ids)
        lengths.append(length)
        if length:
            responses[row, :length] = torch.tensor(token_ids, dtype=dtype)
            response_mask[row, :length] = 1
    return responses, response_mask, lengths


def make_full_sequences(
    *,
    prompts: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.cat([prompts.cpu(), responses.cpu()], dim=1),
        torch.cat([prompt_attention_mask.cpu().long(), response_mask.cpu().long()], dim=1),
    )


def score_teacher(
    *,
    scorer: RemoteTeacherScorer,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    trace: dict[str, Any],
    rows: list[int],
    pad_token_id: int,
    image_token_id: int | None,
    video_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    non_tensor = trace.get("non_tensor_batch", {})
    images_batch = [images_from_non_tensor(non_tensor, row, "vllm_images") for row in rows]
    mm_data_batch = [mm_data_from_non_tensor(non_tensor, row) for row in rows]
    mm_kwargs_batch = [mm_kwargs_from_non_tensor(non_tensor, row) for row in rows]
    logps, ids = scorer.score(
        sequences=sequences,
        attention_mask=attention_mask,
        images_per_sample=images_batch,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        pad_token_id=pad_token_id,
        mm_processor_kwargs_per_sample=mm_kwargs_batch,
        multi_modal_data_per_sample=mm_data_batch,
    )
    return logps.cpu().float(), ids.cpu().long()


def zero_like_if_present(batch: dict[str, Any], key: str) -> None:
    value = batch.get(key)
    if torch.is_tensor(value):
        batch[key] = torch.zeros_like(value)


def response_start(trace: dict[str, Any]) -> int:
    batch = trace["batch"]
    prompts = batch.get("prompts")
    if torch.is_tensor(prompts):
        return int(prompts.shape[1])
    return int(batch["input_ids"].shape[1] - batch["response_mask"].shape[1])


def build_layerc_trace(
    *,
    trace: dict[str, Any],
    trace_path: str,
    trace_index: int,
    llm: LLM,
    sampling_params: SamplingParams,
    scorer: RemoteTeacherScorer,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    batch = trace["batch"]
    sample_count = int(batch["prompts"].shape[0])
    if args.max_rows_per_file > 0:
        sample_count = min(sample_count, args.max_rows_per_file)
    rows = list(range(sample_count))

    response_width = int(batch["responses"].shape[1])
    prompt_width = int(batch["prompts"].shape[1])
    prompt_attention_mask = batch["attention_mask"][:sample_count, :prompt_width].cpu()
    prompts_tensor = batch["prompts"][:sample_count].cpu()

    generated_outputs = []
    for start in range(0, sample_count, args.student_micro_batch_size):
        end = min(start + args.student_micro_batch_size, sample_count)
        vllm_prompts = build_student_prompts(
            trace=trace,
            rows=rows[start:end],
            image_source=args.image_source,
            image_token_id=args.image_token_id,
            video_token_id=args.video_token_id,
        )
        generated_outputs.extend(llm.generate(vllm_prompts, sampling_params, use_tqdm=False))

    responses, response_mask, generated_lengths = pad_responses(
        generated_outputs,
        response_width=response_width,
        pad_token_id=args.pad_token_id,
        dtype=batch["responses"].dtype,
    )
    input_ids, attention_mask = make_full_sequences(
        prompts=prompts_tensor,
        prompt_attention_mask=prompt_attention_mask,
        responses=responses,
        response_mask=response_mask,
    )

    teacher_logps = torch.empty((sample_count, input_ids.shape[1] - 1, args.topk), dtype=torch.float32)
    teacher_ids = torch.empty((sample_count, input_ids.shape[1] - 1, args.topk), dtype=torch.long)
    for start in range(0, sample_count, args.teacher_micro_batch_size):
        end = min(start + args.teacher_micro_batch_size, sample_count)
        logps, ids = score_teacher(
            scorer=scorer,
            sequences=input_ids[start:end],
            attention_mask=attention_mask[start:end],
            trace=trace,
            rows=rows[start:end],
            pad_token_id=args.pad_token_id,
            image_token_id=args.image_token_id,
            video_token_id=args.video_token_id,
        )
        teacher_logps[start:end] = logps
        teacher_ids[start:end] = ids

    out_batch: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out_batch[key] = value[:sample_count].clone()
        else:
            out_batch[key] = value

    out_batch["responses"] = responses.to(dtype=batch["responses"].dtype)
    out_batch["response_mask"] = response_mask.to(dtype=batch["response_mask"].dtype)
    out_batch["input_ids"] = input_ids.to(dtype=batch["input_ids"].dtype)
    out_batch["attention_mask"] = attention_mask.to(dtype=batch["attention_mask"].dtype)
    if "teacher_ids" in out_batch:
        out_batch["teacher_ids"] = teacher_ids.to(dtype=out_batch["teacher_ids"].dtype)
    else:
        out_batch["teacher_ids"] = teacher_ids.to(dtype=torch.int32)
    out_batch["teacher_logprobs"] = teacher_logps.to(dtype=batch.get("teacher_logprobs", teacher_logps).dtype)

    for key in (
        "advantages",
        "old_log_probs",
        "returns",
        "rm_scores",
        "rollout_log_probs",
        "token_level_rewards",
        "token_level_scores",
    ):
        zero_like_if_present(out_batch, key)

    out_trace = dict(trace)
    out_trace["batch"] = out_batch
    out_trace["trace_stage"] = "layerC_smoke_student_rollout_teacher_rescored"
    out_trace["layerC_smoke"] = {
        "source_trace": trace_path,
        "source_trace_index": trace_index,
        "student_model": args.student_model,
        "topk": args.topk,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_response_length,
        "image_source": args.image_source,
        "note": "Student responses were generated by CLight/vLLM, then teacher top-k was rescored by the remote vLLM teacher.",
    }

    start_pos = response_start(out_trace) - 1
    active = response_mask.bool()
    resp_teacher_logps = teacher_logps[:, start_pos : start_pos + response_width]
    resp_teacher_ids = teacher_ids[:, start_pos : start_pos + response_width]
    teacher_mass = resp_teacher_logps.exp().sum(dim=-1)
    metrics = {
        "trace_index": trace_index,
        "source_trace": trace_path,
        "samples": sample_count,
        "response_width": response_width,
        "generated_len_mean": float(torch.tensor(generated_lengths, dtype=torch.float32).mean().item()),
        "generated_len_min": int(min(generated_lengths) if generated_lengths else 0),
        "generated_len_max": int(max(generated_lengths) if generated_lengths else 0),
        "active_response_tokens": int(active.sum().item()),
        "teacher_mass_mean": float(teacher_mass[active].mean().item()) if active.any() else 0.0,
        "teacher_ids_shape": list(resp_teacher_ids.shape),
    }
    return out_trace, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Layer C smoke builder: generate fresh student rollouts from VERL trace prompts, "
            "rescore them with a remote vLLM teacher, and write replay-compatible OPD traces."
        )
    )
    parser.add_argument("traces", nargs="+", help="VERL/Layer-A trace dump(s) or glob pattern(s).")
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29577)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--student-micro-batch-size", type=int, default=4)
    parser.add_argument("--teacher-micro-batch-size", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--max-response-length", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=1537)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-source", choices=("vllm_images", "multi_modal_data"), default="vllm_images")
    parser.add_argument("--image-token-id", type=int, default=151655)
    parser.add_argument("--video-token-id", type=int, default=151656)
    parser.add_argument("--pad-token-id", type=int, default=151643)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-log-stats", action="store_true")
    parser.add_argument("--metrics-output", default=None)
    args = parser.parse_args()

    trace_paths = expand_paths(args.traces)
    if args.max_files > 0:
        trace_paths = trace_paths[: args.max_files]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_output) if args.metrics_output else output_dir / "layerC_smoke_metrics.jsonl"

    print("=== build Layer C smoke traces ===", flush=True)
    print(f"trace_count={len(trace_paths)}", flush=True)
    print(f"student_model={args.student_model}", flush=True)
    print(f"teacher={args.host}:{args.port}", flush=True)
    print(f"output_dir={output_dir}", flush=True)

    llm = LLM(
        model=args.student_model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        disable_log_stats=args.disable_log_stats,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    scorer = RemoteTeacherScorer(host=args.host, port=args.port, timeout=args.timeout, topk=args.topk)

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        for trace_index, trace_path in enumerate(trace_paths):
            trace = load_trace(trace_path)
            out_trace, metrics = build_layerc_trace(
                trace=trace,
                trace_path=trace_path,
                trace_index=trace_index,
                llm=llm,
                sampling_params=sampling_params,
                scorer=scorer,
                args=args,
            )
            output_path = output_dir / Path(trace_path).name
            torch.save(out_trace, output_path)
            metrics["output"] = str(output_path)
            metrics_file.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            metrics_file.flush()
            print(
                "trace={trace_index} | source={source_trace} | output={output} | "
                "samples={samples} | generated_len_mean={generated_len_mean:.2f} | "
                "teacher_mass_mean={teacher_mass_mean:.6f}".format(**metrics),
                flush=True,
            )

    print(f"metrics={metrics_path}", flush=True)
    print("build_layerC_smoke_trace_ok=True", flush=True)


if __name__ == "__main__":
    main()
