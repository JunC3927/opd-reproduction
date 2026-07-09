import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_KEYS = (
    "loss",
    "teacher_mass",
    "student_mass",
    "topk_overlap",
    "grad_norm",
    "param_update_max_abs",
    "param_update_mean_abs",
    "param_update_rel_mean",
    "tokens",
    "samples",
    "local_samples_per_rank",
    "world_size",
    "global_step",
    "chunk_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload CLight VERL replay JSONL metrics to SwanLab.")
    parser.add_argument("--metrics-jsonl", required=True, help="Path to train_full_metrics.jsonl.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--mode", default=None, help="SwanLab mode, e.g. cloud/offline/local if supported.")
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--prefix", default="replay")
    parser.add_argument("--step-key", default="replay_update_step")
    parser.add_argument("--start-step", type=int, default=None, help="Only upload records with step >= this value.")
    parser.add_argument("--end-step", type=int, default=None, help="Only upload records with step <= this value.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--extra-key",
        action="append",
        default=[],
        help="Additional numeric JSON keys to upload. Can be passed multiple times.",
    )
    return parser.parse_args()


def iter_records(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_jsonl)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    import swanlab

    config = {
        "metrics_jsonl": str(metrics_path),
        "prefix": args.prefix,
        "step_key": args.step_key,
        "start_step": args.start_step,
        "end_step": args.end_step,
        "max_records": args.max_records,
    }
    init_kwargs = {
        "project": args.project,
        "experiment_name": args.experiment_name,
        "workspace": args.workspace,
        "mode": args.mode,
        "logdir": args.logdir,
        "save_dir": args.logdir,
        "config": config,
    }
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
    swanlab.init(**init_kwargs)

    keys = tuple(dict.fromkeys(DEFAULT_KEYS + tuple(args.extra_key)))
    uploaded = 0
    first_step = None
    last_step = None
    try:
        for _line_no, record in iter_records(metrics_path):
            step = record.get(args.step_key)
            if step is None:
                step = record.get("file_index")
                if step is not None:
                    step = int(step) + 1
            if step is None:
                raise KeyError(f"Record is missing step key {args.step_key!r} and file_index: {record}")
            step = int(step)
            if args.start_step is not None and step < args.start_step:
                continue
            if args.end_step is not None and step > args.end_step:
                continue

            metrics = {}
            for key in keys:
                value = record.get(key)
                if is_number(value):
                    metrics[f"{args.prefix}/{key}"] = value
            if metrics:
                swanlab.log(metrics, step=step)
                uploaded += 1
                first_step = step if first_step is None else min(first_step, step)
                last_step = step if last_step is None else max(last_step, step)
            if args.max_records is not None and uploaded >= args.max_records:
                break
    finally:
        finish = getattr(swanlab, "finish", None)
        if callable(finish):
            finish()

    print(
        "upload_replay_metrics_to_swanlab_ok=True "
        f"uploaded={uploaded} first_step={first_step} last_step={last_step}"
    )


if __name__ == "__main__":
    main()
