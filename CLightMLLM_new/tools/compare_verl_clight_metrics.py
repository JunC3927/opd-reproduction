import argparse
import json
import math
from pathlib import Path
from typing import Any


METRIC_PAIRS = [
    ("loss", "actor/loss", "clight_loss_vs_verl_actor_loss"),
    ("loss", "actor/distillation/loss", "clight_loss_vs_verl_distill_loss"),
    ("grad_norm", "actor/grad_norm", "grad_norm"),
    ("param_update_max_abs", "actor/param_update_max_abs", "param_update_max_abs"),
    ("param_update_mean_abs", "actor/param_update_mean_abs", "param_update_mean_abs"),
    ("param_update_rel_mean", "actor/param_update_rel_mean", "param_update_rel_mean"),
    ("teacher_mass", "actor/distillation/teacher_mass", "teacher_mass"),
    ("student_mass", "actor/distillation/student_mass", "student_mass"),
    ("topk_overlap", "actor/distillation/overlap_ratio", "topk_overlap"),
]


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL record") from exc
    return records


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def extract_verl(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    out = {
        "global_step": record.get("global_step"),
        "epoch": record.get("epoch"),
    }
    out.update(metrics)
    return out


def index_by_step(records: list[dict[str, Any]], *, is_verl: bool, align: str) -> dict[int, dict[str, Any]]:
    indexed = {}
    for order, record in enumerate(records):
        item = extract_verl(record) if is_verl else record
        if align == "order":
            step_i = order
        else:
            step = item.get("global_step")
            if step is None:
                step = item.get("global_steps")
            try:
                step_i = int(step)
            except (TypeError, ValueError):
                step_i = order
        indexed[step_i] = item
    return indexed


def summarize_diffs(rows: list[tuple[int, float, float]]) -> dict[str, float]:
    diffs = [abs(left - right) for _, left, right in rows]
    rels = [diff / max(abs(right), 1e-12) for diff, (_, left, right) in zip(diffs, rows)]
    return {
        "count": float(len(rows)),
        "mean_left": sum(left for _, left, _ in rows) / len(rows),
        "mean_right": sum(right for _, _, right in rows) / len(rows),
        "mean_abs": sum(diffs) / len(diffs),
        "max_abs": max(diffs),
        "mean_rel": sum(rels) / len(rels),
        "max_rel": max(rels),
    }


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.4e}"
    return f"{value:.6f}"


def print_summary(clight: dict[int, dict[str, Any]], verl: dict[int, dict[str, Any]]) -> None:
    common_steps = sorted(set(clight) & set(verl))
    print("=== metrics compare ===")
    print(f"clight_steps={len(clight)} verl_steps={len(verl)} common_steps={len(common_steps)}")
    if common_steps:
        print(f"step_range={common_steps[0]}..{common_steps[-1]}")
    missing_clight = sorted(set(verl) - set(clight))[:10]
    missing_verl = sorted(set(clight) - set(verl))[:10]
    if missing_clight:
        print(f"missing_in_clight_first10={missing_clight}")
    if missing_verl:
        print(f"missing_in_verl_first10={missing_verl}")
    print()

    print("metric,count,clight_mean,verl_mean,mean_abs,max_abs,mean_rel,max_rel")
    for clight_key, verl_key, label in METRIC_PAIRS:
        rows = []
        for step in common_steps:
            left = as_float(clight[step].get(clight_key))
            right = as_float(verl[step].get(verl_key))
            if left is None or right is None:
                continue
            rows.append((step, left, right))
        if not rows:
            print(f"{label},0,NA,NA,NA,NA,NA,NA")
            continue
        summary = summarize_diffs(rows)
        print(
            ",".join(
                [
                    label,
                    str(int(summary["count"])),
                    fmt(summary["mean_left"]),
                    fmt(summary["mean_right"]),
                    fmt(summary["mean_abs"]),
                    fmt(summary["max_abs"]),
                    fmt(summary["mean_rel"]),
                    fmt(summary["max_rel"]),
                ]
            )
        )


def print_step_table(clight: dict[int, dict[str, Any]], verl: dict[int, dict[str, Any]], max_rows: int) -> None:
    common_steps = sorted(set(clight) & set(verl))[:max_rows]
    if not common_steps:
        return
    print()
    print("=== per-step core table ===")
    header = [
        "step",
        "c_loss",
        "v_loss",
        "c_grad",
        "v_grad",
        "c_up_mean",
        "v_up_mean",
        "c_overlap",
        "v_overlap",
    ]
    print(",".join(header))
    for step in common_steps:
        c = clight[step]
        v = verl[step]
        row = [
            str(step),
            fmt(as_float(c.get("loss"))),
            fmt(as_float(v.get("actor/loss"))),
            fmt(as_float(c.get("grad_norm"))),
            fmt(as_float(v.get("actor/grad_norm"))),
            fmt(as_float(c.get("param_update_mean_abs"))),
            fmt(as_float(v.get("actor/param_update_mean_abs"))),
            fmt(as_float(c.get("topk_overlap"))),
            fmt(as_float(v.get("actor/distillation/overlap_ratio"))),
        ]
        print(",".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CLight replay JSONL metrics with verl metrics JSONL.")
    parser.add_argument("--clight", required=True, help="JSONL produced by replay_verl_opd_trace.py --metrics-output.")
    parser.add_argument("--verl", required=True, help="JSONL produced by VERL_OPD_METRICS_DUMP.")
    parser.add_argument("--align", choices=("step", "order"), default="step")
    parser.add_argument("--max-step-rows", type=int, default=20)
    args = parser.parse_args()

    if not Path(args.clight).exists():
        raise FileNotFoundError(args.clight)
    if not Path(args.verl).exists():
        raise FileNotFoundError(args.verl)

    clight_records = read_jsonl(args.clight)
    verl_records = read_jsonl(args.verl)
    clight = index_by_step(clight_records, is_verl=False, align=args.align)
    verl = index_by_step(verl_records, is_verl=True, align=args.align)

    print_summary(clight, verl)
    print_step_table(clight, verl, args.max_step_rows)
    print("compare_verl_clight_metrics_ok=True")


if __name__ == "__main__":
    main()
