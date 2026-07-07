import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def safetensor_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found under {path}")
    return files


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = {}
    for file in safetensor_files(path):
        state.update(load_file(str(file), device="cpu"))
    return state


def tensor_stats(delta: torch.Tensor, base: torch.Tensor) -> dict[str, float]:
    delta = delta.float()
    base = base.float()
    abs_delta = delta.abs()
    abs_base = base.abs()
    return {
        "numel": float(delta.numel()),
        "max_abs": float(abs_delta.max().item()),
        "mean_abs": float(abs_delta.mean().item()),
        "rms": float(delta.pow(2).mean().sqrt().item()),
        "base_mean_abs": float(abs_base.mean().item()),
        "rel_mean_abs": float((abs_delta.mean() / abs_base.mean().clamp_min(1.0e-12)).item()),
        "changed_ratio": float(abs_delta.gt(0).float().mean().item()),
    }


def bucket_for_name(name: str) -> str:
    if name.startswith("visual.") or ".visual." in name:
        return "visual"
    if "lm_head" in name:
        return "lm_head"
    if "embed_tokens" in name:
        return "embed_tokens"
    if "language_model" in name or "model.layers" in name or ".layers." in name:
        return "language"
    return "other"


def add_weighted(total: dict[str, float], stats: dict[str, float]) -> None:
    numel = stats["numel"]
    total["numel"] = total.get("numel", 0.0) + numel
    total["max_abs"] = max(total.get("max_abs", 0.0), stats["max_abs"])
    for key in ("mean_abs", "rms", "base_mean_abs", "rel_mean_abs", "changed_ratio"):
        total[key] = total.get(key, 0.0) + stats[key] * numel


def finalize_weighted(total: dict[str, float]) -> dict[str, float]:
    numel = total.get("numel", 0.0)
    if numel <= 0:
        return total
    out = dict(total)
    for key in ("mean_abs", "rms", "base_mean_abs", "rel_mean_abs", "changed_ratio"):
        out[key] = out[key] / numel
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare parameter deltas between two HF safetensors checkpoints.")
    parser.add_argument("--base", required=True, help="Base model dir or safetensors file.")
    parser.add_argument("--target", required=True, help="Trained model dir or safetensors file.")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    base_path = Path(args.base).expanduser()
    target_path = Path(args.target).expanduser()
    print(f"loading_base={base_path}", flush=True)
    base_state = load_state_dict(base_path)
    print(f"loading_target={target_path}", flush=True)
    target_state = load_state_dict(target_path)

    common = sorted(set(base_state) & set(target_state))
    base_only = sorted(set(base_state) - set(target_state))
    target_only = sorted(set(target_state) - set(base_state))

    totals: dict[str, float] = {}
    buckets: dict[str, dict[str, float]] = {}
    rows = []
    skipped_shape = []

    for name in common:
        base = base_state[name]
        target = target_state[name]
        if base.shape != target.shape:
            skipped_shape.append((name, tuple(base.shape), tuple(target.shape)))
            continue
        if not torch.is_floating_point(base) or not torch.is_floating_point(target):
            continue
        stats = tensor_stats(target - base, base)
        row = {"name": name, **stats}
        rows.append(row)
        add_weighted(totals, stats)
        bucket = buckets.setdefault(bucket_for_name(name), {})
        add_weighted(bucket, stats)

    rows.sort(key=lambda item: item["mean_abs"], reverse=True)
    result = {
        "base": str(base_path),
        "target": str(target_path),
        "common_tensors": len(common),
        "base_only_count": len(base_only),
        "target_only_count": len(target_only),
        "shape_mismatch_count": len(skipped_shape),
        "overall": finalize_weighted(totals),
        "buckets": {key: finalize_weighted(value) for key, value in sorted(buckets.items())},
        "top_by_mean_abs": rows[: args.top_n],
    }

    print("=== overall ===")
    for key, value in result["overall"].items():
        print(f"{key}={value}")

    print("=== buckets ===")
    for bucket, stats in result["buckets"].items():
        pieces = " ".join(f"{key}={value}" for key, value in stats.items())
        print(f"{bucket}: {pieces}")

    print(f"=== top_by_mean_abs top_n={args.top_n} ===")
    for row in result["top_by_mean_abs"]:
        print(
            row["name"],
            f"mean_abs={row['mean_abs']}",
            f"max_abs={row['max_abs']}",
            f"rel_mean_abs={row['rel_mean_abs']}",
            f"changed_ratio={row['changed_ratio']}",
        )

    if result["base_only_count"] or result["target_only_count"] or result["shape_mismatch_count"]:
        print("=== key diagnostics ===")
        print(f"base_only_count={result['base_only_count']}")
        print(f"target_only_count={result['target_only_count']}")
        print(f"shape_mismatch_count={result['shape_mismatch_count']}")

    if args.json_output:
        output = Path(args.json_output).expanduser()
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved_json={output}")


if __name__ == "__main__":
    main()
