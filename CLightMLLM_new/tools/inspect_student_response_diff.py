import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compact(text: str, limit: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>..."


def token_window(tokens: list[int], center: int, radius: int) -> tuple[int, list[int]]:
    if center < 0:
        center = 0
    start = max(0, center - radius)
    end = min(len(tokens), center + radius + 1)
    return start, tokens[start:end]


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode and inspect VERL vs replayed student response differences.")
    parser.add_argument("--compare-jsonl", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--text-limit", type=int, default=1200)
    parser.add_argument("--window", type=int, default=24)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    rows = read_jsonl(args.compare_jsonl)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    print("=== inspect student response diff ===", flush=True)
    print(f"compare_jsonl={args.compare_jsonl}", flush=True)
    print(f"tokenizer={args.tokenizer}", flush=True)
    print(f"samples={len(rows)}", flush=True)

    for idx, row in enumerate(rows):
        new_ids = row.get("new_generated_token_ids")
        old_ids = row.get("old_generated_token_ids")
        if not isinstance(new_ids, list) or not isinstance(old_ids, list):
            raise RuntimeError(
                "Missing token ids in compare jsonl. Rerun compare_student_vllm_generate_with_verl_io.py "
                "with --save-token-ids."
            )

        first_diff = int(row.get("first_diff", -1))
        prefix = int(row.get("same_prefix_len", 0))
        same_ratio = float(row.get("same_ratio_common", 0.0))
        old_text = tokenizer.decode([int(x) for x in old_ids], skip_special_tokens=False)
        new_text = tokenizer.decode([int(x) for x in new_ids], skip_special_tokens=False)
        old_start, old_window = token_window([int(x) for x in old_ids], first_diff, args.window)
        new_start, new_window = token_window([int(x) for x in new_ids], first_diff, args.window)

        print("=" * 100, flush=True)
        print(
            f"sample={idx} trace_i={row.get('trace_i')} row={row.get('row')} "
            f"exact={row.get('exact_same')} prefix={prefix} first_diff={first_diff} "
            f"same_ratio={same_ratio:.6f}",
            flush=True,
        )
        print(f"file={Path(str(row.get('student_vllm_io'))).name}", flush=True)
        print("- old VERL response:", flush=True)
        print(compact(old_text, args.text_limit), flush=True)
        print("- new replay response:", flush=True)
        print(compact(new_text, args.text_limit), flush=True)
        print("- token window around first diff:", flush=True)
        print(f"old_start={old_start} old_ids={old_window}", flush=True)
        print(f"new_start={new_start} new_ids={new_window}", flush=True)


if __name__ == "__main__":
    main()
