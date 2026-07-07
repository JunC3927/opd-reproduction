#!/usr/bin/env python3

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


def prepare_output_path(path: str, overwrite: bool) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        if output_path.is_dir():
            raise IsADirectoryError(f"Output path is a directory: {output_path}")
        if not overwrite:
            raise FileExistsError(f"output file already exists: {output_path}. Pass --overwrite to replace it.")
        output_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter LLaVA/ShareGPT JSON samples with bad conversations or images.")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--removed_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--print_removed_limit", type=int, default=200)
    parser.add_argument("--no_decode", action="store_true", help="Only check image file existence; skip PIL decode verification.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    return parser.parse_args()


def validate(idx: int, row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if not isinstance(row, dict):
        return False, f"sample is not an object: {type(row).__name__}"

    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return False, "empty or invalid conversations"

    if isinstance(conversations[0], dict) and conversations[0].get("from") == "system":
        conversations = conversations[1:]
    if not conversations or len(conversations) % 2 != 0:
        return False, f"odd number of user/assistant turns: {len(conversations)}"

    placeholders = 0
    for turn_idx, message in enumerate(conversations):
        if not isinstance(message, dict):
            return False, f"turn {turn_idx} is not an object"

        expected = "human" if turn_idx % 2 == 0 else "gpt"
        if message.get("from") != expected:
            return False, f"turn {turn_idx} role {message.get('from')!r} != {expected!r}"

        content = message.get("value")
        if not isinstance(content, str):
            return False, f"turn {turn_idx} value is not a string"
        placeholders += content.count("<image>")

    images = row.get("image")
    images = [] if images is None else images if isinstance(images, list) else [images]
    if len(images) != placeholders:
        return False, f"image placeholders={placeholders}, images={len(images)}"

    for image in images:
        if not isinstance(image, str):
            return False, f"unsupported image value type: {type(image).__name__}"
        path = os.path.expanduser(image)
        if not os.path.isabs(path):
            path = os.path.join(os.path.expanduser(args.image_root), image)
        if not os.path.isfile(path):
            return False, f"image file not found: {path}"
        if not args.no_decode:
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as exc:
                return False, f"bad image file: {path}: {exc}"

    return True, ""


def main() -> None:
    args = parse_args()
    with open(args.input_json, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise TypeError(f"{args.input_json} must be a JSON list.")

    for path in {args.output_json, args.removed_json}:
        prepare_output_path(path, overwrite=args.overwrite)

    kept, removed = [], []
    reasons: Counter[str] = Counter()

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        jobs = ((idx, row, args) for idx, row in enumerate(rows))
        results = pool.map(lambda item: (item[0], item[1], *validate(*item)), jobs)

        pbar = tqdm(results, total=len(rows), desc="filter", unit="sample", dynamic_ncols=True)
        for idx, row, ok, error in pbar:
            if ok:
                normalized_row = dict(row)
                if "id" in normalized_row and normalized_row["id"] is not None:
                    normalized_row["id"] = str(normalized_row["id"])
                kept.append(normalized_row)
                pbar.set_postfix(kept=len(kept), removed=len(removed))
                continue

            sample_id = (
                row.get("id") or row.get("uid") or row.get("question_id") or row.get("image") or idx
                if isinstance(row, dict)
                else idx
            )
            removed.append({"index": idx, "id": sample_id, "error": error, "sample": row})
            reasons[error] += 1
            if args.print_removed_limit <= 0 or len(removed) <= args.print_removed_limit:
                tqdm.write(f"[remove] index={idx} id={sample_id}: {error}")
            elif len(removed) == args.print_removed_limit + 1:
                tqdm.write(
                    f"[remove] reached print_removed_limit={args.print_removed_limit}; "
                    "further removed samples are only saved."
                )
            pbar.set_postfix(kept=len(kept), removed=len(removed))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)
        f.write("\n")
    with open(args.removed_json, "w", encoding="utf-8") as f:
        json.dump(removed, f, ensure_ascii=False)
        f.write("\n")

    print(f"Done. total={len(rows)}, kept={len(kept)}, removed={len(removed)}")
    print(f"Output JSON: {args.output_json}")
    print(f"Removed JSON: {args.removed_json}")
    if reasons:
        print("Top remove reasons:")
        for reason, count in reasons.most_common(20):
            print(f"  {count}: {reason}")


if __name__ == "__main__":
    main()
