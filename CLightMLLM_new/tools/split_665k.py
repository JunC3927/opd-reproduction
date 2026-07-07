#!/usr/bin/env python3

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a JSON list by the first folder in its image path.")
    parser.add_argument("--input_json", required=True, help="Input JSON file containing a top-level list of samples.")
    parser.add_argument("--output_dir", required=True, help="Directory to write split JSON files into.")
    parser.add_argument("--image_col", default="image", help="Column containing image path(s).")
    parser.add_argument("--textonly_name", default="textonly", help="Output group name for samples without images.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear output_dir before writing. By default, the command refuses to write into a non-empty directory.",
    )
    return parser.parse_args()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output path exists but is not a directory: {output_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def validate_output_dir(input_json: Path, output_dir: Path) -> None:
    input_path = input_json.resolve()
    output_path = output_dir.resolve()
    if input_path == output_path or input_path.is_relative_to(output_path):
        raise ValueError(
            f"output_dir {output_dir} contains input_json {input_json}. "
            "Use a dedicated subdirectory, e.g. .../json/split_by_image_folder."
        )


def safe_filename(name: str) -> str:
    name = name.strip() or "empty"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def image_paths(value: Any) -> list[str]:
    if value is None:
        return []

    values = value if isinstance(value, list) else [value]
    paths: list[str] = []
    for item in values:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict):
            path = item.get("path") or item.get("file_name") or item.get("filename")
            if isinstance(path, str):
                paths.append(path)

    return paths


def first_folder(path: str) -> str | None:
    normalized = path.replace("\\", "/").strip()
    parts = [part for part in normalized.split("/") if part and part != "."]
    return parts[0] if parts else None


def group_names(row: dict[str, Any], image_col: str, textonly_name: str) -> list[str]:
    folders = {folder for path in image_paths(row.get(image_col)) if (folder := first_folder(path))}
    if not folders:
        return [textonly_name]
    return sorted(folders)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    if "id" not in row or row["id"] is None:
        return row

    normalized = dict(row)
    normalized["id"] = str(normalized["id"])
    return normalized


def write_json(path: Path, data: Any, indent: int | None = None) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    validate_output_dir(input_json, output_dir)

    with input_json.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise TypeError(f"{input_json} must be a JSON list.")

    prepare_output_dir(output_dir, overwrite=args.overwrite)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_counts: Counter[str] = Counter()

    for row in tqdm(rows, desc="split", unit="sample", dynamic_ncols=True):
        if not isinstance(row, dict):
            raise TypeError(f"sample is not an object: {type(row).__name__}")

        normalized = normalize_row(row)
        for group in group_names(row, args.image_col, args.textonly_name):
            buckets[group].append(normalized)
            group_counts[group] += 1

    for group in sorted(buckets):
        filename = f"{safe_filename(group)}.json"
        path = output_dir / filename
        write_json(path, buckets[group])

    print(f"Done. total={len(rows)}, groups={len(buckets)}")
    print(f"Output dir: {output_dir}")
    for group, count in group_counts.most_common():
        print(f"  {group}: {count}")


if __name__ == "__main__":
    main()
