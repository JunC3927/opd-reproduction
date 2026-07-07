#!/usr/bin/env python3

import argparse
import json
import shutil
import urllib.request as ureq
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify OCR-VQA images.")
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--failed_log", required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--retries", type=int, required=True)
    parser.add_argument("--image_extension", required=True)
    return parser.parse_args()


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must be a JSON object keyed by image id.")
    return data


def verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.convert("RGB")


def download_file(url: str, output_path: Path, timeout: int) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with ureq.urlopen(url, timeout=timeout) as response, tmp_path.open("wb") as f:
            shutil.copyfileobj(response, f)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def download_one(
    key: str,
    item: dict[str, Any],
    output_dir: Path,
    image_extension: str,
    timeout: int,
    retries: int,
) -> tuple[str, str, str | None]:
    url = item.get("imageURL")
    if not isinstance(url, str) or not url:
        return key, "missing_url", None

    output_path = output_dir / f"{key}{image_extension}"
    for attempt in range(retries + 1):
        try:
            if attempt > 0 or not output_path.exists():
                download_file(url, output_path, timeout)
            verify_image(output_path)
            return key, "ok" if attempt == 0 else "re_downloaded", None
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            if attempt >= retries:
                return key, "failed", str(exc)

    return key, "failed", "unknown error"


def write_failures(path: Path, results: list[tuple[str, str, str | None]]) -> None:
    failures = [result for result in results if result[1] not in {"ok", "re_downloaded"}]
    if not failures:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for key, status, message in failures:
            row = {"key": key, "status": status, "message": message}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Failures written to: {path}")


def main() -> None:
    args = parse_args()
    dataset_json = Path(args.dataset_json)
    output_dir = Path(args.output_dir)
    failed_log = Path(args.failed_log)
    image_extension = args.image_extension if args.image_extension.startswith(".") else f".{args.image_extension}"

    data = load_dataset(dataset_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset JSON: {dataset_json}")
    print(f"Output dir: {output_dir}")
    print(f"Total images: {len(data)}")

    results: list[tuple[str, str, str | None]] = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                download_one,
                str(key),
                item,
                output_dir,
                image_extension,
                args.timeout,
                args.retries,
            ): str(key)
            for key, item in data.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            results.append(future.result())

    print("Summary:")
    for status, count in Counter(status for _, status, _ in results).most_common():
        print(f"  {status}: {count}")

    write_failures(failed_log, results)


if __name__ == "__main__":
    main()
