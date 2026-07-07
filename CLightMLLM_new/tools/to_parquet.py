#!/usr/bin/env python3

import argparse
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm import tqdm


def iter_json_or_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"{path}:{line_no} is not a JSON object.")
                yield row
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must be a JSON list or JSONL file.")

    for row in data:
        if not isinstance(row, dict):
            raise TypeError(f"An item in {path} is not a JSON object.")
        yield row


def load_records(args: argparse.Namespace) -> tuple[Iterable[dict[str, Any]], int | None]:
    if args.input_json:
        return iter_json_or_jsonl(args.input_json), None

    from datasets import Image, load_dataset  # type: ignore

    dataset = load_dataset(args.hf_dataset, split=args.split)
    if args.image_col in getattr(dataset, "column_names", []):
        dataset = dataset.cast_column(args.image_col, Image(decode=False))
    return dataset, len(dataset) if hasattr(dataset, "__len__") else None


def batched(records: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    iterator = iter(records)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def validate_messages(row: dict[str, Any], args: argparse.Namespace) -> int:
    messages = row.get(args.messages_col)
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("empty or invalid conversations")
    if len(messages) % 2 != 0:
        raise ValueError(f"odd number of conversation turns: {len(messages)}")

    placeholders = 0
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"conversation turn {idx} is not an object")

        expected_role = args.user_role if idx % 2 == 0 else args.assistant_role
        if message.get(args.role_col) != expected_role:
            raise ValueError(f"turn {idx} role {message.get(args.role_col)!r} != {expected_role!r}")

        content = message.get(args.content_col)
        if not isinstance(content, str):
            raise ValueError(f"turn {idx} content is not a string")
        placeholders += content.count(args.image_placeholder)

    return placeholders


def normalize_images(value: Any, image_root: str | None, allow_missing: bool) -> Any:
    if value is None:
        if allow_missing:
            return None
        raise ValueError("image is missing")

    if isinstance(value, list):
        return [normalize_one_image(item, image_root) for item in value]

    return normalize_one_image(value, image_root)


def normalize_one_image(image: Any, image_root: str | None) -> dict[str, Any]:
    if isinstance(image, dict):
        raw_bytes = image.get("bytes")
        image_path = image.get("path") or image.get("file_name") or image.get("filename")
        if isinstance(raw_bytes, memoryview):
            raw_bytes = raw_bytes.tobytes()
        if isinstance(raw_bytes, bytearray):
            raw_bytes = bytes(raw_bytes)
        if isinstance(raw_bytes, bytes):
            return {"bytes": raw_bytes, "path": str(image_path) if image_path is not None else None}
        if image_path is not None:
            return load_image_path(str(image_path), image_root)
        raise ValueError(f"image dict has neither bytes nor path: keys={list(image.keys())}")

    if isinstance(image, memoryview):
        return {"bytes": image.tobytes(), "path": None}
    if isinstance(image, bytearray):
        return {"bytes": bytes(image), "path": None}
    if isinstance(image, bytes):
        return {"bytes": image, "path": None}
    if isinstance(image, (str, os.PathLike)):
        return load_image_path(os.fspath(image), image_root)

    raise TypeError(f"unsupported image type: {type(image)!r}")


def load_image_path(original_path: str, image_root: str | None) -> dict[str, Any]:
    candidates = [os.path.expanduser(original_path)]
    if image_root:
        candidates.append(os.path.join(os.path.expanduser(image_root), original_path))

    for path in candidates:
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return {"bytes": f.read(), "path": original_path}

    raise FileNotFoundError(f"image file not found: {original_path}")


def count_images(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    return 1


def prepare_row(row: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, str | None]:
    try:
        placeholders = validate_messages(row, args)
        image_value = normalize_images(row.get(args.image_col), args.image_root, allow_missing=placeholders == 0)
        image_count = count_images(image_value)
        if placeholders != image_count:
            raise ValueError(f"image placeholders={placeholders}, images={image_count}")

        out = dict(row)
        out[args.image_col] = image_value
        return out, None
    except Exception as exc:
        return None, str(exc)


class ShardedParquetWriter:
    def __init__(self, output_dir: str | Path, prefix: str, rows_per_file: int, image_col: str) -> None:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        self.pa = pa
        self.pq = pq
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.rows_per_file = rows_per_file
        self.image_col = image_col
        self.schema = None
        self.shard_idx = 0
        self.files: list[Path] = []
        self.pending_rows: list[dict[str, Any]] = []

    def infer_schema(self, rows: list[dict[str, Any]]):
        image_struct = self.pa.struct([self.pa.field("bytes", self.pa.binary()), self.pa.field("path", self.pa.string())])
        image_is_list = any(isinstance(row.get(self.image_col), list) for row in rows)
        image_type = self.pa.list_(image_struct) if image_is_list else image_struct

        rows_without_image = [{key: value for key, value in row.items() if key != self.image_col} for row in rows]
        base_schema = self.pa.Table.from_pylist(rows_without_image).schema
        fields = [field for field in base_schema if field.name != self.image_col]
        fields.append(self.pa.field(self.image_col, image_type))
        return self.pa.schema(fields)

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        if self.schema is None:
            self.schema = self.infer_schema(rows)

        self.pending_rows.extend(rows)
        while len(self.pending_rows) >= self.rows_per_file:
            self.write_shard(self.pending_rows[: self.rows_per_file])
            del self.pending_rows[: self.rows_per_file]

    def write_shard(self, rows: list[dict[str, Any]]) -> None:
        path = self.output_dir / f"{self.prefix}-{self.shard_idx:05d}.parquet"
        table = self.pa.Table.from_pylist(rows, schema=self.schema)
        self.pq.write_table(table, str(path), compression="snappy", use_dictionary=True, row_group_size=len(rows))
        self.validate_shard(path)
        self.files.append(path)
        self.shard_idx += 1

    def validate_shard(self, path: Path) -> None:
        import pyarrow.dataset as ds  # type: ignore

        try:
            for _ in ds.dataset(str(path), format="parquet").to_batches():
                pass
        except Exception as exc:
            raise RuntimeError(f"Unreadable parquet shard: {path}") from exc

    def close(self) -> None:
        if self.pending_rows:
            self.write_shard(self.pending_rows)
            self.pending_rows = []


def reset_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        raise ValueError(f"refusing to clear unsafe output directory: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output path exists but is not a directory: {output_dir}")
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        print(f"Cleared output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf_dataset")
    source.add_argument("--input_json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--image_root")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--rows_per_file", type=int, default=10000)
    parser.add_argument("--prefix", default="train")
    parser.add_argument("--image_col", default="image")
    parser.add_argument("--messages_col", default="conversations")
    parser.add_argument("--role_col", default="from")
    parser.add_argument("--content_col", default="value")
    parser.add_argument("--user_role", default="human")
    parser.add_argument("--assistant_role", default="gpt")
    parser.add_argument("--image_placeholder", default="<image>")
    parser.add_argument("--bad_sample_limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    reset_output_dir(output_dir)

    records, total = load_records(args)
    if args.max_samples is not None:
        records = islice(records, args.max_samples)
        total = min(total, args.max_samples) if total is not None else args.max_samples

    writer = ShardedParquetWriter(output_dir, args.prefix, args.rows_per_file, args.image_col)
    seen = written = skipped = 0
    skip_reasons: Counter[str] = Counter()
    bad_samples: list[tuple[Any, str]] = []

    try:
        with tqdm(total=total, desc="parquet") as pbar:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                for batch in batched(records, args.batch_size):
                    results = list(executor.map(lambda row: prepare_row(row, args), batch))
                    valid_rows = []
                    for row, (prepared, error) in zip(batch, results):
                        seen += 1
                        if prepared is None:
                            skipped += 1
                            reason = error or "unknown error"
                            skip_reasons[reason] += 1
                            if len(bad_samples) < args.bad_sample_limit:
                                bad_samples.append((row.get("id", row.get("uid", "<no id>")), reason))
                            continue

                        written += 1
                        valid_rows.append(prepared)

                    writer.write_rows(valid_rows)
                    pbar.update(len(batch))
    finally:
        writer.close()

    print(f"Done. seen={seen}, written={written}, skipped={skipped}")
    print(f"Output dir: {output_dir}")
    print(f"Parquet shards: {len(writer.files)}")
    if writer.files:
        print(f"First shard: {writer.files[0]}")

    if skipped:
        print("Skipped reason summary:")
        for reason, count in skip_reasons.most_common(20):
            print(f"  {count}: {reason}")
        print("First bad samples:")
        for row_id, reason in bad_samples:
            print(f"  id={row_id}: {reason}")


if __name__ == "__main__":
    main()
