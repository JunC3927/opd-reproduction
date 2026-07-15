#!/usr/bin/env python3
"""Export a self-contained HTML viewer for LLaVA-style parquet datasets."""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_COLUMNS = ("id", "image", "conversations", "data_source")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline HTML viewer for LLaVA-style parquet datasets.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing dataset subdirectories with parquet files.",
    )
    parser.add_argument(
        "--output",
        default="llava_parquet_viewer.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=20,
        help="Maximum number of examples embedded per dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Parquet batch size used while sampling.",
    )
    parser.add_argument(
        "--include-datasets",
        nargs="*",
        default=None,
        help="Optional dataset directory names to include.",
    )
    return parser.parse_args()


def find_dataset_dirs(root: Path, include_names: set[str] | None) -> list[Path]:
    if any(root.glob("*.parquet")):
        return [root]
    dirs = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_dir():
            continue
        if include_names is not None and path.name not in include_names:
            continue
        if any(path.glob("*.parquet")):
            dirs.append(path)
    return dirs


def parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return str(value)


def image_payload(value: Any) -> dict[str, Any]:
    raw = None
    path = None
    if isinstance(value, dict):
        raw = value.get("bytes")
        path = value.get("path")
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)

    if raw is None:
        return {"data_url": None, "path": path, "bytes": 0, "mime": None}

    if isinstance(raw, bytearray):
        raw = bytes(raw)
    if not isinstance(raw, bytes):
        return {"data_url": None, "path": path, "bytes": 0, "mime": None}

    mime = "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        mime = "image/webp"
    elif raw.startswith(b"GIF"):
        mime = "image/gif"

    encoded = base64.b64encode(raw).decode("ascii")
    return {
        "data_url": f"data:{mime};base64,{encoded}",
        "path": path,
        "bytes": len(raw),
        "mime": mime,
    }


def sample_dataset(dataset_dir: Path, limit: int, batch_size: int) -> dict[str, Any]:
    parquet_files = sorted(dataset_dir.glob("*.parquet"))
    total_rows = sum(parquet_row_count(path) for path in parquet_files)
    samples = []
    seen_rows = 0
    schema_columns: list[str] = []

    for parquet_path in parquet_files:
        pf = pq.ParquetFile(parquet_path)
        file_columns = [name for name in DEFAULT_COLUMNS if name in pf.schema_arrow.names]
        if not schema_columns:
            schema_columns = list(pf.schema_arrow.names)
        if not file_columns:
            continue

        for batch in pf.iter_batches(batch_size=batch_size, columns=file_columns):
            rows = batch.to_pylist()
            for row in rows:
                if len(samples) >= limit:
                    break
                image = image_payload(row.get("image"))
                metadata = {
                    key: jsonable(value)
                    for key, value in row.items()
                    if key not in {"image", "conversations"}
                }
                samples.append(
                    {
                        "dataset": dataset_dir.name,
                        "file": parquet_path.name,
                        "row_index_approx": seen_rows,
                        "id": jsonable(row.get("id")),
                        "data_source": jsonable(row.get("data_source")),
                        "image": image,
                        "conversations": jsonable(row.get("conversations")),
                        "metadata": metadata,
                    }
                )
                seen_rows += 1
            if len(samples) >= limit:
                break
        if len(samples) >= limit:
            break

    return {
        "name": dataset_dir.name,
        "path": str(dataset_dir),
        "num_files": len(parquet_files),
        "num_rows": total_rows,
        "columns": schema_columns,
        "samples": samples,
    }


def build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    escaped_json = html.escape(data_json, quote=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLaVA Parquet Dataset Viewer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d9dee7;
      --text: #17202a;
      --muted: #637083;
      --accent: #2563eb;
      --soft: #eef4ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.94);
      backdrop-filter: blur(10px);
      padding: 12px 18px;
    }}
    header h1 {{
      margin: 0 0 4px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    header .sub {{ color: var(--muted); font-size: 13px; }}
    .layout {{
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      min-height: calc(100vh - 66px);
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
      overflow: auto;
    }}
    main {{ padding: 16px; min-width: 0; }}
    input, select, button {{
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--text);
      padding: 8px 10px;
    }}
    button {{
      cursor: pointer;
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }}
    .dataset-list {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 12px;
    }}
    .dataset-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      cursor: pointer;
      background: white;
    }}
    .dataset-card.active {{
      border-color: var(--accent);
      background: var(--soft);
    }}
    .dataset-name {{ font-weight: 700; margin-bottom: 4px; word-break: break-word; }}
    .small {{ color: var(--muted); font-size: 12px; }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
      align-items: center;
    }}
    .sample {{
      display: grid;
      grid-template-columns: minmax(260px, 42%) minmax(0, 1fr);
      gap: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      margin-bottom: 16px;
    }}
    .image-box {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      min-height: 220px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}
    .image-box img {{
      display: block;
      max-width: 100%;
      max-height: 520px;
      object-fit: contain;
    }}
    .meta {{
      display: grid;
      grid-template-columns: 120px minmax(0, 1fr);
      gap: 6px 10px;
      margin-bottom: 12px;
      font-size: 13px;
    }}
    .key {{ color: var(--muted); }}
    .value {{ overflow-wrap: anywhere; }}
    .turn {{
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 8px 0;
      overflow: hidden;
      background: white;
    }}
    .role {{
      background: #f0f3f8;
      border-bottom: 1px solid var(--line);
      padding: 7px 9px;
      font-weight: 700;
      font-size: 13px;
    }}
    pre {{
      margin: 0;
      padding: 9px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      font-size: 13px;
      line-height: 1.45;
    }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); max-height: 280px; }}
      .sample {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <script id="dataset-json" type="application/json">{escaped_json}</script>
  <header>
    <h1>LLaVA Parquet Dataset Viewer</h1>
    <div class="sub" id="summary"></div>
  </header>
  <div class="layout">
    <aside>
      <input id="datasetSearch" type="search" placeholder="Filter datasets" style="width: 100%;" />
      <div class="dataset-list" id="datasetList"></div>
    </aside>
    <main>
      <div class="toolbar">
        <select id="datasetSelect"></select>
        <input id="sampleSearch" type="search" placeholder="Search sampled text/id" style="min-width: 260px;" />
        <button id="prevBtn">Prev</button>
        <button id="nextBtn">Next</button>
        <span class="small" id="position"></span>
      </div>
      <div id="sampleRoot"></div>
    </main>
  </div>
  <script>
    const payload = JSON.parse(document.getElementById("dataset-json").textContent);
    const state = {{ datasetIndex: 0, sampleIndex: 0, datasetFilter: "", sampleFilter: "" }};

    function textOf(value) {{
      if (value == null) return "";
      if (typeof value === "string") return value;
      return JSON.stringify(value, null, 2);
    }}

    function sampleText(sample) {{
      return [
        sample.id,
        sample.data_source,
        sample.file,
        textOf(sample.metadata),
        textOf(sample.conversations)
      ].join("\\n").toLowerCase();
    }}

    function filteredDatasets() {{
      const q = state.datasetFilter.toLowerCase();
      return payload.datasets
        .map((dataset, index) => ({{ dataset, index }}))
        .filter(item => item.dataset.name.toLowerCase().includes(q));
    }}

    function filteredSamples(dataset) {{
      const q = state.sampleFilter.toLowerCase();
      if (!q) return dataset.samples;
      return dataset.samples.filter(sample => sampleText(sample).includes(q));
    }}

    function renderSidebar() {{
      const list = document.getElementById("datasetList");
      list.innerHTML = "";
      for (const item of filteredDatasets()) {{
        const card = document.createElement("div");
        card.className = "dataset-card" + (item.index === state.datasetIndex ? " active" : "");
        card.innerHTML = `
          <div class="dataset-name">${{escapeHtml(item.dataset.name)}}</div>
          <div class="small">${{item.dataset.num_rows.toLocaleString()}} rows · ${{item.dataset.num_files}} files · ${{item.dataset.samples.length}} embedded samples</div>
        `;
        card.onclick = () => {{
          state.datasetIndex = item.index;
          state.sampleIndex = 0;
          render();
        }};
        list.appendChild(card);
      }}
    }}

    function renderSelect() {{
      const select = document.getElementById("datasetSelect");
      select.innerHTML = "";
      payload.datasets.forEach((dataset, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = `${{dataset.name}} (${{dataset.num_rows}} rows)`;
        select.appendChild(option);
      }});
      select.value = String(state.datasetIndex);
    }}

    function renderSample() {{
      const root = document.getElementById("sampleRoot");
      const dataset = payload.datasets[state.datasetIndex];
      const samples = filteredSamples(dataset);
      if (!samples.length) {{
        root.innerHTML = `<div class="sample">No sampled rows match the current filter.</div>`;
        document.getElementById("position").textContent = "0 / 0";
        return;
      }}
      if (state.sampleIndex >= samples.length) state.sampleIndex = samples.length - 1;
      if (state.sampleIndex < 0) state.sampleIndex = 0;
      const sample = samples[state.sampleIndex];
      const imageHtml = sample.image && sample.image.data_url
        ? `<img src="${{sample.image.data_url}}" alt="sample image" />`
        : `<div class="small">No embedded image bytes. ${{escapeHtml(sample.image && sample.image.path || "")}}</div>`;
      const turns = Array.isArray(sample.conversations) ? sample.conversations : [];
      const turnsHtml = turns.map((turn, idx) => `
        <div class="turn">
          <div class="role">${{escapeHtml(turn.from || turn.role || ("turn " + idx))}}</div>
          <pre>${{escapeHtml(textOf(turn.value != null ? turn.value : turn.content))}}</pre>
        </div>
      `).join("");
      const metadataRows = [
        ["dataset", sample.dataset],
        ["file", sample.file],
        ["row approx", sample.row_index_approx],
        ["id", sample.id],
        ["source", sample.data_source],
        ["image", `${{sample.image ? sample.image.mime : ""}} · ${{sample.image ? sample.image.bytes : 0}} bytes`]
      ].map(([key, value]) => `
        <div class="key">${{escapeHtml(String(key))}}</div>
        <div class="value">${{escapeHtml(textOf(value))}}</div>
      `).join("");
      root.innerHTML = `
        <section class="sample">
          <div>
            <div class="image-box">${{imageHtml}}</div>
          </div>
          <div>
            <div class="meta">${{metadataRows}}</div>
            <h3 style="margin: 8px 0;">Conversations</h3>
            ${{turnsHtml || `<pre>${{escapeHtml(textOf(sample.conversations))}}</pre>`}}
          </div>
        </section>
      `;
      document.getElementById("position").textContent = `${{state.sampleIndex + 1}} / ${{samples.length}} sampled rows`;
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    function render() {{
      const totalRows = payload.datasets.reduce((acc, dataset) => acc + dataset.num_rows, 0);
      document.getElementById("summary").textContent =
        `${{payload.datasets.length}} datasets · ${{totalRows.toLocaleString()}} rows · generated from ${{payload.root}}`;
      renderSidebar();
      renderSelect();
      renderSample();
    }}

    document.getElementById("datasetSearch").addEventListener("input", event => {{
      state.datasetFilter = event.target.value;
      renderSidebar();
    }});
    document.getElementById("datasetSelect").addEventListener("change", event => {{
      state.datasetIndex = Number(event.target.value);
      state.sampleIndex = 0;
      render();
    }});
    document.getElementById("sampleSearch").addEventListener("input", event => {{
      state.sampleFilter = event.target.value;
      state.sampleIndex = 0;
      renderSample();
    }});
    document.getElementById("prevBtn").addEventListener("click", () => {{
      state.sampleIndex -= 1;
      renderSample();
    }});
    document.getElementById("nextBtn").addEventListener("click", () => {{
      state.sampleIndex += 1;
      renderSample();
    }});

    render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    include_names = set(args.include_datasets) if args.include_datasets else None

    if args.samples_per_dataset <= 0:
        raise ValueError("--samples-per-dataset must be positive.")
    if not root.exists():
        raise FileNotFoundError(root)

    datasets = []
    for dataset_dir in find_dataset_dirs(root, include_names):
        print(f"[viewer] sampling {dataset_dir.name}", flush=True)
        datasets.append(sample_dataset(dataset_dir, args.samples_per_dataset, args.batch_size))

    payload = {
        "root": str(root),
        "samples_per_dataset": args.samples_per_dataset,
        "datasets": datasets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(payload), encoding="utf-8")
    print(f"[viewer] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
