#!/usr/bin/env python3

from pathlib import Path
import os


DOWNLOAD_DIR = Path("/ppio_net0/download")
HF_ENDPOINT = None  # Example: "https://hf-mirror.com"
HF_TOKEN = None  # None means using the HF_TOKEN environment variable.
MAX_WORKERS = 32
LOCAL_FILES_ONLY = False
FORCE_DOWNLOAD = False

MODELS = [
    {
        "repo_id": "Qwen/Qwen2-VL-2B-Instruct",
        "local_dir": "Qwen2-VL-2B-Instruct",
    },
    # {
    #     "repo_id": "Qwen/Qwen3-VL-2B-Instruct",
    #     "local_dir": "Qwen3-VL-2B-Instruct",
    # },
    # {
    #     "repo_id": "OpenGVLab/InternVL3_5-2B-HF",
    #     "local_dir": "InternVL3_5-2B-HF",
    # },
    # {
    #     "repo_id": "llava-hf/llava-1.5-7b-hf",
    #     "local_dir": "llava-1.5-7b-hf",
    # },
]


def main() -> None:
    if HF_ENDPOINT:
        os.environ["HF_ENDPOINT"] = HF_ENDPOINT

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Please install huggingface_hub before running this tool.") from exc

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    token = HF_TOKEN or os.environ.get("HF_TOKEN")

    for model in MODELS:
        repo_id = model["repo_id"]
        local_dir = DOWNLOAD_DIR / model["local_dir"]
        print(f"Downloading {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_dir),
            token=token,
            max_workers=MAX_WORKERS,
            local_files_only=LOCAL_FILES_ONLY,
            force_download=FORCE_DOWNLOAD,
        )
        print(f"Finished {repo_id}")


if __name__ == "__main__":
    main()
