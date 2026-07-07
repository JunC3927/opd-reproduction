#!/usr/bin/env bash

set -euo pipefail

python tools/download_ocrvqa.py \
  --dataset_json /ppio_net0/datasets/llava_v1_5_mix665k/download/ocr_vqa/dataset.json \
  --output_dir /ppio_net0/datasets/llava_v1_5_mix665k/images/ocr_vqa/images \
  --failed_log /ppio_net0/datasets/llava_v1_5_mix665k/download/ocr_vqa/failed_ocrvqa_downloads.jsonl \
  --num_workers 256 \
  --timeout 30 \
  --retries 5 \
  --image_extension .jpg
