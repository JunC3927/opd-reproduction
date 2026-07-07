#!/usr/bin/env bash

set -euo pipefail

python tools/split_665k.py \
  --input_json /ppio_net0/datasets/llava_v1_5_mix665k/json/llava_v1_5_mix665k_filtered.json \
  --output_dir /ppio_net0/datasets/llava_v1_5_mix665k/json/task_incremental \
  --overwrite
