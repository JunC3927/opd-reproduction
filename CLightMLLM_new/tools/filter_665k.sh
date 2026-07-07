#!/usr/bin/env bash

set -euo pipefail

python tools/filter_665k.py \
  --input_json /ppio_net0/datasets/llava_v1_5_mix665k/json/llava_v1_5_mix665k.json \
  --output_json /ppio_net0/datasets/llava_v1_5_mix665k/json/llava_v1_5_mix665k_filtered.json \
  --removed_json /ppio_net0/datasets/llava_v1_5_mix665k/json/llava_v1_5_mix665k_removed.json \
  --image_root /ppio_net0/datasets/llava_v1_5_mix665k/images \
  --num_workers 64 \
  --print_removed_limit 200 \
  --overwrite
