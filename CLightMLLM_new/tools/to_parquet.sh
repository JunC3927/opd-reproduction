#!/usr/bin/env bash

set -euo pipefail

python tools/to_parquet.py \
  --hf_dataset lmms-lab/LLaVA-NeXT-Data \
  --split train \
  --output_dir /ppio_net0/datasets/parquet/llava779k_debug_snappy \
  --image_col image \
  --messages_col conversations \
  --role_col from \
  --content_col value \
  --user_role human \
  --assistant_role gpt \
  --image_placeholder '<image>' \
  --bad_sample_limit 20 \
  --num_workers 32 \
  --batch_size 1000 \
  --rows_per_file 10000
