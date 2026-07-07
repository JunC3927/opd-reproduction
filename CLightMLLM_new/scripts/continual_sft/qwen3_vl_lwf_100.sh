#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
python train.py --config config/continual_sft/qwen3_vl_lwf_100.yaml 2>&1 | tee logs/train_continual_sft_qwen3_vl_lwf_100.log
