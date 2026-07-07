#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/continual_sft/qwen3_vl.yaml 2>&1 | tee logs/train_continual_sft_qwen3_vl.log
