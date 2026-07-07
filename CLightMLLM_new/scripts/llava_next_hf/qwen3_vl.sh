#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/llava_next_hf/qwen3_vl.yaml 2>&1 | tee logs/train_llava_next_hf_qwen3_vl.log
