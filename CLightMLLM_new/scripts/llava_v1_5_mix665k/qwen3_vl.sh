#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/llava_v1_5_mix665k/qwen3_vl.yaml 2>&1 | tee logs/train_llava_v1_5_mix665k_qwen3_vl.log
