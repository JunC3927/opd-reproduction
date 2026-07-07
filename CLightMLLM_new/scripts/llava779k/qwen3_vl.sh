#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/llava779k/qwen3_vl.yaml 2>&1 | tee logs/train_llava779k_qwen3_vl.log
