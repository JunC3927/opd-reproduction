#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/demo2k/qwen2_vl.yaml 2>&1 | tee logs/train_demo2k_qwen2_vl.log
