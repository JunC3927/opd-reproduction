#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/continual_sft/qwen3_vl_opd_geo3k.yaml 2>&1 | tee logs/train_qwen3_vl_opd_geo3k.log
