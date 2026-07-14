#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/opd/qwen3_vl_geo3k_vllm_student_server.yaml 2>&1 | tee logs/train_qwen3_vl_geo3k_vllm_student_server.log
