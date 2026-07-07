#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/llava_next_hf/llava1_5.yaml 2>&1 | tee logs/train_llava_next_hf_llava1_5.log
