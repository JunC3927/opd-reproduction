#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/llava_next_hf/intern_vl3_5.yaml 2>&1 | tee logs/train_llava_next_hf_intern_vl3_5.log
