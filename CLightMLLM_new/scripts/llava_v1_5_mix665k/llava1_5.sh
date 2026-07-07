#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/llava_v1_5_mix665k/llava1_5.yaml 2>&1 | tee logs/train_llava_v1_5_mix665k_llava1_5.log
