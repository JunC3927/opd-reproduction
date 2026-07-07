#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/continual_sft/llava1_5.yaml 2>&1 | tee logs/train_continual_sft_llava1_5.log
