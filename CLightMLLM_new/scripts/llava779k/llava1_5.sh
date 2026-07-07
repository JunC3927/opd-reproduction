#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/llava779k/llava1_5.yaml 2>&1 | tee logs/train_llava779k_llava1_5.log
