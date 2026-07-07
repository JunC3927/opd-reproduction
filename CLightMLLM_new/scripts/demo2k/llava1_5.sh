#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/demo2k/llava1_5.yaml 2>&1 | tee logs/train_demo2k_llava1_5.log
