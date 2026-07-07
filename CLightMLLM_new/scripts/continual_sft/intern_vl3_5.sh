#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/continual_sft/intern_vl3_5.yaml 2>&1 | tee logs/train_continual_sft_intern_vl3_5.log
