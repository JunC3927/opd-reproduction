#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/llava_v1_5_mix665k_textonly/intern_vl3_5.yaml 2>&1 | tee logs/train_llava_v1_5_mix665k_textonly_intern_vl3_5.log
