#!/usr/bin/env bash

mkdir -p logs
python train.py --config config/demo2k/intern_vl3_5.yaml 2>&1 | tee logs/train_demo2k_intern_vl3_5.log
