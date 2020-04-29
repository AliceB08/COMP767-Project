#!/usr/bin/env bash

mkdir -p experiments/results/
tmp_date=experiments/results/2020-04-15_14-40/
python ./train.py --num_epochs 2000 --save_dir $tmp_date --use_saved_model $tmp_date