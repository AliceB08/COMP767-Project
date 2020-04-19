#!/usr/bin/env bash


tmp_date=experiments/$(date '+%Y-%m-%d_%H-%M')/
mkdir $tmp_date
python ./train_wmaze_path_integration.py --save_dir $tmp_date --num_epochs 1000