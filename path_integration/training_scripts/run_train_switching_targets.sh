#!/usr/bin/env bash

tmp_date=experiments/results/$(date '+%Y-%m-%d_%H-%M')/
mkdir $tmp_date
python ./train.py --save_dir $tmp_date --num_epochs 1000 --nb_switching_targets 3