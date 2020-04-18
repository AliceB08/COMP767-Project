#!/usr/bin/env bash

tmp_date=experiments/results/$(date '+%Y-%m-%d_%H-%M')/
mkdir $tmp_date
python ./train.py --save_dir $tmp_date --num_epochs 1000 --use_saved_model experiments/results/2020-04-15_14-40/ --disable_LSTM_training --seed 9473946