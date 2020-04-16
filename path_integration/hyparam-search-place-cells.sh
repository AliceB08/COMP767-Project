#!/usr/bin/env bash


tmp_date=experiments/results/hyparam_search_place_cells/$(date '+%Y-%m-%d_%H-%M')/
mkdir $tmp_date

python ./train.py --num_epochs 1000 --save_dir $tmp_date --num_place_cells=128 --num_headD_cells=12
python ./train.py --num_epochs 1000 --save_dir $tmp_date --num_place_cells=64 --num_headD_cells=12
python ./train.py --num_epochs 1000 --save_dir $tmp_date --num_place_cells=32 --num_headD_cells=12