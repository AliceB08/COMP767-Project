#!/usr/bin/env bash


tmp_date=experiments/results/hyparam_search_head_cells/$(date '+%Y-%m-%d_%H-%M')/
mkdir $tmp_date

python ./train.py --num_epochs 1000 --save_dir $tmp_date --num_place_cells=256 --num_headD_cells=24
python ./train.py --num_epochs 1000 --save_dir $tmp_date --num_place_cells=256 --num_headD_cells=36
python ./train.py --num_epochs 1000 --save_dir $tmp_date --num_place_cells=256 --num_headD_cells=48