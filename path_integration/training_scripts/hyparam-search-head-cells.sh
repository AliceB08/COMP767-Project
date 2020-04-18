#!/usr/bin/env bash


base_dir=experiments/results/hyparam_search_head_cells/$(date '+%Y-%m-%d_%H-%M')/
mkdir $base_dir

subdir=head_cells_24_
save_dir=$base_dir$subdir
python ./train.py --num_epochs 1000 --save_dir $save_dir --num_place_cells=256 --num_headD_cells=24

subdir=head_cells_36_
save_dir=$base_dir$subdir
python ./train.py --num_epochs 1000 --save_dir $save_dir --num_place_cells=256 --num_headD_cells=36

subdir=head_cells_48_
save_dir=$base_dir$subdir
python ./train.py --num_epochs 1000 --save_dir $save_dir --num_place_cells=256 --num_headD_cells=48