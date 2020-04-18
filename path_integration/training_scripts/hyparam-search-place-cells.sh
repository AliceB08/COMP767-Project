#!/usr/bin/env bash


base_dir=experiments/results/hyparam_search_place_cells/$(date '+%Y-%m-%d_%H-%M')/
mkdir $base_dir

subdir=place_cells_128_
save_dir=$base_dir$subdir
python ./train.py --num_epochs 1000 --save_dir $save_dir --num_place_cells=128 --num_headD_cells=12

subdir=place_cells_64_
save_dir=$base_dir$subdir
python ./train.py --num_epochs 1000 --save_dir $save_dir --num_place_cells=64 --num_headD_cells=12

subdir=place_cells_32_
save_dir=$base_dir$subdir
python ./train.py --num_epochs 1000 --save_dir $save_dir --num_place_cells=32 --num_headD_cells=12