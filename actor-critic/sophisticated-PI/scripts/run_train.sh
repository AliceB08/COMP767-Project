#!/usr/bin/env bash
module load python/3.7
source $HOME/torchenv/bin/activate

tmp_date=experiments/$(date '+%Y-%m-%d_%H-%M')/
mkdir $tmp_date
python train_wmaze_path_integration.py --save_dir $tmp_date --num_epochs 5000 --lr 4e-4 --save_model_freq 200