#!/usr/bin/env bash

tmp_date=$(date '+%Y-%m-%d_%H-%M')
mkdir $tmp_date
python ../train.py --save_dir $tmp_date