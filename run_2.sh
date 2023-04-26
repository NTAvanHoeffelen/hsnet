#!/bin/bash

echo "source activate hsnet" > ~/.bashrc
source /home/user/conda/etc/profile.d/conda.sh
conda activate hsnet
python3 HSnet_wrapper.py "$@"