#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export PYTHONPATH=$(dirname "$0")/..:${PYTHONPATH:-}
accelerate launch --config_file ./packages/accelerate_config.yaml \
  ./dialogue/scripts/main.py \
  --data_dir ./data/processed_data