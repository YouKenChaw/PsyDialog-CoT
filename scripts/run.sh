#!/bin/bash
export PYTHONPATH=$(dirname "$0")/..:${PYTHONPATH:-}
accelerate launch --config_file ./packages/accelerate_config.yaml \
  ./dialogue/scripts/main.py \
  --data_dir ./data/processed_data