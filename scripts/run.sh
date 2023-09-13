#!/bin/bash
export PYTHONPATH=$(dirname "$0")/..:${PYTHONPATH:-}
accelerate launch --config_file ./packages/accelerate_config.yaml \
  ./dialogue/scripts/main.py