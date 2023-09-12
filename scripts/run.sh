#!/bin/bash

export PYTHONPATH=$(dirname "$0")/..:${PYTHONPATH:-}

python ./dialogue/scripts/main.py