#!/usr/bin/env bash
set -euo pipefail

export PYTHONHASHSEED=2023
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

python run_mdt_seg.py "$@"
