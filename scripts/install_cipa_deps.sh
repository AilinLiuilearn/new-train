#!/bin/bash
# 安装 CIPA VMamba baseline 所需依赖
# 在 new-train 目录下运行: bash scripts/install_cipa_deps.sh

set -e
CIPA_ROOT="$(cd "$(dirname "$0")/../../CIPA-main/CIPA-main" && pwd)"
echo "CIPA 根目录: $CIPA_ROOT"

# 1. 安装 selective_scan
echo ">>> 安装 selective_scan..."
cd "$CIPA_ROOT/models/encoders/selective_scan"
pip install -e .
cd - > /dev/null

# 2. 安装 CIPA 其他依赖
echo ">>> 安装 CIPA 依赖..."
pip install easydict einops 2>/dev/null || true

echo ">>> 完成。若需 VMamba 预训练权重，请下载到 $CIPA_ROOT/pretrained/vmamba/"
