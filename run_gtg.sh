#!/bin/bash

# PASS + GTG-Shapley 运行脚本
clear
export HF_ENDPOINT=https://hf-mirror.com
echo "Starting PASS + GTG-Shapley Federated Learning..."

# 使用 conda 环境中的 flwr
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flwr

# 使用主配置文件（已修改为 GTG-Shapley 版本）
flwr run . local-simulation 2>&1 | tee pass_gtg_output.log

echo "Training completed. Check pass_gtg_output.log for details."
