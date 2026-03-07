#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date +%Y%m%d-%H%M%S)"
log_dir="./logs/slurm"
method="baseline"
mkdir -p "$log_dir"

sbatch \
    --job-name=qwen2-vl-7b-eval-${method} \
    --partition=boost_usr_prod \
    --account=EUHPC_D33_243 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --gres=gpu:1 \
    --mem=128G \
    --time=12:00:00 \
    --output="${log_dir}/ttw_eval_${method}_${timestamp}_%j.out" \
    --error="${log_dir}/ttw_eval_${method}_${timestamp}_%j.err" <<'EOT'
#!/bin/bash

# -----------------------------------------------------------
# Baseline Evaluation (no TTW)
#
# Evaluates qwen2-vl-7b directly on classification tasks.
# No test-time warmup - model runs inference as-is.
# -----------------------------------------------------------

# Load the required modules
module load nvhpc/24.5
module load gcc/12.2.0
export CC=gcc
export CXX=g++

# Ensure HuggingFace works offline on compute nodes
export HF_HUB_OFFLINE=1

# Reduce OOM risk (model parallelism via device_map=balanced uses both GPUs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate your environment
source "$(pwd)"/.venv/bin/activate

# Timestamped output directory (same convention as schedule_batch.sh)
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
EVAL_OUTPUT_DIR="logs/schedule/ttw_eval_${TIMESTAMP}"

echo "Starting baseline evaluation on node $(hostname)..."
echo "Results will be saved to: ${EVAL_OUTPUT_DIR}"

# Run evaluation without TTW
python eval_model.py \
    --model qwen2-vl-7b \
    --tasks caltech101,dtd,flowers102,oxford_pets,ucf101 \
    --output_path "${EVAL_OUTPUT_DIR}" \
    --batch_size 64 \
    --log_samples \
    --seed 30

echo "Baseline evaluation finished. Results in: ${EVAL_OUTPUT_DIR}"
EOT
