#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------
# Submit one sbatch job per dataset for offline caption generation.
# Uses vLLM for optimized inference (shared prefill, continuous batching).
# -----------------------------------------------------------

# DATASETS=(caltech101 dtd flowers102 oxford_pets ucf101)
DATASETS=(ucf101)

timestamp="$(date +%Y%m%d-%H%M%S)"
log_dir="./logs/offline_captions"
mkdir -p "$log_dir"

for dataset in "${DATASETS[@]}"; do
    echo "Submitting job for dataset: ${dataset}"
    sbatch \
        --job-name="ttw-vllm-captions${dataset}" \
        --partition=boost_usr_prod \
        --account=EUHPC_D33_243 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --gres=gpu:1 \
        --mem=128G \
        --time=12:00:00 \
        --output="${log_dir}/${dataset}_${timestamp}_%j.out" \
        --error="${log_dir}/${dataset}_${timestamp}_%j.err" <<EOT
#!/bin/bash

# Load the required modules
module load nvhpc/24.5
module load gcc/12.2.0
export CC=gcc
export CXX=g++

# Ensure HuggingFace works offline
export HF_HUB_OFFLINE=1

# Activate your environment
source "\$(pwd)"/.venv/bin/activate

echo "Starting vLLM offline caption generation for ${dataset} on node \$(hostname)..."

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHINDUCTOR_CACHE_DIR=/leonardo_scratch/fast/EUHPC_D33_243/.cache/torchinductor
mkdir -p /leonardo_scratch/fast/EUHPC_D33_243/.cache/torchinductor

python eval_model.py \\
    --model qwen2-vl \\
    --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct \\
    --tasks ${dataset} \\
    --output_path ./offline_captions/ \\
    --ttw_offline_generate \\
    --ttw_use_vllm \\
    --ttw_offline_num_candidates 10 \\
    --ttw_offline_temperature 0.75 \\
    --ttw_offline_max_new_tokens 128 \\
    --seed 30

echo "Job for ${dataset} finished successfully."
EOT
done

echo "All ${#DATASETS[@]} jobs submitted."
