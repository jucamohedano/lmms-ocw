#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date +%Y%m%d-%H%M%S)"
log_dir="./logs/slurm"
method="lora"
num_gpus="${1:-1}"
# Use $# so '' as 2nd arg means "no limit" (full dataset)
if [[ $# -ge 2 ]]; then eval_limit="$2"; else eval_limit="4"; fi
model="qwen2.5-vl-7b-ttw"
limit_suffix="${eval_limit:-full}"
experiment="ttw_${method}"
wandb_args="${EVAL_WANDB_ARGS:-project=lmms-owc,job_type=eval}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
# Optional HF hub id or local path. When unset, omit model_name_or_path from --model_args so
# qwen2.5-vl-7b-ttw uses its default (Qwen/Qwen2.5-VL-7B-Instruct). Passing model_name_or_path=
# would parse as empty string and override that default.
if [[ -n "${MODEL_NAME_OR_PATH}" ]]; then
    model_path_segment="model_name_or_path=${MODEL_NAME_OR_PATH},"
else
    model_path_segment=""
fi
# Build --limit arg only when eval_limit is non-empty.
# When empty, use continuation line (\) so the python command doesn't break.
if [[ -n "${eval_limit}" ]]; then
    limit_line="    --limit ${eval_limit} \\"
else
    limit_line="    \\"
fi
# Build --wandb_args when EVAL_WANDB_ARGS is set (e.g. project=lmms-owc,job_type=eval)
if [[ -n "${wandb_args}" ]]; then
    wandb_line="    --wandb_args \"${wandb_args}\" \\"
else
    wandb_line="    \\"
fi
mkdir -p "$log_dir"

# EVAL_TASKS="caltech101,dtd,flowers102,oxford_pets,ucf101"
EVAL_TASKS="oxford_pets"
# Split comma-separated values into array
IFS=',' read -ra EVAL_TASKS_ARRAY <<< "$EVAL_TASKS"

NUM_JOBS=${#EVAL_TASKS_ARRAY[@]}

if [[ $NUM_JOBS -eq 0 ]]; then
    echo "Error: no jobs found to run." >&2
    exit 1
fi

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${model}-${method}
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D33_243
#SBATCH --nodes=1
#SBATCH --array=1-${NUM_JOBS}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:${num_gpus}
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output="${log_dir}/ttw_eval_${method}_${timestamp}_%A_%a.out"
#SBATCH --error="${log_dir}/ttw_eval_${method}_${timestamp}_%A_%a.err"

# -----------------------------------------------------------
# TTW Evaluation with Offline Captions (LoRA)
#
# Evaluates ${model} on classification tasks using LoRA finetuning.
# Uses SLURM arrays to run one task per job for make_table.py compatibility.
# -----------------------------------------------------------

# Load the required modules
module load nvhpc/24.5
module load gcc/12.2.0
export CC=gcc
export CXX=g++

# Ensure HuggingFace works offline on compute nodes
export HF_HUB_OFFLINE=1
# WandB offline (compute nodes have no internet)
export WANDB_MODE=offline

# Reduce CUDA memory fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Unsloth: disable torch.compile on patched kernels (reduces peak VRAM from
# compiled graph intermediates and avoids compile-cache fragmentation).
# Probe confirmed Qwen2-VL forward is NOT patched with fused CE loss on
# Unsloth 2026.3.4 + Transformers 4.57.6; loss goes through standard
# ForCausalLMLoss. OOM comes from materializing logits + activations for
# batch_size=5 captions simultaneously.
# export UNSLOTH_COMPILE_DISABLE=1
# Diagnostic: Unsloth internal logging to verify loss path at runtime:
# export UNSLOTH_ENABLE_LOGGING=1

# Turn off gpu profiling metrics to csv
export TTW_GPU_MEMORY_CSV=0

# ── CUDA MPS for concurrent TTW warmup workers ──
export CUDA_MPS_PIPE_DIRECTORY=logs/nvidia-mps-\${SLURM_JOB_ID}
export CUDA_MPS_LOG_DIRECTORY=logs/nvidia-mps-log-\${SLURM_JOB_ID}
mkdir -p \${CUDA_MPS_PIPE_DIRECTORY} \${CUDA_MPS_LOG_DIRECTORY}
nvidia-cuda-mps-control -d
echo "CUDA MPS daemon started"

cleanup_mps() {
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    rm -rf \${CUDA_MPS_PIPE_DIRECTORY} \${CUDA_MPS_LOG_DIRECTORY}
    echo "CUDA MPS daemon stopped"
}
trap cleanup_mps EXIT

# Activate your environment
source "\$(pwd)"/.ttw_working_venv/bin/activate

# Calculate the indices based on SLURM_ARRAY_TASK_ID
TASK_INDEX=\$(( \$SLURM_ARRAY_TASK_ID - 1 ))
EVAL_TASKS="$EVAL_TASKS"
IFS=',' read -ra EVAL_TASKS_ARRAY <<< "\$EVAL_TASKS"
task="\${EVAL_TASKS_ARRAY[TASK_INDEX]}"

if [ -z "\$task" ]; then
    echo "Error: Invalid SLURM_ARRAY_TASK_ID (\$SLURM_ARRAY_TASK_ID)"
    exit 1
fi

ACCELERATE_MAIN_PROCESS_PORT=\$((RANDOM % (50000 - 30000 + 1) + 30000))
ACCELERATE_NUM_PROCESSES=\$(nvidia-smi --list-gpus | wc -l)
EVAL_OUTPUT_DIR="logs/schedule/\${task}_${experiment}/${model}"

echo "Starting TTW evaluation on node \$(hostname)..."
echo "Task: \${task}  Model: ${model}  Method: ${method}"
echo "Results will be saved to: \${EVAL_OUTPUT_DIR}"
echo "Accelerate processes: \${ACCELERATE_NUM_PROCESSES}"
echo "Eval limit: ${limit_suffix}"
echo "WandB: ${wandb_args:-disabled}"

python -m accelerate.commands.launch \\
    --main_process_port="\${ACCELERATE_MAIN_PROCESS_PORT}" \\
    --num_processes="\${ACCELERATE_NUM_PROCESSES}" \\
    --mixed_precision=bf16 \\
    -m eval_model \\
    --model ${model} \\
    --model_args offline_caption_dir=./offline_captions/qwen2.5-vl-7b-ttw,${model_path_segment}ttw_finetune_method=${method},ttw_lora_backend=peft,ttw_lr=1e-4,ttw_epochs=5,ttw_concurrent_warmups=2,ttw_grad_accum=True \\
    --tasks "\${task}" \\
    --output_path "\${EVAL_OUTPUT_DIR}" \\
    --batch_size 1 \\
    --log_level DEBUG \\
${limit_line}
${wandb_line}
    --log_samples \\
    --seed 30

echo "TTW evaluation finished for \${task}. Results in: \${EVAL_OUTPUT_DIR}"
EOT

echo "Submitted SLURM array with ${NUM_JOBS} jobs for ${model} (method=${method}, gpus=${num_gpus}, limit=${limit_suffix})"
