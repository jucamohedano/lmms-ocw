#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date +%Y%m%d-%H%M%S)"
log_dir="./logs/slurm"
method="full"
num_gpus="${1:-2}"
# Use $# so '' as 2nd arg means "no limit" (full dataset)
if [[ $# -ge 2 ]]; then eval_limit="$2"; else eval_limit="4"; fi
fsdp_state_dict_type="${FSDP_STATE_DICT_TYPE:-SHARDED_STATE_DICT}"
fsdp_use_orig_params="${FSDP_USE_ORIG_PARAMS:-True}"
ttw_validate_restore="${TTW_VALIDATE_RESTORE:-0}"
model="qwen2-vl-7b-ttw"
# When eval_limit is empty, run on full dataset; use "full" in experiment name
limit_suffix="${eval_limit:-full}"
experiment="debug_test_refactor_ttw_${method}"
wandb_args="${EVAL_WANDB_ARGS:-project=lmms-owc,job_type=eval}"
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
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output="${log_dir}/ttw_eval_${method}_${timestamp}_%A_%a.out"
#SBATCH --error="${log_dir}/ttw_eval_${method}_${timestamp}_%A_%a.err"

# -----------------------------------------------------------
# TTW Evaluation with Offline Captions (Full FT)
#
# Evaluates ${model} on classification tasks using full finetuning.
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

# Reduce CUDA memory fragmentation (helps with full FT OOM over many images)
# Note: PYTORCH_CUDA_ALLOC_CONF was renamed back to PYTORCH_ALLOC_CONF in PyTorch 2.9
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TTW_VALIDATE_RESTORE="${ttw_validate_restore}"

# Activate your environment
source "\$(pwd)"/.venv/bin/activate

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
echo "FSDP state dict type: ${fsdp_state_dict_type}"
echo "FSDP use_orig_params: ${fsdp_use_orig_params}"
echo "TTW restore validation: ${ttw_validate_restore}"
echo "WandB: ${wandb_args:-disabled}"

python -m accelerate.commands.launch \\
    --main_process_port="\${ACCELERATE_MAIN_PROCESS_PORT}" \\
    --num_processes="\${ACCELERATE_NUM_PROCESSES}" \\
    --mixed_precision=bf16 \\
    --use_fsdp \\
    --fsdp_sharding_strategy=FULL_SHARD \\
    --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \\
    --fsdp_transformer_layer_cls_to_wrap=Qwen2VLDecoderLayer \\
    --fsdp_state_dict_type="${fsdp_state_dict_type}" \\
    --fsdp_use_orig_params="${fsdp_use_orig_params}" \\
    -m eval_model \\
    --model ${model} \\
    --model_args offline_caption_dir=./offline_captions/,ttw_finetune_method=${method} \\
    --tasks "\${task}" \\
    --output_path "\${EVAL_OUTPUT_DIR}" \\
    --batch_size 1 \\
${limit_line}
${wandb_line}
    --log_samples \\
    --seed 30
    # To fall back to DDP+grad_accum: remove FSDP flags, add ,ttw_grad_accum=True to --model_args

echo "TTW evaluation finished for \${task}. Results in: \${EVAL_OUTPUT_DIR}"
EOT

echo "Submitted SLURM array with ${NUM_JOBS} jobs for ${model} (method=${method}, gpus=${num_gpus}, limit=${limit_suffix})"
