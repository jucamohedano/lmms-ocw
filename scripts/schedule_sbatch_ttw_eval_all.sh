#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date +%Y%m%d-%H%M%S)"
log_dir="./logs/slurm"
mkdir -p "$log_dir"

EVAL_TASKS="caltech101,dtd,flowers102,oxford_pets,ucf101"
# Split comma-separated values into array
IFS=',' read -ra EVAL_TASKS_ARRAY <<< "$EVAL_TASKS"
NUM_JOBS=${#EVAL_TASKS_ARRAY[@]}

if [[ $NUM_JOBS -eq 0 ]]; then
    echo "Error: no jobs found to run." >&2
    exit 1
fi

methods=(baseline full svf lora)

for method in "${methods[@]}"; do

    # Configure method-specific parameters
    if [ "$method" = "baseline" ]; then
        model="qwen2-vl-7b"
        experiment="vanilla_zero_shot"
        num_gpus=1
        mem="128G"
        time="12:00:00"
        use_accelerate=false
        batch_size=64
        model_args=""
    else
        model="qwen2-vl-7b-ttw"
        experiment="ttw_${method}"
        use_accelerate=true
        batch_size=1
        mem="128G"
        time="24:00:00"

        if [ "$method" = "full" ]; then
            num_gpus=2
            mem="256G"
            model_args="offline_caption_dir=./offline_captions/,ttw_finetune_method=full"
        elif [ "$method" = "lora" ]; then
            num_gpus=1
            model_args="offline_caption_dir=./offline_captions/,ttw_finetune_method=lora,ttw_lora_backend=unsloth,ttw_lr=1e-4,ttw_epochs=5"
        elif [ "$method" = "svf" ]; then
            num_gpus=1
            model_args="offline_caption_dir=./offline_captions/,ttw_finetune_method=svf,ttw_lr=1e-4,ttw_epochs=5"
        fi
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
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --output="${log_dir}/ttw_eval_${method}_${timestamp}_%A_%a.out"
#SBATCH --error="${log_dir}/ttw_eval_${method}_${timestamp}_%A_%a.err"

# -----------------------------------------------------------
# Evaluation Method: ${method^^}
# Model: ${model}
# -----------------------------------------------------------

# Load the required modules
module load nvhpc/24.5
module load gcc/12.2.0
export CC=gcc
export CXX=g++

# Ensure HuggingFace works offline on compute nodes
export HF_HUB_OFFLINE=1

# Reduce OOM risk / fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

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

EVAL_OUTPUT_DIR="logs/schedule/\${task}_${experiment}/${model}"

echo "Starting evaluation on node \$(hostname)..."
echo "Task: \${task}  Model: ${model}  Method: ${method}"
echo "Results will be saved to: \${EVAL_OUTPUT_DIR}"

if [ "${use_accelerate}" = true ]; then
    ACCELERATE_MAIN_PROCESS_PORT=\$((RANDOM % (50000 - 30000 + 1) + 30000))
    ACCELERATE_NUM_PROCESSES=\$(nvidia-smi --list-gpus | wc -l)
    echo "Accelerate processes: \${ACCELERATE_NUM_PROCESSES}"

    python -m accelerate.commands.launch \\
        --main_process_port="\${ACCELERATE_MAIN_PROCESS_PORT}" \\
        --num_processes="\${ACCELERATE_NUM_PROCESSES}" \\
        --mixed_precision=bf16 \\
        -m eval_model \\
        --model ${model} \\
        --model_args "${model_args}" \\
        --tasks "\${task}" \\
        --output_path "\${EVAL_OUTPUT_DIR}" \\
        --batch_size 1 \\
        --log_samples \\
        --seed 30
else
    # Direct execution for baseline
    python eval_model.py \\
        --model ${model} \\
        --tasks "\${task}" \\
        --output_path "\${EVAL_OUTPUT_DIR}" \\
        --batch_size ${batch_size} \\
        --log_samples \\
        --seed 30
fi

echo "Evaluation finished for \${task}. Results in: \${EVAL_OUTPUT_DIR}"
EOT

    echo "Submitted SLURM array with ${NUM_JOBS} jobs for method ${method}."
done

echo "All evaluation jobs submitted."
