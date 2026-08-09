#!/bin/bash
#SBATCH --job-name=embed-train-dense-large
#SBATCH --partition=gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=log/train_dense_retry_%A_%a.out
#SBATCH --error=log/train_dense_retry_%A_%a.err
#SBATCH --array=0-3

# Large/unknown models — H200 141GB needed for training optimizer states
# 0: Mira190/Euler-Legal-Embedding-V1  (unknown size, H200 to be safe)
# 1: Qwen/Qwen3-Embedding-8B
# 2: Qwen/Qwen3-Embedding-4B           (4B training ~64GB with Adam states)
# 3: codefuse-ai/F2LLM-v2-8B

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=(
    "Mira190/Euler-Legal-Embedding-V1"
    "Qwen/Qwen3-Embedding-8B"
    "Qwen/Qwen3-Embedding-4B"
    "codefuse-ai/F2LLM-v2-8B"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Array task $SLURM_ARRAY_TASK_ID — Fine-tuning (large): $MODEL"
echo "Node: $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-?}"

source scripts/_env.sh

python -m experiments.training.train_dense_paragraph --model_name="$MODEL" \
    --training_file_path=data/data_splits/training/ --training_file=anchor_positive_W3.tsv \
    --eval_file_path=data/data_splits/training/ --eval_file=eval_positive_negative_W3.tsv \
    --run_alias=positive_negative_W3 --batch_size=8 --eval_batch_size=8 \
    --output_base="$MODEL_DIR"
