#!/bin/bash
#SBATCH --job-name=embed-train-dense
#SBATCH --partition=gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=log/train_dense_%A_%a.out
#SBATCH --error=log/train_dense_%A_%a.err
#SBATCH --array=0-9

MODELS=(
    "Hanno-Labs/dinghy-law-8b-v1"
    "minetta/nemotron-3-embed-8b-legal"
    "Hanno-Labs/dinghy-law-4b-v1"
    "Mira190/Euler-Legal-Embedding-V1"
    "infgrad/Jasper-Token-Compression-600M"
    "Kingsoft-LLM/QZhou-Embedding"
    "Qwen/Qwen3-Embedding-8B"
    "annamodels/LGAI-Embedding-Preview"
    "Qwen/Qwen3-Embedding-4B"
    "codefuse-ai/F2LLM-v2-8B"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Array task $SLURM_ARRAY_TASK_ID — Fine-tuning: $MODEL"
echo "Node: $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-?}"

source scripts/_env.sh

python -m experiments.training.train_dense_paragraph --model_name="$MODEL" \
    --training_file_path=data/data_splits/training/ --training_file=anchor_positive_W3.tsv \
    --eval_file_path=data/data_splits/training/ --eval_file=eval_positive_negative_W3.tsv \
    --run_alias=positive_negative_W3 --batch_size=4 --eval_batch_size=4 \
    --output_base="$MODEL_DIR"
