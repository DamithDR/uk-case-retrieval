#!/bin/bash
#SBATCH --job-name=embed-eval-zeroshot
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=log/eval_zeroshot_%A_%a.out
#SBATCH --error=log/eval_zeroshot_%A_%a.err
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

MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "Array task $SLURM_ARRAY_TASK_ID — Evaluating (zero-shot): $MODEL_NAME"
echo "Node: $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-?}"

source scripts/_env.sh

echo "--- 1P evaluation ---"
python -m experiments.evaluation.single_para_eval \
    --model_name "$MODEL_NAME" \
    --model_type dense \
    --batch_size 8 \
    --candidates_file_path data/data_splits/ --gold_file_path data/data_splits/ \
    --candidates_file candidates_1P.tsv --gold_file gold_1P.tsv --run_alias 1P_eval_zeroshot

echo "--- 3P evaluation ---"
python -m experiments.evaluation.single_para_eval \
    --model_name "$MODEL_NAME" \
    --model_type dense \
    --batch_size 8 \
    --candidates_file_path data/data_splits/ --gold_file_path data/data_splits/ \
    --candidates_file candidates_3P.tsv --gold_file gold_3P.tsv --run_alias 3P_eval_zeroshot
