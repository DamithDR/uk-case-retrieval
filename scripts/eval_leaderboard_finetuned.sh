#!/bin/bash
#SBATCH --job-name=embed-eval-finetuned-large
#SBATCH --partition=astro
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=log/eval_finetuned_%A_%a.out
#SBATCH --error=log/eval_finetuned_%A_%a.err
#SBATCH --array=0-3

# Large/unknown models (8B or unknown size) — L40S 48GB
# 0: Mira190/Euler-Legal-Embedding-V1  (unknown size, L40S to be safe)
# 1: Qwen/Qwen3-Embedding-8B
# 2: annamodels/LGAI-Embedding-Preview  (unknown size, L40S to be safe)
# 3: codefuse-ai/F2LLM-v2-8B

SCRATCH_MODELS="${MODEL_DIR:-/scratch/hpc/41/dolamull/uk-case-retrieval}/models/models"

MODELS=(
    "$SCRATCH_MODELS/Mira190_Euler-Legal-Embedding-V1_positive_negative_W3/final"
    "$SCRATCH_MODELS/Qwen_Qwen3-Embedding-8B_positive_negative_W3/final"
    "$SCRATCH_MODELS/annamodels_LGAI-Embedding-Preview_positive_negative_W3/final"
    "$SCRATCH_MODELS/codefuse-ai_F2LLM-v2-8B_positive_negative_W3/final"
)

MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "Array task $SLURM_ARRAY_TASK_ID — Evaluating (fine-tuned, large): $MODEL_NAME"
echo "Node: $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-?}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source scripts/_env.sh

echo "--- 1P evaluation ---"
python -m experiments.evaluation.single_para_eval \
    --model_name "$MODEL_NAME" \
    --model_type dense \
    --batch_size 8 \
    --candidates_file_path data/data_splits/ --gold_file_path data/data_splits/ \
    --candidates_file candidates_1P.tsv --gold_file gold_1P.tsv --run_alias 1P_eval_finetuned

echo "--- 3P evaluation ---"
python -m experiments.evaluation.single_para_eval \
    --model_name "$MODEL_NAME" \
    --model_type dense \
    --batch_size 8 \
    --candidates_file_path data/data_splits/ --gold_file_path data/data_splits/ \
    --candidates_file candidates_3P.tsv --gold_file gold_3P.tsv --run_alias 3P_eval_finetuned
