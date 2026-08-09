#!/bin/bash
#SBATCH --job-name=embed-eval-finetuned-small
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=log/eval_finetuned_small_%A_%a.out
#SBATCH --error=log/eval_finetuned_small_%A_%a.err
#SBATCH --array=0-2

# Small models (<=4B) — V100 32GB sufficient for inference
# 0: Hanno-Labs/dinghy-law-4b-v1
# 1: infgrad/Jasper-Token-Compression-600M
# 2: Qwen/Qwen3-Embedding-4B

SCRATCH_MODELS="${MODEL_DIR:-/scratch/hpc/41/dolamull/uk-case-retrieval}/models/models"

MODELS=(
    "$SCRATCH_MODELS/Hanno-Labs_dinghy-law-4b-v1_positive_negative_W3/final"
    "$SCRATCH_MODELS/infgrad_Jasper-Token-Compression-600M_positive_negative_W3/final"
    "$SCRATCH_MODELS/Qwen_Qwen3-Embedding-4B_positive_negative_W3/final"
)

MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "Array task $SLURM_ARRAY_TASK_ID — Evaluating (fine-tuned, small): $MODEL_NAME"
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
