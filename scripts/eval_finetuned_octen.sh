#!/bin/bash
#SBATCH --job-name=embed-eval-octen
#SBATCH --partition=astro
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=log/eval_octen_%j.out
#SBATCH --error=log/eval_octen_%j.err

# Evaluate the fine-tuned litillabs/octen-law-8b-v1 on both paragraph settings:
#   1P — candidates_1P.tsv / gold_1P.tsv
#   3P — candidates_3P.tsv / gold_3P.tsv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source scripts/_env.sh

MODEL_NAME="$MODEL_DIR/models/litillabs_octen-law-8b-v1_positive_negative_W3/final"

echo "Evaluating (fine-tuned): $MODEL_NAME"
echo "Node: $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-?}"

if [ ! -d "$MODEL_NAME" ]; then
    echo "ERROR: fine-tuned model not found at $MODEL_NAME — did training finish?" >&2
    exit 1
fi

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
