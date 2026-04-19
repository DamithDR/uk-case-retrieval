#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --output=log/output_%A_%a.log
#SBATCH --mem=40G

source venv/bin/activate

MODELS=(
    "models/BAAI_bge-m3_positive_negative_W1/final"
    "models/intfloat_e5-large-v2_positive_negative_W1/final"
    "models/nomic-ai_nomic-embed-text-v1.5_positive_negative_W1/final"
    "models/Qwen_Qwen3-Embedding-0.6B_positive_negative_W1/final"
    "models/Snowflake_snowflake-arctic-embed-l-v2.0_positive_negative_W1/final"
)

for MODEL_NAME in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Evaluating: $MODEL_NAME"
    echo "=========================================="
    python -m experiments.evaluation.single_para_eval \
        --model_name "$MODEL_NAME" \
        --model_type dense \
        --batch_size 8 \
        --candidates_file_path data/data_splits/ --gold_file_path data/data_splits/ \
        --candidates_file candidates_1P.tsv --gold_file gold_1P.tsv --run_alias 1P_eval
done