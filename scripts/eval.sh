#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --output=log/output_%A_%a.log
#SBATCH --mem=40G
#SBATCH --array=0-3

MODELS=(
    "models/google-bert_bert-base-uncased_positive_negative_W3/final"
    "models/nlpaueb_legal-bert-base-uncased_positive_negative_W3/final"
    "models/BAAI_bge-m3_positive_negative_W3/final"
    "models/nomic-ai_nomic-embed-text-v1.5_positive_negative_W3/final"
    "models/Qwen_Qwen3-Embedding-0.6B_positive_negative_W3/final"
    "models/Snowflake_snowflake-arctic-embed-l-v2.0_positive_negative_W3/final"
)

MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "Array task $SLURM_ARRAY_TASK_ID — Evaluating: $MODEL_NAME"

source venv/bin/activate

python -m experiments.evaluation.single_para_eval \
    --model_name "$MODEL_NAME" \
    --model_type dense \
    --batch_size 8 \
    --candidates_file_path data/data_splits/ --gold_file_path data/data_splits/ \
    --candidates_file candidates_3P.tsv --gold_file gold_3P.tsv --run_alias 3P_eval