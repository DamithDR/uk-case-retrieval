#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --output=log/output_%A_%a.log
#SBATCH --mem=40G
#SBATCH --array=0-2  # adjust upper bound to (number of models - 1)

source venv/bin/activate

MODELS=(
    "nomic-ai/nomic-embed-text-v1.5"
    "intfloat/e5-large-v2"
    "Snowflake/snowflake-arctic-embed-l-v2.0"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Running model: $MODEL (array task $SLURM_ARRAY_TASK_ID)"

python -m experiments.training.single_paragraph_transformer --model_name="$MODEL" \
    --training_file_path=data/data_splits/training/ --training_file=anchor_positive_W1.tsv \
    --eval_file_path=data/data_splits/training/ --eval_file=eval_positive_negative_W1.tsv \
    --run_alias=positive_negative_W1 --batch_size=2 --eval_batch_size=2