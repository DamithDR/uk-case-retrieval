#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --output=log/output_%A_%a.log
#SBATCH --mem=40G

if [ -z "$1" ]; then
    echo "Usage: sbatch scripts/eval.sh <model_name>"
    exit 1
fi

MODEL_NAME="$1"

source venv/bin/activate

python -m experiments.evaluation.single_para_eval \
--model_name "$MODEL_NAME" \
--candidates_file_path data/data_splits/ --gold_file_path data/data_splits/ \
--candidates_file candidates_1P.tsv --gold_file gold_1P.tsv --run_alias 1P_eval