#!/bin/bash
# SUPERSEDED by scripts/train_sep_h200.sh, which now trains all 8 models on H200.
# Kept only for reference; scripts/submit_sep_1P_3P.sh no longer calls it.
#SBATCH --job-name=sep-train-astro
#SBATCH --partition=astro
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=log/sep_train_astro_%A_%a.out
#SBATCH --error=log/sep_train_astro_%A_%a.err
#SBATCH --array=0-7%4

# Fine-tune the <=1B models on BOTH paragraph windows, one model per window.
# astro / L40S 48GB is ample for a full fine-tune at this size (24h queue cap).
#
# Array layout: task = model_index_within_class * 2 + window_index
#   even tasks -> W1 (scored on 1P),  odd tasks -> W3 (scored on 3P)

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source scripts/_models_sep.sh

IDX=($(sep_indices_for_class astro))
MODEL_I=${IDX[$((SLURM_ARRAY_TASK_ID / 2))]}
WINDOW=${SEP_WINDOWS[$((SLURM_ARRAY_TASK_ID % 2))]}

MODEL=$(sep_model "$MODEL_I")
EVAL_BS=$(sep_eval_batch "$MODEL_I")
ALIAS=$(sep_run_alias "$WINDOW")

echo "=== task $SLURM_ARRAY_TASK_ID — fine-tune $MODEL on $WINDOW (-> $(sep_setting_for_window "$WINDOW")) ==="
echo "Node: $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-?} | train_bs=$SEP_TRAIN_BS eval_bs=$EVAL_BS"

source scripts/_env.sh

python -m experiments.training.train_dense_paragraph --model_name="$MODEL" \
    --training_file_path=data/data_splits/training/ --training_file="anchor_positive_${WINDOW}.tsv" \
    --eval_file_path=data/data_splits/training/ --eval_file="eval_positive_negative_${WINDOW}.tsv" \
    --run_alias="$ALIAS" --batch_size="$SEP_TRAIN_BS" --eval_batch_size="$EVAL_BS" \
    --output_base="$MODEL_DIR"

echo "=== done: $(sep_model_path "$MODEL" "$WINDOW") ==="
