#!/bin/bash
#SBATCH --job-name=sep-train-h200
#SBATCH --partition=gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --output=log/sep_train_h200_%A_%a.out
#SBATCH --error=log/sep_train_h200_%A_%a.err
#SBATCH --array=0-7%4

# Fine-tune the >=4B models on BOTH paragraph windows, one model per window.
# These need an H200 141GB: a full fine-tune of a 7.5B model is ~15GB bf16
# weights + ~60GB AdamW fp32 states + ~15GB grads, which does not fit an
# L40S 48GB. Throttled to %4 because the cluster has only 4 H200 GPUs total.
#
# Array layout: task = model_index_within_class * 2 + window_index
#   even tasks -> W1 (scored on 1P),  odd tasks -> W3 (scored on 3P)

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source scripts/_models_sep.sh

IDX=($(sep_indices_for_class h200))
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
