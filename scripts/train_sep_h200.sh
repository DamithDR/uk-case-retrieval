#!/bin/bash
#SBATCH --job-name=sep-train
#SBATCH --partition=gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --array=0-7%4

# Fine-tune ALL 8 models on BOTH paragraph windows -- every task on an H200.
#
# This script replaced the earlier astro/h200 split (scripts/train_sep_astro.sh,
# now superseded): the 48GB L40S turned out to be the direct cause of two of the
# previous run's failures, and at 141GB the H200 removes both without touching
# precision, batch size or the data, so the numbers stay comparable with the
# fine-tunes that already succeeded.
#
# Array layout: task = model_index * len(SEP_WINDOWS) + window_index, with
# model_index indexing SEP_TABLE directly. 3P is disabled, so SEP_WINDOWS is
# just (W1) and that collapses to task == model_index, 0-7 -> W1 (scored on 1P).
# The arithmetic stays general, so re-enabling W3 needs no change here.
#
# RESUMABLE: a task whose final checkpoint already exists exits 0 without
# retraining, so the full 0-15 array can be resubmitted safely and will only
# run what is actually missing. Set FORCE_RETRAIN=1 to retrain regardless.

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source scripts/_models_sep.sh

NWIN=${#SEP_WINDOWS[@]}
MODEL_I=$((SLURM_ARRAY_TASK_ID / NWIN))
WINDOW=${SEP_WINDOWS[$((SLURM_ARRAY_TASK_ID % NWIN))]}

MODEL=$(sep_model "$MODEL_I")
EVAL_BS=$(sep_eval_batch "$MODEL_I")
OPTIM=$(sep_optim "$MODEL_I")
ALIAS=$(sep_run_alias "$WINDOW")

echo "=== task $SLURM_ARRAY_TASK_ID — fine-tune $MODEL on $WINDOW (-> $(sep_setting_for_window "$WINDOW")) ==="
echo "Node: $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-?} | train_bs=$SEP_TRAIN_BS eval_bs=$EVAL_BS optim=$OPTIM"

source scripts/_env.sh

CKPT=$(sep_model_path "$MODEL" "$WINDOW")
echo "Checkpoint: $CKPT"

# Skip work that is already done, so the whole array can be resubmitted.
if [ -d "$CKPT" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "SKIP: checkpoint already exists -- nothing to retrain (FORCE_RETRAIN=1 overrides)."
    exit 0
fi

python -m experiments.training.train_dense_paragraph --model_name="$MODEL" \
    --training_file_path=data/data_splits/training/ --training_file="anchor_positive_${WINDOW}.tsv" \
    --eval_file_path=data/data_splits/training/ --eval_file="eval_positive_negative_${WINDOW}.tsv" \
    --run_alias="$ALIAS" --batch_size="$SEP_TRAIN_BS" --eval_batch_size="$EVAL_BS" \
    --optim="$OPTIM" --output_base="$MODEL_DIR"
rc=$?

# Report failure honestly. The previous run used a bare `echo done` here, so an
# OOM'd task still printed a success line, named a checkpoint it never wrote,
# and exited 0 -- which is why SLURM recorded every failed task as COMPLETED.
if [ "$rc" -ne 0 ]; then
    echo "=== FAILED: $MODEL / $WINDOW (python exit $rc) ===" >&2
    exit "$rc"
fi
if [ ! -d "$CKPT" ]; then
    echo "=== FAILED: $MODEL / $WINDOW exited 0 but wrote no checkpoint at $CKPT ===" >&2
    exit 1
fi

echo "=== done: $CKPT ==="
