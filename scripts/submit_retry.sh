#!/bin/bash
# Retry submission — runs only the jobs that failed in the first round.
# Same dependency chain as submit_leaderboard.sh:
#   eval_zeroshot_retry starts immediately
#   train_dense_retry starts immediately (in parallel)
#   eval_leaderboard_finetuned waits for train_dense_retry to finish
#
# NOTE: Before running this, upgrade transformers on the login node:
#   pip install --upgrade transformers --no-cache-dir
# This is required for Hanno-Labs/dinghy-law-8b-v1 and minetta/nemotron-3-embed-8b-legal
# (model_type 'ministral3' is not recognised by the older installed version).
#
# Usage:
#   bash scripts/submit_retry.sh
#   DRY_RUN=1 bash scripts/submit_retry.sh

set -euo pipefail
cd "$(dirname "$0")/.."

LOGDIR="${LOGDIR:-log}"
mkdir -p "$LOGDIR"

dry() { [ "${DRY_RUN:-0}" = "1" ]; }

submit() {
  local label="$1"; local script="$2"; shift 2
  if dry; then
    echo "[DRY RUN] sbatch $* $script"
    echo "fake-job-id-${label}"
    return
  fi
  sbatch --parsable "$@" "$script"
}

echo "=== Retry submission ==="
echo "LOGDIR=$LOGDIR | DRY_RUN=${DRY_RUN:-0}"
echo

echo ">>> Submitting zero-shot eval retry..."
ZEROSHOT_ID=$(submit zeroshot scripts/eval_zeroshot_retry.sh \
  --output="$LOGDIR/eval_zeroshot_retry_%A_%a.out" \
  --error="$LOGDIR/eval_zeroshot_retry_%A_%a.err")
echo "    job id: $ZEROSHOT_ID (array 0-8)"

echo ">>> Submitting dense fine-tuning retry..."
TRAIN_ID=$(submit train scripts/train_dense_retry.sh \
  --output="$LOGDIR/train_dense_retry_%A_%a.out" \
  --error="$LOGDIR/train_dense_retry_%A_%a.err")
echo "    job id: $TRAIN_ID (array 0-8)"

echo ">>> Submitting fine-tuned eval (depends on train job $TRAIN_ID)..."
FINETUNED_ID=$(submit finetuned scripts/eval_leaderboard_finetuned.sh \
  --dependency=afterok:"$TRAIN_ID" \
  --kill-on-invalid-dep=yes \
  --output="$LOGDIR/eval_finetuned_retry_%A_%a.out" \
  --error="$LOGDIR/eval_finetuned_retry_%A_%a.err")
echo "    job id: $FINETUNED_ID (array 0-8, held until $TRAIN_ID completes)"

echo
echo "=== All retry jobs submitted ==="
echo "  zero-shot eval  : $ZEROSHOT_ID"
echo "  dense training  : $TRAIN_ID"
echo "  fine-tuned eval : $FINETUNED_ID (after $TRAIN_ID)"
echo
echo "Monitor with:  squeue -u \$USER"
