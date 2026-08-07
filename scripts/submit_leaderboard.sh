#!/bin/bash
# Submit all leaderboard experiments in one shot.
#
#   bash scripts/submit_leaderboard.sh
#
# Job order:
#   1. eval_zeroshot      — runs immediately (no fine-tuning, no deps)
#   2. train_dense        — runs immediately in parallel with zero-shot eval
#   3. eval_finetuned     — submitted with --dependency=afterok on train_dense;
#                           fires automatically once all 4 training tasks succeed
#
# Env overrides:
#   QUEUE    gpu-short | gpu-medium | gpu-long  (default: as per each job's script)
#   LOGDIR   where .out/.err go                 (default: log)
#   DRY_RUN=1  print sbatch commands without submitting

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root so relative paths resolve

LOGDIR="${LOGDIR:-log}"
mkdir -p "$LOGDIR"

dry() { [ "${DRY_RUN:-0}" = "1" ]; }

submit() {
  # submit <label> <script> [extra sbatch flags...]
  local label="$1"; local script="$2"; shift 2
  if dry; then
    echo "[DRY RUN] sbatch $* $script"
    echo "fake-job-id-${label}"
    return
  fi
  sbatch --parsable "$@" "$script"
}

echo "=== Leaderboard experiment submission ==="
echo "LOGDIR=$LOGDIR | DRY_RUN=${DRY_RUN:-0}"
echo

# 1. Zero-shot evaluation — no dependencies, starts immediately
echo ">>> Submitting zero-shot eval..."
ZEROSHOT_ID=$(submit zeroshot scripts/eval_zeroshot.sh \
  --output="$LOGDIR/eval_zeroshot_%A_%a.out" \
  --error="$LOGDIR/eval_zeroshot_%A_%a.err")
echo "    job id: $ZEROSHOT_ID (array 0-3)"

# 2. Dense fine-tuning — no dependencies, runs in parallel with zero-shot
echo ">>> Submitting dense fine-tuning..."
TRAIN_ID=$(submit train scripts/train_dense_parallel.sh \
  --output="$LOGDIR/train_dense_%A_%a.out" \
  --error="$LOGDIR/train_dense_%A_%a.err")
echo "    job id: $TRAIN_ID (array 0-3)"

# 3. Fine-tuned evaluation — waits for ALL training array tasks to succeed
echo ">>> Submitting fine-tuned eval (depends on train job $TRAIN_ID)..."
FINETUNED_ID=$(submit finetuned scripts/eval_leaderboard_finetuned.sh \
  --dependency=afterok:"$TRAIN_ID" \
  --kill-on-invalid-dep=yes \
  --output="$LOGDIR/eval_finetuned_%A_%a.out" \
  --error="$LOGDIR/eval_finetuned_%A_%a.err")
echo "    job id: $FINETUNED_ID (array 0-3, held until $TRAIN_ID completes)"

echo
echo "=== All jobs submitted ==="
echo "  zero-shot eval  : $ZEROSHOT_ID"
echo "  dense training  : $TRAIN_ID"
echo "  fine-tuned eval : $FINETUNED_ID (after $TRAIN_ID)"
echo
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $LOGDIR/train_dense_${TRAIN_ID}_*.out"
