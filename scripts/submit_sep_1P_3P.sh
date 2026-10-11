#!/bin/bash
# Single entry point: fine-tune all 8 models from instructions/ins separately
# for the 1P and 3P settings, evaluate each on its own setting, and collect the
# results — one command, fully chained by SLURM dependencies.
#
#     bash scripts/submit_sep_1P_3P.sh
#     DRY_RUN=1 bash scripts/submit_sep_1P_3P.sh     # print, don't submit
#
# Pipeline (16 fine-tunes -> 16 evaluations -> 1 collect):
#
#   train_sep_astro  array 0-7%4   astro / L40S 48GB  24h   4 models x W1,W3  (<=1B)
#   train_sep_h200   array 0-7%4   gpu-medium / H200  48h   4 models x W1,W3  (>=4B)
#   eval_sep         array 0-15%6  astro / L40S 48GB  24h   8 models x 1P,3P
#   collect_sep                    astro (cpu only)   15m
#
# Dependencies use afterany, not afterok, and each eval task checks for its
# checkpoint first — so one failed fine-tune cannot cancel the other 15
# evaluations. collect_sep then reports whatever is missing.
#
# Env overrides:
#   DRY_RUN=1            print sbatch commands without submitting
#   LOGDIR=...           log destination                (default: log)
#   TRAIN_SMALL_QUEUE=   override small-model partition (default: astro)
#   TRAIN_SMALL_TIME=    override small-model walltime  (default: 24:00:00)
#   SEP_TRAIN_BS=n       training batch size            (default: 4, uniform)
#
# If a W3 small-model fine-tune hits the astro 24h wall, resubmit just that job
# on an H200 with the longer queue:
#   TRAIN_SMALL_QUEUE=gpu-medium TRAIN_SMALL_TIME=48:00:00 \
#     bash scripts/submit_sep_1P_3P.sh

set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/_models_sep.sh

LOGDIR="${LOGDIR:-log}"
mkdir -p "$LOGDIR" "results/${SEP_RESULTS_SUBDIR}"

dry() { [ "${DRY_RUN:-0}" = "1" ]; }

submit() {
    local label="$1" script="$2"; shift 2
    if dry; then
        echo "[DRY RUN] sbatch $* $script" >&2
        echo "dryrun-${label}"
        return
    fi
    sbatch --parsable "$@" "$script"
}

echo "=== separate 1P / 3P fine-tuning + evaluation ==="
echo "models: $(sep_count)  |  windows: ${SEP_WINDOWS[*]}  |  train_bs: $SEP_TRAIN_BS"
echo "LOGDIR=$LOGDIR  DRY_RUN=${DRY_RUN:-0}"
echo

n=$(sep_count)
for ((i = 0; i < n; i++)); do
    printf '  [%d] %-40s %-5s eval_bs=%s\n' "$i" "$(sep_model "$i")" "$(sep_class "$i")" "$(sep_eval_batch "$i")"
done
echo

# 1. small models (<=1B) — astro / L40S
echo ">>> training, small models (astro / L40S)..."
ASTRO_ID=$(submit train-astro scripts/train_sep_astro.sh \
    ${TRAIN_SMALL_QUEUE:+--partition="$TRAIN_SMALL_QUEUE"} \
    ${TRAIN_SMALL_TIME:+--time="$TRAIN_SMALL_TIME"} \
    --export=ALL,SEP_TRAIN_BS="$SEP_TRAIN_BS" \
    --output="$LOGDIR/sep_train_astro_%A_%a.out" \
    --error="$LOGDIR/sep_train_astro_%A_%a.err")
echo "    job $ASTRO_ID  (array 0-7, 4 models x 2 windows)"

# 2. large models (>=4B) — gpu-medium / H200
echo ">>> training, large models (gpu-medium / H200)..."
H200_ID=$(submit train-h200 scripts/train_sep_h200.sh \
    --export=ALL,SEP_TRAIN_BS="$SEP_TRAIN_BS" \
    --output="$LOGDIR/sep_train_h200_%A_%a.out" \
    --error="$LOGDIR/sep_train_h200_%A_%a.err")
echo "    job $H200_ID  (array 0-7, 4 models x 2 windows)"

# 3. evaluation — astro, after BOTH training arrays finish
echo ">>> evaluation (astro / L40S, after $ASTRO_ID and $H200_ID)..."
EVAL_ID=$(submit eval scripts/eval_sep.sh \
    --dependency=afterany:"$ASTRO_ID":"$H200_ID" \
    --output="$LOGDIR/sep_eval_%A_%a.out" \
    --error="$LOGDIR/sep_eval_%A_%a.err")
echo "    job $EVAL_ID  (array 0-15, 8 models x 2 settings)"

# 4. collect — merge per-model CSVs into the final named outputs
echo ">>> collect (after $EVAL_ID)..."
COLLECT_ID=$(submit collect scripts/collect_sep.sh \
    --dependency=afterany:"$EVAL_ID" \
    --output="$LOGDIR/sep_collect_%j.out" \
    --error="$LOGDIR/sep_collect_%j.err")
echo "    job $COLLECT_ID"

echo
echo "=== submitted ==="
echo "  train (small) : $ASTRO_ID"
echo "  train (large) : $H200_ID"
echo "  evaluate      : $EVAL_ID    after $ASTRO_ID, $H200_ID"
echo "  collect       : $COLLECT_ID    after $EVAL_ID"
echo
echo "Results when finished:"
echo "  results/${SEP_OUT_PREFIX}_1P_results.csv"
echo "  results/${SEP_OUT_PREFIX}_3P_results.csv"
echo "  results/${SEP_OUT_PREFIX}_recall_summary.csv   <- R@1,5,10,20,50"
echo
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f $LOGDIR/sep_train_h200_${H200_ID}_*.out"
