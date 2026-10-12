#!/bin/bash
# Fine-tune and evaluate ONE model for 1P: litillabs/octen-law-8b-v1, the only
# model in SEP_TABLE that needs bitsandbytes.
#
#     bash scripts/submit_octen_1P.sh
#     DRY_RUN=1 bash scripts/submit_octen_1P.sh      # print, don't submit
#
# Why this model is separate: at 7.05B params a full fine-tune in fp32 needs
# ~28.2GB weights + ~28.2GB grads + ~56.4GB fused-AdamW moments = ~113GB before
# a single activation, and it OOM'd on a 141GB H200 at 139.72/139.80 GiB
# (log/sep_train_h200_25111800_4.err). SEP_TABLE therefore gives it
# optim=adamw_bnb_8bit, which takes the moments from ~56.4GB to ~14GB while
# leaving batch_size (and so the in-batch negative pool) and the fp32 master
# weights alone. That optimizer is the one bitsandbytes dependency in the run.
#
# This is a thin wrapper: it submits the same scripts/train_sep_h200.sh and
# scripts/eval_sep.sh the full pipeline uses, restricted to this model's single
# array task, so there is no duplicated training or evaluation logic to drift.
# scripts/submit_sep_1P_3P.sh reaches the same place on its own now that the
# other models are scored -- this one just says so explicitly.
#
# It adapts to state: training is skipped if the checkpoint already exists, and
# it refuses to re-score a model that already has results unless forced.
#
# Env overrides:
#   DRY_RUN=1         print sbatch commands without submitting
#   MODEL=...         a different model id from SEP_TABLE
#   LOGDIR=...        log destination               (default: log)
#   TRAIN_TIME=       override training walltime    (default: script's 48:00:00)
#   SEP_TRAIN_BS=n    training batch size           (default: 4, uniform)
#   FORCE_RETRAIN=1   retrain even if a checkpoint exists
#   FORCE_REEVAL=1    re-score even if results exist
#   NO_COLLECT=1      don't chain scripts/collect_sep.sh afterwards

set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/_models_sep.sh

MODEL="${MODEL:-litillabs/octen-law-8b-v1}"
LOGDIR="${LOGDIR:-log}"
mkdir -p "$LOGDIR" "results/${SEP_RESULTS_SUBDIR}"

# Same resolution scripts/_env.sh uses, without sourcing it (that would run a
# pip install, which has no business happening on a login node).
: "${SCRATCH_DIR:=/scratch/hpc/41/dolamull}"
: "${MODEL_DIR:=$SCRATCH_DIR/uk-case-retrieval/models}"

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

# --- locate the model and its array task ----------------------------------
# Looked up by id rather than hardcoded, so reordering SEP_TABLE cannot silently
# point this at the wrong model.
MODEL_I=-1
n=$(sep_count)
for ((i = 0; i < n; i++)); do
    if [ "$(sep_model "$i")" = "$MODEL" ]; then MODEL_I=$i; break; fi
done
if [ "$MODEL_I" -lt 0 ]; then
    echo "ERROR: '$MODEL' is not in SEP_TABLE (scripts/_models_sep.sh). Known models:" >&2
    for ((i = 0; i < n; i++)); do echo "         $(sep_model "$i")" >&2; done
    exit 1
fi

# 1P is the W1 window. Find its slot rather than assuming 0, since the array
# layout is model_index * len(SEP_WINDOWS) + window_index.
NWIN=${#SEP_WINDOWS[@]}
WIN_I=-1
for ((w = 0; w < NWIN; w++)); do
    if [ "${SEP_WINDOWS[$w]}" = "W1" ]; then WIN_I=$w; break; fi
done
if [ "$WIN_I" -lt 0 ]; then
    echo "ERROR: W1 is not in SEP_WINDOWS (${SEP_WINDOWS[*]}), so 1P cannot be run." >&2
    exit 1
fi

TASK=$((MODEL_I * NWIN + WIN_I))
OPTIM=$(sep_optim "$MODEL_I")
EVAL_BS=$(sep_eval_batch "$MODEL_I")
CKPT=$(sep_model_path "$MODEL" W1)
RESULTS="results/$(sep_eval_alias "$MODEL" 1P)_results.csv"

echo "=== 1P run for a single model ==="
echo "  model      : $MODEL"
echo "  array task : $TASK  (model index $MODEL_I, window W1)"
echo "  optimizer  : $OPTIM"
echo "  batch      : train=$SEP_TRAIN_BS  eval=$EVAL_BS"
echo "  checkpoint : $CKPT"
echo "  results    : $RESULTS"
echo "  LOGDIR=$LOGDIR  DRY_RUN=${DRY_RUN:-0}"
echo

case "$OPTIM" in
    *bnb*|*8bit*)
        echo "  NOTE: optim=$OPTIM needs bitsandbytes in the conda env. The training"
        echo "        job checks for it before loading the model and exits immediately"
        echo "        with a clear message if it is missing, rather than failing later."
        echo
        ;;
esac

if [ ! -d "$MODEL_DIR" ]; then
    echo "WARNING: MODEL_DIR is not visible from here:" >&2
    echo "         $MODEL_DIR" >&2
    echo "         The checkpoint check below cannot be trusted. Run on the cluster." >&2
    echo >&2
fi

# --- already finished? ----------------------------------------------------
if [ -s "$RESULTS" ] && [ "${FORCE_REEVAL:-0}" != "1" ]; then
    echo "Nothing to do: $RESULTS already exists."
    echo "Re-score it with:  FORCE_REEVAL=1 bash $0"
    exit 0
fi

# --- 1. train, unless the checkpoint is already there ---------------------
TRAIN_ID=""
if [ -d "$CKPT" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo ">>> checkpoint already exists -- skipping training (FORCE_RETRAIN=1 overrides)"
else
    echo ">>> training task $TASK (gpu-medium / H200)..."
    TRAIN_ID=$(submit octen-train scripts/train_sep_h200.sh \
        --array="$TASK" \
        ${TRAIN_TIME:+--time="$TRAIN_TIME"} \
        --export=ALL,SEP_TRAIN_BS="$SEP_TRAIN_BS",FORCE_RETRAIN="${FORCE_RETRAIN:-0}" \
        --output="$LOGDIR/octen_train_%A_%a.out" \
        --error="$LOGDIR/octen_train_%A_%a.err")
    echo "    job $TRAIN_ID"
fi

# --- 2. evaluate ----------------------------------------------------------
echo ">>> evaluating task $TASK on 1P${TRAIN_ID:+ (after $TRAIN_ID)}..."
EVAL_ID=$(submit octen-eval scripts/eval_sep.sh \
    --array="$TASK" \
    ${TRAIN_ID:+--dependency=afterany:"$TRAIN_ID"} \
    --export=ALL,FORCE_REEVAL="${FORCE_REEVAL:-0}" \
    --output="$LOGDIR/octen_eval_%A_%a.out" \
    --error="$LOGDIR/octen_eval_%A_%a.err")
echo "    job $EVAL_ID"

# --- 3. merge into the final tables --------------------------------------
COLLECT_ID=""
if [ "${NO_COLLECT:-0}" != "1" ]; then
    echo ">>> collect (after $EVAL_ID)..."
    COLLECT_ID=$(submit octen-collect scripts/collect_sep.sh \
        --dependency=afterany:"$EVAL_ID" \
        --output="$LOGDIR/octen_collect_%j.out" \
        --error="$LOGDIR/octen_collect_%j.err")
    echo "    job $COLLECT_ID"
fi

echo
echo "=== submitted ==="
[ -n "$TRAIN_ID" ]   && echo "  train    : $TRAIN_ID"
echo "  evaluate : $EVAL_ID"
[ -n "$COLLECT_ID" ] && echo "  collect  : $COLLECT_ID"
echo
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f $LOGDIR/octen_train_*.out"
echo "  grep -l FAILED $LOGDIR/octen_*.err"
echo
echo "The collect step rewrites results/${SEP_OUT_PREFIX}_1P_results.csv and"
echo "results/${SEP_OUT_PREFIX}_recall_summary.csv from every per-model CSV in"
echo "results/${SEP_RESULTS_SUBDIR}/, so the other models stay in the final tables."
