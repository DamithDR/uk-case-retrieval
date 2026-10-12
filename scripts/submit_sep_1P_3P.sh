#!/bin/bash
# Single entry point: fine-tune the models from instructions/ins for the 1P
# setting, evaluate each on 1P, and collect the results -- one command, chained
# by SLURM dependencies.
#
#     bash scripts/submit_sep_1P_3P.sh
#     DRY_RUN=1 bash scripts/submit_sep_1P_3P.sh     # print the plan, don't submit
#
# 1P ONLY. 3P is disabled in scripts/_models_sep.sh (SEP_WINDOWS=("W1")); every
# array sizes itself off that list, so nothing here needs changing to turn 3P
# back on.
#
# Everything runs on H200 (gpu-medium). The earlier astro/L40S split caused
# three of the previous run's five failures purely by being a 48GB card; 141GB
# removes them without changing precision, batch size or data, so results stay
# comparable with the fine-tunes that already succeeded.
#
# SUBMITS ONLY WHAT IS MISSING. This script inspects scratch and results/ on the
# login node and builds each --array= list from what is actually absent, so a
# model that already has results is not submitted at all -- no task, no GPU
# allocation, not even a no-op. The per-task skips inside the job scripts remain
# as a backstop against races. Stages:
#
#   eval_sep  (pass 1)   score checkpoints already on disk -- NO dependency
#   train_sep            fine-tune whatever has no checkpoint
#   eval_sep  (pass 2)   score those, afterany:train
#   collect_sep          merge, afterany both eval passes
#
# Pass 1 carries no dependency so an already-trained model is scored within
# minutes rather than queueing behind fresh fine-tunes.
#
# Env overrides:
#   DRY_RUN=1            print the plan and sbatch commands without submitting
#   LOGDIR=...           log destination                (default: log)
#   TRAIN_QUEUE=         override training partition    (default: gpu-medium)
#   TRAIN_TIME=          override training walltime     (default: 48:00:00)
#   SEP_TRAIN_BS=n       training batch size            (default: 4, uniform)
#   THROTTLE=n           max concurrent array tasks     (default: 4 = H200 count)
#   FORCE_RETRAIN=1      retrain even where a checkpoint exists
#   FORCE_REEVAL=1       re-score even where results exist

set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/_models_sep.sh

LOGDIR="${LOGDIR:-log}"
THROTTLE="${THROTTLE:-4}"
mkdir -p "$LOGDIR" "results/${SEP_RESULTS_SUBDIR}"

# Resolve MODEL_DIR the same way scripts/_env.sh does, but without sourcing it
# (that would run a pip install, which has no business happening on a login node).
: "${SCRATCH_DIR:=/scratch/hpc/41/dolamull}"
: "${MODEL_DIR:=$SCRATCH_DIR/uk-case-retrieval/models}"

dry() { [ "${DRY_RUN:-0}" = "1" ]; }
join_by() { local IFS=,; echo "$*"; }

submit() {
    local label="$1" script="$2"; shift 2
    if dry; then
        echo "[DRY RUN] sbatch $* $script" >&2
        echo "dryrun-${label}"
        return
    fi
    sbatch --parsable "$@" "$script"
}

echo "=== separate 1P fine-tuning + evaluation (all jobs on H200) ==="
echo "models: $(sep_count)  |  windows: ${SEP_WINDOWS[*]}  |  train_bs: $SEP_TRAIN_BS"
echo "MODEL_DIR=$MODEL_DIR"
echo "LOGDIR=$LOGDIR  THROTTLE=$THROTTLE  DRY_RUN=${DRY_RUN:-0}"
echo

if [ ! -d "$MODEL_DIR" ]; then
    echo "WARNING: MODEL_DIR does not exist or is not visible from here:" >&2
    echo "         $MODEL_DIR" >&2
    echo "         Every checkpoint will look missing, so this would submit a" >&2
    echo "         full retrain. Run this on the cluster, or set MODEL_DIR." >&2
    echo >&2
fi

# --- work out what is actually missing ------------------------------------
TRAIN_TASKS=(); EVAL_NOW=(); EVAL_AFTER=(); NOTHING=()
NWIN=${#SEP_WINDOWS[@]}
n=$(sep_count)

row() { printf '  %-4s %-40s %-8s %s\n' "$1" "$2" "$3" "$4"; }
row task model status action
row ---- ---------------------------------------- -------- ------

for ((i = 0; i < n; i++)); do
    for ((w = 0; w < NWIN; w++)); do
        task=$((i * NWIN + w))
        window=${SEP_WINDOWS[$w]}
        setting=$(sep_setting_for_window "$window")
        model=$(sep_model "$i")
        results="results/$(sep_eval_alias "$model" "$setting")_results.csv"
        ckpt=$(sep_model_path "$model" "$window")

        has_results=0; [ -s "$results" ] && has_results=1
        has_ckpt=0;    [ -d "$ckpt" ]    && has_ckpt=1
        [ "${FORCE_REEVAL:-0}"  = "1" ] && has_results=0
        [ "${FORCE_RETRAIN:-0}" = "1" ] && has_ckpt=0

        if [ "$has_results" = 1 ]; then
            NOTHING+=("$task")
            row "$task" "$model" scored "skip (results exist)"
        elif [ "$has_ckpt" = 1 ]; then
            EVAL_NOW+=("$task")
            row "$task" "$model" trained "evaluate now"
        else
            TRAIN_TASKS+=("$task"); EVAL_AFTER+=("$task")
            row "$task" "$model" missing "train + evaluate (optim=$(sep_optim "$i"))"
        fi
    done
done

echo
echo "  to train: ${#TRAIN_TASKS[@]}   to evaluate: $(( ${#EVAL_NOW[@]} + ${#EVAL_AFTER[@]} ))   already done: ${#NOTHING[@]}"
echo

if [ "${#TRAIN_TASKS[@]}" -eq 0 ] && [ "${#EVAL_NOW[@]}" -eq 0 ]; then
    echo "Nothing to submit -- every model already has 1P results."
    echo "Merge them with:  bash scripts/collect_sep.sh"
    exit 0
fi

DEPS=()

# --- 1. score checkpoints that already exist (no dependency) ---------------
if [ "${#EVAL_NOW[@]}" -gt 0 ]; then
    echo ">>> evaluation pass 1 -- tasks $(join_by "${EVAL_NOW[@]}") (no dependency)..."
    PASS1_ID=$(submit eval1 scripts/eval_sep.sh \
        --array="$(join_by "${EVAL_NOW[@]}")%$THROTTLE" \
        --export=ALL,FORCE_REEVAL="${FORCE_REEVAL:-0}" \
        --output="$LOGDIR/sep_eval1_%A_%a.out" \
        --error="$LOGDIR/sep_eval1_%A_%a.err")
    echo "    job $PASS1_ID"
    DEPS+=("$PASS1_ID")
fi

# --- 2. train what has no checkpoint --------------------------------------
if [ "${#TRAIN_TASKS[@]}" -gt 0 ]; then
    echo ">>> training -- tasks $(join_by "${TRAIN_TASKS[@]}") (gpu-medium / H200)..."
    TRAIN_ID=$(submit train scripts/train_sep_h200.sh \
        --array="$(join_by "${TRAIN_TASKS[@]}")%$THROTTLE" \
        ${TRAIN_QUEUE:+--partition="$TRAIN_QUEUE"} \
        ${TRAIN_TIME:+--time="$TRAIN_TIME"} \
        --export=ALL,SEP_TRAIN_BS="$SEP_TRAIN_BS",FORCE_RETRAIN="${FORCE_RETRAIN:-0}" \
        --output="$LOGDIR/sep_train_%A_%a.out" \
        --error="$LOGDIR/sep_train_%A_%a.err")
    echo "    job $TRAIN_ID"

    echo ">>> evaluation pass 2 -- tasks $(join_by "${EVAL_AFTER[@]}") (after $TRAIN_ID)..."
    PASS2_ID=$(submit eval2 scripts/eval_sep.sh \
        --array="$(join_by "${EVAL_AFTER[@]}")%$THROTTLE" \
        --dependency=afterany:"$TRAIN_ID" \
        --export=ALL,FORCE_REEVAL="${FORCE_REEVAL:-0}" \
        --output="$LOGDIR/sep_eval2_%A_%a.out" \
        --error="$LOGDIR/sep_eval2_%A_%a.err")
    echo "    job $PASS2_ID"
    DEPS+=("$PASS2_ID")
fi

# --- 3. collect, after every evaluation that was submitted ----------------
echo ">>> collect (after $(join_by "${DEPS[@]}"))..."
COLLECT_ID=$(submit collect scripts/collect_sep.sh \
    --dependency=afterany:"$(IFS=:; echo "${DEPS[*]}")" \
    --output="$LOGDIR/sep_collect_%j.out" \
    --error="$LOGDIR/sep_collect_%j.err")
echo "    job $COLLECT_ID"

echo
echo "=== submitted ==="
[ -n "${PASS1_ID:-}" ] && echo "  evaluate (now)  : $PASS1_ID    tasks $(join_by "${EVAL_NOW[@]}")"
[ -n "${TRAIN_ID:-}" ] && echo "  train           : $TRAIN_ID    tasks $(join_by "${TRAIN_TASKS[@]}")"
[ -n "${PASS2_ID:-}" ] && echo "  evaluate (rest) : $PASS2_ID    after $TRAIN_ID"
echo "  collect         : $COLLECT_ID"
echo
echo "Results when finished:"
echo "  results/${SEP_OUT_PREFIX}_1P_results.csv"
echo "  results/${SEP_OUT_PREFIX}_recall_summary.csv   <- R@1,5,10,20,50"
echo
echo "Partial results land in results/${SEP_RESULTS_SUBDIR}/ as each model finishes;"
echo "run 'bash scripts/collect_sep.sh' on the login node any time to merge what exists."
echo
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  grep -l FAILED $LOGDIR/sep_*.err"
