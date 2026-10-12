#!/bin/bash
#SBATCH --job-name=sep-eval
#SBATCH --partition=gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=8:00:00
#SBATCH --array=0-7%4

# Evaluate each fine-tuned model on ITS OWN setting only:
#   the W1-trained checkpoint is scored on 1P, the W3-trained one on 3P.
#
# Now on H200, not astro. Encoding runs in fp32 (single_para_eval.py does not
# request a dtype), so a 4B model needs ~40GB at eval_bs=32 -- which overran the
# 44GB usable on an L40S and killed the Qwen3-4B and dinghy-law-4b evaluations
# last run, *after* their 6.2h and 6.4h fine-tunes had succeeded. The H200's
# 141GB covers it with precision left untouched, so these scores stay directly
# comparable with the four evaluations that already completed in fp32.
#
# Array layout: task = model_index * len(SEP_WINDOWS) + setting_index. 3P is
# disabled, so this collapses to task == model_index, 0-7 -> 1P (W1 checkpoint).
#
# Each task writes its own results CSV (results/sep_runs/{setting}_{slug}_results.csv)
# so 16 concurrent tasks never append to the same file. scripts/collect_sep.sh
# merges them afterwards.
#
# RESUMABLE, twice over: a task skips if its results CSV already exists, and
# skips if its checkpoint does not exist yet. So the full 0-15 array can be
# submitted immediately to pick up whatever checkpoints are ready, and again
# after training to collect the rest, without duplicating any rows.
# Set FORCE_REEVAL=1 to re-score a model that already has results.

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source scripts/_models_sep.sh

NWIN=${#SEP_WINDOWS[@]}
MODEL_I=$((SLURM_ARRAY_TASK_ID / NWIN))
WINDOW=${SEP_WINDOWS[$((SLURM_ARRAY_TASK_ID % NWIN))]}
SETTING=$(sep_setting_for_window "$WINDOW")

MODEL=$(sep_model "$MODEL_I")
EVAL_BS=$(sep_eval_batch "$MODEL_I")

ALIAS=$(sep_eval_alias "$MODEL" "$SETTING")
RESULTS="results/${ALIAS}_results.csv"
mkdir -p "results/${SEP_RESULTS_SUBDIR}"

echo "=== task $SLURM_ARRAY_TASK_ID — evaluate $MODEL ($WINDOW checkpoint) on $SETTING ==="
echo "Results:    $RESULTS"

# Already scored. single_para_eval.py *appends*, so without this a resubmission
# would add a second set of rows for the same model. Checked before the env is
# built so a no-op task releases its GPU in seconds rather than sitting through
# the conda activate and pip install in _env.sh.
if [ -s "$RESULTS" ] && [ "${FORCE_REEVAL:-0}" != "1" ]; then
    echo "SKIP: $RESULTS already exists (FORCE_REEVAL=1 overrides)."
    exit 0
fi

source scripts/_env.sh

CKPT=$(sep_model_path "$MODEL" "$WINDOW")
echo "Node: $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-?} | batch=$EVAL_BS"
echo "Checkpoint: $CKPT"

# Depend on the training array with afterany, so one failed fine-tune does not
# cancel the other evaluations -- skip cleanly instead.
if [ ! -d "$CKPT" ]; then
    echo "SKIP: no fine-tuned checkpoint at $CKPT (did the $WINDOW training task for $MODEL succeed?)" >&2
    exit 0
fi

python -m experiments.evaluation.single_para_eval \
    --model_name "$CKPT" \
    --model_type dense \
    --batch_size "$EVAL_BS" \
    --candidates_file_path data/data_splits/ --gold_file_path data/data_splits/ \
    --candidates_file "candidates_${SETTING}.tsv" --gold_file "gold_${SETTING}.tsv" \
    --run_alias "$ALIAS"
rc=$?

if [ "$rc" -ne 0 ]; then
    echo "=== FAILED: $MODEL / $SETTING (python exit $rc) ===" >&2
    exit "$rc"
fi
if [ ! -s "$RESULTS" ]; then
    echo "=== FAILED: $MODEL / $SETTING exited 0 but wrote no results to $RESULTS ===" >&2
    exit 1
fi

echo "=== done: $MODEL / $SETTING ==="
