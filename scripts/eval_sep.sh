#!/bin/bash
#SBATCH --job-name=sep-eval
#SBATCH --partition=astro
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=log/sep_eval_%A_%a.out
#SBATCH --error=log/sep_eval_%A_%a.err
#SBATCH --array=0-15%6

# Evaluate each fine-tuned model on ITS OWN setting only:
#   the W1-trained checkpoint is scored on 1P, the W3-trained one on 3P.
#
# Everything runs on astro — evaluation is inference only (~15GB for an 8B
# model in bf16), so an L40S 48GB is comfortable for every model here.
#
# Array layout: task = model_index * 2 + setting_index
#   even tasks -> 1P (W1 checkpoint),  odd tasks -> 3P (W3 checkpoint)
#
# Each task writes its own results CSV (results/sep_runs/{setting}_{slug}_results.csv)
# so 16 concurrent tasks never append to the same file. scripts/collect_sep.sh
# merges them afterwards.

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source scripts/_models_sep.sh

MODEL_I=$((SLURM_ARRAY_TASK_ID / 2))
WINDOW=${SEP_WINDOWS[$((SLURM_ARRAY_TASK_ID % 2))]}
SETTING=$(sep_setting_for_window "$WINDOW")

MODEL=$(sep_model "$MODEL_I")
EVAL_BS=$(sep_eval_batch "$MODEL_I")

source scripts/_env.sh

CKPT=$(sep_model_path "$MODEL" "$WINDOW")
ALIAS=$(sep_eval_alias "$MODEL" "$SETTING")
mkdir -p "results/${SEP_RESULTS_SUBDIR}"

echo "=== task $SLURM_ARRAY_TASK_ID — evaluate $MODEL ($WINDOW checkpoint) on $SETTING ==="
echo "Node: $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-?} | batch=$EVAL_BS"
echo "Checkpoint: $CKPT"
echo "Results:    results/${ALIAS}_results.csv"

# Depend on the training arrays with afterany, so one failed fine-tune does not
# cancel the other 15 evaluations — skip cleanly instead.
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

echo "=== done: $MODEL / $SETTING ==="
