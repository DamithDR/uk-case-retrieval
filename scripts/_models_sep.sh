# Shared model table for the separate-1P/3P fine-tuning run — SOURCED, not executed.
#
# Columns:  hf_model_id | gpu_class | eval_batch | train_optim
#
#   gpu_class   Retained for bookkeeping only — it records which GPU each model
#               *would* need for a full fine-tune. Every job in this run (train
#               and eval alike) is submitted to the H200 partition, so nothing
#               reads this column to choose a queue any more.
#                 astro = fits an L40S 48GB for a FULL fine-tune (<=1B params)
#                 h200  = needs an H200 141GB for a full fine-tune (>=4B params)
#
#   eval_batch  encoding / TripletEvaluator batch size. Throughput only — it has
#               no effect on gradients or the loss, so it is tuned per model size.
#               The *training* batch size is deliberately uniform (SEP_TRAIN_BS)
#               so the recipe is identical across models and results stay
#               comparable.
#
#   train_optim HF optimizer name. Uniform `adamw_torch_fused` everywhere except
#               octen-law-8b-v1: at 7.05B params, fp32 weights (28.2GB) + fp32
#               grads (28.2GB) + fused-AdamW fp32 moments (56.4GB) is ~113GB
#               before a single activation, and it OOM'd on a 141GB H200 at
#               139.72/139.80 GiB (log/sep_train_h200_25111800_4.err).
#               `adamw_bnb_8bit` takes the moments from 56.4GB to ~14GB, which
#               fits, while leaving batch_size (and therefore the in-batch
#               negative pool) and the fp32 master weights untouched.
#               NOTE: octen is then the one model not trained with the exact
#               optimizer the other seven use — worth a footnote in the writeup.
#
# Model ids resolved from instructions/ins against the ids already used in
# scripts/train_dense_parallel.sh and scripts/train_parallel.sh.

SEP_TABLE=(
    "BAAI/bge-m3|astro|64|adamw_torch_fused"
    "nomic-ai/nomic-embed-text-v1.5|astro|64|adamw_torch_fused"
    "Qwen/Qwen3-Embedding-0.6B|astro|64|adamw_torch_fused"
    "infgrad/Jasper-Token-Compression-600M|astro|64|adamw_torch_fused"
    "Qwen/Qwen3-Embedding-4B|h200|32|adamw_torch_fused"
    "Hanno-Labs/dinghy-law-4b-v1|h200|32|adamw_torch_fused"
    "litillabs/octen-law-8b-v1|h200|16|adamw_bnb_8bit"
    "annamodels/LGAI-Embedding-Preview|h200|16|adamw_torch_fused"
)

# Uniform training batch size — keeps MultipleNegativesRankingLoss in-batch
# negatives identical for every model. Override with SEP_TRAIN_BS=n if needed.
: "${SEP_TRAIN_BS:=4}"

# Paragraph windows, and the evaluation setting each one is scored on.
#   W1 (anchor_positive_W1.tsv) -> 1P (candidates_1P.tsv / gold_1P.tsv)
#   W3 (anchor_positive_W3.tsv) -> 3P (candidates_3P.tsv / gold_3P.tsv)
#
# 3P IS DISABLED BY DEFAULT. The main run is 1P only, so W3 is not in
# SEP_WINDOWS and the pipeline neither trains nor evaluates it: the arrays size
# themselves off this list, and collect_sep.sh only emits tables for the windows
# named here. The W3 data and the mapping below are untouched.
#
# SEP_WINDOWS_OVERRIDE (a space-separated string, since bash arrays cannot be
# exported through sbatch --export) replaces the list for one invocation. That
# is how a single-model 3P run gets its results into a 3P table without
# re-enabling 3P for everything:
#     SEP_WINDOWS_OVERRIDE="W1 W3" bash scripts/collect_sep.sh
#
# Do NOT change the default to ("W1" "W3") while array jobs are queued. This
# file is sourced at job *runtime*, not spooled by SLURM, so lengthening the
# list renumbers every array task retroactively: a queued task 6 means
# octen/W1 under one window and Jasper/W1 under two.
if [ -n "${SEP_WINDOWS_OVERRIDE:-}" ]; then
    read -r -a SEP_WINDOWS <<< "$SEP_WINDOWS_OVERRIDE"
else
    SEP_WINDOWS=("W1")
fi
sep_setting_for_window() { case "$1" in W1) echo "1P";; W3) echo "3P";; esac; }

# Distinct run alias so this run does not overwrite the earlier
# *_positive_negative_W3 checkpoints from the previous leaderboard run.
sep_run_alias() { echo "sep_$1"; }

# --- accessors -------------------------------------------------------------
sep_model()      { echo "${SEP_TABLE[$1]}" | cut -d'|' -f1; }
sep_class()      { echo "${SEP_TABLE[$1]}" | cut -d'|' -f2; }
sep_eval_batch() { echo "${SEP_TABLE[$1]}" | cut -d'|' -f3; }
sep_optim()      { echo "${SEP_TABLE[$1]}" | cut -d'|' -f4; }
sep_slug()       { echo "${1//\//_}"; }
sep_count()      { echo "${#SEP_TABLE[@]}"; }

# Indices of all models in a given gpu class, in table order.
sep_indices_for_class() {
    local want="$1" i
    for i in "${!SEP_TABLE[@]}"; do
        [ "$(sep_class "$i")" = "$want" ] && printf '%s ' "$i"
    done
}

# Final fine-tuned checkpoint path produced by train_dense_paragraph.py:
#   {output_base}/models/{model_slug}_{run_alias}/final
sep_model_path() {
    local model="$1" window="$2"
    echo "${MODEL_DIR}/models/$(sep_slug "$model")_$(sep_run_alias "$window")/final"
}

# Per-model results file written by single_para_eval.py as
#   results/{run_alias}_results.csv
SEP_RESULTS_SUBDIR="sep_runs"
sep_eval_alias() {
    local model="$1" setting="$2"
    echo "${SEP_RESULTS_SUBDIR}/${setting}_$(sep_slug "$model")"
}

# Final merged outputs (the "new name" for this run).
SEP_OUT_PREFIX="finetuned_sep"
