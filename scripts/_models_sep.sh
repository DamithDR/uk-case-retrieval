# Shared model table for the separate-1P/3P fine-tuning run — SOURCED, not executed.
#
# Columns:  hf_model_id | gpu_class | eval_batch
#
#   gpu_class   astro = fits an L40S 48GB for a FULL fine-tune (<=1B params)
#               h200  = needs an H200 141GB for a full fine-tune (>=4B params)
#               (every model fits astro for *inference*, so all evals use astro)
#
#   eval_batch  encoding / TripletEvaluator batch size. Throughput only — it has
#               no effect on gradients or the loss, so it is tuned per model size.
#               The *training* batch size is deliberately uniform (SEP_TRAIN_BS)
#               so the recipe is identical across models and results stay
#               comparable.
#
# Model ids resolved from instructions/ins against the ids already used in
# scripts/train_dense_parallel.sh and scripts/train_parallel.sh.

SEP_TABLE=(
    "BAAI/bge-m3|astro|64"
    "nomic-ai/nomic-embed-text-v1.5|astro|64"
    "Qwen/Qwen3-Embedding-0.6B|astro|64"
    "infgrad/Jasper-Token-Compression-600M|astro|64"
    "Qwen/Qwen3-Embedding-4B|h200|32"
    "Hanno-Labs/dinghy-law-4b-v1|h200|32"
    "litillabs/octen-law-8b-v1|h200|16"
    "annamodels/LGAI-Embedding-Preview|h200|16"
)

# Uniform training batch size — keeps MultipleNegativesRankingLoss in-batch
# negatives identical for every model. Override with SEP_TRAIN_BS=n if needed.
: "${SEP_TRAIN_BS:=4}"

# Paragraph windows, and the evaluation setting each one is scored on.
#   W1 (anchor_positive_W1.tsv) -> 1P (candidates_1P.tsv / gold_1P.tsv)
#   W3 (anchor_positive_W3.tsv) -> 3P (candidates_3P.tsv / gold_3P.tsv)
SEP_WINDOWS=("W1" "W3")
sep_setting_for_window() { case "$1" in W1) echo "1P";; W3) echo "3P";; esac; }

# Distinct run alias so this run does not overwrite the earlier
# *_positive_negative_W3 checkpoints from the previous leaderboard run.
sep_run_alias() { echo "sep_$1"; }

# --- accessors -------------------------------------------------------------
sep_model()      { echo "${SEP_TABLE[$1]}" | cut -d'|' -f1; }
sep_class()      { echo "${SEP_TABLE[$1]}" | cut -d'|' -f2; }
sep_eval_batch() { echo "${SEP_TABLE[$1]}" | cut -d'|' -f3; }
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
