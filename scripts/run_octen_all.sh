#!/bin/bash
#SBATCH --job-name=octen-all
#SBATCH --partition=gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --output=log/octen_all_%j.out
#SBATCH --error=log/octen_all_%j.err

# ONE job, everything for litillabs/octen-law-8b-v1, in order:
#
#   1. zero-shot  1P   base model off the hub, scored on 1P
#   2. zero-shot  3P   base model off the hub, scored on 3P
#   3. fine-tune  W3   full fine-tune on the 3-paragraph window
#   4. evaluate   3P   that checkpoint, scored on 3P
#   5. collect         merge into the final 1P/3P tables
#
#     sbatch scripts/run_octen_all.sh
#     bash scripts/run_octen_all.sh      # same thing, interactively/on a GPU node
#
# Deliberately a single serial job rather than a dependency chain: one thing to
# submit, one log to read, one allocation. Phases 1-2 are inference only and
# cost ~1-3h; phase 3 dominates at ~20h (W3 is 2.12x the training data of W1),
# so the whole thing fits the 48h wall with room to spare.
#
# RESUMABLE. Every phase checks for its own output and skips if it is already
# there, so if the job is pre-empted or hits the wall you can resubmit the exact
# same command and it continues where it stopped. Phases are also independent:
# a zero-shot failure does not cost you the fine-tune, and the script reports a
# non-zero exit at the end naming whatever failed.
#
# Why this model needs bitsandbytes: at 7.05B params a full fp32 fine-tune needs
# ~28.2GB weights + ~28.2GB grads + ~56.4GB fused-AdamW moments = ~113GB before
# a single activation, and it OOM'd on a 141GB H200 at 139.72/139.80 GiB
# (log/sep_train_h200_25111800_4.err). SEP_TABLE gives it optim=adamw_bnb_8bit,
# taking the moments to ~14GB while leaving batch_size (and so the in-batch
# negative pool) and the fp32 master weights untouched. The trainer checks the
# import before loading the model, so a missing package fails in seconds.
#
# Env overrides:
#   MODEL=...          another model id from SEP_TABLE
#   FORCE_ZEROSHOT=1   redo the zero-shot evaluations
#   FORCE_RETRAIN=1    redo the fine-tune
#   FORCE_REEVAL=1     redo the fine-tuned evaluation
#   SKIP_COLLECT=1     leave the merged tables alone
#   EVAL_MAX_LEN=n     encoder sequence cap (default 4096, matching training)

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Locate the repo root. Under sbatch, "$0" is SLURM's spooled copy of this
# script in /var/spool/slurmd/jobNNN/slurm_script, so dirname "$0" says nothing
# about where the repo is -- it resolves to /var/spool/slurmd and every relative
# path below then fails. SLURM instead starts the job in the directory sbatch
# was run from, which is what SLURM_SUBMIT_DIR holds and what the rest of this
# pipeline already assumes is the repo root. The dirname branch is the fallback
# for running this directly with bash.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/_models_sep.sh" ]; then
    cd "$SLURM_SUBMIT_DIR" || exit 1
elif [ -f "$(dirname "$0")/_models_sep.sh" ]; then
    cd "$(dirname "$0")/.." || exit 1
fi

# Fail with one clear message rather than a cascade of "command not found".
if [ ! -f scripts/_models_sep.sh ]; then
    echo "ERROR: cannot find the repo root (pwd=$PWD, \$0=$0)." >&2
    echo "       Submit or run this from the repo root, e.g." >&2
    echo "         cd /path/to/uk-case-retrieval && sbatch scripts/$(basename "$0")" >&2
    exit 1
fi
source scripts/_models_sep.sh

MODEL="${MODEL:-litillabs/octen-law-8b-v1}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-4096}"

# --- resolve the model from the shared table ------------------------------
# By id, not a hardcoded index, so reordering SEP_TABLE cannot point this at a
# different model.
MODEL_I=-1
for ((i = 0; i < $(sep_count); i++)); do
    if [ "$(sep_model "$i")" = "$MODEL" ]; then MODEL_I=$i; break; fi
done
if [ "$MODEL_I" -lt 0 ]; then
    echo "ERROR: '$MODEL' is not in SEP_TABLE (scripts/_models_sep.sh)." >&2
    exit 1
fi

EVAL_BS=$(sep_eval_batch "$MODEL_I")
OPTIM=$(sep_optim "$MODEL_I")
SLUG=$(sep_slug "$MODEL")

echo "================================================================"
echo " model      : $MODEL"
echo " node       : $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-?}"
echo " batch      : train=$SEP_TRAIN_BS  eval=$EVAL_BS"
echo " optimizer  : $OPTIM"
echo " eval maxlen: $EVAL_MAX_LEN"
echo "================================================================"

source scripts/_env.sh

CKPT_W3=$(sep_model_path "$MODEL" W3)
FT_3P_RESULTS="results/$(sep_eval_alias "$MODEL" 3P)_results.csv"
mkdir -p "results/${SEP_RESULTS_SUBDIR}"

FAILED=()

# A model's zero-shot rows go into a results file shared by every model, and
# single_para_eval.py appends, so presence is tested by looking for this model's
# own rows rather than for the file.
zeroshot_done() {
    [ -s "$1" ] && grep -q "^${MODEL}," "$1"
}

# --- phase 1 + 2: zero-shot on 1P and 3P ---------------------------------
# Same run_alias convention as scripts/eval_zeroshot.sh, so these rows land in
# the existing zero-shot leaderboards alongside the other models.
for setting in 1P 3P; do
    alias="${setting}_eval_zeroshot"
    out="results/${alias}_results.csv"

    echo
    echo "=== phase: zero-shot $setting ==="
    echo "    results: $out"

    if zeroshot_done "$out" && [ "${FORCE_ZEROSHOT:-0}" != "1" ]; then
        echo "    SKIP: $MODEL already has rows there (FORCE_ZEROSHOT=1 overrides)."
        continue
    fi

    python -m experiments.evaluation.single_para_eval \
        --model_name "$MODEL" \
        --model_type dense \
        --batch_size "$EVAL_BS" \
        --max_seq_length "$EVAL_MAX_LEN" \
        --candidates_file_path data/data_splits/ --gold_file_path data/data_splits/ \
        --candidates_file "candidates_${setting}.tsv" --gold_file "gold_${setting}.tsv" \
        --run_alias "$alias"
    rc=$?

    if [ "$rc" -ne 0 ]; then
        echo "    === FAILED: zero-shot $setting (python exit $rc) ===" >&2
        FAILED+=("zeroshot-$setting")
    elif ! zeroshot_done "$out"; then
        echo "    === FAILED: zero-shot $setting exited 0 but wrote no rows to $out ===" >&2
        FAILED+=("zeroshot-$setting")
    else
        echo "    done: zero-shot $setting"
    fi
done

# --- phase 3: fine-tune on W3 --------------------------------------------
echo
echo "=== phase: fine-tune on W3 ==="
echo "    checkpoint: $CKPT_W3"

TRAIN_OK=1
if [ -d "$CKPT_W3" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "    SKIP: checkpoint already exists (FORCE_RETRAIN=1 overrides)."
else
    python -m experiments.training.train_dense_paragraph \
        --model_name="$MODEL" \
        --training_file_path=data/data_splits/training/ --training_file=anchor_positive_W3.tsv \
        --eval_file_path=data/data_splits/training/ --eval_file=eval_positive_negative_W3.tsv \
        --run_alias="$(sep_run_alias W3)" \
        --batch_size="$SEP_TRAIN_BS" --eval_batch_size="$EVAL_BS" \
        --optim="$OPTIM" --output_base="$MODEL_DIR"
    rc=$?

    # Report failure honestly. The previous run's scripts printed a success line
    # unconditionally, so an OOM'd task named a checkpoint it never wrote and
    # exited 0 -- which is why SLURM recorded every failure as COMPLETED.
    if [ "$rc" -ne 0 ]; then
        echo "    === FAILED: fine-tune W3 (python exit $rc) ===" >&2
        FAILED+=("train-W3"); TRAIN_OK=0
    elif [ ! -d "$CKPT_W3" ]; then
        echo "    === FAILED: fine-tune W3 exited 0 but wrote no checkpoint ===" >&2
        FAILED+=("train-W3"); TRAIN_OK=0
    else
        echo "    done: $CKPT_W3"
    fi
fi

# --- phase 4: evaluate the fine-tuned checkpoint on 3P -------------------
echo
echo "=== phase: evaluate fine-tuned W3 checkpoint on 3P ==="
echo "    results: $FT_3P_RESULTS"

if [ "$TRAIN_OK" -ne 1 ]; then
    echo "    SKIP: the fine-tune did not produce a checkpoint." >&2
    FAILED+=("eval-3P-finetuned(no checkpoint)")
elif [ -s "$FT_3P_RESULTS" ] && [ "${FORCE_REEVAL:-0}" != "1" ]; then
    echo "    SKIP: results already exist (FORCE_REEVAL=1 overrides)."
else
    python -m experiments.evaluation.single_para_eval \
        --model_name "$CKPT_W3" \
        --model_type dense \
        --batch_size "$EVAL_BS" \
        --candidates_file_path data/data_splits/ --gold_file_path data/data_splits/ \
        --candidates_file candidates_3P.tsv --gold_file gold_3P.tsv \
        --run_alias "$(sep_eval_alias "$MODEL" 3P)"
    rc=$?

    if [ "$rc" -ne 0 ]; then
        echo "    === FAILED: fine-tuned 3P evaluation (python exit $rc) ===" >&2
        FAILED+=("eval-3P-finetuned")
    elif [ ! -s "$FT_3P_RESULTS" ]; then
        echo "    === FAILED: fine-tuned 3P evaluation wrote no results ===" >&2
        FAILED+=("eval-3P-finetuned")
    else
        echo "    done: $FT_3P_RESULTS"
    fi
fi

# --- phase 5: merge the per-model CSVs into the final tables -------------
# SEP_WINDOWS_OVERRIDE so a 3P table is emitted too; the default list is 1P
# only. The 3P table will show MISSING for the models not run on 3P, which is
# expected. Pure awk, so this costs seconds.
if [ "${SKIP_COLLECT:-0}" != "1" ]; then
    echo
    echo "=== phase: collect ==="
    SEP_WINDOWS_OVERRIDE="W1 W3" bash scripts/collect_sep.sh || FAILED+=("collect")
fi

# --- summary -------------------------------------------------------------
echo
echo "================================================================"
echo " results for $MODEL"
echo "   zero-shot 1P   : results/1P_eval_zeroshot_results.csv"
echo "   zero-shot 3P   : results/3P_eval_zeroshot_results.csv"
echo "   fine-tuned 3P  : $FT_3P_RESULTS"
echo "   merged tables  : results/${SEP_OUT_PREFIX}_1P_results.csv"
echo "                    results/${SEP_OUT_PREFIX}_3P_results.csv"
echo "                    results/${SEP_OUT_PREFIX}_recall_summary.csv"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo
    echo " FAILED PHASES: ${FAILED[*]}"
    echo " Resubmit the same command to retry only those -- finished phases skip."
    echo "================================================================"
    exit 1
fi
echo
echo " all phases completed"
echo "================================================================"
