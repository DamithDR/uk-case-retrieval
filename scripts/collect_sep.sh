#!/bin/bash
#SBATCH --job-name=sep-collect
#SBATCH --partition=astro
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=log/sep_collect_%j.out
#SBATCH --error=log/sep_collect_%j.err

# Merge the per-model evaluation CSVs from results/sep_runs/ into the final
# named outputs for this run.
#
# Pure awk: it needs no GPU and no python env. It nonetheless requests
# gres=gpu:nvidia_l40s:1 above, because the astro partition rejects a
# submission that names no gres -- and that unsatisfiable submission is why
# the collect step never ran at all. The L40S then sits idle for the few
# seconds of awk, so prefer the login node when you just want the tables now:
#
#     bash scripts/collect_sep.sh
#
# Outputs (results/) -- one table per window in SEP_WINDOWS; 3P is disabled:
#   finetuned_sep_1P_results.csv       full metrics, 1P models on 1P
#   finetuned_sep_recall_summary.csv   R@1,5,10,20,50
#
# Re-runnable: it rebuilds the outputs from scratch each time, and if a model
# was evaluated more than once it keeps only the most recent rows.

set -uo pipefail
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

SUMMARY_KS="1,5,10,20,50"
RUNS_DIR="results/${SEP_RESULTS_SUBDIR}"
SUMMARY="results/${SEP_OUT_PREFIX}_recall_summary.csv"

# Pull the last occurrence of one metric row out of a per-model results file,
# projecting only the requested k columns (by header name, not position).
extract() {
    # extract <file> <metric> <k-list|all>
    awk -F, -v metric="$2" -v ks="$3" '
        NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; ncol = NF; next }
        $2 == metric { map = $3; for (i = 1; i <= ncol; i++) v[i] = $i; found = 1 }
        END {
            if (!found) exit 1
            if (ks == "all") {
                out = map
                for (i = 4; i <= ncol; i++) out = out "," v[i]
            } else {
                n = split(ks, want, ",")
                out = map
                for (j = 1; j <= n; j++) {
                    c = col["k_" want[j]]
                    out = out "," (c ? v[c] : "NA")
                }
            }
            print out
        }
    ' "$1"
}

echo "=== collecting results from $RUNS_DIR ==="
if [ ! -d "$RUNS_DIR" ]; then
    echo "ERROR: $RUNS_DIR does not exist — no evaluations have written results yet." >&2
    exit 1
fi

# --- summary: R@k for both settings ---------------------------------------
{
    printf 'Setting,Model,MAP'
    for k in ${SUMMARY_KS//,/ }; do printf ',R@%s' "$k"; done
    printf '\n'
} > "$SUMMARY"

missing=0
total=0

for window in "${SEP_WINDOWS[@]}"; do
    setting=$(sep_setting_for_window "$window")
    full="results/${SEP_OUT_PREFIX}_${setting}_results.csv"
    : > "$full"

    n=$(sep_count)
    for ((i = 0; i < n; i++)); do
        model=$(sep_model "$i")
        f="${RUNS_DIR}/${setting}_$(sep_slug "$model")_results.csv"
        total=$((total + 1))

        if [ ! -s "$f" ]; then
            echo "  MISSING  $setting  $model"
            missing=$((missing + 1))
            continue
        fi

        # header for the full file, once, taken from the first file we see
        if [ ! -s "$full" ]; then
            head -1 "$f" > "$full"
        fi

        for metric in F1 Precision Recall; do
            row=$(extract "$f" "$metric" all) || continue
            echo "${model},${metric},${row}" >> "$full"
        done

        rrow=$(extract "$f" Recall "$SUMMARY_KS") \
            && echo "${setting},${model},${rrow}" >> "$SUMMARY" \
            && echo "  ok       $setting  $model"
    done

    echo "--> $full"
done

echo "--> $SUMMARY"
echo
echo "=== Recall summary ==="
column -s, -t < "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo
echo "collected $((total - missing))/$total model-setting pairs"
[ "$missing" -gt 0 ] && echo "NOTE: $missing missing — check log/sep_eval*.err and log/sep_train_*.err"
exit 0
