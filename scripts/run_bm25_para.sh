#!/bin/bash
#SBATCH --job-name=bm25-para-eval
#SBATCH --partition=cpu-48h
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=log/bm25_para_%j.out
#SBATCH --error=log/bm25_para_%j.err

# BM25 baseline for the paragraph retrieval settings. CPU only — no GPU needed.

echo "Node: $(hostname)"

source scripts/_env.sh

echo "--- 1P evaluation ---"
python -m experiments.evaluation.bm25_para_eval \
    --model_name BM25 \
    --candidates_file_path data/data_splits/ --gold_file_path data/data_splits/ \
    --candidates_file candidates_1P.tsv --gold_file gold_1P.tsv --run_alias 1P_eval_bm25

echo "--- 3P evaluation ---"
python -m experiments.evaluation.bm25_para_eval \
    --model_name BM25 \
    --candidates_file_path data/data_splits/ --gold_file_path data/data_splits/ \
    --candidates_file candidates_3P.tsv --gold_file gold_3P.tsv --run_alias 3P_eval_bm25
