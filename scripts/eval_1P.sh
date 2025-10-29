#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --exclude=hex-p3-g19
#SBATCH --output=log/output_%A_%a.log
#SBATCH --mem=40G

source venv/bin/activate

python -m experiments.evaluation.single_para_eval \
--model_name /models/google-bert_bert-base-uncased_positive_negative_W1/final/ \
--candidates_file_path data/data_splits/ --gold_file_path data/data_splits/ \
--candidates_file candidates_1P.tsv --gold_file gold_1P.tsv --run_alias 1P_eval