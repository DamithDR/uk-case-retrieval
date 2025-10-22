#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --mem-per-gpu=24G
#SBATCH --output=log/output_%A_%a.log

source venv/bin/activate

python3 -m experiments.training.single_paragraph_transformer --model_name=nlpaueb/legal-bert-base-uncased \
--training_file_path=data/data_splits/training/ --training_file=anchor_positive_W1.tsv \
--eval_file_path=data/data_splits/training/ --eval_file=eval_positive_negative_W1.tsv \
--run_alias=positive_negative_W1 --batch_size=16 --eval_batch_size=16