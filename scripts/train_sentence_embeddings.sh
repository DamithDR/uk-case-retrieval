#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --exclude=hex-p3-g19
#SBATCH --output=log/output_%A_%a.log
#SBATCH --mem=40G

source venv/bin/activate

python3 -m experiments.training.single_paragraph_transformer --model_name=answerdotai/ModernBERT-base \
--training_file_path=data/data_splits/training/ --training_file=anchor_positive_W1.tsv \
--eval_file_path=data/data_splits/training/ --eval_file=eval_positive_negative_W1.tsv \
--run_alias=positive_negative_W1 --batch_size=2 --eval_batch_size=2