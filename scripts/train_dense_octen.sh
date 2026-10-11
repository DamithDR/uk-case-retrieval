#!/bin/bash
#SBATCH --job-name=embed-train-octen
#SBATCH --partition=gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=log/train_octen_%j.out
#SBATCH --error=log/train_octen_%j.err

# Fine-tune litillabs/octen-law-8b-v1 (8B, qwen3 arch, sentence-transformers).
# Trains once on the W3 paragraph window — same convention as the other 8B
# models in train_dense_parallel.sh; 1P and 3P are evaluation settings
# (see scripts/eval_finetuned_octen.sh).

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="litillabs/octen-law-8b-v1"

echo "Fine-tuning: $MODEL"
echo "Node: $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-?}"

source scripts/_env.sh

python -m experiments.training.train_dense_paragraph --model_name="$MODEL" \
    --training_file_path=data/data_splits/training/ --training_file=anchor_positive_W3.tsv \
    --eval_file_path=data/data_splits/training/ --eval_file=eval_positive_negative_W3.tsv \
    --run_alias=positive_negative_W3 --batch_size=4 --eval_batch_size=4 \
    --output_base="$MODEL_DIR"
