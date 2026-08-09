#!/bin/bash
#SBATCH --job-name=embed-train-dense-small
#SBATCH --partition=gpu-medium
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=log/train_dense_small_%A_%a.out
#SBATCH --error=log/train_dense_small_%A_%a.err
#SBATCH --array=0

# Small model (600M) — V100 32GB is sufficient for training
# 0: infgrad/Jasper-Token-Compression-600M

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=(
    "infgrad/Jasper-Token-Compression-600M"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Array task $SLURM_ARRAY_TASK_ID — Fine-tuning (small): $MODEL"
echo "Node: $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-?}"

source scripts/_env.sh

python -m experiments.training.train_dense_paragraph --model_name="$MODEL" \
    --training_file_path=data/data_splits/training/ --training_file=anchor_positive_W3.tsv \
    --eval_file_path=data/data_splits/training/ --eval_file=eval_positive_negative_W3.tsv \
    --run_alias=positive_negative_W3 --batch_size=8 --eval_batch_size=8 \
    --output_base="$MODEL_DIR"
