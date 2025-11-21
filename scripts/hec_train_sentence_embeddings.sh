#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d.dolamullage@lancaster.ac.uk
#SBATCH --output=/storage/hpc/41/dolamull/experiments/uk-case-retrieval/qwen0.6_output.log
#SBATCH --error=/storage/hpc/41/dolamull/experiments/uk-case-retrieval/qwen0.6_error.log

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/41/dolamull/conda_envs/llm_env
export HF_HOME=/scratch/hpc/41/dolamull/hf_cache
export PIP_CACHE_DIR=/scratch/hpc/41/dolamull/pip_cache
export TRITON_CACHE_DIR=/scratch/hpc/41/dolamull/triton-cache


while IFS='=' read -r key value; do
  if [[ -n "$key" && "$key" != \#* ]]; then
    export "$key"="$value"
  fi
done < .env

huggingface-cli login --token $HUGGINGFACE_TOKEN

python -m experiments.training.single_paragraph_transformer --model_name=Qwen/Qwen3-Embedding-0.6B \
--training_file_path=data/data_splits/training/ --training_file=anchor_positive_W1.tsv \
--eval_file_path=data/data_splits/training/ --eval_file=eval_positive_negative_W1.tsv \
--run_alias=positive_negative_W1 --batch_size=2 --eval_batch_size=2