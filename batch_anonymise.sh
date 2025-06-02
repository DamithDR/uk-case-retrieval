#!/bin/bash
#SBATCH --partition=cpu-6h
#SBATCH --output=log/output_%A_%a.log
#SBATCH --array=0-15
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

source venv/bin/activate

python3 -m util.anonymise_parallel
