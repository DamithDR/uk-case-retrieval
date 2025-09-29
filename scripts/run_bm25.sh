#!/bin/bash
#SBATCH --partition=cpu-6h
#SBATCH --output=log/output_%A_%a.log

source venv/bin/activate

python3 -m module.BM25
