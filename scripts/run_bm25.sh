#!/bin/bash
#SBATCH --partition=cpu-48h
#SBATCH --output=log/output.log
#SBATCH --mem=64G

source venv/bin/activate

python3 -m module.BM25
