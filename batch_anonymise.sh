#SBATCH --partition=cpu-6h
#SBATCH --output=log/output_%A_%a.log
#SBATCH --array=0-16
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

python -m util.anonymise_parallel
