#SBATCH --partition=cpu-6h
#SBATCH --output=log/output_%A_%a.log
#SBATCH --array=0-15
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

set -x

source venv/bin/activate

python -m util.anonymise_parallel
