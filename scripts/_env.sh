# Shared environment activation — SOURCED (not executed) by SLURM jobs.
# Loads cluster modules and activates the conda env.
# Every value can be overridden from the environment before sourcing.

export LANG="${LANG:-C.UTF-8}"
export LC_ALL=C.UTF-8
export PYTHONUTF8=1

: "${MODULE_MINIFORGE:=miniforge/20251003}"
: "${MODULE_CUDA:=}"
: "${MODULE_GCC:=}"
: "${EMBED_ENV:=/storage/hpc/41/dolamull/conda_envs/llm_env}"

set +u
source /etc/profile 2>/dev/null || true
module purge 2>/dev/null || true
[ -n "${MODULE_MINIFORGE}" ] && module load "${MODULE_MINIFORGE}"
[ -n "${MODULE_CUDA}" ]      && module load "${MODULE_CUDA}"
[ -n "${MODULE_GCC}" ]       && module load "${MODULE_GCC}"
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${EMBED_ENV}" 2>/dev/null || true
[[ "${EMBED_ENV}" == /* && -d "${EMBED_ENV}/bin" ]] && export PATH="${EMBED_ENV}/bin:${PATH}"

if [[ "${EMBED_ENV}" == /* ]] && [[ "$(command -v python)" != "${EMBED_ENV}"/* ]]; then
  echo "ERROR: env not found at ${EMBED_ENV} (python=$(command -v python)). Check EMBED_ENV / that the env exists." >&2
  exit 1
fi

export SCRATCH_DIR="${SCRATCH_DIR:-/scratch/hpc/41/dolamull}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_DIR/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export MODEL_DIR="${MODEL_DIR:-$SCRATCH_DIR/uk-case-retrieval/models}"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$MODEL_DIR" 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
[ -n "${HF_TOKEN:-}" ] && export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# Ensure transformers supports ministral3 and qwen3 architectures (requires >=4.51)
pip install -q "transformers>=4.51.0"

echo "env: $(python -V 2>&1) @ $(command -v python) | conda=${EMBED_ENV} | hf_token=$([ -n "${HF_TOKEN:-}" ] && echo set || echo unset)"
