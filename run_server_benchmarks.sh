#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="${1:-}"

if [[ -z "${CONDA_ENV_NAME}" ]]; then
  echo "Usage: bash run_server_benchmarks.sh <conda_env_name> [extra benchmark args...]"
  exit 1
fi

shift

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

python run_benchmarks.py \
  --datasets ogbn-arxiv cora pubmed amazon-photo \
  --num_iterate 5 \
  "$@"
