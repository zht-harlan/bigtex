#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="${1:-bigtex}"
PURIFIER_MODE="${PURIFIER_MODE:-mock}"
PURIFIER_MODEL_NAME="${PURIFIER_MODEL_NAME:-}"
API_URL="${API_URL:-}"
API_KEY="${API_KEY:-}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required in PATH"
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

for ds in ogbn-arxiv cora pubmed amazon-photo; do
  cmd=(
    python run_full_pipeline.py "$ds"
    --purifier_mode "${PURIFIER_MODE}"
    --backbone_name scibert
    --pooling cls
    --text_batch_size 32
    --text_max_length 256
    --hidden_dim 256
    --gnn_type sage
    --num_layers 2
    --joint_batch_size 64
    --joint_epochs 20
    --joint_lr 2e-4
    --weight_decay 5e-4
    --num_quantizers 3
    --codebook_size 128
    --max_text_length 256
    --lora_r 8
    --lora_alpha 32
    --lora_dropout 0.1
    --num_runs 3
  )

  if [[ -n "${PURIFIER_MODEL_NAME}" ]]; then
    cmd+=(--purifier_model_name "${PURIFIER_MODEL_NAME}")
  fi
  if [[ -n "${API_URL}" ]]; then
    cmd+=(--api_url "${API_URL}")
  fi
  if [[ -n "${API_KEY}" ]]; then
    cmd+=(--api_key "${API_KEY}")
  fi

  echo "Running dataset: ${ds}"
  "${cmd[@]}"
done
