#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-bigtex}"
PURIFIER_MODE="${PURIFIER_MODE:-mock}"
API_URL="${API_URL:-}"
API_KEY="${API_KEY:-}"
PURIFIER_MODEL_NAME="${PURIFIER_MODEL_NAME:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

for ds in ogbn-arxiv cora pubmed amazon-photo; do
  cmd=(
    python run_full_pipeline_diveq.py "$ds"
    --purifier_mode "$PURIFIER_MODE"
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
    --codebook_size 128
    --lora_r 8
    --lora_alpha 32
    --lora_dropout 0.1
    --num_runs 3
  )

  if [[ -n "$PURIFIER_MODEL_NAME" ]]; then
    cmd+=(--purifier_model_name "$PURIFIER_MODEL_NAME")
  fi
  if [[ -n "$API_URL" ]]; then
    cmd+=(--api_url "$API_URL")
  fi
  if [[ -n "$API_KEY" ]]; then
    cmd+=(--api_key "$API_KEY")
  fi

  "${cmd[@]}"
done
