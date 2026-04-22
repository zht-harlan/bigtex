#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-bigtex}"
DATA_ROOT="${DATA_ROOT:-../dataset}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-offline_artifacts}"
RESULTS_ROOT="${RESULTS_ROOT:-grid_search_results_diveq_tail3}"
PURIFIER_MODE="${PURIFIER_MODE:-mock}"
BACKBONE_NAME="${BACKBONE_NAME:-scibert}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-32}"
TEXT_MAX_LENGTH="${TEXT_MAX_LENGTH:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
GNN_TYPE="${GNN_TYPE:-sage}"
NUM_LAYERS="${NUM_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.2}"
JOINT_BATCH_SIZE="${JOINT_BATCH_SIZE:-64}"
JOINT_EPOCHS="${JOINT_EPOCHS:-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-32}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
RUNS="${RUNS:-1}"
SEED_BASE="${SEED_BASE:-42}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

datasets=(children history photo)
aux_weights=(0.05 0.1)

declare -A lr_grid
lr_grid[children]="5e-5 1e-4 2e-4"
lr_grid[history]="5e-5 1e-4 2e-4"
lr_grid[photo]="1e-4 2e-4 3e-4"

for ds in "${datasets[@]}"; do
  python build_offline_text_artifacts.py "$ds" \
    --output_root "$ARTIFACT_ROOT" \
    --data_root "$DATA_ROOT" \
    --purifier_mode "$PURIFIER_MODE" \
    --encoder_name "$BACKBONE_NAME" \
    --pooling cls \
    --batch_size "$TEXT_BATCH_SIZE" \
    --max_length "$TEXT_MAX_LENGTH"
done

for ds in "${datasets[@]}"; do
  for lr in ${lr_grid[$ds]}; do
    for aux_weight in "${aux_weights[@]}"; do
      lr_tag="${lr//./p}"
      lr_tag="${lr_tag//-/m}"
      aux_tag="${aux_weight//./p}"

      python train_joint_graph_text_classifier_diveq.py "$ds" \
        --data_root "$DATA_ROOT" \
        --artifact_root "$ARTIFACT_ROOT" \
        --results_dir "${RESULTS_ROOT}/lr_${lr_tag}__aux_${aux_tag}" \
        --hidden-dim "$HIDDEN_DIM" \
        --num-layers "$NUM_LAYERS" \
        --gnn_type "$GNN_TYPE" \
        --dropout "$DROPOUT" \
        --batch_size "$JOINT_BATCH_SIZE" \
        --epochs "$JOINT_EPOCHS" \
        --lr "$lr" \
        --weight_decay "$WEIGHT_DECAY" \
        --runs "$RUNS" \
        --seed_base "$SEED_BASE" \
        --codebook_size "$CODEBOOK_SIZE" \
        --backbone_name "$BACKBONE_NAME" \
        --max_text_length "$TEXT_MAX_LENGTH" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --enable_vq_aux_head \
        --vq_aux_weight "$aux_weight" \
        --node_codes_path "${RESULTS_ROOT}/lr_${lr_tag}__aux_${aux_tag}/${ds}/${ds}_diveq_node_codes.csv"
    done
  done
done
