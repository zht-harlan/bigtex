#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-bigtex}"
DATA_ROOT="${DATA_ROOT:-../dataset}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-offline_artifacts}"
RESULTS_ROOT="${RESULTS_ROOT:-analysis_results_diveq}"
PURIFIER_MODE="${PURIFIER_MODE:-mock}"
BACKBONE_NAME="${BACKBONE_NAME:-scibert}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-32}"
TEXT_MAX_LENGTH="${TEXT_MAX_LENGTH:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
GNN_TYPE="${GNN_TYPE:-sage}"
NUM_LAYERS="${NUM_LAYERS:-1}"
DROPOUT="${DROPOUT:-0.2}"
JOINT_BATCH_SIZE="${JOINT_BATCH_SIZE:-64}"
JOINT_EPOCHS="${JOINT_EPOCHS:-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
RUNS="${RUNS:-3}"
SEED_BASE="${SEED_BASE:-42}"
VQ_AUX_WEIGHT="${VQ_AUX_WEIGHT:-0.1}"
TEXT_AUX_WEIGHT="${TEXT_AUX_WEIGHT:-0.1}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

datasets=(children photo)
codebook_sizes=(8 16 32 64 128)
lora_ranks=(4 8 16 32)

declare -A lr_map
lr_map[children]="1e-4"
lr_map[photo]="2e-4"

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
  lr="${lr_map[$ds]}"

  for codebook_size in "${codebook_sizes[@]}"; do
    python run_full_pipeline_diveq.py "$ds" \
      --data_root "$DATA_ROOT" \
      --artifact_root "$ARTIFACT_ROOT" \
      --results_dir "${RESULTS_ROOT}/codebook" \
      --purifier_mode "$PURIFIER_MODE" \
      --backbone_name "$BACKBONE_NAME" \
      --pooling cls \
      --text_batch_size "$TEXT_BATCH_SIZE" \
      --text_max_length "$TEXT_MAX_LENGTH" \
      --num-layers "$NUM_LAYERS" \
      --hidden-dim "$HIDDEN_DIM" \
      --dropout "$DROPOUT" \
      --joint_batch_size "$JOINT_BATCH_SIZE" \
      --joint_epochs "$JOINT_EPOCHS" \
      --lr "$lr" \
      --weight_decay "$WEIGHT_DECAY" \
      --codebook_size "$codebook_size" \
      --lora_r 8 \
      --lora_alpha 32 \
      --lora_dropout 0.1 \
      --enable_vq_aux_head \
      --vq_aux_weight "$VQ_AUX_WEIGHT" \
      --enable_text_aux_head \
      --text_aux_weight "$TEXT_AUX_WEIGHT" \
      --runs "$RUNS"
  done

  for lora_r in "${lora_ranks[@]}"; do
    lora_alpha=$((lora_r * 4))
    python run_full_pipeline_diveq.py "$ds" \
      --data_root "$DATA_ROOT" \
      --artifact_root "$ARTIFACT_ROOT" \
      --results_dir "${RESULTS_ROOT}/lora" \
      --purifier_mode "$PURIFIER_MODE" \
      --backbone_name "$BACKBONE_NAME" \
      --pooling cls \
      --text_batch_size "$TEXT_BATCH_SIZE" \
      --text_max_length "$TEXT_MAX_LENGTH" \
      --num-layers "$NUM_LAYERS" \
      --hidden-dim "$HIDDEN_DIM" \
      --dropout "$DROPOUT" \
      --joint_batch_size "$JOINT_BATCH_SIZE" \
      --joint_epochs "$JOINT_EPOCHS" \
      --lr "$lr" \
      --weight_decay "$WEIGHT_DECAY" \
      --codebook_size 32 \
      --lora_r "$lora_r" \
      --lora_alpha "$lora_alpha" \
      --lora_dropout 0.1 \
      --enable_vq_aux_head \
      --vq_aux_weight "$VQ_AUX_WEIGHT" \
      --enable_text_aux_head \
      --text_aux_weight "$TEXT_AUX_WEIGHT" \
      --runs "$RUNS"
  done
done
