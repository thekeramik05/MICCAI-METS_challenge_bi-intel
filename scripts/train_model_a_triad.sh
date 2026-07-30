#!/usr/bin/env bash
# Model A — Triad(PlainConvUNet-MAE)-initialised nnU-Net, two-stage LP -> FT.
#
#   stage 1 (LP100): encoder frozen, 100 epochs, warms up the decoder
#   stage 2 (FT900): full fine-tune for 900 epochs, initialised from stage 1
#
# Usage: bash scripts/train_model_a_triad.sh <fold>
set -euo pipefail

FOLD="${1:?usage: train_model_a_triad.sh <fold 0-4>}"

export nnUNet_raw="${nnUNet_raw:?set nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:?set nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:?set nnUNet_results}"

# Foundation encoder used to initialise the nnU-Net encoder.
export TRIAD_CKPT="${TRIAD_CKPT:?set TRIAD_CKPT to Triad-PlainConvUNet-MAE.pth}"

DATASET=1
CONFIG=3d_fullres

STAGE1=nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3_Frozen__3_LP100
STAGE2=nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3_FT900

echo "[stage 1/2] linear probing (encoder frozen, 100 epochs) — fold ${FOLD}"
nnUNetv2_train "${DATASET}" "${CONFIG}" "${FOLD}" -tr "${STAGE1}" -p nnUNetPlans

echo "[stage 2/2] full fine-tuning (900 epochs) — fold ${FOLD}"
nnUNetv2_train "${DATASET}" "${CONFIG}" "${FOLD}" -tr "${STAGE2}" -p nnUNetPlans \
  -pretrained_weights "${nnUNet_results}/Dataset001_BraTS/${STAGE1}__nnUNetPlans__${CONFIG}/fold_${FOLD}/checkpoint_final.pth"

echo "model A fold ${FOLD} done"
