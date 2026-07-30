#!/usr/bin/env bash
# Model B — BrainIAC feature-concat nnU-Net.
#
# A frozen ViT-B/16 (BrainIAC) encodes each of the four modalities at 96^3, the
# averaged 768-d token grid is projected to 4 channels, upsampled to the patch
# resolution and concatenated to the 4 MRI channels (8 input channels total).
# The nnU-Net itself is trained from scratch on that augmented input.
#
# Usage: bash scripts/train_model_b_brainiac.sh <fold>
set -euo pipefail

FOLD="${1:?usage: train_model_b_brainiac.sh <fold 0-4>}"

export nnUNet_raw="${nnUNet_raw:?set nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:?set nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:?set nnUNet_results}"

export BRAINIAC_CKPT="${BRAINIAC_CKPT:?set BRAINIAC_CKPT to BrainIAC.ckpt}"
export BRAINIAC_MODALITY_INDICES="${BRAINIAC_MODALITY_INDICES:-0,1,2,3}"
export BRAINIAC_FEATURE_CHANNELS="${BRAINIAC_FEATURE_CHANNELS:-4}"
export BRAINIAC_FREEZE="${BRAINIAC_FREEZE:-1}"

nnUNetv2_train 1 3d_fullres "${FOLD}" \
  -tr nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__3 -p nnUNetPlans

echo "model B fold ${FOLD} done"
