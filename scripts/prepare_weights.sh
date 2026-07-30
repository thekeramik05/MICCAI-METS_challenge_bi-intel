#!/usr/bin/env bash
# Stage the checkpoints the Docker image needs into ./weights/.
#
# Only the folds that are actually used at inference time are copied:
#   model A (Triad init)  : folds 0-4
#   model B (BrainIAC)    : folds 1, 3
#
# weights/ is git-ignored — it exists purely as Docker build context.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Where the trained models currently live on the training machine.
SRC_RESULTS="${SRC_RESULTS:-/home/irteam/data-vol1/2026_MICCAI_challenge/MICCAI_task_1/nnUNet/nnUNet_results}"
SRC_ENCODERS="${SRC_ENCODERS:-/home/irteam/data-vol1/2026_MICCAI_challenge/MICCAI_task_1/outputs/postprocess/foundation_encoder_checkpoints}"

DATASET="Dataset001_BraTS"
MODEL_A="nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3_FT900__nnUNetPlans__3d_fullres"
MODEL_B="nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__3__nnUNetPlans__3d_fullres"
FOLDS_A=(0 1 2 3 4)
FOLDS_B=(1 3)

DST_RESULTS="${REPO_ROOT}/weights/nnUNet_results/${DATASET}"
DST_ENCODERS="${REPO_ROOT}/weights/foundation_encoders"

copy_model() {
  local model_name="$1"; shift
  local folds=("$@")
  local src="${SRC_RESULTS}/${DATASET}/${model_name}"
  local dst="${DST_RESULTS}/${model_name}"

  [[ -d "${src}" ]] || { echo "FATAL: missing model dir ${src}" >&2; exit 1; }
  mkdir -p "${dst}"

  # plans.json / dataset.json are required to rebuild the network.
  for meta in plans.json dataset.json; do
    [[ -f "${src}/${meta}" ]] || { echo "FATAL: missing ${src}/${meta}" >&2; exit 1; }
    cp -f "${src}/${meta}" "${dst}/"
  done

  for fold in "${folds[@]}"; do
    local ckpt="${src}/fold_${fold}/checkpoint_final.pth"
    [[ -f "${ckpt}" ]] || { echo "FATAL: missing ${ckpt}" >&2; exit 1; }
    mkdir -p "${dst}/fold_${fold}"
    cp -f "${ckpt}" "${dst}/fold_${fold}/checkpoint_final.pth"
    echo "  fold_${fold} ok"
  done
  echo "staged ${model_name}"
}

echo "staging model A (Triad init, folds ${FOLDS_A[*]})"
copy_model "${MODEL_A}" "${FOLDS_A[@]}"

echo "staging model B (BrainIAC wrapper, folds ${FOLDS_B[*]})"
copy_model "${MODEL_B}" "${FOLDS_B[@]}"

echo "staging foundation encoder checkpoints"
mkdir -p "${DST_ENCODERS}"
for enc in BrainIAC.ckpt Triad-PlainConvUNet-MAE.pth; do
  [[ -f "${SRC_ENCODERS}/${enc}" ]] || { echo "FATAL: missing ${SRC_ENCODERS}/${enc}" >&2; exit 1; }
  cp -f "${SRC_ENCODERS}/${enc}" "${DST_ENCODERS}/"
done

echo
echo "done. staged size:"
du -sh "${REPO_ROOT}/weights"
