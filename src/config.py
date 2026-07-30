"""Central configuration for the BraTS-METS inference pipeline.

Every tunable that affects the produced segmentation lives here so that the
Docker image, the local CLI and the analysis scripts stay in sync.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all overridable by environment variable so the same code runs both
# inside the challenge container and on a local workstation)
# --------------------------------------------------------------------------
INPUT_DIR = Path(os.environ.get("BRATS_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("BRATS_OUTPUT_DIR", "/output"))

# nnU-Net results root that holds the two trained model folders
NNUNET_RESULTS = Path(os.environ.get("nnUNet_results", "/opt/weights/nnUNet_results"))
DATASET_NAME = os.environ.get("BRATS_DATASET_NAME", "Dataset001_BraTS")

# Scratch space. MUST NOT be /input (challenge forbids writing there).
WORK_DIR = Path(os.environ.get("BRATS_WORK_DIR", "/tmp/brats_work"))

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# Model A ("D1"): Triad(PlainConvUNet-MAE)-initialised nnU-Net, LP100 -> FT900.
MODEL_A_DIR = (
    NNUNET_RESULTS
    / DATASET_NAME
    / "nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3_FT900__nnUNetPlans__3d_fullres"
)
MODEL_A_FOLDS = (0, 1, 2, 3, 4)

# Model B: BrainIAC (frozen ViT-B/16) feature-concat wrapper around nnU-Net.
# Only folds 1 and 3 are used: on internal cross-validation these two were the
# only folds that consistently improved the ensemble over model A alone.
MODEL_B_DIR = (
    NNUNET_RESULTS
    / DATASET_NAME
    / "nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__3__nnUNetPlans__3d_fullres"
)
MODEL_B_FOLDS = (1, 3)

CHECKPOINT_NAME = os.environ.get("BRATS_CHECKPOINT", "checkpoint_final.pth")

# --------------------------------------------------------------------------
# Input modality order. Must match dataset.json channel_names:
#   0 = t1c, 1 = t1n, 2 = t2f, 3 = t2w
# --------------------------------------------------------------------------
MODALITY_SUFFIXES = ("t1c", "t1n", "t2f", "t2w")

# T2W has been non-mandatory in BraTS-METS since 2025 (some cases carry a native
# T2, some a synthetic one, some none). The trained networks always expect four
# channels, so a missing T2W is filled with the FLAIR volume — the closest
# available T2-weighted contrast — instead of aborting the run.
SUBSTITUTABLE_MODALITIES = {"t2w": "t2f"}

# --------------------------------------------------------------------------
# Label map (BraTS-METS)
# --------------------------------------------------------------------------
LABELS = (1, 2, 3, 4)
LABEL_NAMES = {0: "background", 1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}

# --------------------------------------------------------------------------
# Ensemble weights: weight given to MODEL A per softmax channel.
# Model B receives (1 - w). Channel index == label index.
#
# Rationale: on the official hidden validation set BrainIAC clearly beat Triad
# on the resection-cavity (RC) region, while the two were statistically
# indistinguishable elsewhere. We therefore keep a plain 50/50 average for
# every channel and only shift RC towards BrainIAC.
# --------------------------------------------------------------------------
ENSEMBLE_W_MODEL_A = {
    0: 0.50,  # background
    1: 0.50,  # NETC
    2: 0.50,  # SNFH
    3: 0.50,  # ET
    4: 0.15,  # RC  -> BrainIAC weighted 0.85
}

# --------------------------------------------------------------------------
# Post-processing: size-gated, class-wise confidence filtering.
#
# Only connected components smaller than SMALL_SIZE_CUTOFF voxels are examined;
# such a component is deleted when its mean softmax confidence falls below the
# class threshold. Components >= the cutoff are always kept.
#
# Thresholds were fitted on the fold 1 + fold 3 validation split by histogram /
# threshold-sweep analysis of false-positive component confidence, stratified by
# component size (see scripts/analysis/02_*, 04_*).
# --------------------------------------------------------------------------
ENABLE_POSTPROCESSING = os.environ.get("BRATS_ENABLE_PP", "1") not in ("0", "false", "False")
SMALL_SIZE_CUTOFF = int(os.environ.get("BRATS_PP_SIZE_CUTOFF", "50"))
CONFIDENCE_THRESHOLD = {
    1: float(os.environ.get("BRATS_PP_T_NETC", "0.6")),
    2: float(os.environ.get("BRATS_PP_T_SNFH", "0.6")),
    3: float(os.environ.get("BRATS_PP_T_ET", "0.6")),
    4: float(os.environ.get("BRATS_PP_T_RC", "0.5")),
}

# --------------------------------------------------------------------------
# Inference runtime knobs (see README for the A10G 24 GB / 12 h budget)
# --------------------------------------------------------------------------
# Test-time augmentation (8x mirroring) costs ~8x network time. Disabled by
# default so the 12 hour budget holds with a large margin on an A10G.
DISABLE_TTA = os.environ.get("BRATS_DISABLE_TTA", "1") not in ("0", "false", "False")
TILE_STEP_SIZE = float(os.environ.get("BRATS_STEP_SIZE", "0.5"))
