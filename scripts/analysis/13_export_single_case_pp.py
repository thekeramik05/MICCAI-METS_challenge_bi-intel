#!/usr/bin/env python3
"""BraTS-MET-00703-001 (fold_3) simple ensemble + PP 결과를 nii.gz로 저장."""
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(os.environ.get("MICCAI_PROJECT_ROOT", "."))

BASE = PROJECT_ROOT / "nnUNet"
GT_DIR = BASE / "nnUNet_raw/Dataset001_BraTS/labelsTr"
D1_FT = BASE / "nnUNet_results/Dataset001_BraTS/nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3_FT900__nnUNetPlans__3d_fullres"
BRAINIAC = BASE / "nnUNet_results/Dataset001_BraTS/nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__3__nnUNetPlans__3d_fullres"
OUT_DIR = PROJECT_ROOT / "outputs/fold_validation_pp_examples"

CASE = "BraTS-MET-00703-001"
FOLD = 3
LABELS = [1, 2, 3, 4]
CONF_THRESH = {1: 0.6, 2: 0.6, 3: 0.6, 4: 0.5}
SMALL_SIZE_CUTOFF = 50
STRUCT = np.ones((3, 3, 3), dtype=np.int8)


def load_probs(npz_path: Path) -> np.ndarray:
    probs = np.load(npz_path, allow_pickle=True)["probabilities"]
    return np.transpose(probs, (0, 3, 2, 1))


def apply_pp(pred, conf):
    pp = pred.copy()
    for lab in LABELS:
        mask = pred == lab
        labeled, n = ndimage.label(mask, structure=STRUCT)
        if n == 0:
            continue
        sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
        mean_conf = ndimage.mean(conf, labeled, index=np.arange(1, n + 1))
        for comp_id in range(1, n + 1):
            if sizes[comp_id - 1] < SMALL_SIZE_CUTOFF and mean_conf[comp_id - 1] < CONF_THRESH[lab]:
                pp[labeled == comp_id] = 0
    return pp


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = nib.load(GT_DIR / f"{CASE}.nii.gz")

    d1p = load_probs(D1_FT / f"fold_{FOLD}/validation/{CASE}.npz")
    brp = load_probs(BRAINIAC / f"fold_{FOLD}/validation/{CASE}.npz")
    ens = (d1p + brp) / 2.0
    pred_zyx = ens.argmax(0).astype(np.uint8)
    conf_zyx = ens.max(0)
    pp_zyx = apply_pp(pred_zyx, conf_zyx)

    pred_xyz = np.transpose(pred_zyx, (2, 1, 0))
    pp_xyz = np.transpose(pp_zyx, (2, 1, 0))

    raw_path = OUT_DIR / f"{CASE}_ensemble_raw.nii.gz"
    pp_path = OUT_DIR / f"{CASE}_ensemble_pp.nii.gz"
    nib.save(nib.Nifti1Image(pred_xyz, ref.affine, ref.header), str(raw_path))
    nib.save(nib.Nifti1Image(pp_xyz, ref.affine, ref.header), str(pp_path))

    print(f"saved -> {raw_path}")
    print(f"saved -> {pp_path}")
    print(f"GT     -> {GT_DIR / f'{CASE}.nii.gz'}")


if __name__ == "__main__":
    main()
