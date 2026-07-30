#!/usr/bin/env python3
"""BraTS-MET-00703-001 (fold_3, PP 적용 후 meanDice=0.9711) GT vs Pred(PP) 시각화."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(os.environ.get("MICCAI_PROJECT_ROOT", "."))

BASE = PROJECT_ROOT / "nnUNet"
GT_DIR = BASE / "nnUNet_raw/Dataset001_BraTS/labelsTr"
IMG_DIR = BASE / "nnUNet_raw/Dataset001_BraTS/imagesTr"
D1_FT = BASE / "nnUNet_results/Dataset001_BraTS/nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3_FT900__nnUNetPlans__3d_fullres"
BRAINIAC = BASE / "nnUNet_results/Dataset001_BraTS/nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__3__nnUNetPlans__3d_fullres"

CASE = "BraTS-MET-00703-001"
FOLD = 3
LABELS = [1, 2, 3, 4]
LABEL_NAMES = {0: "background", 1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
CONF_THRESH = {1: 0.6, 2: 0.6, 3: 0.6, 4: 0.5}
SMALL_SIZE_CUTOFF = 50
STRUCT = np.ones((3, 3, 3), dtype=np.int8)
OUT_DIR = Path("/tmp/claude-500/-home1/50ed32c3-c2b5-4799-b9e4-b6e297c5e101/scratchpad")


def load_probs(npz_path: Path) -> np.ndarray:
    probs = np.load(npz_path, allow_pickle=True)["probabilities"]
    return np.transpose(probs, (0, 3, 2, 1))  # -> (C, Z, Y, X) zyx internal


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
    gt_img = nib.load(GT_DIR / f"{CASE}.nii.gz")
    gt = np.asarray(gt_img.dataobj).astype(np.uint8)  # xyz

    d1p = load_probs(D1_FT / f"fold_{FOLD}/validation/{CASE}.npz")
    brp = load_probs(BRAINIAC / f"fold_{FOLD}/validation/{CASE}.npz")
    ens = (d1p + brp) / 2.0
    pred_zyx = ens.argmax(0).astype(np.uint8)
    conf_zyx = ens.max(0)
    pp_zyx = apply_pp(pred_zyx, conf_zyx)
    pred_xyz = np.transpose(pred_zyx, (2, 1, 0))
    pp_xyz = np.transpose(pp_zyx, (2, 1, 0))

    t1c_path = IMG_DIR / f"{CASE}_0000.nii.gz"
    if not t1c_path.exists():
        t1c_path = sorted(IMG_DIR.glob(f"{CASE}_*.nii.gz"))[0]
    t1c = np.asarray(nib.load(t1c_path).dataobj).astype(np.float32)

    fg = gt > 0
    z_sizes = fg.sum(axis=(0, 1))
    best_z = int(np.argmax(z_sizes))

    cmap = mcolors.ListedColormap(["none", "#e03131", "#f08c00", "#2f9e44", "#1971c2"])
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    def overlay_mask(lab_map):
        m = lab_map.copy().astype(float)
        m[m == 0] = np.nan
        return m

    panels = [
        ("T1c (raw)", t1c[:, :, best_z], None),
        ("Ground Truth", t1c[:, :, best_z], gt[:, :, best_z]),
        ("Pred (raw, no PP)", t1c[:, :, best_z], pred_xyz[:, :, best_z]),
        ("Pred (after PP)", t1c[:, :, best_z], pp_xyz[:, :, best_z]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (title, bg, lab_map) in zip(axes, panels):
        ax.imshow(np.rot90(bg), cmap="gray")
        if lab_map is not None:
            ax.imshow(np.rot90(overlay_mask(lab_map)), cmap=cmap, norm=norm, alpha=0.6)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#e03131", "#f08c00", "#2f9e44", "#1971c2"]]
    fig.legend(handles, ["NETC", "SNFH", "ET", "RC"], loc="lower center", ncol=4, fontsize=10)
    fig.suptitle(f"{CASE} (fold_{FOLD}) — axial slice z={best_z} (largest GT foreground)\n"
                 f"Simple(D1+BrainIAC) ensemble, PP-applied mean multiclass Dice = 0.9711", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    out_path = OUT_DIR / "best_case_BraTS-MET-00703-001_visualization.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
