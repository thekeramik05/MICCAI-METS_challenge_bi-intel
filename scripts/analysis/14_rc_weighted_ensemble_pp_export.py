#!/usr/bin/env python3
"""D1+BrainIAC total_val 앙상블: RC만 가중치(D1=0.15/BrainIAC=0.85) 조정, 나머지는 5:5.
그 위에 크기<50 + 클래스별 confidence threshold 후처리(PP)까지 적용.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

OUT_ROOT = Path("/home/irteam/data-vol1/2026_MICCAI_challenge/MICCAI_task_1/outputs")
D1_DIR = OUT_ROOT / "total_val_D1_5fold_probs"
BRAINIAC_DIR = OUT_ROOT / "total_val_brainiac_f13_probs"
RAW_OUT_DIR = OUT_ROOT / "total_val_D1_brainiac_ensemble_rcweighted"
PP_OUT_DIR = OUT_ROOT / "total_val_D1_brainiac_ensemble_rcweighted_pp"

LABELS = [1, 2, 3, 4]
LABEL_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
W_D1 = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.15}  # RC만 D1=0.15 / BrainIAC=0.85
CONF_THRESH = {1: 0.6, 2: 0.6, 3: 0.6, 4: 0.5}
SMALL_SIZE_CUTOFF = 50
STRUCT = np.ones((3, 3, 3), dtype=np.int8)


def apply_pp(pred, conf):
    pp = pred.copy()
    total_checked = {lab: 0 for lab in LABELS}
    total_removed = {lab: 0 for lab in LABELS}
    for lab in LABELS:
        mask = pred == lab
        labeled, n = ndimage.label(mask, structure=STRUCT)
        if n == 0:
            continue
        sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
        mean_conf = ndimage.mean(conf, labeled, index=np.arange(1, n + 1))
        for comp_id in range(1, n + 1):
            sz = sizes[comp_id - 1]
            if sz >= SMALL_SIZE_CUTOFF:
                continue
            total_checked[lab] += 1
            if mean_conf[comp_id - 1] < CONF_THRESH[lab]:
                pp[labeled == comp_id] = 0
                total_removed[lab] += 1
    return pp, total_checked, total_removed


def main():
    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    PP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = sorted(p.stem for p in D1_DIR.glob("*.npz"))
    print(f"{len(cases)} cases", flush=True)

    agg_checked = {lab: 0 for lab in LABELS}
    agg_removed = {lab: 0 for lab in LABELS}

    for i, case in enumerate(cases, 1):
        d1p = np.load(D1_DIR / f"{case}.npz")["probabilities"]
        brp = np.load(BRAINIAC_DIR / f"{case}.npz")["probabilities"]

        weighted = np.zeros_like(d1p)
        for c in range(d1p.shape[0]):
            w = W_D1.get(c, 0.5)
            weighted[c] = w * d1p[c] + (1 - w) * brp[c]

        pred_zyx = weighted.argmax(0).astype(np.uint8)
        conf_zyx = weighted.max(0)

        pp_zyx, checked, removed = apply_pp(pred_zyx, conf_zyx)
        for lab in LABELS:
            agg_checked[lab] += checked[lab]
            agg_removed[lab] += removed[lab]

        raw_xyz = np.transpose(pred_zyx, (2, 1, 0))
        pp_xyz = np.transpose(pp_zyx, (2, 1, 0))

        ref = nib.load(D1_DIR.parent / "total_val_D1_brainiac_ensemble_fixed" / f"{case}.nii.gz") \
            if (D1_DIR.parent / "total_val_D1_brainiac_ensemble_fixed" / f"{case}.nii.gz").exists() else None
        if ref is None:
            raise RuntimeError("reference affine source not found")

        nib.save(nib.Nifti1Image(raw_xyz, ref.affine, ref.header), str(RAW_OUT_DIR / f"{case}.nii.gz"))
        nib.save(nib.Nifti1Image(pp_xyz, ref.affine, ref.header), str(PP_OUT_DIR / f"{case}.nii.gz"))

        if i % 20 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)} done", flush=True)

    print(f"\n===== PP로 제거된 컴포넌트 (크기<{SMALL_SIZE_CUTOFF}, 클래스별 threshold 미달) =====")
    for lab in LABELS:
        c, r = agg_checked[lab], agg_removed[lab]
        pct = r / c * 100 if c else 0
        print(f"  {LABEL_NAMES[lab]}: 검사대상 {c}개 중 {r}개 제거 ({pct:.1f}%)")

    print(f"\nraw(가중치만) -> {RAW_OUT_DIR}")
    print(f"pp(가중치+후처리) -> {PP_OUT_DIR}")


if __name__ == "__main__":
    main()
