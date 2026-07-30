#!/usr/bin/env python3
"""후처리된 simple ensemble(total_val)이 다른 추론 결과들과 얼마나 유사한지 (foreground Dice/IoU)."""
import os
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(os.environ.get("MICCAI_PROJECT_ROOT", "."))

OUT_ROOT = PROJECT_ROOT / "outputs"
PP_DIR = OUT_ROOT / "total_val_D1_brainiac_ensemble_simple_pp"

TARGETS = {
    "RAW_Simple(후처리전)": OUT_ROOT / "total_val_D1_brainiac_ensemble_fixed",
    "Weighted_ensemble": OUT_ROOT / "total_val_D1_brainiac_ensemble_weighted",
    "D1_native(5fold)": OUT_ROOT / "total_val_D1_5fold_probs",
    "BrainIAC_native(f1,3)": OUT_ROOT / "total_val_brainiac_f13_probs",
    "5fold_final(참고baseline)": OUT_ROOT / "BraTS_validation/nnUNet/5fold_final",
    "Refined_weighted(RC/NETC 재조정)": OUT_ROOT / "total_val_D1_brainiac_ensemble_refined",
    "Old_ensemble5fold_pp_min10(예전 nnunet+triad)": OUT_ROOT / "total_val_ensemble5fold_pp_min10",
}


def dice_iou(a, b):
    fa, fb = a > 0, b > 0
    inter = np.logical_and(fa, fb).sum()
    union = np.logical_or(fa, fb).sum()
    dice = 2 * inter / (fa.sum() + fb.sum()) if (fa.sum() + fb.sum()) else 1.0
    iou = inter / union if union else 1.0
    return dice, iou


def main():
    pp_cases = sorted(p.name for p in PP_DIR.glob("*.nii.gz"))
    print(f"PP 케이스 수: {len(pp_cases)}")

    for name, tdir in TARGETS.items():
        common = [c for c in pp_cases if (tdir / c).exists()]
        if not common:
            print(f"\n{name}: 비교 가능 케이스 없음 (경로 확인 필요: {tdir})")
            continue
        dices, ious = [], []
        for c in common:
            a = np.asarray(nib.load(PP_DIR / c).dataobj)
            b = np.asarray(nib.load(tdir / c).dataobj)
            d, i = dice_iou(a, b)
            dices.append(d)
            ious.append(i)
        dices, ious = np.array(dices), np.array(ious)
        print(f"\n== PP_Simple vs {name} ({len(common)}케이스) ==")
        print(f"  Dice: mean={dices.mean():.4f} median={np.median(dices):.4f} min={dices.min():.4f}")
        print(f"  IoU : mean={ious.mean():.4f} median={np.median(ious):.4f} min={ious.min():.4f}")


if __name__ == "__main__":
    main()
