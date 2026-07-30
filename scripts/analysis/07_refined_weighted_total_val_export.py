#!/usr/bin/env python3
"""실제 hidden validation set 결과(ET/TC/WT는 50/50 이득, RC/WT는 50/50 손해)에 맞춰
클래스별 가중치를 재조정한 D1+BrainIAC total_val 최종 앙상블.

조정 근거:
- RC: BrainIAC(0.551) >> Triad(0.521) > Ensemble(0.517, 최저) -> BrainIAC 쪽으로 강하게(w_D1=0.15)
- WT(NETC+SNFH+ET 합산): Triad(0.666) > Ensemble=Brainiac(0.663) -> NETC를 Triad 쪽으로(w_D1=0.7)
- SNFH: 근소하게 Triad 유지(w_D1=0.55)
- ET: Ensemble이 이미 최고(0.678) -> 50/50 유지
- 좌표계 버그 없는 검증된 방식(argmax+transpose) 사용.
"""
from pathlib import Path

import nibabel as nib
import numpy as np

OUT_ROOT = Path("/home/irteam/data-vol1/2026_MICCAI_challenge/MICCAI_task_1/outputs")
D1_DIR = OUT_ROOT / "total_val_D1_5fold_probs"
BRAINIAC_DIR = OUT_ROOT / "total_val_brainiac_f13_probs"
OUT_DIR = OUT_ROOT / "total_val_D1_brainiac_ensemble_refined"

W_D1 = {0: 0.5, 1: 0.70, 2: 0.55, 3: 0.50, 4: 0.15}  # NETC, SNFH, ET, RC


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = sorted(p.stem for p in D1_DIR.glob("*.npz"))
    print(f"{len(cases)} candidate cases")

    done = 0
    for i, case in enumerate(cases, 1):
        d1_npz = D1_DIR / f"{case}.npz"
        br_npz = BRAINIAC_DIR / f"{case}.npz"
        d1_nii_path = D1_DIR / f"{case}.nii.gz"
        if not (br_npz.exists() and d1_nii_path.exists()):
            print(f"  SKIP {case}: missing file")
            continue

        d1p = np.load(d1_npz)["probabilities"]
        brp = np.load(br_npz)["probabilities"]
        C = d1p.shape[0]

        ens = np.zeros_like(d1p)
        for c in range(C):
            w = W_D1.get(c, 0.5)
            ens[c] = w * d1p[c] + (1 - w) * brp[c]

        pred = np.transpose(ens.argmax(0), (2, 1, 0)).astype(np.uint8)

        ref = nib.load(d1_nii_path)
        out_img = nib.Nifti1Image(pred, ref.affine, ref.header)
        nib.save(out_img, str(OUT_DIR / f"{case}.nii.gz"))

        done += 1
        if i % 20 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)} done")

    print(f"saved {done} segmentations -> {OUT_DIR}")


if __name__ == "__main__":
    main()
