#!/usr/bin/env python3
"""그리드서치 클래스별 가중치를 적용한 D1(5-fold)+BrainIAC(fold1,3) total_val 최종 앙상블.

좌표계 버그 수정판과 동일한 방식 사용: export_prediction_from_logits를 쓰지 않고
단순 argmax + transpose(2,1,0) + 원본 이미지 affine 재사용 (검증 완료된 방법).
"""
from pathlib import Path

import nibabel as nib
import numpy as np

OUT_ROOT = Path("/home/irteam/data-vol1/2026_MICCAI_challenge/MICCAI_task_1/outputs")
D1_DIR = OUT_ROOT / "total_val_D1_5fold_probs"
BRAINIAC_DIR = OUT_ROOT / "total_val_brainiac_f13_probs"
OUT_DIR = OUT_ROOT / "total_val_D1_brainiac_ensemble_weighted"

# 그리드서치 결과 (fold_1+3 coordinate-wise 근사, w_D1 기준). background(0)=0.5 기본.
W_D1 = {0: 0.5, 1: 0.20, 2: 0.45, 3: 0.50, 4: 0.50}


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

        # 검증된 좌표계 변환: argmax -> transpose(Z,Y,X)->(X,Y,Z)
        pred = np.transpose(ens.argmax(0), (2, 1, 0)).astype(np.uint8)

        ref = nib.load(d1_nii_path)  # 정확한 affine/header 재사용
        out_img = nib.Nifti1Image(pred, ref.affine, ref.header)
        nib.save(out_img, str(OUT_DIR / f"{case}.nii.gz"))

        done += 1
        if i % 20 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)} done")

    print(f"saved {done} segmentations -> {OUT_DIR}")


if __name__ == "__main__":
    main()
