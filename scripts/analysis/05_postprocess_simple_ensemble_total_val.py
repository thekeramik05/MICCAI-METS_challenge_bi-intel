#!/usr/bin/env python3
"""Simple(50/50) D1+BrainIAC total_val 앙상블에 클래스별/크기별 confidence 후처리 적용.

규칙: 컴포넌트 크기 < SMALL_SIZE_CUTOFF(50복셀) 인 것만 대상으로,
클래스별 confidence 임계값(CONF_THRESH) 미달 시 제거. 50복셀 이상은 그대로 둠
(fold_1+3 검증에서 confidence가 더 이상 안 갈리는 구간으로 확인됨).
"""
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(os.environ.get("MICCAI_PROJECT_ROOT", "."))

OUT_ROOT = PROJECT_ROOT / "outputs"
D1_DIR = OUT_ROOT / "total_val_D1_5fold_probs"
BRAINIAC_DIR = OUT_ROOT / "total_val_brainiac_f13_probs"
RAW_OUT_DIR = OUT_ROOT / "total_val_D1_brainiac_ensemble_fixed"  # 후처리 전 simple ensemble (이미 존재)
PP_OUT_DIR = OUT_ROOT / "total_val_D1_brainiac_ensemble_simple_pp"

LABELS = [1, 2, 3, 4]
LABEL_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
CONF_THRESH = {1: 0.6, 2: 0.6, 3: 0.6, 4: 0.5}
SMALL_SIZE_CUTOFF = 50  # 이 미만 크기의 컴포넌트만 confidence 검사 대상
STRUCT = np.ones((3, 3, 3), dtype=np.int8)


def main():
    PP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = sorted(p.stem for p in D1_DIR.glob("*.npz"))
    print(f"{len(cases)} cases", flush=True)

    total_removed = {lab: 0 for lab in LABELS}
    total_checked = {lab: 0 for lab in LABELS}

    for i, case in enumerate(cases, 1):
        d1p = np.load(D1_DIR / f"{case}.npz")["probabilities"]
        brp = np.load(BRAINIAC_DIR / f"{case}.npz")["probabilities"]
        ens = (d1p + brp) / 2.0

        pred_zyx = ens.argmax(0).astype(np.uint8)  # (Z,Y,X) 내부 좌표계
        conf_zyx = ens.max(0)

        pp_pred = pred_zyx.copy()
        for lab in LABELS:
            mask = pred_zyx == lab
            labeled, n = ndimage.label(mask, structure=STRUCT)
            if n == 0:
                continue
            sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
            mean_conf = ndimage.mean(conf_zyx, labeled, index=np.arange(1, n + 1))

            for comp_id in range(1, n + 1):
                sz = sizes[comp_id - 1]
                if sz >= SMALL_SIZE_CUTOFF:
                    continue
                total_checked[lab] += 1
                if mean_conf[comp_id - 1] < CONF_THRESH[lab]:
                    pp_pred[labeled == comp_id] = 0  # background로 제거
                    total_removed[lab] += 1

        pred_xyz = np.transpose(pp_pred, (2, 1, 0))

        ref = nib.load(RAW_OUT_DIR / f"{case}.nii.gz")
        out_img = nib.Nifti1Image(pred_xyz, ref.affine, ref.header)
        nib.save(out_img, str(PP_OUT_DIR / f"{case}.nii.gz"))

        if i % 20 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)} done", flush=True)

    print(f"\n===== 후처리로 제거된 컴포넌트 (크기<{SMALL_SIZE_CUTOFF}, class별 T 미달) =====")
    for lab in LABELS:
        c, r = total_checked[lab], total_removed[lab]
        pct = r / c * 100 if c else 0
        print(f"  {LABEL_NAMES[lab]}: 검사대상 {c}개 중 {r}개 제거 ({pct:.1f}%)")

    print(f"\nsaved -> {PP_OUT_DIR}")


if __name__ == "__main__":
    main()
