#!/usr/bin/env python3
"""ET/NETC/RC 컴포넌트가 SNFH(label 2)와 붙어있지 않은 경우("계층 위반")가
실제로 FP와 상관있는지 fold_1+3 검증셋에서 확인.

SNFH를 1복셀 dilation한 마스크와 겹치지 않는(=인접하지 않는) 컴포넌트를
"계층 위반"으로 분류하고, TP/FP 비율과 confidence를 계층 위반/정상으로 나눠 비교.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(os.environ.get("MICCAI_PROJECT_ROOT", "."))

BASE = PROJECT_ROOT / "nnUNet"
GT_DIR = BASE / "nnUNet_raw/Dataset001_BraTS/labelsTr"
D1_FT = BASE / "nnUNet_results/Dataset001_BraTS/nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3_FT900__nnUNetPlans__3d_fullres"
BRAINIAC = BASE / "nnUNet_results/Dataset001_BraTS/nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__3__nnUNetPlans__3d_fullres"

FOLDS = [1, 3]
CHECK_LABELS = [1, 3, 4]  # NETC, ET, RC (SNFH=2 자체는 검사 대상 아님)
LABEL_NAMES = {1: "NETC", 3: "ET", 4: "RC"}
SNFH_LABEL = 2
STRUCT = np.ones((3, 3, 3), dtype=np.int8)
MAX_WORKERS = 16


def load_probs(npz_path: Path) -> np.ndarray:
    probs = np.load(npz_path, allow_pickle=True)["probabilities"]
    return np.transpose(probs, (0, 3, 2, 1))


def process_case(args):
    fold, case_stem = args
    gt = np.asarray(nib.load(GT_DIR / f"{case_stem}.nii.gz").dataobj).astype(np.uint8)
    d1 = load_probs(D1_FT / f"fold_{fold}/validation/{case_stem}.npz")
    br = load_probs(BRAINIAC / f"fold_{fold}/validation/{case_stem}.npz")
    ens = (d1 + br) / 2.0

    pred = ens.argmax(0).astype(np.uint8)
    conf = ens.max(0)

    snfh_mask = pred == SNFH_LABEL
    snfh_dilated = ndimage.binary_dilation(snfh_mask, structure=STRUCT, iterations=1)

    out = {lab: {"violate": {"tp": [], "fp": [], "size": []}, "ok": {"tp": [], "fp": [], "size": []}} for lab in CHECK_LABELS}
    for lab in CHECK_LABELS:
        pred_mask = pred == lab
        gt_mask = gt == lab
        labeled, n = ndimage.label(pred_mask, structure=STRUCT)
        if n == 0:
            continue
        sizes = ndimage.sum(pred_mask, labeled, index=np.arange(1, n + 1))
        mean_conf = ndimage.mean(conf, labeled, index=np.arange(1, n + 1))
        overlap = ndimage.sum(gt_mask, labeled, index=np.arange(1, n + 1))
        touches_snfh = ndimage.sum(snfh_dilated, labeled, index=np.arange(1, n + 1)) > 0

        for i in range(n):
            key = "ok" if touches_snfh[i] else "violate"
            c = float(mean_conf[i])
            out[lab][key]["size"].append(float(sizes[i]))
            if overlap[i] > 0:
                out[lab][key]["tp"].append(c)
            else:
                out[lab][key]["fp"].append(c)
    return out


def main():
    agg = {lab: {"violate": {"tp": [], "fp": [], "size": []}, "ok": {"tp": [], "fp": [], "size": []}} for lab in CHECK_LABELS}

    for f in FOLDS:
        cases = sorted(p.stem for p in (D1_FT / f"fold_{f}/validation").glob("*.npz"))
        args = [(f, c) for c in cases]
        print(f"fold_{f}: {len(cases)} cases", flush=True)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for i, res in enumerate(ex.map(process_case, args), 1):
                for lab in CHECK_LABELS:
                    for key in ("violate", "ok"):
                        for k in ("tp", "fp", "size"):
                            agg[lab][key][k].extend(res[lab][key][k])
                if i % 100 == 0 or i == len(cases):
                    print(f"  fold_{f}: {i}/{len(cases)} done", flush=True)

    print("\n===== SNFH 비인접(계층 위반) vs 인접(정상) 컴포넌트의 TP/FP 비율 =====")
    for lab in CHECK_LABELS:
        print(f"\n--- {LABEL_NAMES[lab]} ---")
        for key, label_kr in [("violate", "SNFH 비인접(위반)"), ("ok", "SNFH 인접(정상)")]:
            tp = len(agg[lab][key]["tp"])
            fp = len(agg[lab][key]["fp"])
            total = tp + fp
            sizes = np.array(agg[lab][key]["size"])
            fp_conf = np.array(agg[lab][key]["fp"])
            tp_conf = np.array(agg[lab][key]["tp"])
            fp_rate = fp / total * 100 if total else float("nan")
            print(f"  {label_kr}: 총 {total}개 (TP {tp}, FP {fp}, FP율 {fp_rate:.1f}%), "
                  f"평균크기 {sizes.mean() if len(sizes) else float('nan'):.1f}복셀, "
                  f"FP평균conf {fp_conf.mean() if len(fp_conf) else float('nan'):.3f}, "
                  f"TP평균conf {tp_conf.mean() if len(tp_conf) else float('nan'):.3f}")


if __name__ == "__main__":
    main()
