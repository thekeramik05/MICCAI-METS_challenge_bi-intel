#!/usr/bin/env python3
"""D1(Triad FT900) 단독: min_size=10 적용 시 사라지는(size<10) 컴포넌트의 클래스별 confidence 평균."""
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

FOLDS = [1, 3]
LABELS = [1, 2, 3, 4]
LABEL_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
MIN_SIZE = 10
STRUCT = np.ones((3, 3, 3), dtype=np.int8)
MAX_WORKERS = 16


def load_probs(npz_path: Path) -> np.ndarray:
    probs = np.load(npz_path, allow_pickle=True)["probabilities"]
    return np.transpose(probs, (0, 3, 2, 1))


def process_case(args):
    fold, case_stem = args
    gt = np.asarray(nib.load(GT_DIR / f"{case_stem}.nii.gz").dataobj).astype(np.uint8)
    d1 = load_probs(D1_FT / f"fold_{fold}/validation/{case_stem}.npz")
    pred = d1.argmax(0).astype(np.uint8)
    conf = d1.max(0)

    out = {lab: {"all": [], "tp": [], "fp": []} for lab in LABELS}
    for lab in LABELS:
        pred_mask = pred == lab
        gt_mask = gt == lab
        labeled, n = ndimage.label(pred_mask, structure=STRUCT)
        if n == 0:
            continue
        sizes = ndimage.sum(pred_mask, labeled, index=np.arange(1, n + 1))
        mean_conf = ndimage.mean(conf, labeled, index=np.arange(1, n + 1))
        overlap = ndimage.sum(gt_mask, labeled, index=np.arange(1, n + 1))

        for i in range(n):
            if sizes[i] < MIN_SIZE:
                c = float(mean_conf[i])
                out[lab]["all"].append(c)
                if overlap[i] > 0:
                    out[lab]["tp"].append(c)
                else:
                    out[lab]["fp"].append(c)
    return out


def main():
    agg = {lab: {"all": [], "tp": [], "fp": []} for lab in LABELS}

    for f in FOLDS:
        cases = sorted(p.stem for p in (D1_FT / f"fold_{f}/validation").glob("*.npz"))
        args = [(f, c) for c in cases]
        print(f"fold_{f}: {len(cases)} cases", flush=True)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for i, res in enumerate(ex.map(process_case, args), 1):
                for lab in LABELS:
                    for k in ("all", "tp", "fp"):
                        agg[lab][k].extend(res[lab][k])
                if i % 100 == 0 or i == len(cases):
                    print(f"  fold_{f}: {i}/{len(cases)} done", flush=True)

    print(f"\n===== D1 단독, min_size={MIN_SIZE} 적용 시 사라지는 컴포넌트의 클래스별 confidence =====")
    print(f"{'class':>6s} {'n_all':>6s} {'mean_all':>9s} | {'n_tp':>5s} {'mean_tp':>8s} | {'n_fp':>5s} {'mean_fp':>8s}")
    for lab in LABELS:
        a = np.array(agg[lab]["all"])
        t = np.array(agg[lab]["tp"])
        fp = np.array(agg[lab]["fp"])
        print(f"{LABEL_NAMES[lab]:>6s} {len(a):6d} {a.mean() if len(a) else float('nan'):9.3f} | "
              f"{len(t):5d} {t.mean() if len(t) else float('nan'):8.3f} | "
              f"{len(fp):5d} {fp.mean() if len(fp) else float('nan'):8.3f}")


if __name__ == "__main__":
    main()
