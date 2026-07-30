#!/usr/bin/env python3
"""Simple(50/50) vs Weighted 앙상블, 클래스x크기bin별 FP confidence 히스토그램 + 임계값별 제거."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

BASE = Path("/home/irteam/data-vol1/2026_MICCAI_challenge/MICCAI_task_1/nnUNet")
GT_DIR = BASE / "nnUNet_raw/Dataset001_BraTS/labelsTr"
D1_FT = BASE / "nnUNet_results/Dataset001_BraTS/nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3_FT900__nnUNetPlans__3d_fullres"
BRAINIAC = BASE / "nnUNet_results/Dataset001_BraTS/nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__3__nnUNetPlans__3d_fullres"

FOLDS = [1, 3]
LABELS = [1, 2, 3, 4]
LABEL_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
W_D1_WEIGHTED = {0: 0.5, 1: 0.20, 2: 0.45, 3: 0.50, 4: 0.50}

STRUCT = np.ones((3, 3, 3), dtype=np.int8)
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HIST_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BIN_ORDER = ["1", "2-4", "5-9", "10-14", "15-19", "20-49", "50-99", "100+"]
MAX_WORKERS = 16


def size_bin(v: int) -> str:
    if v == 1:
        return "1"
    if 2 <= v <= 4:
        return "2-4"
    if 5 <= v <= 9:
        return "5-9"
    if 10 <= v <= 14:
        return "10-14"
    if 15 <= v <= 19:
        return "15-19"
    if 20 <= v <= 49:
        return "20-49"
    if 50 <= v <= 99:
        return "50-99"
    return "100+"


def load_probs(npz_path: Path) -> np.ndarray:
    probs = np.load(npz_path, allow_pickle=True)["probabilities"]
    return np.transpose(probs, (0, 3, 2, 1))


def process_case(args):
    fold, case_stem = args
    gt = np.asarray(nib.load(GT_DIR / f"{case_stem}.nii.gz").dataobj).astype(np.uint8)
    d1 = load_probs(D1_FT / f"fold_{fold}/validation/{case_stem}.npz")
    br = load_probs(BRAINIAC / f"fold_{fold}/validation/{case_stem}.npz")
    C = d1.shape[0]

    simple = (d1 + br) / 2.0
    weighted = np.zeros_like(d1)
    for c in range(C):
        w = W_D1_WEIGHTED.get(c, 0.5)
        weighted[c] = w * d1[c] + (1 - w) * br[c]

    out = {}
    for name, probs in [("Simple5050", simple), ("Weighted", weighted)]:
        pred = probs.argmax(0).astype(np.uint8)
        conf = probs.max(0)
        fp_by_bin = {lab: {b: [] for b in BIN_ORDER} for lab in LABELS}

        for lab in LABELS:
            gt_mask = gt == lab
            pred_mask = pred == lab
            pred_labeled, n_pred = ndimage.label(pred_mask, structure=STRUCT)
            if n_pred > 0:
                psizes = ndimage.sum(pred_mask, pred_labeled, index=np.arange(1, n_pred + 1))
                poverlap = ndimage.sum(gt_mask, pred_labeled, index=np.arange(1, n_pred + 1))
                pred_conf = ndimage.mean(conf, pred_labeled, index=np.arange(1, n_pred + 1))
                for i in range(n_pred):
                    if poverlap[i] == 0:
                        b = size_bin(int(psizes[i]))
                        fp_by_bin[lab][b].append(float(pred_conf[i]))
        out[name] = fp_by_bin
    return out


def main():
    agg = {name: {lab: {b: [] for b in BIN_ORDER} for lab in LABELS} for name in ["Simple5050", "Weighted"]}

    for f in FOLDS:
        cases = sorted(p.stem for p in (D1_FT / f"fold_{f}/validation").glob("*.npz"))
        args = [(f, c) for c in cases]
        print(f"fold_{f}: {len(cases)} cases", flush=True)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for i, res in enumerate(ex.map(process_case, args), 1):
                for name in agg:
                    for lab in LABELS:
                        for b in BIN_ORDER:
                            agg[name][lab][b].extend(res[name][lab][b])
                if i % 100 == 0 or i == len(cases):
                    print(f"  fold_{f}: {i}/{len(cases)} done", flush=True)

    for lab in LABELS:
        print(f"\n########## {LABEL_NAMES[lab]}: 크기bin별 FP confidence 히스토그램 + 임계값 제거 ##########")
        for b in BIN_ORDER:
            s_vals = np.array(agg["Simple5050"][lab][b])
            w_vals = np.array(agg["Weighted"][lab][b])
            if len(s_vals) == 0 and len(w_vals) == 0:
                continue
            print(f"\n===== {LABEL_NAMES[lab]} / size={b} =====")
            for name, vals in [("Simple5050", s_vals), ("Weighted", w_vals)]:
                n = len(vals)
                print(f"-- {name} (n={n}) --")
                if n == 0:
                    continue
                counts, edges = np.histogram(vals, bins=HIST_BINS)
                for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
                    bar = "#" * int(round(c / max(counts.max(), 1) * 20))
                    print(f"  [{lo:.1f},{hi:.1f}): {c:3d} {bar}")
                print(f"  mean={vals.mean():.3f} median={np.median(vals):.3f}")
                sweep = []
                for t in THRESHOLDS:
                    removed = int((vals < t).sum())
                    sweep.append(f"T{t}:{removed}/{n}({removed/n:.0%})")
                print("  " + "  ".join(sweep))


if __name__ == "__main__":
    main()
