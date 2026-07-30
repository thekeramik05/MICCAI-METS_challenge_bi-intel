#!/usr/bin/env python3
"""Simple(50/50) vs 클래스별 가중치 앙상블(그리드서치 결과), fold_1+3.
joint argmax 기준 실제 Dice + 크기bin별 TP/FN + FP confidence를 한 번에 계산."""
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
LABELS = [1, 2, 3, 4]
LABEL_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
W_D1_WEIGHTED = {0: 0.5, 1: 0.20, 2: 0.45, 3: 0.50, 4: 0.50}  # 그리드서치 결과

OVERLAP_THRESHOLD = 0.1
STRUCT = np.ones((3, 3, 3), dtype=np.int8)
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

        dice_stats = {}
        bin_stats = {lab: {b: {"tp": 0, "fn": 0} for b in BIN_ORDER} for lab in LABELS}
        conf_stats = {lab: {"tp": [], "fn": [], "fp": []} for lab in LABELS}

        for lab in LABELS:
            gt_mask = gt == lab
            pred_mask = pred == lab

            tp_vox = int(np.logical_and(pred_mask, gt_mask).sum())
            fp_vox = int(np.logical_and(pred_mask, ~gt_mask).sum())
            fn_vox = int(np.logical_and(~pred_mask, gt_mask).sum())
            dice_stats[lab] = (tp_vox, fp_vox, fn_vox)

            gt_labeled, n_gt = ndimage.label(gt_mask, structure=STRUCT)
            if n_gt > 0:
                gsizes = ndimage.sum(gt_mask, gt_labeled, index=np.arange(1, n_gt + 1))
                overlap = ndimage.sum(pred_mask, gt_labeled, index=np.arange(1, n_gt + 1))
                gt_conf = ndimage.mean(conf, gt_labeled, index=np.arange(1, n_gt + 1))
                ratio = overlap / gsizes
                detected = ratio >= OVERLAP_THRESHOLD
                for i in range(n_gt):
                    b = size_bin(int(gsizes[i]))
                    key = "tp" if detected[i] else "fn"
                    bin_stats[lab][b][key] += 1
                    conf_stats[lab][key].append(float(gt_conf[i]))

            pred_labeled, n_pred = ndimage.label(pred_mask, structure=STRUCT)
            if n_pred > 0:
                poverlap = ndimage.sum(gt_mask, pred_labeled, index=np.arange(1, n_pred + 1))
                pred_conf = ndimage.mean(conf, pred_labeled, index=np.arange(1, n_pred + 1))
                for i in range(n_pred):
                    if poverlap[i] == 0:
                        conf_stats[lab]["fp"].append(float(pred_conf[i]))

        out[name] = {"dice": dice_stats, "bins": bin_stats, "conf": conf_stats}
    return out


def dice(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else float("nan")


def new_agg():
    agg = {}
    for name in ["Simple5050", "Weighted"]:
        agg[name] = {
            "dice": {lab: [0, 0, 0] for lab in LABELS},
            "bins": {lab: {b: {"tp": 0, "fn": 0} for b in BIN_ORDER} for lab in LABELS},
            "conf": {lab: {"tp": [], "fn": [], "fp": []} for lab in LABELS},
        }
    return agg


def merge(agg, res):
    for name in ["Simple5050", "Weighted"]:
        for lab in LABELS:
            tp, fp, fn = res[name]["dice"][lab]
            agg[name]["dice"][lab][0] += tp
            agg[name]["dice"][lab][1] += fp
            agg[name]["dice"][lab][2] += fn
            for b in BIN_ORDER:
                agg[name]["bins"][lab][b]["tp"] += res[name]["bins"][lab][b]["tp"]
                agg[name]["bins"][lab][b]["fn"] += res[name]["bins"][lab][b]["fn"]
            for k in ("tp", "fn", "fp"):
                agg[name]["conf"][lab][k].extend(res[name]["conf"][lab][k])


def main():
    agg = new_agg()
    for f in FOLDS:
        cases = sorted(p.stem for p in (D1_FT / f"fold_{f}/validation").glob("*.npz"))
        args = [(f, c) for c in cases]
        print(f"fold_{f}: {len(cases)} cases", flush=True)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for i, res in enumerate(ex.map(process_case, args), 1):
                merge(agg, res)
                if i % 100 == 0 or i == len(cases):
                    print(f"  fold_{f}: {i}/{len(cases)} done", flush=True)

    print(f"\n===== 전체 Dice (joint argmax, fold_1+3) =====")
    print(f"{'class':>6s} {'Simple5050':>11s} {'Weighted':>10s} {'차이':>8s}")
    for lab in LABELS:
        s = dice(*agg["Simple5050"]["dice"][lab])
        w = dice(*agg["Weighted"]["dice"][lab])
        print(f"{LABEL_NAMES[lab]:>6s} {s:11.4f} {w:10.4f} {w-s:+8.4f}")

    for lab in LABELS:
        print(f"\n===== {LABEL_NAMES[lab]}: 크기bin별 검출률 (Simple5050 vs Weighted) =====")
        print(f"{'bin':>8s} {'n':>5s} {'S_TP':>5s} {'S_FN':>5s} {'S_det%':>7s} | {'W_TP':>5s} {'W_FN':>5s} {'W_det%':>7s}")
        for b in BIN_ORDER:
            s = agg["Simple5050"]["bins"][lab][b]
            w = agg["Weighted"]["bins"][lab][b]
            n = s["tp"] + s["fn"]
            if n == 0:
                continue
            sr = s["tp"] / n * 100
            wr = w["tp"] / n * 100
            print(f"{b:>8s} {n:5d} {s['tp']:5d} {s['fn']:5d} {sr:6.1f}% | {w['tp']:5d} {w['fn']:5d} {wr:6.1f}%")

        print(f"-- {LABEL_NAMES[lab]} FP confidence --")
        for name in ["Simple5050", "Weighted"]:
            vals = agg[name]["conf"][lab]["fp"]
            if vals:
                print(f"  {name}: n={len(vals)} mean={np.mean(vals):.3f} median={np.median(vals):.3f}")
            else:
                print(f"  {name}: n=0")


if __name__ == "__main__":
    main()
