#!/usr/bin/env python3
"""
1) D1(Triad) 단독 vs BrainIAC 단독, fold_1+3 검증셋 클래스별 특징 비교
   (Dice, FP 컴포넌트 수/평균크기/평균confidence, TP 컴포넌트 평균크기)
2) Simple(50/50) 앙상블 + 최종 후처리(크기<50 + 클래스별 confidence threshold) 적용 후,
   fold_1+3 검증셋에서 케이스별 multi-class mean Dice 랭킹 (best cases)
"""
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
CONF_THRESH = {1: 0.6, 2: 0.6, 3: 0.6, 4: 0.5}
SMALL_SIZE_CUTOFF = 50
STRUCT = np.ones((3, 3, 3), dtype=np.int8)
MAX_WORKERS = 16


def load_probs(npz_path: Path) -> np.ndarray:
    probs = np.load(npz_path, allow_pickle=True)["probabilities"]
    return np.transpose(probs, (0, 3, 2, 1))


def dice(pred_mask, gt_mask):
    inter = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return None
    return 2 * inter / denom


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


def process_case(args):
    fold, case_stem = args
    gt = np.asarray(nib.load(GT_DIR / f"{case_stem}.nii.gz").dataobj).astype(np.uint8)
    d1p = load_probs(D1_FT / f"fold_{fold}/validation/{case_stem}.npz")
    brp = load_probs(BRAINIAC / f"fold_{fold}/validation/{case_stem}.npz")

    d1_pred = d1p.argmax(0).astype(np.uint8)
    br_pred = brp.argmax(0).astype(np.uint8)

    # --- part 1: D1 alone / BrainIAC alone per-class stats ---
    stats = {"D1": {lab: {"dice": None, "fp_n": 0, "fp_size": [], "fp_conf": [], "tp_size": []} for lab in LABELS},
             "BrainIAC": {lab: {"dice": None, "fp_n": 0, "fp_size": [], "fp_conf": [], "tp_size": []} for lab in LABELS}}

    for name, pred, probs in [("D1", d1_pred, d1p), ("BrainIAC", br_pred, brp)]:
        conf = probs.max(0)
        for lab in LABELS:
            gt_mask = gt == lab
            pred_mask = pred == lab
            d = dice(pred_mask, gt_mask)
            stats[name][lab]["dice"] = d

            labeled, n = ndimage.label(pred_mask, structure=STRUCT)
            if n == 0:
                continue
            sizes = ndimage.sum(pred_mask, labeled, index=np.arange(1, n + 1))
            mean_conf = ndimage.mean(conf, labeled, index=np.arange(1, n + 1))
            overlap = ndimage.sum(gt_mask, labeled, index=np.arange(1, n + 1))
            for i in range(n):
                if overlap[i] == 0:
                    stats[name][lab]["fp_n"] += 1
                    stats[name][lab]["fp_size"].append(float(sizes[i]))
                    stats[name][lab]["fp_conf"].append(float(mean_conf[i]))
                else:
                    stats[name][lab]["tp_size"].append(float(sizes[i]))

    # --- part 2: simple ensemble + PP, per-case mean multiclass dice ---
    ens = (d1p + brp) / 2.0
    pred = ens.argmax(0).astype(np.uint8)
    conf = ens.max(0)
    pp_pred = apply_pp(pred, conf)

    class_dices = []
    for lab in LABELS:
        gt_mask = gt == lab
        if gt_mask.sum() == 0:
            continue
        d = dice(pp_pred == lab, gt_mask)
        if d is not None:
            class_dices.append(d)
    mean_dice = float(np.mean(class_dices)) if class_dices else None

    return case_stem, fold, stats, mean_dice, len(class_dices)


def main():
    all_stats = {"D1": {lab: {"dice": [], "fp_n": 0, "fp_size": [], "fp_conf": [], "tp_size": []} for lab in LABELS},
                  "BrainIAC": {lab: {"dice": [], "fp_n": 0, "fp_size": [], "fp_conf": [], "tp_size": []} for lab in LABELS}}
    case_results = []

    for f in FOLDS:
        cases = sorted(p.stem for p in (D1_FT / f"fold_{f}/validation").glob("*.npz"))
        args = [(f, c) for c in cases]
        print(f"fold_{f}: {len(cases)} cases", flush=True)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for i, (case_stem, fold, stats, mean_dice, n_classes) in enumerate(ex.map(process_case, args), 1):
                for name in ("D1", "BrainIAC"):
                    for lab in LABELS:
                        if stats[name][lab]["dice"] is not None:
                            all_stats[name][lab]["dice"].append(stats[name][lab]["dice"])
                        all_stats[name][lab]["fp_n"] += stats[name][lab]["fp_n"]
                        all_stats[name][lab]["fp_size"].extend(stats[name][lab]["fp_size"])
                        all_stats[name][lab]["fp_conf"].extend(stats[name][lab]["fp_conf"])
                        all_stats[name][lab]["tp_size"].extend(stats[name][lab]["tp_size"])
                if mean_dice is not None:
                    case_results.append((case_stem, fold, mean_dice, n_classes))
                if i % 100 == 0 or i == len(cases):
                    print(f"  fold_{f}: {i}/{len(cases)} done", flush=True)

    print("\n===== Part 1: D1(Triad) 단독 vs BrainIAC 단독 클래스별 특징 (fold_1+3) =====")
    for lab in LABELS:
        print(f"\n--- {LABEL_NAMES[lab]} ---")
        for name in ("D1", "BrainIAC"):
            s = all_stats[name][lab]
            dices = np.array(s["dice"])
            fp_size = np.array(s["fp_size"])
            fp_conf = np.array(s["fp_conf"])
            tp_size = np.array(s["tp_size"])
            print(f"  {name:9s}: meanDice={dices.mean():.3f} | FP개수={s['fp_n']:4d} "
                  f"FP평균크기={fp_size.mean() if len(fp_size) else float('nan'):8.1f} "
                  f"FP평균conf={fp_conf.mean() if len(fp_conf) else float('nan'):.3f} | "
                  f"TP평균크기={tp_size.mean() if len(tp_size) else float('nan'):8.1f}")

    case_results.sort(key=lambda x: x[2], reverse=True)
    print("\n===== Part 2: Simple ensemble + PP 적용 후 fold_1+3 케이스별 mean multiclass Dice TOP 15 =====")
    for case_stem, fold, mean_dice, n_classes in case_results[:15]:
        print(f"  fold_{fold}  {case_stem}  meanDice={mean_dice:.4f}  (present_classes={n_classes})")

    print(f"\n전체 케이스 수: {len(case_results)}, mean_dice==1.0000 케이스 수: {sum(1 for c in case_results if c[2] >= 0.9999)}")


if __name__ == "__main__":
    main()
