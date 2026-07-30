#!/usr/bin/env python3
"""후처리 임계값 근거 시각화: 클래스별 FP confidence 히스토그램(전체) +
클래스x크기bin FP confidence 분포. Simple(50/50) 앙상블, fold_1+3 검증셋 기준."""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
CONF_THRESH = {1: 0.6, 2: 0.6, 3: 0.6, 4: 0.5}
SMALL_SIZE_CUTOFF = 50
STRUCT = np.ones((3, 3, 3), dtype=np.int8)
MAX_WORKERS = 16
OUT_DIR = Path("/tmp/claude-500/-home1/50ed32c3-c2b5-4799-b9e4-b6e297c5e101/scratchpad")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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

    out = {lab: {"fp_small": [], "fp_large": [], "tp_small": [],
                 "scatter_size": [], "scatter_conf": [], "scatter_is_tp": []} for lab in LABELS}
    for lab in LABELS:
        gt_mask = gt == lab
        pred_mask = pred == lab
        labeled, n = ndimage.label(pred_mask, structure=STRUCT)
        if n == 0:
            continue
        sizes = ndimage.sum(pred_mask, labeled, index=np.arange(1, n + 1))
        mean_conf = ndimage.mean(conf, labeled, index=np.arange(1, n + 1))
        overlap = ndimage.sum(gt_mask, labeled, index=np.arange(1, n + 1))

        for i in range(n):
            c = float(mean_conf[i])
            sz = float(sizes[i])
            is_tp = overlap[i] > 0
            small = sz < SMALL_SIZE_CUTOFF
            if not is_tp:
                out[lab]["fp_small" if small else "fp_large"].append(c)
            elif small:
                out[lab]["tp_small"].append(c)

            out[lab]["scatter_size"].append(sz)
            out[lab]["scatter_conf"].append(c)
            out[lab]["scatter_is_tp"].append(is_tp)
    return out


def main():
    keys = ("fp_small", "fp_large", "tp_small", "scatter_size", "scatter_conf", "scatter_is_tp")
    agg = {lab: {k: [] for k in keys} for lab in LABELS}

    for f in FOLDS:
        cases = sorted(p.stem for p in (D1_FT / f"fold_{f}/validation").glob("*.npz"))
        args = [(f, c) for c in cases]
        print(f"fold_{f}: {len(cases)} cases", flush=True)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for i, res in enumerate(ex.map(process_case, args), 1):
                for lab in LABELS:
                    for k in keys:
                        agg[lab][k].extend(res[lab][k])
                if i % 100 == 0 or i == len(cases):
                    print(f"  fold_{f}: {i}/{len(cases)} done", flush=True)

    np.savez(OUT_DIR / "fp_confidence_data.npz",
             **{f"{LABEL_NAMES[lab]}_{k}": np.array(v)
                for lab in LABELS for k, v in agg[lab].items()})

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    bins = np.linspace(0, 1, 21)
    for ax, lab in zip(axes.flat, LABELS):
        name = LABEL_NAMES[lab]
        fp_small = np.array(agg[lab]["fp_small"])
        tp_small = np.array(agg[lab]["tp_small"])
        thr = CONF_THRESH[lab]

        ax.hist(tp_small, bins=bins, alpha=0.55, color="#2b8a3e", label=f"TP (n={len(tp_small)})", density=True)
        ax.hist(fp_small, bins=bins, alpha=0.55, color="#c92a2a", label=f"FP (n={len(fp_small)})", density=True)
        ax.axvline(thr, color="black", linestyle="--", linewidth=1.5, label=f"threshold={thr}")
        ax.set_title(f"{name} (size < {SMALL_SIZE_CUTOFF} voxel components)")
        ax.set_xlabel("mean softmax confidence")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    fig.suptitle("Confidence distribution of small (<50 voxel) components: TP vs FP\n(D1+BrainIAC simple ensemble, fold_1+3 validation)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "fp_confidence_histogram.png", dpi=140)
    print(f"\nsaved -> {OUT_DIR / 'fp_confidence_histogram.png'}")
    plt.close(fig)

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
    for ax, lab in zip(axes2.flat, LABELS):
        name = LABEL_NAMES[lab]
        sizes = np.array(agg[lab]["scatter_size"])
        confs = np.array(agg[lab]["scatter_conf"])
        is_tp = np.array(agg[lab]["scatter_is_tp"], dtype=bool)
        thr = CONF_THRESH[lab]

        sizes_plot = np.clip(sizes, 0.5, None)
        ax.scatter(sizes_plot[is_tp], confs[is_tp], s=8, alpha=0.35, color="#2b8a3e", label=f"TP (n={is_tp.sum()})")
        ax.scatter(sizes_plot[~is_tp], confs[~is_tp], s=8, alpha=0.35, color="#c92a2a", label=f"FP (n={(~is_tp).sum()})")
        ax.axhline(thr, color="black", linestyle="--", linewidth=1.2, label=f"threshold={thr}")
        ax.axvline(SMALL_SIZE_CUTOFF, color="#1971c2", linestyle=":", linewidth=1.2, label=f"size cutoff={SMALL_SIZE_CUTOFF}")
        ax.set_xscale("log")
        ax.set_xlabel("component size (voxels, log scale)")
        ax.set_ylabel("mean softmax confidence")
        ax.set_title(name)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, loc="lower right")

    fig2.suptitle("Component size vs confidence, colored by TP/FP\n(D1+BrainIAC simple ensemble, fold_1+3 validation, all component sizes)", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig(OUT_DIR / "size_confidence_scatter.png", dpi=140)
    print(f"saved -> {OUT_DIR / 'size_confidence_scatter.png'}")

    for lab in LABELS:
        name = LABEL_NAMES[lab]
        fps = np.array(agg[lab]["fp_small"])
        tps = np.array(agg[lab]["tp_small"])
        thr = CONF_THRESH[lab]
        removed_fp = (fps < thr).sum()
        removed_tp = (tps < thr).sum()
        print(f"{name}: threshold={thr} -> FP 제거 {removed_fp}/{len(fps)} ({removed_fp/max(len(fps),1):.1%}), "
              f"TP 손실 {removed_tp}/{len(tps)} ({removed_tp/max(len(tps),1):.1%})")


if __name__ == "__main__":
    main()
