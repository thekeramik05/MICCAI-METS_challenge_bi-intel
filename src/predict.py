#!/usr/bin/env python3
"""BraTS-METS 2026 Task 1 — containerised inference entrypoint.

Pipeline
--------
1. Discover every case folder under /input (read-only, never written to).
2. Run two independently trained nnU-Net v2 models:
     A. Triad(PlainConvUNet-MAE)-initialised nnU-Net   — folds 0-4
     B. BrainIAC (frozen ViT-B/16) feature-concat wrapper — folds 1, 3
3. Average the two softmax maps with per-class weights (RC leans to model B).
4. Apply size-gated, class-wise confidence post-processing.
5. Write one flat ``.nii.gz`` per case to /output in the exact input geometry.

All reading and writing goes through nnU-Net's own SimpleITKIO so that voxel
spacing, origin and direction are carried over byte-for-byte from the input.
Every output is additionally re-opened and verified against its input; any
mismatch aborts the whole run loudly rather than shipping a broken submission.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch

from config import (
    CHECKPOINT_NAME,
    DISABLE_TTA,
    ENABLE_POSTPROCESSING,
    INPUT_DIR,
    LABELS,
    LABEL_NAMES,
    MODALITY_SUFFIXES,
    MODEL_A_DIR,
    MODEL_A_FOLDS,
    MODEL_B_DIR,
    MODEL_B_FOLDS,
    OUTPUT_DIR,
    SUBSTITUTABLE_MODALITIES,
    TILE_STEP_SIZE,
)
from postprocess import confidence_postprocess, weighted_average

# Output must end in <5-digit id>-<3-digit timepoint>.nii.gz
CASE_ID_PATTERN = re.compile(r"(\d{5})-(\d{3})$")

# Tolerance for float comparison of spacing / origin / direction
GEOMETRY_ATOL = 1e-4


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------
def discover_cases(input_dir: Path) -> List[Tuple[str, List[Path]]]:
    """Return [(case_id, [t1c, t1n, t2f, t2w]), ...] sorted by case id."""
    if not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")

    cases: List[Tuple[str, List[Path]]] = []
    for case_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        case_id = case_dir.name

        if not CASE_ID_PATTERN.search(case_id):
            print(f"[warn] skipping '{case_id}': does not end in <5 digits>-<3 digits>", flush=True)
            continue

        found: dict[str, Path] = {}
        for suffix in MODALITY_SUFFIXES:
            matches = sorted(case_dir.glob(f"*-{suffix}.nii.gz"))
            if len(matches) > 1:
                raise RuntimeError(
                    f"Case '{case_id}': found {len(matches)} files matching '*-{suffix}.nii.gz', expected one"
                )
            if matches:
                found[suffix] = matches[0]

        # T2W is optional in BraTS-METS since 2025 — some cases have native T2,
        # some synthetic, some none at all. The networks need all four channels,
        # so substitute the FLAIR (the closest available T2-weighted contrast)
        # rather than crashing: a crash invalidates the entire submission.
        missing = [s for s in MODALITY_SUFFIXES if s not in found]
        for suffix in missing:
            if suffix not in SUBSTITUTABLE_MODALITIES:
                raise RuntimeError(
                    f"Case '{case_id}': required modality '{suffix}' is missing and cannot be substituted"
                )
            fallback = SUBSTITUTABLE_MODALITIES[suffix]
            if fallback not in found:
                raise RuntimeError(
                    f"Case '{case_id}': '{suffix}' missing and fallback '{fallback}' is absent too"
                )
            print(
                f"[warn] {case_id}: no '{suffix}' image; substituting '{fallback}'",
                flush=True,
            )
            found[suffix] = found[fallback]

        cases.append((case_id, [found[s] for s in MODALITY_SUFFIXES]))

    if not cases:
        raise RuntimeError(f"No valid case folders found under {input_dir}")
    return cases


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------
def build_predictor(model_dir: Path, folds: Sequence[int]):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    if not model_dir.is_dir():
        raise RuntimeError(f"Model directory not found: {model_dir}")
    for fold in folds:
        checkpoint = model_dir / f"fold_{fold}" / CHECKPOINT_NAME
        if not checkpoint.is_file():
            raise RuntimeError(f"Missing checkpoint: {checkpoint}")

    predictor = nnUNetPredictor(
        tile_step_size=TILE_STEP_SIZE,
        use_gaussian=True,
        use_mirroring=not DISABLE_TTA,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=tuple(folds), checkpoint_name=CHECKPOINT_NAME
    )
    return predictor


# ---------------------------------------------------------------------------
# Spatial verification
# ---------------------------------------------------------------------------
def verify_geometry(reference_file: Path, produced_file: Path, case_id: str) -> None:
    """Abort if the written segmentation does not match the input geometry."""
    reference = sitk.ReadImage(str(reference_file))
    produced = sitk.ReadImage(str(produced_file))

    problems: List[str] = []
    if reference.GetSize() != produced.GetSize():
        problems.append(f"size {produced.GetSize()} != {reference.GetSize()}")
    if not np.allclose(reference.GetSpacing(), produced.GetSpacing(), atol=GEOMETRY_ATOL):
        problems.append(f"spacing {produced.GetSpacing()} != {reference.GetSpacing()}")
    if not np.allclose(reference.GetOrigin(), produced.GetOrigin(), atol=GEOMETRY_ATOL):
        problems.append(f"origin {produced.GetOrigin()} != {reference.GetOrigin()}")
    if not np.allclose(reference.GetDirection(), produced.GetDirection(), atol=GEOMETRY_ATOL):
        problems.append(f"direction {produced.GetDirection()} != {reference.GetDirection()}")

    if problems:
        raise RuntimeError(
            f"Spatial metadata mismatch for case '{case_id}':\n  " + "\n  ".join(problems)
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="BraTS-METS Task 1 inference")
    parser.add_argument("--input", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list discovered cases and verify model files, then exit.",
    )
    args = parser.parse_args()

    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    print("=" * 78, flush=True)
    print("BraTS-METS 2026 Task 1 — ensemble inference", flush=True)
    print(f"  input            : {args.input}", flush=True)
    print(f"  output           : {args.output}", flush=True)
    print(f"  model A folds    : {MODEL_A_FOLDS}", flush=True)
    print(f"  model B folds    : {MODEL_B_FOLDS}", flush=True)
    print(f"  TTA (mirroring)  : {'off' if DISABLE_TTA else 'on'}", flush=True)
    print(f"  post-processing  : {'on' if ENABLE_POSTPROCESSING else 'off'}", flush=True)
    print(f"  CUDA available   : {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU              : {name} ({total_gb:.1f} GiB)", flush=True)
    print("=" * 78, flush=True)

    cases = discover_cases(args.input)
    print(f"discovered {len(cases)} case(s)", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)

    predictor_a = build_predictor(MODEL_A_DIR, MODEL_A_FOLDS)
    predictor_b = build_predictor(MODEL_B_DIR, MODEL_B_FOLDS)
    print("both models initialised", flush=True)

    if args.dry_run:
        for case_id, files in cases:
            print(f"  {case_id}: {[f.name for f in files]}", flush=True)
        print("dry run complete", flush=True)
        return 0

    reader = SimpleITKIO()
    pp_totals: Dict[int, List[int]] = {label: [0, 0] for label in LABELS}
    started = time.time()

    for index, (case_id, modality_files) in enumerate(cases, start=1):
        case_started = time.time()

        image, properties = reader.read_images([str(f) for f in modality_files])

        _, probs_a = predictor_a.predict_single_npy_array(
            image, properties, None, None, True
        )
        _, probs_b = predictor_b.predict_single_npy_array(
            image, properties, None, None, True
        )

        ensembled = weighted_average(np.asarray(probs_a), np.asarray(probs_b))
        segmentation = ensembled.argmax(0).astype(np.uint8)

        if ENABLE_POSTPROCESSING:
            confidence = ensembled.max(0)
            segmentation, stats = confidence_postprocess(segmentation, confidence)
            for label, (checked, removed) in stats.items():
                pp_totals[label][0] += checked
                pp_totals[label][1] += removed

        output_file = args.output / f"{case_id}.nii.gz"
        reader.write_seg(segmentation, str(output_file), properties)

        # Fail loudly rather than submitting a geometrically wrong mask.
        verify_geometry(modality_files[0], output_file, case_id)

        elapsed = time.time() - case_started
        total_elapsed = time.time() - started
        eta = (total_elapsed / index) * (len(cases) - index)
        print(
            f"[{index}/{len(cases)}] {case_id}  {elapsed:6.1f}s  "
            f"elapsed {total_elapsed / 60:6.1f}m  eta {eta / 60:6.1f}m",
            flush=True,
        )

    if ENABLE_POSTPROCESSING:
        print("\npost-processing summary (components < cutoff):", flush=True)
        for label in LABELS:
            checked, removed = pp_totals[label]
            pct = (removed / checked * 100) if checked else 0.0
            print(
                f"  {LABEL_NAMES[label]:>5s}: removed {removed}/{checked} ({pct:.1f}%)",
                flush=True,
            )

    produced = sorted(args.output.glob("*.nii.gz"))
    print(f"\nwrote {len(produced)} segmentation(s) in {(time.time() - started) / 60:.1f} min", flush=True)
    if len(produced) != len(cases):
        raise RuntimeError(f"Expected {len(cases)} outputs, found {len(produced)}")

    # /output must stay flat.
    subdirs = [p for p in args.output.iterdir() if p.is_dir()]
    if subdirs:
        raise RuntimeError(f"Output must be flat but contains subfolders: {subdirs}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
