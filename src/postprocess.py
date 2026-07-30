"""Ensemble averaging and size-gated confidence post-processing."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage

from config import (
    CONFIDENCE_THRESHOLD,
    ENSEMBLE_W_MODEL_A,
    LABELS,
    SMALL_SIZE_CUTOFF,
)

# 26-connectivity
_STRUCT = np.ones((3, 3, 3), dtype=np.int8)


def weighted_average(probs_a: np.ndarray, probs_b: np.ndarray) -> np.ndarray:
    """Per-channel weighted average of two softmax maps.

    Both inputs must already live in the same voxel grid (they do: nnU-Net
    returns probabilities resampled back to the original image geometry).
    """
    if probs_a.shape != probs_b.shape:
        raise RuntimeError(
            f"Probability shape mismatch between models: {probs_a.shape} vs {probs_b.shape}"
        )

    out = np.empty_like(probs_a)
    for c in range(probs_a.shape[0]):
        w = ENSEMBLE_W_MODEL_A.get(c, 0.5)
        out[c] = w * probs_a[c] + (1.0 - w) * probs_b[c]
    return out


def confidence_postprocess(
    segmentation: np.ndarray,
    confidence: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
    """Delete small, low-confidence connected components.

    A component is removed only when BOTH hold:
      * it is smaller than ``SMALL_SIZE_CUTOFF`` voxels, and
      * its mean softmax confidence is below the class threshold.

    Returns the cleaned segmentation and, per label, a (checked, removed) pair.
    """
    cleaned = segmentation.copy()
    stats: Dict[int, Tuple[int, int]] = {}

    for label in LABELS:
        mask = segmentation == label
        labeled, n_components = ndimage.label(mask, structure=_STRUCT)
        checked = removed = 0

        if n_components:
            index = np.arange(1, n_components + 1)
            sizes = ndimage.sum(mask, labeled, index=index)
            mean_conf = ndimage.mean(confidence, labeled, index=index)
            threshold = CONFIDENCE_THRESHOLD[label]

            for i, component_id in enumerate(index):
                if sizes[i] >= SMALL_SIZE_CUTOFF:
                    continue
                checked += 1
                if mean_conf[i] < threshold:
                    cleaned[labeled == component_id] = 0
                    removed += 1

        stats[label] = (checked, removed)

    return cleaned, stats
