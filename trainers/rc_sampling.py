# Copyright 2026 the authors of MICCAI-METS_challenge_bi-intel.
# Licensed under the Apache License, Version 2.0 - see LICENSE at the repository root.
#
# Extends nnU-Net v2 (MIC-DKFZ, Apache-2.0): https://github.com/MIC-DKFZ/nnUNet

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np


RC_LABEL = 4
DEFAULT_RC_WEIGHT = 2.0
DEFAULT_BASE_WEIGHT = 1.0


def case_has_rc(identifier: str, preprocessed_folder: Path | str) -> bool:

    properties_file = Path(preprocessed_folder) / f"{identifier}.pkl"

    if not properties_file.is_file():
        raise FileNotFoundError(
            f"Missing preprocessed properties file: {properties_file}"
        )

    with open(properties_file, "rb") as f:
        properties = pickle.load(f)

    class_locations = properties.get("class_locations", {})
    rc_locations = class_locations.get(RC_LABEL, [])

    return len(rc_locations) > 0


def build_rc_sampling_probabilities(
    case_ids: Sequence[str],
    preprocessed_folder: Path | str,
    rc_weight: float = DEFAULT_RC_WEIGHT,
    base_weight: float = DEFAULT_BASE_WEIGHT,
) -> np.ndarray:
    
    weights = np.array(
        [
            rc_weight if case_has_rc(case_id, preprocessed_folder) else base_weight
            for case_id in case_ids
        ],
        dtype=np.float64,
    )

    weight_sum = weights.sum()

    if weight_sum <= 0:
        raise RuntimeError("Sampling weights sum to zero.")

    return weights / weight_sum