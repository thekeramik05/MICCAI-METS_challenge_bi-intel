from pathlib import Path
import os

import numpy as np
import pandas as pd

DEFAULT_CASE_INDEX_CSV = Path(
    os.environ.get(
        "MICCAI_CASE_INDEX_CSV",
        "dataset/train/training_files_index.csv",
    )
)

SMALL_LESION_GROUPS = frozenset({"tiny", "small"})
DEFAULT_SMALL_LESION_WEIGHT = 3.0
DEFAULT_BASE_WEIGHT = 1.0


def build_small_lesion_sampling_probabilities(
    case_ids: list[str],
    csv_path: Path | str = DEFAULT_CASE_INDEX_CSV,
    small_lesion_groups: frozenset[str] = SMALL_LESION_GROUPS,
    small_weight: float = DEFAULT_SMALL_LESION_WEIGHT,
    base_weight: float = DEFAULT_BASE_WEIGHT,
) -> np.ndarray:
    """
    tiny/small lesion case에 더 높은 확률을 부여하는 case-level sampling weight 생성.
    nnUNetDataLoader.sampling_probabilities는 합이 1인 1D array여야 함.
    """
    df = pd.read_csv(csv_path)
    if "path" not in df.columns or "lesion_size_group" not in df.columns:
        raise KeyError("training_files_index.csv must contain 'path' and 'lesion_size_group' columns")

    case_to_group = {
        Path(str(path)).name: str(group)
        for path, group in zip(df["path"], df["lesion_size_group"])
    }

    weights = np.array(
        [
            small_weight if case_to_group.get(case_id, "") in small_lesion_groups else base_weight
            for case_id in case_ids
        ],
        dtype=np.float64,
    )
    if weights.sum() <= 0:
        raise RuntimeError("sampling weights sum to zero")
    return weights / weights.sum()
