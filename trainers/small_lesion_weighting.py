# Copyright 2026 the authors of MICCAI-METS_challenge_bi-intel.
# Licensed under the Apache License, Version 2.0 - see LICENSE at the repository root.
#
# Extends nnU-Net v2 (MIC-DKFZ, Apache-2.0): https://github.com/MIC-DKFZ/nnUNet

import numpy as np
from scipy import ndimage
import torch

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


SMALL_LESION_VOXEL_THRESHOLD = 20
SMALL_LESION_CE_WEIGHT = 10.0


def _build_weight_map_2d_or_3d(
    seg_map: np.ndarray,
    target_labels: tuple,
    voxel_threshold: int,
    small_weight: float,
) -> np.ndarray:
    weight = np.ones_like(seg_map, dtype=np.float32)
    for lbl in target_labels:
        mask = seg_map == lbl
        if not mask.any():
            continue
        labeled, _ = ndimage.label(mask)
        for comp_id in range(1, labeled.max() + 1):
            comp_mask = labeled == comp_id
            if comp_mask.sum() <= voxel_threshold:
                weight[comp_mask] = small_weight
    return weight


class AddSmallLesionWeightMapTransform(BasicTransform):
    """
    segmentation을 보고 voxel-wise CE weight map을 만들어 data_dict['voxel_weight']에 추가.
    deep supervision으로 segmentation이 list인 경우, 가장 높은 해상도(index 0)에만 weight를 부여.
    """

    def __init__(
        self,
        target_labels=(4,),
        voxel_threshold=SMALL_LESION_VOXEL_THRESHOLD,
        small_weight=SMALL_LESION_CE_WEIGHT,
    ):
        super().__init__()
        self.target_labels = target_labels
        self.voxel_threshold = voxel_threshold
        self.small_weight = small_weight

    def apply(self, data_dict, **params):
        seg = data_dict.get("segmentation")
        if seg is None:
            return data_dict

        if isinstance(seg, (list, tuple)):
            weight_maps = []
            for scale_idx, s in enumerate(seg):
                if scale_idx == 0:
                    weight_maps.append(self._build_tensor_weight(s))
                else:
                    weight_maps.append(torch.ones_like(s, dtype=torch.float32))
            data_dict["voxel_weight"] = weight_maps
        else:
            data_dict["voxel_weight"] = self._build_tensor_weight(seg)

        return data_dict

    def _build_tensor_weight(self, seg: torch.Tensor) -> torch.Tensor:
        seg_np = seg[0].detach().cpu().numpy().astype(np.int32)
        weight_np = _build_weight_map_2d_or_3d(
            seg_np,
            target_labels=self.target_labels,
            voxel_threshold=self.voxel_threshold,
            small_weight=self.small_weight,
        )
        return torch.from_numpy(weight_np).to(device=seg.device, dtype=torch.float32).unsqueeze(0)
