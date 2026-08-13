# Copyright 2026 the authors of MICCAI-METS_challenge_bi-intel.
# Licensed under the Apache License, Version 2.0 - see LICENSE at the repository root.
#
# Extends nnU-Net v2 (MIC-DKFZ, Apache-2.0): https://github.com/MIC-DKFZ/nnUNet

import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast

from scipy.ndimage import label as cc_label

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context

from nnUNetTrainerBraTS_RCOversample import nnUNetTrainerBraTS_RCOversample
from rc_sampling import DEFAULT_RC_WEIGHT, build_rc_sampling_probabilities
from custom_dataloaders import get_dataloaders_with_train_sampling
from weighted_losses import DC_and_VoxelWeightedCE_loss


class nnUNetTrainerBraTS_SmallLesionWeightedCE(nnUNetTrainerBraTS_RCOversample):
    rc_weight: float = DEFAULT_RC_WEIGHT

    small_lesion_target_labels: tuple = (1, 2, 3, 4)
    small_lesion_max_voxels: int = 20
    small_lesion_weight: float = 2.0
    normalize_voxel_weight: bool = True

    def get_dataloaders(self):
        dataset_tr, _ = self.get_tr_and_val_datasets()

        sampling_probabilities = build_rc_sampling_probabilities(
            case_ids=dataset_tr.identifiers,
            preprocessed_folder=self.preprocessed_dataset_folder,
            rc_weight=self.rc_weight,
        )

        return get_dataloaders_with_train_sampling(
            self,
            train_sampling_probabilities=sampling_probabilities,
            small_lesion_target_labels=self.small_lesion_target_labels,
        )

    def _build_loss(self):
        return DC_and_VoxelWeightedCE_loss(
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
            },
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
        )

    @staticmethod
    def _to_device_maybe_list(x, device):
        if isinstance(x, (list, tuple)):
            return [item.to(device, non_blocking=True) for item in x]

        return x.to(device, non_blocking=True)

    @staticmethod
    def _resize_like(
        x: torch.Tensor,
        reference: torch.Tensor,
        mode: str = "nearest",
    ) -> torch.Tensor:
        if x.shape[2:] == reference.shape[2:]:
            return x

        original_dtype = x.dtype

        if mode == "nearest":
            resized = F.interpolate(
                x.float(),
                size=reference.shape[2:],
                mode="nearest",
            )
        else:
            resized = F.interpolate(
                x.float(),
                size=reference.shape[2:],
                mode=mode,
                align_corners=False,
            )

        return resized.to(dtype=original_dtype)

    def _make_single_voxel_weight(self, target: torch.Tensor) -> torch.Tensor:
        if target.ndim < 4:
            raise RuntimeError(f"Unexpected target shape: {target.shape}")

        device = target.device
        target_np = target.detach().cpu().numpy().astype(np.int16)
        weight_np = np.ones_like(target_np, dtype=np.float32)

        for batch_idx in range(target_np.shape[0]):
            seg = target_np[batch_idx, 0]

            for class_label in self.small_lesion_target_labels:
                class_mask = seg == class_label

                if not np.any(class_mask):
                    continue

                structure = np.ones((3,) * class_mask.ndim, dtype=np.int8)
                labeled_components, num_components = cc_label(
                    class_mask,
                    structure=structure,
                )

                if num_components == 0:
                    continue

                component_sizes = np.bincount(labeled_components.ravel())

                small_component_ids = np.where(
                    (component_sizes > 0)
                    & (component_sizes <= self.small_lesion_max_voxels)
                )[0]

                small_component_ids = small_component_ids[small_component_ids != 0]

                if len(small_component_ids) == 0:
                    continue

                small_component_mask = np.isin(
                    labeled_components,
                    small_component_ids,
                )

                weight_np[batch_idx, 0][small_component_mask] = (
                    self.small_lesion_weight
                )

        return torch.from_numpy(weight_np).to(
            device=device,
            dtype=torch.float32,
        )

    def _make_voxel_weight(self, target):
        if isinstance(target, (list, tuple)):
            return [self._make_single_voxel_weight(t) for t in target]

        return self._make_single_voxel_weight(target)

    def _normalize_single_voxel_weight(
        self,
        voxel_weight: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if not self.normalize_voxel_weight:
            return voxel_weight

        voxel_weight = voxel_weight.float()
        ignore_label = self.label_manager.ignore_label

        if ignore_label is None:
            valid_mask = torch.ones_like(target, dtype=torch.float32)
        else:
            valid_mask = (target != ignore_label).float()

        if valid_mask.shape != voxel_weight.shape:
            valid_mask = self._resize_like(valid_mask, voxel_weight, mode="nearest")

        weighted_sum = (voxel_weight * valid_mask).sum()
        valid_count = valid_mask.sum()

        if valid_count <= 0 or weighted_sum <= 0:
            return voxel_weight

        return voxel_weight * (valid_count / weighted_sum.clamp_min(1e-8))

    def _normalize_voxel_weight(self, voxel_weight, target):
        if isinstance(voxel_weight, (list, tuple)):
            return [
                self._normalize_single_voxel_weight(w, t)
                for w, t in zip(voxel_weight, target)
            ]

        return self._normalize_single_voxel_weight(voxel_weight, target)

    def _maybe_expand_to_deep_supervision_list(self, output, target, voxel_weight):
        if not isinstance(output, (list, tuple)):
            if isinstance(target, (list, tuple)):
                target = target[0]
            if isinstance(voxel_weight, (list, tuple)):
                voxel_weight = voxel_weight[0]

            target = self._resize_like(target, output, mode="nearest")
            voxel_weight = self._resize_like(voxel_weight, output, mode="nearest")

            return output, target, voxel_weight

        output_list = list(output)

        if isinstance(target, (list, tuple)):
            target_list = list(target)
        else:
            target_list = [
                self._resize_like(target, o, mode="nearest")
                for o in output_list
            ]

        if isinstance(voxel_weight, (list, tuple)):
            voxel_weight_list = list(voxel_weight)
        else:
            voxel_weight_list = [
                self._resize_like(voxel_weight, o, mode="nearest")
                for o in output_list
            ]

        target_list = [
            self._resize_like(t, o, mode="nearest")
            for t, o in zip(target_list, output_list)
        ]

        voxel_weight_list = [
            self._resize_like(w, o, mode="nearest")
            for w, o in zip(voxel_weight_list, output_list)
        ]

        return output_list, target_list, voxel_weight_list

    @staticmethod
    def _get_deep_supervision_weights(num_outputs: int, device) -> torch.Tensor:
        if num_outputs == 1:
            return torch.ones(1, device=device, dtype=torch.float32)

        weights = torch.tensor(
            [1.0 / (2**i) for i in range(num_outputs)],
            device=device,
            dtype=torch.float32,
        )

        weights[-1] = 0.0
        weights = weights / weights.sum().clamp_min(1e-8)

        return weights

    def _compute_loss(self, output, target):
        voxel_weight = self._make_voxel_weight(target)

        output, target, voxel_weight = self._maybe_expand_to_deep_supervision_list(
            output,
            target,
            voxel_weight,
        )

        voxel_weight = self._normalize_voxel_weight(voxel_weight, target)

        if not isinstance(output, (list, tuple)):
            return self.loss(output, target, voxel_weight)

        deep_supervision_weights = self._get_deep_supervision_weights(
            num_outputs=len(output),
            device=output[0].device,
        )

        total_loss = None

        for o, t, w, ds_weight in zip(
            output,
            target,
            voxel_weight,
            deep_supervision_weights,
        ):
            if ds_weight == 0:
                continue

            scale_loss = self.loss(o, t, w)
            weighted_loss = ds_weight * scale_loss

            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss

        return total_loss

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_maybe_list(batch["target"], self.device)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self._compute_loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_maybe_list(batch["target"], self.device)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            del data

            loss = self._compute_loss(output, target)

        output_for_metric = output[0] if isinstance(output, (list, tuple)) else output
        target_for_metric = target[0] if isinstance(target, (list, tuple)) else target

        target_for_metric = self._resize_like(
            target_for_metric,
            output_for_metric,
            mode="nearest",
        )

        axes = [0] + list(range(2, output_for_metric.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (
                torch.sigmoid(output_for_metric) > 0.5
            ).long()
        else:
            output_seg = output_for_metric.argmax(1)[:, None]

            predicted_segmentation_onehot = torch.zeros(
                output_for_metric.shape,
                device=output_for_metric.device,
                dtype=torch.float16,
            )

            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_for_metric != self.label_manager.ignore_label).float()
                target_for_metric[target_for_metric == self.label_manager.ignore_label] = 0
            else:
                if target_for_metric.dtype == torch.bool:
                    mask = ~target_for_metric[:, -1:]
                else:
                    mask = 1 - target_for_metric[:, -1:]

                target_for_metric = target_for_metric[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot,
            target_for_metric,
            axes=axes,
            mask=mask,
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }


class nnUNetTrainerBraTS_SmallLesionWeightedCE__2(
    nnUNetTrainerBraTS_SmallLesionWeightedCE
):
    rc_weight: float = 2.0


class nnUNetTrainerBraTS_SmallLesionWeightedCE__3(
    nnUNetTrainerBraTS_SmallLesionWeightedCE
):
    rc_weight: float = 3.0


class nnUNetTrainerBraTS_SmallLesionWeightedCE__5(
    nnUNetTrainerBraTS_SmallLesionWeightedCE
):
    rc_weight: float = 5.0


class nnUNetTrainerBraTS_SmallLesionWeightedCE_CEw2__3(
    nnUNetTrainerBraTS_SmallLesionWeightedCE
):
    rc_weight: float = 3.0
    small_lesion_weight: float = 2.0


class nnUNetTrainerBraTS_SmallLesionWeightedCE_CEw3__3(
    nnUNetTrainerBraTS_SmallLesionWeightedCE
):
    rc_weight: float = 3.0
    small_lesion_weight: float = 3.0


class nnUNetTrainerBraTS_SmallLesionWeightedCE_CEw5__3(
    nnUNetTrainerBraTS_SmallLesionWeightedCE
):
    rc_weight: float = 3.0
    small_lesion_weight: float = 5.0


class nnUNetTrainerBraTS_SmallLesionWeightedCE_RCOnly_CEw3__3(
    nnUNetTrainerBraTS_SmallLesionWeightedCE
):
    rc_weight: float = 3.0
    small_lesion_weight: float = 3.0
    small_lesion_target_labels: tuple = (4,)


class nnUNetTrainerBraTS_SmallLesionWeightedCE_ETRC_CEw3__3(
    nnUNetTrainerBraTS_SmallLesionWeightedCE
):
    rc_weight: float = 3.0
    small_lesion_weight: float = 3.0
    small_lesion_target_labels: tuple = (3, 4)


class nnUNetTrainerBraTS_SmallLesionWeightedCE1234(
    nnUNetTrainerBraTS_SmallLesionWeightedCE
):
    small_lesion_target_labels: tuple = (1, 2, 3, 4)


class nnUNetTrainerBraTS_SmallLesionWeightedCE1234__3(
    nnUNetTrainerBraTS_SmallLesionWeightedCE1234
):
    rc_weight: float = 3.0


class nnUNetTrainerBraTS_SmallLesionWeightedCE1234_CEw10__3(
    nnUNetTrainerBraTS_SmallLesionWeightedCE1234
):
    """RC oversample x3 + small lesion CE weight 10 on labels 1,2,3,4."""

    rc_weight: float = 3.0
    small_lesion_weight: float = 10.0

    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        device=torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.num_val_iterations_per_epoch = 25