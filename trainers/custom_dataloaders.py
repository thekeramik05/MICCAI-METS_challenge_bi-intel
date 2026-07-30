import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class WeightednnUNetDataLoader(nnUNetDataLoader):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = None
        seg_all = None
        weight_all = None

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)

                    data, seg, seg_prev, properties = self._data.load_case(i)
                    shape = data.shape[1:]

                    bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties["class_locations"])
                    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]

                    data_cropped = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    seg_cropped = torch.from_numpy(
                        crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16)
                    ).to(torch.int16)
                    if seg_prev is not None:
                        seg_prev_cropped = torch.from_numpy(
                            crop_and_pad_nd(seg_prev, bbox, -1, cast_cropped_to=np.int16)
                        ).to(torch.int16)
                        seg_cropped = torch.cat((seg_cropped, seg_prev_cropped[None]), dim=0)

                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        seg_cropped = seg_cropped[:, 0]

                    weight_sample = None
                    if self.transforms is not None:
                        transformed = self.transforms(**{"image": data_cropped, "segmentation": seg_cropped})
                        data_sample = transformed["image"]
                        seg_sample = transformed["segmentation"]
                        weight_sample = transformed.get("voxel_weight")
                    else:
                        data_sample = data_cropped
                        seg_sample = seg_cropped

                    if data_all is None:
                        data_all = torch.empty((self.batch_size, *data_sample.shape), dtype=torch.float32)
                    data_all[j] = data_sample

                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [torch.empty((self.batch_size, *s.shape), dtype=s.dtype) for s in seg_sample]
                        for s_idx, s in enumerate(seg_sample):
                            seg_all[s_idx][j] = s
                    else:
                        if seg_all is None:
                            seg_all = torch.empty((self.batch_size, *seg_sample.shape), dtype=seg_sample.dtype)
                        seg_all[j] = seg_sample

                    if weight_sample is not None:
                        if isinstance(weight_sample, list):
                            if weight_all is None:
                                weight_all = [
                                    torch.empty((self.batch_size, *w.shape), dtype=w.dtype) for w in weight_sample
                                ]
                            for w_idx, w in enumerate(weight_sample):
                                weight_all[w_idx][j] = w
                        else:
                            if weight_all is None:
                                weight_all = torch.empty(
                                    (self.batch_size, *weight_sample.shape), dtype=weight_sample.dtype
                                )
                            weight_all[j] = weight_sample

        batch = {"data": data_all, "target": seg_all, "keys": selected_keys}
        if weight_all is not None:
            batch["voxel_weight"] = weight_all
        return batch


def get_dataloaders_with_train_sampling(
    trainer,
    train_sampling_probabilities=None,
    small_lesion_target_labels=None,
):
    """
    nnUNetTrainer.get_dataloaders()와 동일하지만 train loader에만 case-level sampling 적용.
    small_lesion_target_labels가 주어지면 train/val transform에 small lesion voxel weight map 생성 transform을 추가.
    """
    if trainer.dataset_class is None:
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

        trainer.dataset_class = infer_dataset_class(trainer.preprocessed_dataset_folder)

    patch_size = trainer.configuration_manager.patch_size
    deep_supervision_scales = trainer._get_deep_supervision_scales()
    (
        rotation_for_DA,
        do_dummy_2d_data_aug,
        initial_patch_size,
        mirror_axes,
    ) = trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

    tr_transforms = trainer.get_training_transforms(
        patch_size,
        rotation_for_DA,
        deep_supervision_scales,
        mirror_axes,
        do_dummy_2d_data_aug,
        use_mask_for_norm=trainer.configuration_manager.use_mask_for_norm,
        is_cascaded=trainer.is_cascaded,
        foreground_labels=trainer.label_manager.foreground_labels,
        regions=trainer.label_manager.foreground_regions if trainer.label_manager.has_regions else None,
        ignore_label=trainer.label_manager.ignore_label,
    )

    val_transforms = trainer.get_validation_transforms(
        deep_supervision_scales,
        is_cascaded=trainer.is_cascaded,
        foreground_labels=trainer.label_manager.foreground_labels,
        regions=trainer.label_manager.foreground_regions if trainer.label_manager.has_regions else None,
        ignore_label=trainer.label_manager.ignore_label,
    )

    loader_cls = nnUNetDataLoader
    if small_lesion_target_labels is not None:
        from nnunetv2.training.nnUNetTrainer.variants.small_lesion_weighting import AddSmallLesionWeightMapTransform

        weight_transform = AddSmallLesionWeightMapTransform(target_labels=small_lesion_target_labels)
        tr_transforms = ComposeTransforms([tr_transforms, weight_transform])
        val_transforms = ComposeTransforms([val_transforms, weight_transform])
        loader_cls = WeightednnUNetDataLoader

    dataset_tr, dataset_val = trainer.get_tr_and_val_datasets()
    dl_tr = loader_cls(
        dataset_tr,
        trainer.batch_size,
        initial_patch_size,
        trainer.configuration_manager.patch_size,
        trainer.label_manager,
        oversample_foreground_percent=trainer.oversample_foreground_percent,
        sampling_probabilities=train_sampling_probabilities,
        pad_sides=None,
        transforms=tr_transforms,
        probabilistic_oversampling=trainer.probabilistic_oversampling,
    )
    dl_val = loader_cls(
        dataset_val,
        trainer.batch_size,
        trainer.configuration_manager.patch_size,
        trainer.configuration_manager.patch_size,
        trainer.label_manager,
        oversample_foreground_percent=trainer.oversample_foreground_percent,
        sampling_probabilities=None,
        pad_sides=None,
        transforms=val_transforms,
        probabilistic_oversampling=trainer.probabilistic_oversampling,
    )

    allowed_num_processes = get_allowed_n_proc_DA()
    if allowed_num_processes == 0:
        mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
        mt_gen_val = SingleThreadedAugmenter(dl_val, None)
    else:
        mt_gen_train = NonDetMultiThreadedAugmenter(
            data_loader=dl_tr,
            transform=None,
            num_processes=allowed_num_processes,
            num_cached=max(6, allowed_num_processes // 2),
            seeds=None,
            pin_memory=trainer.device.type == "cuda",
            wait_time=0.002,
        )
        mt_gen_val = NonDetMultiThreadedAugmenter(
            data_loader=dl_val,
            transform=None,
            num_processes=max(1, allowed_num_processes // 2),
            num_cached=max(3, allowed_num_processes // 4),
            seeds=None,
            pin_memory=trainer.device.type == "cuda",
            wait_time=0.002,
        )

    _ = next(mt_gen_train)
    _ = next(mt_gen_val)
    return mt_gen_train, mt_gen_val
