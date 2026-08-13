# Copyright 2026 the authors of MICCAI-METS_challenge_bi-intel.
# Licensed under the Apache License, Version 2.0 - see LICENSE at the repository root.
#
# Extends nnU-Net v2 (MIC-DKFZ, Apache-2.0): https://github.com/MIC-DKFZ/nnUNet
#
# Initialises the encoder from the Triad PlainConvUNet-MAE checkpoint
# (https://github.com/wangshansong1/Triad). Triad is used under its upstream
# licence and is modified here by fine-tuning. Verify that licence before
# redistributing any derived weights.

import os
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from nnUNetTrainerBraTS_SmallLesionWeightedCE import nnUNetTrainerBraTS_SmallLesionWeightedCE
from nnUNetTrainerBraTS_TriadInit_RCOversample import (
    TRIAD_CKPT_DEFAULT,
    _freeze_encoder,
    _freeze_encoder_except_stem,
    _load_triad_encoder_into_network,
    _print_trainable_summary,
)
from rc_sampling import DEFAULT_RC_WEIGHT


class nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE(nnUNetTrainerBraTS_SmallLesionWeightedCE):
    """
    Triad encoder init + RC oversampling + all-label small-lesion voxel-weighted CE.

    Combines:
    - Triad pretrained encoder initialization (full fine-tuning by default)
    - RC(label=4) case-level oversampling
    - CE voxel weight on small connected components for all foreground labels

    Small lesion definition:
    - label in small_lesion_target_labels
    - connected component size <= small_lesion_max_voxels (default: 20)
    """

    triad_ckpt: str = str(TRIAD_CKPT_DEFAULT)

    # "none" | "encoder" | "stem_only"
    freeze_mode: str = "none"

    rc_weight: float = DEFAULT_RC_WEIGHT

    # all foreground labels (NET, ED, ET, RC)
    small_lesion_target_labels: tuple = (1, 2, 3, 4)

    small_lesion_max_voxels: int = 20
    small_lesion_weight: float = 2.0
    normalize_voxel_weight: bool = True

    def initialize(self):
        super().initialize()

        checkpoint_path = Path(os.environ.get("TRIAD_CKPT", self.triad_ckpt))

        _load_triad_encoder_into_network(
            network=self.network,
            checkpoint_path=checkpoint_path,
        )

        if self.freeze_mode == "none":
            print("[TriadInit] freeze_mode=none. Full fine-tuning enabled.")

        elif self.freeze_mode == "encoder":
            print("[TriadInit] freeze_mode=encoder. Freezing entire encoder.")
            _freeze_encoder(self.network)

        elif self.freeze_mode == "stem_only":
            print(
                "[TriadInit] freeze_mode=stem_only. "
                "Freezing encoder except first conv block."
            )
            _freeze_encoder_except_stem(self.network)

        else:
            raise ValueError(
                f"Unknown freeze_mode={self.freeze_mode}. "
                "Use one of: none, encoder, stem_only."
            )

        _print_trainable_summary(self.network)


class nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE__2(
    nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE
):
    rc_weight: float = 2.0
    freeze_mode: str = "none"


class nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE__2p5(
    nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE
):
    rc_weight: float = 2.5
    freeze_mode: str = "none"


class nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE__3(
    nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE
):
    """
    Main recommended experiment.

    Triad encoder init
    + full fine-tuning
    + RC oversampling x3
    + all-label small lesion CE weight (default CE weight=2.0)
    """
    rc_weight: float = 3.0
    freeze_mode: str = "none"


class nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw2__3(
    nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE
):
    rc_weight: float = 3.0
    small_lesion_weight: float = 2.0
    freeze_mode: str = "none"


class nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3(
    nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE
):
    """
    Triad init + RC oversampling x3 + small lesion CE weight x3 on all labels.
    """
    rc_weight: float = 3.0
    small_lesion_weight: float = 3.0
    freeze_mode: str = "none"


class nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw5__3(
    nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE
):
    rc_weight: float = 3.0
    small_lesion_weight: float = 5.0
    freeze_mode: str = "none"


class nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_RCOnly_CEw10__3(
    nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE
):
    """
    Triad init + RC oversampling x3 + small lesion CE weight 10 on label 4 only.
    """
    rc_weight: float = 3.0
    small_lesion_weight: float = 10.0
    small_lesion_target_labels: tuple = (4,)
    freeze_mode: str = "none"


class nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_Frozen__3(
    nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE
):
    rc_weight: float = 3.0
    freeze_mode: str = "encoder"


class nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3_Frozen__3(
    nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3
):
    """
    Representation probe for the CEw3 recipe.

    Triad encoder init + encoder frozen + decoder trained
    + RC oversampling x3 + small lesion CE weight x3 on all labels.
    """
    freeze_mode: str = "encoder"


class nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3_Frozen__3_LP100(
    nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3_Frozen__3
):
    """
    LP stage for LP-FT, CEw3 recipe.

    Same as CEw3_Frozen__3 (encoder frozen, decoder trained)
    but capped at 100 epochs instead of the default 1000.

    Usage:
    1. Train this class to get a short decoder warm-up checkpoint.
    2. Feed that checkpoint via -pretrained_weights into
       nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3 (freeze_mode="none")
       for the FT stage.
    """

    checkpoint_interval: int = 100

    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        device=torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.save_every = 100

    def on_epoch_end(self):
        epoch_just_finished = self.current_epoch  # current_epoch increments inside super()
        super().on_epoch_end()

        if (epoch_just_finished + 1) % self.checkpoint_interval == 0:
            snapshot_path = os.path.join(
                self.output_folder, f"checkpoint_epoch{epoch_just_finished + 1}.pth"
            )
            self.save_checkpoint(snapshot_path)
            self.print_to_log_file(f"Saved epoch snapshot: {snapshot_path}")


class nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3_FT900(
    nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3
):
    """
    FT stage for LP-FT, CEw3 recipe.

    Same as CEw3__3 (encoder unfrozen, full fine-tuning) but capped at
    900 epochs instead of the default 1000, so that LP100 (100 epochs)
    + this stage (900 epochs) totals 1000 epochs.

    Usage:
    Feed the checkpoint produced by
    nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3_Frozen__3_LP100
    via -pretrained_weights when launching this trainer.
    """

    checkpoint_interval: int = 100

    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        device=torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 900

    def on_epoch_end(self):
        epoch_just_finished = self.current_epoch  # current_epoch increments inside super()
        super().on_epoch_end()

        if (epoch_just_finished + 1) % self.checkpoint_interval == 0:
            snapshot_path = os.path.join(
                self.output_folder, f"checkpoint_epoch{epoch_just_finished + 1}.pth"
            )
            self.save_checkpoint(snapshot_path)
            self.print_to_log_file(f"Saved epoch snapshot: {snapshot_path}")


class nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_StemOnly__3(
    nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE
):
    rc_weight: float = 3.0
    freeze_mode: str = "stem_only"
