import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from custom_dataloaders import get_dataloaders_with_train_sampling
from small_lesion_sampling import (
    DEFAULT_SMALL_LESION_WEIGHT,
    build_small_lesion_sampling_probabilities,
)


class nnUNetTrainerBraTS_SmallLesionOversample(nnUNetTrainer):
    """
    tiny/small lesion case를 더 자주 뽑는 case-level oversampling만 적용.
    loss는 nnUNet 기본 Dice+CE 유지.
    EMA pseudo Dice 기준으로 patience epoch 동안 개선 없으면 early stopping.
    """

    small_lesion_weight: float = DEFAULT_SMALL_LESION_WEIGHT
    early_stopping_patience: int = 10

    def on_train_start(self):
        super().on_train_start()
        self._epochs_without_improvement = 0
        self.print_to_log_file(
            f"Early stopping enabled: patience={self.early_stopping_patience} epochs (ema_fg_dice)"
        )

    def get_dataloaders(self):
        dataset_tr, _ = self.get_tr_and_val_datasets()
        sampling_probs = build_small_lesion_sampling_probabilities(
            dataset_tr.identifiers,
            small_weight=self.small_lesion_weight,
        )
        n_small = int((sampling_probs > (1.0 / max(len(sampling_probs), 1))).sum())
        self.print_to_log_file(
            f"Small lesion oversampling enabled: weight={self.small_lesion_weight}, "
            f"train_cases={len(dataset_tr.identifiers)}, boosted_cases~={n_small}"
        )
        return get_dataloaders_with_train_sampling(self, train_sampling_probabilities=sampling_probs)

    def on_epoch_end(self):
        prev_best_ema = self._best_ema
        super().on_epoch_end()

        if prev_best_ema is None or (self._best_ema is not None and self._best_ema > prev_best_ema):
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1
            self.print_to_log_file(
                f"No improvement in ema_fg_dice for {self._epochs_without_improvement}/"
                f"{self.early_stopping_patience} epochs"
            )

        if self._epochs_without_improvement >= self.early_stopping_patience:
            self.print_to_log_file(
                f"Early stopping triggered after {self.current_epoch} epochs. "
                f"Best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}"
            )
            self.num_epochs = self.current_epoch
