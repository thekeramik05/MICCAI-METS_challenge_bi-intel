from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from custom_dataloaders import get_dataloaders_with_train_sampling
from rc_sampling import (
    DEFAULT_RC_WEIGHT,
    build_rc_sampling_probabilities,
    case_has_rc,
)


class nnUNetTrainerBraTS_RCOversample(nnUNetTrainer):

    rc_weight: float = DEFAULT_RC_WEIGHT

    def get_dataloaders(self):
        dataset_tr, _ = self.get_tr_and_val_datasets()

        sampling_probabilities = build_rc_sampling_probabilities(
            case_ids=dataset_tr.identifiers,
            preprocessed_folder=self.preprocessed_dataset_folder,
            rc_weight=self.rc_weight,
        )

        num_rc_cases = sum(
            case_has_rc(case_id, self.preprocessed_dataset_folder)
            for case_id in dataset_tr.identifiers
        )

        self.print_to_log_file(
            "RC oversampling enabled: "
            f"rc_weight={self.rc_weight}, "
            f"train_cases={len(dataset_tr.identifiers)}, "
            f"rc_cases={num_rc_cases}"
        )

        return get_dataloaders_with_train_sampling(
            self,
            train_sampling_probabilities=sampling_probabilities,
        )


class nnUNetTrainerBraTS_RCOversample__2(nnUNetTrainerBraTS_RCOversample):
    """RC case sampling weight = 2.0"""

    rc_weight: float = 2.0


class nnUNetTrainerBraTS_RCOversample__2p5(nnUNetTrainerBraTS_RCOversample):
    """RC case sampling weight = 2.5"""

    rc_weight: float = 2.5


class nnUNetTrainerBraTS_RCOversample__3(nnUNetTrainerBraTS_RCOversample):
    """RC case sampling weight = 3.0"""

    rc_weight: float = 3.0