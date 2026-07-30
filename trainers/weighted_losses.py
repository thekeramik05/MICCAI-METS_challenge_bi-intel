import torch
import torch.nn as nn

from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class DC_and_VoxelWeightedCE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ignore_label = ignore_label
        self.ce = nn.CrossEntropyLoss(reduction='none')  # voxel-wise weight 곱하기 위해 reduction 없이
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output, target, voxel_weight):
        """
        net_output: (B, C, H, W, D)
        target: (B, 1, H, W, D)  - 정수 라벨
        voxel_weight: (B, 1, H, W, D)  - voxel별 CE weight (small lesion=10.0, 나머지=1.0)
        """
        target = target.long()

        if self.ignore_label is not None:
            mask = (target != self.ignore_label).float()
            target_ce = target.clone()
            target_ce[target == self.ignore_label] = 0
        else:
            mask = torch.ones_like(target, dtype=torch.float32)
            target_ce = target

        ce_loss = self.ce(net_output, target_ce[:, 0])  # (B, H, W, D)
        ce_loss = ce_loss * voxel_weight[:, 0] * mask[:, 0]
        ce_loss = ce_loss.sum() / mask.sum().clamp(min=1e-8)

        dc_loss = self.dc(net_output, target, loss_mask=mask)

        return self.weight_ce * ce_loss + self.weight_dice * dc_loss