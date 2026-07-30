import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class FocalTverskyLoss(nn.Module):
    """
    Multi-class Focal Tversky loss (Abraham & Khan, 2019).
    TI_c = (TP_c + s) / (TP_c + alpha*FN_c + beta*FP_c + s)
    FTL  = mean_c( (1 - TI_c)^gamma )
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 4 / 3,
        smooth: float = 1e-5,
        do_bg: bool = False,
        batch_dice: bool = False,
        ddp: bool = True,
        apply_nonlin=None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.ddp = ddp
        self.apply_nonlin = apply_nonlin if apply_nonlin is not None else softmax_helper_dim1

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor | None = None):
        x = self.apply_nonlin(net_output)

        if target.ndim != x.ndim:
            target = target.view((target.shape[0], 1, *target.shape[1:]))

        with torch.no_grad():
            if x.shape == target.shape:
                target_onehot = target.float()
            else:
                target_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                target_onehot.scatter_(1, target.long(), 1)

            if not self.do_bg:
                target_onehot = target_onehot[:, 1:]
                x = x[:, 1:]

        axes = tuple(range(2, x.ndim))
        if self.batch_dice:
            axes = (0,) + axes

        if loss_mask is None:
            tp = (x * target_onehot).sum(axes, dtype=torch.float32)
            fp = (x * (1 - target_onehot)).sum(axes, dtype=torch.float32)
            fn = ((1 - x) * target_onehot).sum(axes, dtype=torch.float32)
        else:
            mask = loss_mask.to(dtype=x.dtype)
            tp = (x * target_onehot * mask).sum(axes, dtype=torch.float32)
            fp = (x * (1 - target_onehot) * mask).sum(axes, dtype=torch.float32)
            fn = ((1 - x) * target_onehot * mask).sum(axes, dtype=torch.float32)

        if self.ddp and self.batch_dice:
            from nnunetv2.utilities.ddp_allgather import AllGatherGrad

            tp = AllGatherGrad.apply(tp).sum(0, dtype=torch.float32)
            fp = AllGatherGrad.apply(fp).sum(0, dtype=torch.float32)
            fn = AllGatherGrad.apply(fn).sum(0, dtype=torch.float32)

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        ).clamp_min(1e-8)
        focal_tversky = torch.pow(1 - tversky_index, self.gamma)
        return focal_tversky.mean()


class FTL_and_CE_loss(nn.Module):
    def __init__(
        self,
        ftl_kwargs: dict,
        ce_kwargs: dict,
        weight_ce: float = 1.0,
        weight_ftl: float = 1.0,
        ignore_label: int | None = None,
    ):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs = dict(ce_kwargs)
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_ce = weight_ce
        self.weight_ftl = weight_ftl
        self.ignore_label = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.ftl = FocalTverskyLoss(**ftl_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, "ignore label is only supported for index targets"
            mask = target != self.ignore_label
            target_ftl = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            mask = None
            target_ftl = target
            num_fg = None

        ftl_loss = self.ftl(net_output, target_ftl, loss_mask=mask) if self.weight_ftl != 0 else 0
        ce_loss = (
            self.ce(net_output, target[:, 0])
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )
        return self.weight_ce * ce_loss + self.weight_ftl * ftl_loss


def build_focal_tversky_loss(
    batch_dice: bool,
    ignore_label: int | None,
    enable_deep_supervision: bool,
    get_deep_supervision_scales,
    is_ddp: bool,
    do_compile: bool,
):
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

    loss = FTL_and_CE_loss(
        ftl_kwargs={
            "alpha": 0.7,
            "beta": 0.3,
            "gamma": 4 / 3,
            "smooth": 1e-5,
            "do_bg": False,
            "batch_dice": batch_dice,
            "ddp": is_ddp,
        },
        ce_kwargs={},
        weight_ce=1.0,
        weight_ftl=1.0,
        ignore_label=ignore_label,
    )

    if do_compile:
        loss.ftl = torch.compile(loss.ftl)

    if enable_deep_supervision:
        deep_supervision_scales = get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        if is_ddp and not do_compile:
            weights[-1] = 1e-6
        else:
            weights[-1] = 0
        weights = weights / weights.sum()
        loss = DeepSupervisionWrapper(loss, weights)

    return loss
