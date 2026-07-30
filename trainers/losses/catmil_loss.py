"""CATMIL: Component-Adaptive Tversky + lesion-level MIL supervision.

Reproduces (best-effort, from the paper's stated equations; exact alpha/beta/lambda
values were not disclosed in the source text, so sensible defaults are used and
documented below):

  w_i = (|C_k| + eps_w)^(-gamma)   if voxel i belongs to GT component C_k (class c)
      = w_bg                        if voxel i is background for class c
  I_CAT = (TP_w + delta) / (TP_w + alpha*FN_w + beta*FP_w + delta)
  L_CAT = 1 - I_CAT

  s_k = max_{i in C_k} p_i(class c)          (soft detection score per GT lesion)
  L_MIL_pos = mean_k [ -log(s_k + eps_mil) ]

  NOT in the paper text extracted for this project: the positive-only MIL term
  above has zero gradient pressure on any background voxel, so nothing in CATMIL
  penalizes false positives at the instance level (only the base CE/Dice + CAT's
  beta*FP_w term do, and apparently not enough -- an initial LP run with this
  positive-only MIL collapsed to ~186 FP lesions/case). Added a symmetric
  hard-negative term: per sample, the top-K most confident BACKGROUND voxels
  (excluding true-negative-labeled voxels via bg_mask) are penalized the same way
  MIL rewards its best positive voxel per instance -- a per-sample online
  hard-negative-mining term, cheap (no connected components needed, pure top-k).

  L_MIL_neg = mean_over_topK_bg_voxels [ -log(1 - p_fg_total + eps_mil) ]

  L_CATMIL = L_Dice + L_CE
             + lambda_cat(t)*L_CAT
             + lambda_mil_pos(t)*L_MIL_pos
             + lambda_mil_neg(t)*L_MIL_neg

v2 (lambda_cat=1.0, single lambda_mil=0.1 on positive-only MIL) collapsed a
100-epoch frozen-decoder LP run to ~186 FP lesions/case (precision 0.02): MIL had
zero downward pressure on background, and lambda_cat=1.0 (as large as the ENTIRE
base loss) was too strong for a stage with little capacity/time to self-correct.
v3 (current): positive/negative MIL weighted independently so their balance is
directly tunable, lambda_cat/lambda_mil_pos/lambda_mil_neg all lowered, and
mil_neg_topk raised 10->64 (the FP explosion was hundreds of SEPARATE scattered
blobs per case; a top-10 hard-negative pool was too narrow to reach most of
them).

lambda_cat/lambda_mil_pos/lambda_mil_neg are ramped linearly from 0 to their
final value over `warmup_epochs` (paper only specifies this schedule for
lambda_cat; the same warmup is applied to the MIL terms here as a conservative
stability choice, since all three operate on sparse per-lesion/per-voxel signal
that is noisy on a randomly initialized/early-training network).

alpha/beta default to 0.5/0.5 (= plain Tversky = Dice) rather than the
recall-biased 0.7/0.3 used elsewhere in this project's FocalTversky experiment,
so this run isolates the NEW mechanism (component-size weighting + MIL) from the
already-tested recall-bias trick. Pass alpha/beta explicitly to combine both.
"""
import numpy as np
import torch
import torch.nn as nn

from scipy.ndimage import label as cc_label

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.utilities.helpers import softmax_helper_dim1


def _components_and_weights(target_np, num_fg_classes, gamma, eps_w, w_bg, boost, roi_margin):
    """
    target_np: (B, 1, *spatial) int array, integer labels 0..num_fg_classes.

    Returns:
      weight: (B, num_fg_classes, *spatial) float32 array, CAT per-voxel weight.
        Zeroed outside a per-INSTANCE local ROI (that instance's bbox padded by
        `roi_margin` voxels) -- see ROI note below.
      instance_ids: (B, *spatial) int64 array. GT labels are mutually exclusive
        (each voxel belongs to at most one class), so per-batch-item instance ids
        from every class's connected components can share ONE flat id space
        (0 = background/no lesion, 1..K = individual lesion instances) without
        collision. This lets the MIL term use a single vectorized scatter-max
        instead of a Python loop over every component.
      n_instances: list[int], number of instances per batch item (for id offsets).

    ROI note (root-cause fix, see ComponentAdaptiveTverskyMILLoss docstring):
    without this, fp_w sums residual softmax probability over the ENTIRE patch
    (millions of voxels) for every class, so a tiny lesion's tp_w (bounded by its
    own voxel count) can never outweigh background leakage summed over the whole
    volume, regardless of per-voxel weight boosting -- Tversky collapses toward 0
    (verified: even a confidently/correctly segmented 5-voxel lesion needed an
    unrealistically sharp softmax, logit gap >~25, to avoid saturating). Cropping
    fp/fn/tp to a local neighborhood around each instance (not the whole volume)
    removes millions of irrelevant far-background voxels from the sum entirely,
    so a small lesion's signal is compared against a LOCAL, comparable-scale
    background instead of the whole patch. Padded per-INSTANCE (not per-class or
    per-sample) since BraTS-MET is multifocal -- lesions of the same class can sit
    in unrelated parts of the brain, so a single shared bbox per class/sample
    would often re-include most of the volume anyway.
    """
    B = target_np.shape[0]
    spatial_shape = target_np.shape[2:]
    weight = np.zeros((B, num_fg_classes) + spatial_shape, dtype=np.float32)
    instance_ids = np.zeros((B,) + spatial_shape, dtype=np.int64)
    n_instances = [0] * B
    structure = np.ones((3,) * len(spatial_shape), dtype=np.int8)
    ndim = len(spatial_shape)

    for b in range(B):
        seg = target_np[b, 0]
        next_id = 1
        for c in range(1, num_fg_classes + 1):
            class_mask = seg == c
            if not np.any(class_mask):
                continue
            idx = np.where(class_mask)
            sl = tuple(slice(i.min(), i.max() + 1) for i in idx)
            sub_mask = class_mask[sl]
            labeled_sub, n = cc_label(sub_mask, structure=structure)
            if n == 0:
                continue
            sizes = np.bincount(labeled_sub.ravel())
            class_offset = [s.start for s in sl]
            for k in range(1, n + 1):
                m_sub = labeled_sub == k
                size_k = int(sizes[k])
                # BUG FIX: lesion weight must be >= w_bg always, or tp_w gets
                # crushed relative to fp_w for any lesion bigger than a few
                # voxels. Small lesions get a large boost on top of the w_bg
                # baseline; large lesions decay toward w_bg (neutral).
                w_val = w_bg + boost * (size_k + eps_w) ** (-gamma)
                weight[b, c - 1][sl][m_sub] = w_val
                instance_ids[b][sl][m_sub] = next_id
                next_id += 1

                # local ROI: this instance's own bbox (in GLOBAL coordinates),
                # padded by roi_margin and clipped to the volume, gets the
                # w_bg baseline (so the local background around this lesion
                # still competes fairly); everywhere else stays 0 (excluded).
                comp_idx = np.where(m_sub)
                roi_slices = tuple(
                    slice(
                        max(0, class_offset[d] + comp_idx[d].min() - roi_margin),
                        min(spatial_shape[d], class_offset[d] + comp_idx[d].max() + 1 + roi_margin),
                    )
                    for d in range(ndim)
                )
                roi_view = weight[b, c - 1][roi_slices]
                np.maximum(roi_view, w_bg, out=roi_view)
                # re-stamp this instance's own (possibly larger) weight, in case
                # the w_bg floor above overwrote it for a tiny/thin component
                weight[b, c - 1][sl][m_sub] = w_val
        n_instances[b] = next_id - 1

    return weight, instance_ids, n_instances


class ComponentAdaptiveTverskyMILLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        eps_w: float = 1.0,
        w_bg: float = 1.0,
        boost: float = 4.0,
        roi_margin: int = 5,
        delta: float = 1e-5,
        eps_mil: float = 1e-6,
        mil_neg_topk: int = 64,
        do_bg: bool = False,
        apply_nonlin=None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps_w = eps_w
        self.w_bg = w_bg
        self.boost = boost
        self.roi_margin = roi_margin
        self.delta = delta
        self.eps_mil = eps_mil
        self.mil_neg_topk = mil_neg_topk
        self.do_bg = do_bg
        self.apply_nonlin = apply_nonlin if apply_nonlin is not None else softmax_helper_dim1

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        x = self.apply_nonlin(net_output)

        if target.ndim != x.ndim:
            target = target.view((target.shape[0], 1, *target.shape[1:]))
        target = target.long()

        num_classes = x.shape[1]
        num_fg = num_classes - 1 if not self.do_bg else num_classes
        class_offset = 1 if not self.do_bg else 0

        target_np = target.detach().cpu().numpy()
        weight_np, instance_ids_np, n_instances = _components_and_weights(
            target_np, num_fg, self.gamma, self.eps_w, self.w_bg, self.boost, self.roi_margin
        )
        weight = torch.from_numpy(weight_np).to(device=x.device, dtype=x.dtype)

        target_onehot = torch.zeros(
            (x.shape[0], num_classes) + x.shape[2:], device=x.device, dtype=torch.float32
        )
        target_onehot.scatter_(1, target, 1)

        x_fg = x[:, class_offset:]
        g_fg = target_onehot[:, class_offset:]

        axes = tuple(range(2, x_fg.ndim))
        tp_w = (x_fg * g_fg * weight).sum(axes)
        fp_w = (x_fg * (1 - g_fg) * weight).sum(axes)
        fn_w = ((1 - x_fg) * g_fg * weight).sum(axes)

        tversky = (tp_w + self.delta) / (
            tp_w + self.alpha * fn_w + self.beta * fp_w + self.delta
        ).clamp_min(1e-8)

        # BUG FIX: for a (sample, class) with NO GT of that class in this patch
        # (tp_w=fn_w=0, the common case -- each patch usually has only 1-2 of the
        # 4 fg classes), tversky collapses to ~delta/(beta*fp_w+delta). fp_w sums
        # residual softmax probability over the ENTIRE patch (millions of
        # background voxels), so even a tiny, totally normal per-voxel leak (e.g.
        # 0.006) overwhelms delta=1e-5 and forces tversky->0 (max penalty) EVEN
        # FOR A NEAR-PERFECT SEGMENTATION. Verified: near-perfect prediction with
        # normal softmax noise on absent channels gave l_cat=0.9998 (saturated).
        # Fix: exclude GT-absent (sample, class) entries from the mean entirely --
        # CAT should only judge classes that actually appear in this patch.
        gt_present = g_fg.sum(axes) > 0  # (B, num_fg)
        if gt_present.any():
            l_cat = (1 - tversky)[gt_present].mean()
        else:
            l_cat = torch.zeros((), device=x.device, dtype=x.dtype)

        total_instances = sum(n_instances)
        if total_instances == 0:
            l_mil_pos = torch.zeros((), device=x.device, dtype=x.dtype)
        else:
            # p_true: predicted probability of each voxel's OWN GT class (0 where
            # background). Single vectorized op, replaces per-class channel lookup.
            p_true = (x_fg * g_fg).sum(dim=1)  # (B, *spatial)

            # offset instance ids per batch item so one flat scatter covers the
            # whole batch, then a single scatter_reduce gives every lesion's max
            # predicted probability in one vectorized (differentiable) GPU op.
            offsets = np.concatenate(([0], np.cumsum(n_instances)[:-1]))
            instance_ids_np_offset = instance_ids_np.copy()
            for b in range(x_fg.shape[0]):
                mask = instance_ids_np_offset[b] > 0
                instance_ids_np_offset[b][mask] += offsets[b]
            instance_ids = torch.from_numpy(instance_ids_np_offset).to(device=x.device, dtype=torch.long)

            flat_ids = instance_ids.reshape(-1)
            flat_p = p_true.reshape(-1)
            fg_mask = flat_ids > 0
            flat_ids = flat_ids[fg_mask]
            flat_p = flat_p[fg_mask]

            s_k_init = torch.zeros(total_instances + 1, device=x.device, dtype=x.dtype)
            s_k = s_k_init.scatter_reduce(0, flat_ids, flat_p, reduce="amax", include_self=False)
            s_k = s_k[1:]  # drop the unused index-0 slot

            l_mil_pos = (-torch.log(s_k + self.eps_mil)).mean()

        # hard-negative term: per sample, penalize the top-K most confident
        # BACKGROUND voxels (mirrors the positive term's "best voxel per instance"
        # with "worst K voxels per sample among true negatives"). Always computed
        # (doesn't need any GT lesion to exist), cheap (no connected components).
        p_fg_total = x_fg.sum(dim=1)  # (B, *spatial), prob of ANY fg class
        bg_mask = g_fg.sum(dim=1) == 0  # (B, *spatial)
        neg_candidates = p_fg_total.masked_fill(~bg_mask, -1.0)
        flat_neg = neg_candidates.reshape(neg_candidates.shape[0], -1)
        k = min(self.mil_neg_topk, flat_neg.shape[1])
        topk_vals, _ = flat_neg.topk(k, dim=1)
        valid = topk_vals > -0.5
        if valid.any():
            topk_vals = topk_vals.clamp(min=0.0, max=1.0 - 1e-6)
            l_mil_neg = (-torch.log(1 - topk_vals[valid] + self.eps_mil)).mean()
        else:
            l_mil_neg = torch.zeros((), device=x.device, dtype=x.dtype)

        return l_cat, l_mil_pos, l_mil_neg


class CATMILLoss(nn.Module):
    """
    Base (DC+CE) loss is deep-supervised across all scales, exactly like nnU-Net's
    default recipe. CAT/MIL are computed ONCE per iteration, on the finest-resolution
    output/target only: connected components computed at coarse deep-supervision
    scales are both (a) redundant (the finest-scale weight map is the meaningful
    one; lesion identity is clearest at full resolution, since small lesions merge
    or vanish under downsampling) and (b) expensive (a naive per-scale recompute
    multiplies the CPU-side scipy connected-component cost by the number of
    supervision scales, ~5-6x, on every training iteration).
    """

    def __init__(
        self,
        base_kwargs: dict,
        cat_mil_kwargs: dict,
        lambda_cat: float = 0.3,
        lambda_mil_pos: float = 0.04,
        lambda_mil_neg: float = 0.06,
        warmup_epochs: int = 10,
        ds_weights=None,
    ):
        super().__init__()
        self.base = DC_and_CE_loss(**base_kwargs)
        self.cat_mil = ComponentAdaptiveTverskyMILLoss(**cat_mil_kwargs)
        self.lambda_cat = lambda_cat
        self.lambda_mil_pos = lambda_mil_pos
        self.lambda_mil_neg = lambda_mil_neg
        self.warmup_epochs = max(1, warmup_epochs)
        self.current_epoch = 0
        self.ds_weights = ds_weights

        # last-iteration component values, for cheap per-epoch logging by the
        # trainer (all already computed as part of the normal forward pass, so
        # reading these back adds zero extra cost).
        self.last_base_loss = float("nan")
        self.last_weighted_cat = float("nan")
        self.last_weighted_mil_pos = float("nan")
        self.last_weighted_mil_neg = float("nan")

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _warmup_factor(self):
        return min(self.current_epoch / self.warmup_epochs, 1.0)

    def forward(self, net_output, target):
        if isinstance(net_output, (list, tuple)):
            assert self.ds_weights is not None and len(self.ds_weights) == len(net_output)
            base_loss = None
            for o, t, dsw in zip(net_output, target, self.ds_weights):
                if dsw == 0:
                    continue
                scale_loss = dsw * self.base(o, t)
                base_loss = scale_loss if base_loss is None else base_loss + scale_loss
            finest_output, finest_target = net_output[0], target[0]
        else:
            base_loss = self.base(net_output, target)
            finest_output, finest_target = net_output, target

        l_cat, l_mil_pos, l_mil_neg = self.cat_mil(finest_output, finest_target)
        w = self._warmup_factor()
        weighted_cat = w * self.lambda_cat * l_cat
        weighted_mil_pos = w * self.lambda_mil_pos * l_mil_pos
        weighted_mil_neg = w * self.lambda_mil_neg * l_mil_neg

        self.last_base_loss = float(base_loss.detach())
        self.last_weighted_cat = float(weighted_cat.detach())
        self.last_weighted_mil_pos = float(weighted_mil_pos.detach())
        self.last_weighted_mil_neg = float(weighted_mil_neg.detach())

        return base_loss + weighted_cat + weighted_mil_pos + weighted_mil_neg


def build_catmil_loss(
    batch_dice: bool,
    ignore_label,
    enable_deep_supervision: bool,
    get_deep_supervision_scales,
    is_ddp: bool,
    do_compile: bool,
    alpha: float = 0.5,
    beta: float = 0.5,
    lambda_cat: float = 0.3,
    lambda_mil_pos: float = 0.04,
    lambda_mil_neg: float = 0.06,
    warmup_epochs: int = 10,
    mil_neg_topk: int = 64,
    roi_margin: int = 5,
):
    ce_kwargs = {}
    if ignore_label is not None:
        ce_kwargs["ignore_index"] = ignore_label

    ds_weights = None
    if enable_deep_supervision:
        deep_supervision_scales = get_deep_supervision_scales()
        weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
        if is_ddp and not do_compile:
            weights[-1] = 1e-6
        else:
            weights[-1] = 0
        ds_weights = (weights / weights.sum()).tolist()

    return CATMILLoss(
        base_kwargs={
            "soft_dice_kwargs": {"batch_dice": batch_dice, "smooth": 1e-5, "do_bg": False},
            "ce_kwargs": ce_kwargs,
            "weight_ce": 1,
            "weight_dice": 1,
            "ignore_label": ignore_label,
        },
        cat_mil_kwargs={
            "alpha": alpha, "beta": beta, "do_bg": False,
            "mil_neg_topk": mil_neg_topk, "roi_margin": roi_margin,
        },
        lambda_cat=lambda_cat,
        lambda_mil_pos=lambda_mil_pos,
        lambda_mil_neg=lambda_mil_neg,
        warmup_epochs=warmup_epochs,
        ds_weights=ds_weights,
    )
