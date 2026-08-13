# Copyright 2026 the authors of MICCAI-METS_challenge_bi-intel.
# Licensed under the Apache License, Version 2.0 - see LICENSE at the repository root.
#
# Extends nnU-Net v2 (MIC-DKFZ, Apache-2.0): https://github.com/MIC-DKFZ/nnUNet
#
# Loads the BrainIAC ViT-B/16 checkpoint (https://github.com/AIM-KannLab/BrainIAC)
# as a frozen auxiliary feature extractor. BrainIAC is licensed for non-commercial
# academic research only; commercial use requires a separate licence from
# Mass General Brigham.
#
# NOTE: checkpoints saved by this trainer embed the full BrainIAC backbone
# (116.7 M parameters under the `brainiac.*` prefix, of 204.9 M total) and
# therefore inherit that restriction. Redistributing them redistributes BrainIAC.

"""
Usage example (PROJECT_ROOT is wherever this repository is checked out):

export nnUNet_results=$PROJECT_ROOT/nnUNet/nnUNet_results

export BRAINIAC_CKPT=$PROJECT_ROOT/weights/foundation_encoders/BrainIAC.ckpt
export BRAINIAC_MODALITY_INDICES=0,1,2,3
export BRAINIAC_FEATURE_CHANNELS=4
export BRAINIAC_FREEZE=1

nnUNetv2_predict \
  -i $PROJECT_ROOT/dataset/total_val \
  -o $PROJECT_ROOT/outputs/BraTS_validation/BrainIACWrapper/fold0 \
  -d Dataset001_BraTS \
  -c 3d_fullres \
  -tr nnUNetTrainerBraTS_BrainIACWrapper_RCOversample \
  -p nnUNetPlans \
  -f 0 \
  -chk checkpoint_final.pth
"""

import inspect
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerBraTS_RCOversample import (
    nnUNetTrainerBraTS_RCOversample,
)


# Fallback only. Set BRAINIAC_CKPT to the encoder checkpoint; the Docker image
# does exactly that (see Dockerfile).
BRAINIAC_CKPT_DEFAULT = Path("weights/foundation_encoders/BrainIAC.ckpt")

BRAINIAC_TARGET_SHAPE = (96, 96, 96)
BRAINIAC_PATCH_GRID = (6, 6, 6)


def _get_raw_network(network: nn.Module) -> nn.Module:
    if hasattr(network, "module"):
        return network.module
    return network


def _unwrap_checkpoint(ckpt):
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    for key in ("state_dict", "model_state_dict", "network_weights", "net", "model", "module"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]

    if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    raise RuntimeError(f"Could not find state_dict. Top-level keys={list(ckpt.keys())[:20]}")


def _load_brainiac_backbone_state_dict(model: nn.Module, checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"BrainIAC checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_checkpoint(ckpt)

    backbone_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            backbone_state_dict[key[len("backbone."):]] = value
        elif key.startswith("module.backbone."):
            backbone_state_dict[key[len("module.backbone."):]] = value
        elif key.startswith("model.backbone."):
            backbone_state_dict[key[len("model.backbone."):]] = value

    if len(backbone_state_dict) == 0:
        backbone_state_dict = state_dict

    missing, unexpected = model.backbone.load_state_dict(backbone_state_dict, strict=False)

    print("\n[BrainIAC] checkpoint load report")
    print(f"[BrainIAC] checkpoint path      : {checkpoint_path}")
    print(f"[BrainIAC] loaded tensors       : {len(backbone_state_dict)}")
    print(f"[BrainIAC] missing keys         : {len(missing)}")
    print(f"[BrainIAC] unexpected keys      : {len(unexpected)}")


class BrainIACEncoder3D(nn.Module):
    def __init__(
        self,
        checkpoint_path: Path = BRAINIAC_CKPT_DEFAULT,
        freeze: bool = True,
    ):
        super().__init__()

        from monai.networks.nets import ViT

        self.backbone = ViT(
            in_channels=1,
            img_size=BRAINIAC_TARGET_SHAPE,
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            save_attn=False,
        )

        _load_brainiac_backbone_state_dict(self, checkpoint_path)

        self.freeze = bool(freeze)
        if self.freeze:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    @staticmethod
    def _normalize_per_sample(x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(2, x.ndim))
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def forward_single_modality(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, D, H, W]
        returns: [B, 768, 6, 6, 6]
        """
        x = self._normalize_per_sample(x)

        if tuple(x.shape[2:]) != BRAINIAC_TARGET_SHAPE:
            x = F.interpolate(
                x,
                size=BRAINIAC_TARGET_SHAPE,
                mode="trilinear",
                align_corners=False,
            )

        hidden = self.backbone(x)[0]

        if hidden.ndim != 3:
            raise RuntimeError(f"Unexpected BrainIAC hidden shape: {tuple(hidden.shape)}")

        b, n, c = hidden.shape
        expected_n = BRAINIAC_PATCH_GRID[0] * BRAINIAC_PATCH_GRID[1] * BRAINIAC_PATCH_GRID[2]

        if n == expected_n + 1:
            hidden = hidden[:, 1:, :]
            n = expected_n

        if n != expected_n:
            raise RuntimeError(
                f"Unexpected BrainIAC token count: got {n}, expected {expected_n} "
                f"or {expected_n + 1} with CLS token."
            )

        d, h, w = BRAINIAC_PATCH_GRID
        feat = hidden.reshape(b, d, h, w, c)
        feat = feat.permute(0, 4, 1, 2, 3).contiguous()
        return feat

    def forward(self, x: torch.Tensor, modality_indices: Sequence[int]) -> torch.Tensor:
        """
        x: [B, C, D, H, W]
        returns averaged feature: [B, 768, 6, 6, 6]
        """
        feats = []

        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            for idx in modality_indices:
                if idx < 0 or idx >= x.shape[1]:
                    raise ValueError(
                        f"Invalid modality index {idx}. Input has {x.shape[1]} channels."
                    )
                feats.append(self.forward_single_modality(x[:, idx:idx + 1]))

        return torch.stack(feats, dim=0).mean(dim=0)


def _find_first_conv3d_parent(
    module: nn.Module,
    expected_in_channels: Optional[int] = None,
) -> Tuple[nn.Module, str, nn.Conv3d]:
    fallback = None

    for _parent_name, parent in module.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Conv3d):
                if fallback is None:
                    fallback = (parent, child_name, child)
                if expected_in_channels is None or child.in_channels == expected_in_channels:
                    return parent, child_name, child

    if fallback is not None:
        return fallback

    raise RuntimeError("Could not find any nn.Conv3d in nnU-Net network.")


def _replace_module_recursively(root: nn.Module, old_mod: nn.Module, new_mod: nn.Module):
    for name, child in root.named_children():
        if child is old_mod:
            setattr(root, name, new_mod)
        else:
            _replace_module_recursively(child, old_mod, new_mod)


def _replace_first_conv3d_for_extra_channels(
    network: nn.Module,
    extra_channels: int,
    expected_old_in_channels: Optional[int] = 4,
) -> Tuple[int, int]:
    """
    Patch first Conv3d to accept original MRI channels + BrainIAC feature channels.

    Existing channel weights are copied.
    Extra channel weights are initialized to zero.
    """
    raw = _get_raw_network(network)

    _, _, old_conv = _find_first_conv3d_parent(
        raw,
        expected_in_channels=expected_old_in_channels,
    )

    old_in = old_conv.in_channels
    new_in = old_in + int(extra_channels)

    new_conv = nn.Conv3d(
        in_channels=new_in,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )

    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :old_in].copy_(old_conv.weight)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    _replace_module_recursively(raw, old_conv, new_conv)

    print("\n[BrainIACWrapper] Patched first Conv3d")
    print(f"[BrainIACWrapper] in_channels  : {old_in} -> {new_in}")
    print(f"[BrainIACWrapper] out_channels : {old_conv.out_channels}")
    print("[BrainIACWrapper] extra-channel weights initialized to zero")

    return old_in, new_in


class BrainIACFeatureConcatWrapper(nn.Module):
    def __init__(
        self,
        base_network: nn.Module,
        brainiac_checkpoint: Path,
        modality_indices: Sequence[int] = (0, 1, 2, 3),
        feature_channels: int = 4,
        freeze_brainiac: bool = True,
    ):
        super().__init__()

        if feature_channels <= 0:
            raise ValueError("feature_channels must be positive.")

        self.base_network = base_network
        self.modality_indices = tuple(modality_indices)
        self.feature_channels = int(feature_channels)

        self.brainiac = BrainIACEncoder3D(
            checkpoint_path=brainiac_checkpoint,
            freeze=freeze_brainiac,
        )

        self.proj = nn.Conv3d(
            in_channels=768,
            out_channels=self.feature_channels,
            kernel_size=1,
            bias=True,
        )

    @property
    def decoder(self):
        """
        nnU-Net expects network.decoder for deep supervision handling.
        """
        return _get_raw_network(self.base_network).decoder

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, D, H, W]
        """
        with torch.autocast(x.device.type, enabled=False):
            feat = self.brainiac(x.float(), self.modality_indices)
            feat = self.proj(feat)

        feat = F.interpolate(
            feat,
            size=tuple(x.shape[2:]),
            mode="trilinear",
            align_corners=False,
        )

        x_aug = torch.cat([x, feat.to(dtype=x.dtype)], dim=1)
        return self.base_network(x_aug)


def _parse_modality_indices(value: str) -> Tuple[int, ...]:
    value = value.strip()
    if value.lower() in ("all", ""):
        return (0, 1, 2, 3)
    return tuple(int(v.strip()) for v in value.split(",") if v.strip() != "")


def _print_trainable_summary(network: nn.Module, max_lines: int = 100):
    raw = _get_raw_network(network)

    total = sum(p.numel() for p in raw.parameters())
    trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    frozen = total - trainable

    print("\n[BrainIACWrapper] Parameter summary")
    print(f"[BrainIACWrapper] total params     : {total:,}")
    print(f"[BrainIACWrapper] trainable params : {trainable:,}")
    print(f"[BrainIACWrapper] frozen params    : {frozen:,}")

    print(f"[BrainIACWrapper] trainable parameter names, first {max_lines}:")
    count = 0
    for name, p in raw.named_parameters():
        if p.requires_grad:
            print(f"  [TRAIN] {name} {tuple(p.shape)}")
            count += 1
            if count >= max_lines:
                print("  ...")
                break


def _extract_num_input_channels_from_build_args(args, kwargs) -> int:
    """
    nnU-Net v2 versions may call build_network_architecture with slightly
    different signatures. In both common forms, num_input_channels is the
    4th positional argument.

    Modern:
      plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision=True

    Older/other wrapper style:
      architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
      num_input_channels, num_output_channels, enable_deep_supervision=True
    """
    if "num_input_channels" in kwargs:
        return int(kwargs["num_input_channels"])

    if len(args) >= 4:
        return int(args[3])

    raise RuntimeError(
        "Could not infer num_input_channels from build_network_architecture arguments. "
        f"args={args}, kwargs={kwargs}"
    )


def _extract_enable_deep_supervision(args, kwargs) -> bool:
    if "enable_deep_supervision" in kwargs:
        return bool(kwargs["enable_deep_supervision"])

    if len(args) >= 5 and isinstance(args[-1], bool):
        return bool(args[-1])

    return True


def _call_parent_build_network(
    parent_cls,
    args,
    kwargs,
    enable_deep_supervision: bool,
):
    """
    nnUNet predictor calls custom build_network_architecture as:

      args[0] = plans_manager
      args[1] = dataset_json
      args[2] = configuration_manager
      args[3] = num_input_channels
      args[4] = num_output_channels

    But parent nnUNetTrainerBraTS_RCOversample.build_network_architecture expects:

      plans_manager,
      configuration_manager,
      num_input_channels,
      num_output_channels,
      enable_deep_supervision=True

    Therefore dataset_json must be skipped.
    """
    parent_fn = parent_cls.build_network_architecture
    parent_sig = inspect.signature(parent_fn)

    print(f"[BrainIACWrapper] parent build signature: {parent_sig}")

    if len(args) >= 5:
        plans_manager = args[0]
        configuration_manager = args[2]
        num_input_channels = args[3]
        num_output_channels = args[4]

        return parent_fn(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
        )

    if len(args) >= 4:
        return parent_fn(
            *args[:4],
            enable_deep_supervision=enable_deep_supervision,
        )

    raise RuntimeError(
        "Expected at least 4 positional arguments for parent build_network_architecture, "
        f"got args={args}, kwargs={kwargs}"
    )


class nnUNetTrainerBraTS_BrainIACWrapper_RCOversample(nnUNetTrainerBraTS_RCOversample):
    brainiac_ckpt: str = str(BRAINIAC_CKPT_DEFAULT)
    brainiac_feature_channels: int = 4
    brainiac_modality_indices: Tuple[int, ...] = (0, 1, 2, 3)
    freeze_brainiac: bool = True

    def _do_i_compile(self):
        return False

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        print("\n[DEBUG] BrainIAC custom build_network_architecture CALLED\n")
        print(f"[BrainIACWrapper] num_input_channels      : {num_input_channels}")
        print(f"[BrainIACWrapper] num_output_channels     : {num_output_channels}")
        print(f"[BrainIACWrapper] enable_deep_supervision : {enable_deep_supervision}")

        network = nnUNetTrainerBraTS_RCOversample.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
        )

        ckpt = Path(os.environ.get("BRAINIAC_CKPT", str(BRAINIAC_CKPT_DEFAULT)))

        feature_channels = int(
            os.environ.get(
                "BRAINIAC_FEATURE_CHANNELS",
                str(nnUNetTrainerBraTS_BrainIACWrapper_RCOversample.brainiac_feature_channels),
            )
        )

        modality_indices = _parse_modality_indices(
            os.environ.get(
                "BRAINIAC_MODALITY_INDICES",
                ",".join(
                    map(
                        str,
                        nnUNetTrainerBraTS_BrainIACWrapper_RCOversample.brainiac_modality_indices,
                    )
                ),
            )
        )

        freeze_value = os.environ.get(
            "BRAINIAC_FREEZE",
            "1" if nnUNetTrainerBraTS_BrainIACWrapper_RCOversample.freeze_brainiac else "0",
        )
        freeze_brainiac = freeze_value not in ("0", "false", "False", "no", "NO")

        print("\n[BrainIACWrapper] build config")
        print(f"[BrainIACWrapper] checkpoint       : {ckpt}")
        print(f"[BrainIACWrapper] modality_indices : {modality_indices}")
        print(f"[BrainIACWrapper] feature_channels : {feature_channels}")
        print(f"[BrainIACWrapper] freeze_brainiac  : {freeze_brainiac}")

        _replace_first_conv3d_for_extra_channels(
            network=network,
            extra_channels=feature_channels,
            expected_old_in_channels=num_input_channels,
        )

        network = BrainIACFeatureConcatWrapper(
            base_network=network,
            brainiac_checkpoint=ckpt,
            modality_indices=modality_indices,
            feature_channels=feature_channels,
            freeze_brainiac=freeze_brainiac,
        )

        return network

    def initialize(self):
        """
        Do not wrap the network here.

        The wrapper must be created inside build_network_architecture so that
        training and inference construct the same network structure.
        """
        super().initialize()
        _print_trainable_summary(self.network)


class nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__1(
    nnUNetTrainerBraTS_BrainIACWrapper_RCOversample
):
    rc_weight: float = 1.0


class nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__2(
    nnUNetTrainerBraTS_BrainIACWrapper_RCOversample
):
    rc_weight: float = 2.0


class nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__2p5(
    nnUNetTrainerBraTS_BrainIACWrapper_RCOversample
):
    rc_weight: float = 2.5


class nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__3(
    nnUNetTrainerBraTS_BrainIACWrapper_RCOversample
):
    rc_weight: float = 3.0