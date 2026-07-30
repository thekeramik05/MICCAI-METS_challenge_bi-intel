import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from nnUNetTrainerBraTS_RCOversample import nnUNetTrainerBraTS_RCOversample


# Fallback only. Set TRIAD_CKPT to the encoder checkpoint; the Docker image
# does exactly that (see Dockerfile).
TRIAD_CKPT_DEFAULT = Path("weights/foundation_encoders/Triad-PlainConvUNet-MAE.pth")


def _unwrap_checkpoint(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Supports:
    - raw state_dict
    - {"state_dict": ...}
    - {"model_state_dict": ...}
    - {"network_weights": ...}
    - {"net": ...}
    - {"model": ...}
    """
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    candidate_keys = [
        "state_dict",
        "model_state_dict",
        "network_weights",
        "net",
        "model",
        "module",
    ]

    for key in candidate_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]

    if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    raise RuntimeError(
        "Could not find a valid state_dict in checkpoint. "
        f"Top-level keys: {list(ckpt.keys())[:20]}"
    )


def _strip_known_prefixes(
    state_dict: Dict[str, torch.Tensor],
    prefixes=("module.", "model.", "net.", "network."),
) -> Dict[str, torch.Tensor]:
    out = {}

    for key, value in state_dict.items():
        new_key = key

        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        out[new_key] = value

    return out


def _is_likely_input_weight_key(key: str) -> bool:
    patterns = [
        "encoder.stages.0",
        "conv_blocks_context.0",
        "all_modules.0.weight",
        "conv.weight",
        "patch_embedding",
        "patch_embed",
        "proj.weight",
    ]

    return any(p in key for p in patterns) and key.endswith("weight")


def _expand_input_weight(
    weight: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    """
    Expand 1-channel pretrained conv weight to 4-channel input.

    Important:
    repeat만 하면 activation scale이 커질 수 있으므로 / dst_in 적용.
    """
    if weight.shape == target_shape:
        return weight

    if weight.ndim == 5 and len(target_shape) == 5:
        src_in = weight.shape[1]
        dst_in = target_shape[1]

        if src_in == dst_in:
            return weight

        if src_in == 1 and dst_in > 1:
            return weight.repeat(1, dst_in, 1, 1, 1) / float(dst_in)

        if src_in < dst_in:
            repeat_factor = dst_in // src_in
            if src_in * repeat_factor == dst_in:
                return weight.repeat(1, repeat_factor, 1, 1, 1) / float(repeat_factor)

        if src_in > dst_in:
            return weight[:, :dst_in].contiguous()

    return weight


def _get_raw_network(network: nn.Module) -> nn.Module:
    """
    Handles possible wrappers: DDP's `.module` and torch.compile's `._orig_mod`.
    Without unwrapping `_orig_mod`, named_parameters()/state_dict() keys get a
    "_orig_mod." prefix once torch.compile has wrapped the network, which makes
    every "encoder."-prefixed key lookup below silently fail.
    """
    while True:
        if hasattr(network, "_orig_mod"):
            network = network._orig_mod
        elif hasattr(network, "module"):
            network = network.module
        else:
            break
    return network


def _load_triad_encoder_into_network(
    network: nn.Module,
    checkpoint_path: Path,
    min_loaded_tensors_warn: int = 10,
) -> Tuple[list, list]:
    network = _get_raw_network(network)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Triad checkpoint not found: {checkpoint_path}")

    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_checkpoint(raw_ckpt)
    state_dict = _strip_known_prefixes(state_dict)

    encoder_sd = {
        key: value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }

    current = network.state_dict()

    filtered = {}
    skipped_missing_key = []
    skipped_shape = []
    adapted_input_keys = []

    for key, value in encoder_sd.items():
        if key not in current:
            skipped_missing_key.append(key)
            continue

        target_shape = current[key].shape
        original_shape = value.shape

        if value.shape != target_shape:
            if _is_likely_input_weight_key(key):
                value = _expand_input_weight(value, target_shape)

                if value.shape == target_shape:
                    adapted_input_keys.append(
                        (key, tuple(original_shape), tuple(target_shape))
                    )

            if value.shape != target_shape:
                skipped_shape.append(
                    (key, tuple(original_shape), tuple(target_shape))
                )
                continue

        filtered[key] = value

    missing, unexpected = network.load_state_dict(filtered, strict=False)

    print("\n[TriadInit] Encoder initialization report")
    print(f"[TriadInit] checkpoint path          : {checkpoint_path}")
    print(f"[TriadInit] checkpoint encoder tensors: {len(encoder_sd)}")
    print(f"[TriadInit] loaded tensors           : {len(filtered)}")
    print(f"[TriadInit] skipped missing key      : {len(skipped_missing_key)}")
    print(f"[TriadInit] skipped shape mismatch   : {len(skipped_shape)}")
    print(f"[TriadInit] adapted input weights    : {len(adapted_input_keys)}")
    print(f"[TriadInit] model missing after load  : {len(missing)}")
    print(f"[TriadInit] unexpected after load     : {len(unexpected)}")

    if len(filtered) > 0:
        print("[TriadInit] first loaded keys:")
        for k in list(filtered.keys())[:15]:
            print(f"  {k}: {tuple(filtered[k].shape)}")

    if len(adapted_input_keys) > 0:
        print("[TriadInit] adapted input keys:")
        for k, src_shape, dst_shape in adapted_input_keys[:10]:
            print(f"  {k}: ckpt={src_shape} -> model={dst_shape}")

    if len(skipped_shape) > 0:
        print("[TriadInit] first skipped shape mismatches:")
        for k, src_shape, dst_shape in skipped_shape[:15]:
            print(f"  {k}: ckpt={src_shape}, model={dst_shape}")

    if len(filtered) < min_loaded_tensors_warn:
        print(
            "[WARNING][TriadInit] Very few tensors were loaded. "
            "This may mean the nnU-Net architecture does not match the Triad checkpoint."
        )

    return missing, unexpected


def _freeze_encoder(network: nn.Module):
    network = _get_raw_network(network)

    for name, param in network.named_parameters():
        if name.startswith("encoder."):
            param.requires_grad = False


def _freeze_encoder_except_stem(network: nn.Module):
    """
    Freeze encoder, but keep first encoder conv block trainable.

    Purpose:
    - Preserve most foundation encoder representation.
    - Allow 1-channel pretrained stem to adapt to 4-channel MRI.
    """
    network = _get_raw_network(network)

    for name, param in network.named_parameters():
        if name.startswith("encoder."):
            param.requires_grad = False

    stem_prefixes = (
        "encoder.stages.0.0.convs.0",
    )

    for name, param in network.named_parameters():
        if name.startswith(stem_prefixes):
            param.requires_grad = True


def _print_trainable_summary(network: nn.Module, max_lines: int = 80):
    network = _get_raw_network(network)

    total = sum(p.numel() for p in network.parameters())
    trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    frozen = total - trainable

    print("\n[TriadInit] Parameter summary")
    print(f"[TriadInit] total params     : {total:,}")
    print(f"[TriadInit] trainable params : {trainable:,}")
    print(f"[TriadInit] frozen params    : {frozen:,}")

    print(f"[TriadInit] trainable parameter names, first {max_lines}:")
    count = 0
    for name, param in network.named_parameters():
        if param.requires_grad:
            print(f"  [TRAIN] {name} {tuple(param.shape)}")
            count += 1
            if count >= max_lines:
                print("  ...")
                break

    if count == 0:
        print("[WARNING][TriadInit] No trainable parameters found.")


class nnUNetTrainerBraTS_TriadInit_RCOversample(nnUNetTrainerBraTS_RCOversample):
    """
    Triad encoder init + RC oversampling.

    Default behavior:
    - Load Triad pretrained encoder into nnU-Net network.
    - Do NOT freeze encoder.
    - Full fine-tuning.
    - RC oversampling behavior inherited from nnUNetTrainerBraTS_RCOversample.

    This is the recommended first experiment:
    foundation initialization + nnU-Net training recipe + RC oversampling.
    """

    triad_ckpt: str = str(TRIAD_CKPT_DEFAULT)

    # Options:
    # "none"       : full fine-tuning
    # "encoder"    : freeze entire encoder
    # "stem_only"  : freeze encoder except first conv block
    freeze_mode: str = "none"

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


# ---------------------------------------------------------------------
# Recommended full fine-tuning variants
# ---------------------------------------------------------------------

class nnUNetTrainerBraTS_TriadInit_RCOversample__2(nnUNetTrainerBraTS_TriadInit_RCOversample):
    rc_weight: float = 2.0
    freeze_mode: str = "none"


class nnUNetTrainerBraTS_TriadInit_RCOversample__2p5(nnUNetTrainerBraTS_TriadInit_RCOversample):
    rc_weight: float = 2.5
    freeze_mode: str = "none"


class nnUNetTrainerBraTS_TriadInit_RCOversample__3(nnUNetTrainerBraTS_TriadInit_RCOversample):
    """
    Main recommended experiment.

    Triad encoder init
    + full fine-tuning
    + RC oversampling x3
    """
    rc_weight: float = 3.0
    freeze_mode: str = "none"


# ---------------------------------------------------------------------
# Frozen encoder variants
# ---------------------------------------------------------------------

class nnUNetTrainerBraTS_TriadInit_RCOversample_Frozen__3(nnUNetTrainerBraTS_TriadInit_RCOversample):
    """
    Representation probe.

    Triad encoder init
    + encoder frozen
    + decoder trained
    + RC oversampling x3
    """
    rc_weight: float = 3.0
    freeze_mode: str = "encoder"


class nnUNetTrainerBraTS_TriadInit_RCOversample_Frozen__3_LP100(
    nnUNetTrainerBraTS_TriadInit_RCOversample_Frozen__3
):
    """
    LP stage for LP-FT.

    Same as Frozen__3 (encoder frozen, decoder trained)
    but capped at 100 epochs instead of the default 1000.

    Usage:
    1. Train this class to get a short decoder warm-up checkpoint.
    2. Feed that checkpoint via -pretrained_weights into
       nnUNetTrainerBraTS_TriadInit_RCOversample__3 (freeze_mode="none")
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


class nnUNetTrainerBraTS_TriadInit_RCOversample_StemOnly__3(nnUNetTrainerBraTS_TriadInit_RCOversample):
    """
    Middle-ground probe.

    Triad encoder init
    + freeze encoder except first conv block
    + decoder trained
    + RC oversampling x3

    Useful because Triad checkpoint is 1-channel while current input is 4-channel.
    """
    rc_weight: float = 3.0
    freeze_mode: str = "stem_only"


# ---------------------------------------------------------------------
# Optional: no extra RC oversampling baseline if rc_weight=1 means neutral
# ---------------------------------------------------------------------

class nnUNetTrainerBraTS_TriadInit_RCOversample__1(nnUNetTrainerBraTS_TriadInit_RCOversample):
    """
    Triad init with neutral RC sampling weight.

    Use only if your rc_sampling implementation treats rc_weight=1.0 as no oversampling.
    """
    rc_weight: float = 1.0
    freeze_mode: str = "none"