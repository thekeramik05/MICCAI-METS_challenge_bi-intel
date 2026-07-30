#!/usr/bin/env python3
"""Install the custom nnU-Net trainers into the installed package tree.

nnU-Net resolves trainer classes with ``recursive_find_python_class``, which
walks the physical ``variants/`` directory inside the installed nnunetv2
package. Putting the files on PYTHONPATH is not enough — they have to live
there. Several of the trainers additionally import their helpers by bare module
name, so the variants directory itself is made importable through a .pth file.

Run at image build time; fails loudly if either required trainer cannot be
resolved afterwards.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import nnunetv2

SOURCE = Path(os.environ.get("TRAINER_SOURCE_DIR", "/opt/trainers"))
REQUIRED_TRAINERS = (
    "nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3_FT900",
    "nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__3",
)


def main() -> int:
    package_root = Path(nnunetv2.__file__).parent
    variants = package_root / "training" / "nnUNetTrainer" / "variants"
    variants.mkdir(parents=True, exist_ok=True)

    if not SOURCE.is_dir():
        print(f"FATAL: {SOURCE} not found", file=sys.stderr)
        return 1

    for item in sorted(SOURCE.iterdir()):
        if item.name == "__pycache__":
            continue
        target = variants / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
        print(f"  installed {item.name}")

    # Make bare imports such as "from rc_sampling import ..." resolvable.
    site_packages = package_root.parent
    (site_packages / "nnunet_brats_variants.pth").write_text(f"{variants}\n")
    print(f"registered {variants} on sys.path via .pth")

    # Verify discovery in a fresh interpreter-visible state.
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

    search_root = os.path.join(str(package_root), "training", "nnUNetTrainer")
    for name in REQUIRED_TRAINERS:
        cls = recursive_find_python_class(
            search_root, name, "nnunetv2.training.nnUNetTrainer"
        )
        if cls is None:
            print(f"FATAL: trainer not discoverable: {name}", file=sys.stderr)
            return 1
        print(f"  OK {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
