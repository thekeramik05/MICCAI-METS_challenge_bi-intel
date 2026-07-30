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
import subprocess
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

    # The .pth is only processed at interpreter start-up, so it is NOT active in
    # this process. Several trainers import their helpers by bare module name
    # ("from rc_sampling import ...") and would raise ModuleNotFoundError if we
    # tried to resolve them here. Verify in a fresh interpreter instead, which
    # is also the state the container actually runs in.
    check = (
        "import os, sys\n"
        "from nnunetv2.utilities.find_class_by_name import recursive_find_python_class\n"
        "import nnunetv2\n"
        "root = os.path.join(os.path.dirname(nnunetv2.__file__), 'training', 'nnUNetTrainer')\n"
        "missing = [n for n in sys.argv[1:]\n"
        "           if recursive_find_python_class(root, n, 'nnunetv2.training.nnUNetTrainer') is None]\n"
        "print('\\n'.join('  OK ' + n for n in sys.argv[1:] if n not in missing))\n"
        "sys.exit('FATAL: trainer not discoverable: ' + ', '.join(missing) if missing else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", check, *REQUIRED_TRAINERS],
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
