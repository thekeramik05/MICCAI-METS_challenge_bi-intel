# ---------------------------------------------------------------------------
# BraTS-METS 2026 Challenge — Task 1 (Brain Metastases Segmentation)
#
# Copyright 2026 The BraTS-METS submission authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
#
# Runtime target: NVIDIA A10G (24 GiB VRAM), CUDA <= 13.0, 48 GiB RAM,
#                 16 GiB /dev/shm, no network access, 12 h wall-clock budget.
#
# Build:
#   bash scripts/prepare_weights.sh          # stage checkpoints into weights/
#   docker build -t brats-mets-task1:latest .
# ---------------------------------------------------------------------------

# torch 2.6.0 + CUDA 12.4 runtime. sm_86 (A10G) is covered by this build and
# 12.4 stays below the challenge's CUDA 13.0 ceiling.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL org.opencontainers.image.title="BraTS-METS 2026 Task 1 — Triad + BrainIAC ensemble"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# --- Python dependencies (baked in; the container never reaches the network) --
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

# --- Custom nnU-Net trainers -----------------------------------------------
# Installed into the physical variants/ directory of the installed nnunetv2
# package; see docker/install_trainers.py for why PYTHONPATH is not sufficient.
# The script also asserts that both required trainers resolve, so a broken
# trainer set fails the build instead of the submission.
COPY trainers/ /opt/trainers/
COPY docker/install_trainers.py /opt/install_trainers.py
RUN python /opt/install_trainers.py && rm -rf /opt/trainers /opt/install_trainers.py

# --- Model weights ----------------------------------------------------------
# Populated by scripts/prepare_weights.sh (kept out of git, see .gitignore).
COPY weights/nnUNet_results/ /opt/weights/nnUNet_results/
COPY weights/foundation_encoders/ /opt/weights/foundation_encoders/

# --- Inference code ---------------------------------------------------------
COPY src/ /opt/app/src/
COPY docker/entrypoint.sh /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh

# --- nnU-Net environment ----------------------------------------------------
# raw/preprocessed are unused at inference time but nnU-Net warns loudly when
# they are unset, so point them at a writable scratch location.
ENV nnUNet_results=/opt/weights/nnUNet_results \
    nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    BRATS_WORK_DIR=/tmp/brats_work

# The BrainIAC wrapper reconstructs its frozen ViT-B/16 backbone inside
# build_network_architecture and needs the original encoder checkpoint present
# even though the trained fold weights overwrite it immediately afterwards.
ENV BRAINIAC_CKPT=/opt/weights/foundation_encoders/BrainIAC.ckpt \
    BRAINIAC_MODALITY_INDICES=0,1,2,3 \
    BRAINIAC_FEATURE_CHANNELS=4 \
    BRAINIAC_FREEZE=1 \
    TRIAD_CKPT=/opt/weights/foundation_encoders/Triad-PlainConvUNet-MAE.pth

# --- Inference behaviour (override with -e at run time if needed) -----------
ENV BRATS_INPUT_DIR=/input \
    BRATS_OUTPUT_DIR=/output \
    BRATS_DISABLE_TTA=1 \
    BRATS_ENABLE_PP=1

RUN mkdir -p /tmp/nnUNet_raw /tmp/nnUNet_preprocessed /tmp/brats_work

WORKDIR /opt/app
ENTRYPOINT ["/opt/app/entrypoint.sh"]
