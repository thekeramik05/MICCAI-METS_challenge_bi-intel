# BraTS-METS 2026 — Task 1: Brain Metastases Segmentation

An ensemble of two nnU-Net v2 models — one initialised from a pretrained MRI
encoder, one given auxiliary input channels — followed by size-gated, class-wise
confidence post-processing.

Code in this repository is licensed under the [Apache License 2.0](LICENSE).
The model weights are distributed separately and are **not** covered by that
licence: Model B is governed by the BrainIAC Research-Only License and is
restricted to non-commercial academic research and educational use. See
[§7 Third-party components](#7-third-party-components-and-licensing).

---

## 1. Method overview

| | Model A ("Triad") | Model B ("BrainIAC") |
|---|---|---|
| Backbone | nnU-Net v2 `3d_fullres` (PlainConvUNet) | nnU-Net v2 `3d_fullres` (PlainConvUNet) |
| Pretrained component | Triad PlainConvUNet-MAE checkpoint used to **initialise the encoder weights** | Frozen ViT-B/16 (BrainIAC) used as an **auxiliary feature extractor** |
| Coupling | encoder weights loaded, then trained | ViT features projected to 4 channels and concatenated to the 4 MRI channels (8 input channels) |
| Training | two-stage: LP100 (encoder frozen) → FT900 (full fine-tune) | single stage, RC oversampling |
| Loss | Dice + voxel-weighted CE (small-lesion weighting, CE weight 3) | Dice + CE, RC oversampling |
| Folds used at inference | 0, 1, 2, 3, 4 | **1, 3 only** |

The two coupling strategies are deliberately different: Triad replaces the
encoder initialisation, whereas BrainIAC never touches the U-Net encoder — it
only widens the input tensor. The two models therefore start from different
initialisations and see different inputs, which is what makes ensembling useful
here. Section 1.1 states what this does and does not establish about the
pretrained components themselves.

Only folds 1 and 3 of Model B are used. Two separate constraints set this: the
number of folds was capped at two by the inference time budget, and folds 1 and
3 were the pair selected on internal cross-validation performance when ensembled
with Model A.

### 1.1 What the pretrained components explain

A control was run for Model B on fold 1 (n = 259 cases; RC reference components
present in 41 of them). Holding the 8-channel input fixed and replacing the four
BrainIAC-derived channels with **random features** of the same shape gives, for
the resection cavity region:

| Comparison | ΔDice (bootstrap 95% CI) |
|---|---|
| random-feature control − plain nnU-Net | **+0.0911 [+0.0205, +0.1684]** |
| BrainIAC − random-feature control | −0.0137 [−0.0417, +0.0098] (n.s.) |
| duplicated-input control − random-feature control | −0.0728 (worse) |

Two conclusions follow. The auxiliary channels do help RC, and they have to
carry independent variation: duplicating existing input channels performs worse
than noise. But **we find no evidence that BrainIAC's pretrained representation
outperforms random features of the same shape.** The effect we can demonstrate
belongs to the auxiliary-channel design, not to the specific pretraining.

Model A's Triad initialisation was not ablated, so no claim is made about its
contribution in either direction.

This control is a single fold and constrains interpretation rather than settling
it; it is not a full ablation study.

### 1.2 Ensembling

The two softmax maps are averaged **per class**:

| Channel | Model A weight | Model B weight |
|---|---|---|
| background | 0.50 | 0.50 |
| NETC | 0.50 | 0.50 |
| SNFH | 0.50 | 0.50 |
| ET | 0.50 | 0.50 |
| **RC** | **0.15** | **0.85** |

Everything stays at a plain 50/50 average except the resection cavity. On the
official hidden validation set Model B was clearly better on RC while the two
models were statistically indistinguishable on the other regions, so RC is the
only channel where a deviation from 50/50 is supported by measured evidence.

A per-class weight grid search was also carried out. Under the *true*
joint-argmax decision rule its predicted gains almost completely disappeared
(ΔDice ≈ +0.0006 / +0.0000 / −0.0001 / +0.0060 for NETC / SNFH / ET / RC), so
those weights were **not** adopted.

### 1.3 Post-processing

Per class, 26-connected components are extracted from the argmax label map.
A component is deleted only when **both** conditions hold:

1. it is smaller than **50 voxels**, and
2. its mean softmax confidence is below the class threshold:

| Class | Threshold |
|---|---|
| NETC | 0.60 |
| SNFH | 0.60 |
| ET | 0.60 |
| RC | 0.50 |

Components of 50 voxels or more are always kept, regardless of confidence.

The cutoff and the thresholds were fitted on the fold 1 + fold 3 validation
split (`scripts/analysis/02_*`, `04_*`, `10_*`) from the confidence distribution
of false-positive components stratified by component size. Two patterns drove
the choice: confidence rises with component size, and RC sits about 0.1 lower
than every other class across all size bins.

This is deliberately *not* a plain `min_size` filter. Measured on Model A alone,
a naive "delete everything under 10 voxels" rule would have removed 99 NETC and
237 ET true-positive components whose mean confidence (0.757 and 0.773) was
higher than that of the false positives it was meant to remove.

No other post-processing is applied — no largest-component selection, no hole
filling, and no label-hierarchy enforcement. A hierarchy rule ("ET must touch
SNFH") was tested and rejected: on fold 1 + 3 the SNFH-non-adjacent ET
components contained 823 true positives against only 203 false positives, so
enforcing it would have cost far more than it gained.

---

## 2. Repository layout

```
.
├── Dockerfile                 # challenge submission image
├── docker/
│   ├── entrypoint.sh          # container entrypoint
│   └── install_trainers.py    # copies trainers into the nnU-Net package
├── src/
│   ├── config.py              # all paths, weights and thresholds
│   ├── predict.py             # inference entrypoint (case discovery → output)
│   └── postprocess.py         # ensembling + confidence post-processing
├── trainers/                  # custom nnU-Net trainer classes
├── scripts/
│   ├── prepare_weights.sh     # stage checkpoints into weights/ before build
│   ├── train_model_a_triad.sh
│   ├── train_model_b_brainiac.sh
│   └── analysis/              # validation and threshold-fitting experiments
├── splits/
│   └── splits_final.json      # the five-fold split used throughout
├── environment.yaml
├── requirements.txt
└── LICENSE
```

---

## 3. Model weights

The fine-tuned checkpoints are hosted on Hugging Face:

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model_Weights-blue)](https://huggingface.co/keramik05/MICCAI-METS_challenge_bi-intel_model)

Download `weights.zip` and unzip it at the repository root:

```bash
unzip weights.zip          # run from the repository root
```

```
weights/
└── nnUNet_results/Dataset001_BraTS/
    ├── nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3_FT900__nnUNetPlans__3d_fullres/
    │   ├── fold_0/checkpoint_final.pth ... fold_4/checkpoint_final.pth   # Model A, 5 folds
    │   ├── dataset.json
    │   └── plans.json
    └── nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__3__nnUNetPlans__3d_fullres/
        ├── fold_1/checkpoint_final.pth, fold_3/checkpoint_final.pth      # Model B, 2 folds
        ├── dataset.json
        └── plans.json
```

Seven fold checkpoints in total (five for Model A, two for Model B), matching
`MODEL_A_FOLDS` and `MODEL_B_FOLDS` in `src/config.py`. Verify with:

```bash
find weights -name checkpoint_final.pth | wc -l     # expect 7
```

If you already hold the checkpoints on a training machine,
`scripts/prepare_weights.sh` stages them into the same layout instead.

### 3.1 Pretrained encoders — obtained separately

The Triad and BrainIAC checkpoints are **not redistributed here**. Download them
from their authors and point the environment variables at them:

| Variable | Checkpoint | Source |
|---|---|---|
| `TRIAD_CKPT` | `Triad-PlainConvUNet-MAE.pth` | https://github.com/wangshansong1/Triad |
| `BRAINIAC_CKPT` | `BrainIAC.ckpt` | https://github.com/AIM-KannLab/BrainIAC |

`TRIAD_CKPT` is required for training Model A only. `BRAINIAC_CKPT` is required
for both training and inference of Model B, because the wrapper reconstructs the
ViT before loading the fold checkpoint.

### 3.2 What the Model B checkpoints contain

Model B checkpoints are not standalone nnU-Net weights. Each contains a
complete, frozen copy of the BrainIAC ViT-B/16 backbone:

| Module prefix | Parameters | Origin |
|---|---|---|
| `base_network.*` | 88.2 M | nnU-Net, trained here |
| `brainiac.*` | **116.7 M** | **BrainIAC ViT-B/16, verbatim** |
| `proj.*` | 3 K | projection layer, trained here |
| total | 204.9 M | |

Downloading Model B therefore means obtaining BrainIAC weights, and use is
restricted to non-commercial academic research accordingly. Model A contains no
third-party weights: the Triad checkpoint was used only to initialise the
encoder, which was then fully fine-tuned, leaving a plain 88.2 M-parameter
nnU-Net `encoder`/`decoder` structure.

To inspect a checkpoint yourself:

```python
import collections, torch
w = torch.load("fold_1/checkpoint_final.pth", map_location="cpu")["network_weights"]
print(collections.Counter(k.split(".")[0] for k in w))
```

---

## 4. Environment setup (conda, no sudo)

```bash
conda env create -f environment.yaml
conda activate brats-mets

# CUDA 12.4 torch build
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

The custom trainers must be visible to nnU-Net. nnU-Net resolves trainer classes
with `recursive_find_python_class`, which walks the **physical** `variants/`
directory inside the installed package — setting `PYTHONPATH` alone is not
sufficient:

```bash
VARIANTS=$(python -c "import nnunetv2,os;print(os.path.join(os.path.dirname(nnunetv2.__file__),'training','nnUNetTrainer','variants'))")
cp trainers/*.py "$VARIANTS/"
export PYTHONPATH="$VARIANTS:${PYTHONPATH:-}"   # needed for the bare imports inside the trainers
```

Set the nnU-Net paths:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

---

## 5. Training

Dataset `Dataset001_BraTS`, `3d_fullres`, patch size `[112, 160, 128]`,
batch size 2, SGD (momentum 0.99, Nesterov), initial LR 0.01. The five-fold
split is fixed in `splits/splits_final.json`.

```bash
export TRIAD_CKPT=/path/to/Triad-PlainConvUNet-MAE.pth
export BRAINIAC_CKPT=/path/to/BrainIAC.ckpt

# Model A — folds 0-4 (LP100 then FT900)
for f in 0 1 2 3 4; do bash scripts/train_model_a_triad.sh $f; done

# Model B — folds 1 and 3 are the ones used at inference
for f in 1 3; do bash scripts/train_model_b_brainiac.sh $f; done
```

---

## 6. Inference

### 6.1 Local

```bash
export nnUNet_results=/path/to/nnUNet_results
export BRAINIAC_CKPT=/path/to/BrainIAC.ckpt

python src/predict.py --input /path/to/Validation --output /path/to/results
```

`--dry-run` lists the discovered cases and verifies that every checkpoint is
present without running the network.

Input layout (one folder per case, as shipped by the challenge):

```
Validation/
└── BraTS-MET-12345-100/
    ├── BraTS-MET-12345-100-t1c.nii.gz
    ├── BraTS-MET-12345-100-t1n.nii.gz
    ├── BraTS-MET-12345-100-t2f.nii.gz
    └── BraTS-MET-12345-100-t2w.nii.gz
```

Output is flat, one file per case, named after the case folder
(`BraTS-MET-12345-100.nii.gz`), which satisfies the required
`<5-digit id>-<3-digit timepoint>.nii.gz` ending.

### 6.2 Docker

```bash
bash scripts/prepare_weights.sh      # only if weights/ is empty
docker build -t brats-mets-task1:latest .
```

Run with the official command:

```bash
docker run \
  --rm \
  --network none \
  --gpus=all \
  --volume $PWD/Validation/:/input:ro \
  --volume $PWD/results/:/output:rw \
  --memory=48G --shm-size=16G \
  brats-mets-task1:latest
```

`--network none` is worth keeping in any local test: a container that reaches
the network during inference will work on a developer machine and fail on an
isolated grader.

### 6.3 Runtime

The models were trained on an H200 (143 GiB), but inference is memory-light:
nnU-Net keeps every fold's weights in CPU RAM and swaps them into a *single* GPU
network one at a time, so VRAM is driven by the sliding-window buffers for a
`112×160×128` patch, not by the number of folds. Peak usage measured well under
24 GiB; the extra ViT-B/16 in Model B adds roughly 0.4 GiB.

Wall-clock is the tighter constraint. The entrypoint runs seven network passes
per case (five folds of Model A, two of Model B). Measured per case, TTA off:

| GPU | per case | source |
|---|---|---|
| H200 (143 GiB) | ≈19.6 s | measured |
| **RTX 4090 (24 GiB)** | **16.19 s** (σ 1.04, n = 7 after warm-up) | **measured** |
| A10G (24 GiB) | 23–35 s | extrapolated |

Most of the per-case time is CPU-bound preprocessing (resampling, normalisation)
and export back to the native grid; only a small fraction is GPU network time,
which is why the 4090 — despite far more compute than an A10G — lands close to
the H200. The A10G range scales the GPU component by its FP16 throughput ratio
and allows for a host CPU up to 2× slower than the measured machine.

At 303 cases a 12 h budget corresponds to 142.6 s/case, so the projected
1.9–3.0 h leaves a 4–6× margin. Enabling 8× mirroring multiplies only the GPU
component and lands near 8.5 h, cutting the margin to about 1.4× on an unmeasured
device. TTA is therefore **disabled by default**; re-enable it only after
verifying timing on the target GPU:

```bash
docker run ... -e BRATS_DISABLE_TTA=0 ...
```

Both models share the same plans, so preprocessing and export currently run
twice per case. Doing them once and reusing the result would remove most of that
overhead. It was left out deliberately: the current path is validated against the
reference pipeline (below) and the run already fits the budget with a wide margin.

All knobs are overridable with `-e`:

| Variable | Default | Meaning |
|---|---|---|
| `BRATS_DISABLE_TTA` | `1` | disable 8× mirroring |
| `BRATS_STEP_SIZE` | `0.5` | sliding-window step (raise to 0.7 for more speed) |
| `BRATS_ENABLE_PP` | `1` | enable confidence post-processing |
| `BRATS_PP_SIZE_CUTOFF` | `50` | component size gate |
| `BRATS_PP_T_NETC/SNFH/ET/RC` | `0.6/0.6/0.6/0.5` | per-class confidence thresholds |

### 6.4 Spatial correctness

All I/O goes through nnU-Net's `SimpleITKIO`, so spacing, origin and direction
are carried over from the input image rather than reconstructed. Every written
segmentation is re-opened and compared against its input `t1c`; any mismatch in
size, spacing, origin or direction aborts the run instead of producing a silently
misaligned submission. The pipeline never writes to `/input` and asserts that
`/output` stays flat.

This matters because an earlier version of the offline ensembling code fed
already-resampled probability maps to `export_prediction_from_logits`, which
expects *pre*-crop/*pre*-resample logits and re-applies the stored cropping
itself. The result was catastrophically misaligned (IoU ≈ 0.075 against the same
model's own native output) while still producing plausible-looking files. Routing
everything through `SimpleITKIO` plus the post-write geometry assertion removes
that entire class of failure.

**Validation.** Running this entrypoint on a case that had also been produced by
the offline reference pipeline gives 3 differing voxels out of ~37 000 foreground
voxels (foreground Dice 0.999973); the residual is float summation order. With
TTA disabled the agreement is Dice ≈ 0.996, the expected TTA difference.

---

## 7. Third-party components and licensing

The Apache-2.0 licence covers **the code in this repository only**. The
pretrained encoders are not redistributed here, and the released fold
checkpoints carry their own terms.

| Component | Licence | Source |
|---|---|---|
| nnU-Net v2 | Apache-2.0 | https://github.com/MIC-DKFZ/nnUNet |
| Triad (PlainConvUNet-MAE) | MIT | https://github.com/wangshansong1/Triad |
| BrainIAC (ViT-B/16) | Non-commercial academic research only | https://github.com/AIM-KannLab/BrainIAC |

Triad is MIT-licensed and imposes no restriction on the derived Model A weights,
which were produced by initialising the encoder from `Triad-PlainConvUNet-MAE.pth`
and then fully fine-tuning it on BraTS-METS data. Triad itself builds on
[VoCo v2](https://github.com/Luffy03/Large-Scale-Medical).

Model B embeds the BrainIAC backbone (§3.2) and is therefore governed by the
**BrainIAC Research-Only License** rather than by a standard open licence. That
licence permits use, copying, modification and redistribution *solely for
non-commercial academic research and educational purposes*, which is the basis
on which the Model B checkpoints are published here. Commercial use — including
use in clinical workflows, decision-support systems, healthcare operations, or
any fee-bearing product or service — is prohibited without a separate written
licence from Mass General Brigham, and all commercial rights are reserved by
them. A copy of the licence ships with the weights archive.

Model A is unaffected: Triad is MIT-licensed, so those checkpoints carry no
comparable restriction.

---

## 8. Citation

If you use this code, please cite the challenge and the evaluation
infrastructure:

```bibtex
@article{moawad2023brainmetastasis,
  title   = {The Brain Tumor Segmentation (BraTS-METS) Challenge 2023:
             Brain Metastasis Segmentation on Pre-treatment MRI},
  author  = {Moawad, Ahmed W. and others},
  journal = {arXiv preprint arXiv:2306.00838},
  year    = {2023},
  doi     = {10.48550/arXiv.2306.00838},
  url     = {https://doi.org/10.48550/arXiv.2306.00838}
}

@article{karargyris2023medperf,
  title   = {Federated benchmarking of medical artificial intelligence
             with MedPerf},
  author  = {Karargyris, Alexandros and others},
  journal = {Nature Machine Intelligence},
  volume  = {5},
  pages   = {799--810},
  year    = {2023},
  doi     = {10.1038/s42256-023-00652-2}
}
```

Please also cite the method components this work builds on:

```bibtex
@article{isensee2021nnunet,
  title     = {nnU-Net: a self-configuring method for deep learning-based
               biomedical image segmentation},
  author    = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and
               Petersen, Jens and Maier-Hein, Klaus H.},
  journal   = {Nature Methods},
  volume    = {18},
  pages     = {203--211},
  year      = {2021},
  doi       = {10.1038/s41592-020-01008-z}
}

@article{wang2026triad,
  title   = {Vision foundation model for 3D magnetic resonance imaging
             segmentation, classification, and registration},
  author  = {Wang, Shansong and Safari, Mojtaba and Li, Qiang and
             Chang, Chih-Wei and Qiu, Richard LJ and Roper, Justin and
             Yu, David S. and Yang, Xiaofeng},
  journal = {Medical Image Analysis},
  volume  = {110},
  pages   = {103992},
  year    = {2026},
  issn    = {1361-8415},
  doi     = {10.1016/j.media.2026.103992},
  url     = {https://www.sciencedirect.com/science/article/pii/S1361841526000617}
}

@article{tak2026generalizable,
  title     = {A generalizable foundation model for analysis of human brain MRI},
  author    = {Tak, Divyanshu and Gormosa, B. A. and Zapaishchykova, A. and others},
  journal   = {Nature Neuroscience},
  year      = {2026},
  publisher = {Springer Nature},
  doi       = {10.1038/s41593-026-02202-6},
  url       = {https://doi.org/10.1038/s41593-026-02202-6}
}
```

---

## 9. Acknowledgements

Data used in this publication were obtained as part of the Challenge project
through Synapse ID (syn74274097).
