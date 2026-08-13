# BraTS-METS 2026 — Task 1: Brain Metastases Segmentation

Ensemble of two foundation-model-augmented nnU-Net v2 models, followed by
size-gated, class-wise confidence post-processing.

Licensed under the [Apache License 2.0](LICENSE).

---

## 1. Method overview

| | Model A ("Triad") | Model B ("BrainIAC") |
|---|---|---|
| Backbone | nnU-Net v2 `3d_fullres` (PlainConvUNet) | nnU-Net v2 `3d_fullres` (PlainConvUNet) |
| Foundation model | Triad PlainConvUNet-MAE checkpoint used to **initialise the encoder weights** | Frozen ViT-B/16 (BrainIAC) used as an **auxiliary feature extractor** |
| Coupling | encoder weights loaded, then trained | ViT features projected to 4 channels and concatenated to the 4 MRI channels (8 input channels) |
| Training | two-stage: LP100 (encoder frozen) → FT900 (full fine-tune) | single stage, RC oversampling |
| Loss | Dice + voxel-weighted CE (small-lesion weighting, CE weight 3) | Dice + CE, RC oversampling |
| Folds used at inference | 0, 1, 2, 3, 4 | **1, 3 only** |

The two coupling strategies are deliberately different: Triad replaces the
encoder initialisation, whereas BrainIAC never touches the U-Net encoder — it
only widens the input tensor. This keeps the two models decorrelated enough for
ensembling to help.

Only folds 1 and 3 of model B are used. Two separate constraints set this: the
number of folds was capped at two by the inference time budget, and folds 1 and
3 were the pair selected on internal cross-validation performance when
ensembled with model A.

### Ensembling

The two softmax maps are averaged **per class**:

| Channel | Model A weight | Model B weight |
|---|---|---|
| background | 0.50 | 0.50 |
| NETC | 0.50 | 0.50 |
| SNFH | 0.50 | 0.50 |
| ET | 0.50 | 0.50 |
| **RC** | **0.15** | **0.85** |

Everything stays at a plain 50/50 average except the resection cavity. On the
official hidden validation set BrainIAC was clearly better on RC while the two
models were statistically indistinguishable on the other regions, so RC is the
only channel where a deviation from 50/50 is justified by measured evidence.

A per-class weight grid search was also carried out. Under the *true*
joint-argmax decision rule its predicted gains almost completely disappeared
(ΔDice ≈ +0.0006 / +0.0000 / −0.0001 / +0.0060 for NETC / SNFH / ET / RC), so
those weights were **not** adopted.

### Post-processing

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

This is deliberately *not* a plain `min_size` filter. Measured on model A alone,
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
├── Dockerfile                 # challenge submission image (Apache-2.0 header)
├── docker/entrypoint.sh       # container entrypoint
├── src/
│   ├── config.py              # all paths, weights and thresholds
│   ├── predict.py             # inference entrypoint (case discovery → output)
│   └── postprocess.py         # ensembling + confidence post-processing
├── trainers/                  # custom nnU-Net trainer classes
├── scripts/
│   ├── prepare_weights.sh     # stage checkpoints into weights/ before build
│   ├── train_model_a_triad.sh
│   ├── train_model_b_brainiac.sh
│   └── analysis/              # validation / threshold-fitting experiments
├── environment.yaml
├── requirements.txt
└── LICENSE
```

## 💾 Model Weights

The pre-trained checkpoints are hosted on Hugging Face:

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model_Weights-blue)](https://huggingface.co/keramik05/MICCAI-METS_challenge_bi-intel_model)

Download `weights.zip` and unzip it at the repository root. The archive already
contains the layout the build expects, so nothing has to be moved by hand:

```
weights/
├── nnUNet_results/Dataset001_BraTS/
│   ├── nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3_FT900__nnUNetPlans__3d_fullres/
│   │   ├── fold_0/checkpoint_final.pth ... fold_4/checkpoint_final.pth   # Model A, 5 folds
│   │   ├── dataset.json
│   │   └── plans.json
│   └── nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__3__nnUNetPlans__3d_fullres/
│       ├── fold_1/checkpoint_final.pth, fold_3/checkpoint_final.pth      # Model B, 2 folds
│       ├── dataset.json
│       └── plans.json
└── foundation_encoders/
    ├── Triad-PlainConvUNet-MAE.pth
    └── BrainIAC.ckpt
```

```bash
unzip weights.zip          # run from the repository root
docker build -t brats-mets-task1:latest .
```

Seven fold checkpoints in total (five for Model A, two for Model B), matching
`MODEL_A_FOLDS` and `MODEL_B_FOLDS` in `src/config.py`. If you already have the
checkpoints on a local training machine, `scripts/prepare_weights.sh` stages them
into the same layout instead — see §6.

---

## 3. Environment setup (conda, no sudo)

```bash
conda env create -f environment.yaml
conda activate brats-mets

# CUDA 12.4 torch build
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

The custom trainers must be visible to nnU-Net. nnU-Net resolves trainer
classes with `recursive_find_python_class`, which walks the **physical**
`variants/` directory inside the installed package — setting `PYTHONPATH` alone
is not sufficient:

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

## 4. Training

Dataset `Dataset001_BraTS`, `3d_fullres`, patch size `[112, 160, 128]`,
batch size 2, SGD (momentum 0.99, Nesterov), initial LR 0.01.

```bash
export TRIAD_CKPT=/path/to/Triad-PlainConvUNet-MAE.pth
export BRAINIAC_CKPT=/path/to/BrainIAC.ckpt

# Model A — folds 0-4 (LP100 then FT900)
for f in 0 1 2 3 4; do bash scripts/train_model_a_triad.sh $f; done

# Model B — folds 1 and 3 are the ones used at inference
for f in 1 3; do bash scripts/train_model_b_brainiac.sh $f; done
```

---

## 5. Inference

### Local

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

### Docker

```bash
# Only needed if weights/ is empty. The script reads from the training machine's
# nnUNet_results, so skip it when the checkpoints are already staged — check with
#   find weights -name checkpoint_final.pth | wc -l     # expect 7
bash scripts/prepare_weights.sh

docker build -t brats-mets-task1:latest .
```

For build, local verification and Synapse submission on a machine that already has
the staged `weights/`, follow `DOCKER_HANDOVER.md` instead — it covers the checks
this section leaves out (trainer-discovery assertion, output geometry validation,
both documented memory limits).

Official run command:

```bash
docker run \
  --rm \
  --network none \
  --gpus=all \
  --volume $PWD/Validation/:/input:ro \
  --volume $PWD/results/:/output:rw \
  --memory=48G --shm-size=16G \
  docker.synapse.org/PROJECT_ID/IMAGE_NAME:TAG
```

### Runtime notes (A10G, 24 GiB, 12 h budget)

The models were trained on an H200 (143 GiB) but inference is memory-light:
nnU-Net keeps every fold's weights in CPU RAM and swaps them into a *single*
GPU network one at a time, so VRAM is driven by the sliding-window buffers for
a `112×160×128` patch, not by the number of folds. Peak usage measured well
under 24 GiB; the extra ViT-B/16 in model B adds roughly 0.4 GiB.

Wall-clock is the tighter constraint. This entrypoint runs 7 network passes per
case (5 folds of model A + 2 of model B). Measured per case, TTA off:

| GPU | per case | source |
|---|---|---|
| H200 (143 GiB) | ≈19.6 s | measured |
| **RTX 4090 (24 GiB)** | **16.19 s** (σ 1.04, n=7 after warm-up) | **measured** |
| A10G (24 GiB) | 23–35 s | extrapolated |

Most of the per-case time is CPU-bound preprocessing (resampling,
normalisation) and export back to the native grid; only a small fraction is GPU
network time, which is why the 4090 — despite far more compute than an A10G —
lands close to the H200. The A10G range above scales the GPU component by its
FP16 throughput ratio and allows for a host CPU up to 2× slower than the
measured machine.

At 303 cases the 12 h limit corresponds to 142.6 s/case, so the projected
**1.9–3.0 h leaves a 4–6× margin**. Enabling 8× mirroring multiplies only the
GPU component and lands near 8.5 h, cutting the margin to about 1.4× on an
unmeasured device; TTA is therefore **disabled by default**.

Known optimisation, not applied: both models share the same plans, so the
preprocessing and the export currently run twice per case. Preprocessing once
and reusing the result for both networks would remove most of that overhead. It
was left out deliberately — the current path is validated byte-for-byte against
the reference pipeline (see below) and the run already fits the budget with a
wide margin.

To re-enable TTA (only if you have verified the timing on the target GPU):

```bash
docker run ... -e BRATS_DISABLE_TTA=0 ...
```

Other knobs, all overridable with `-e`:

| Variable | Default | Meaning |
|---|---|---|
| `BRATS_DISABLE_TTA` | `1` | disable 8× mirroring |
| `BRATS_STEP_SIZE` | `0.5` | sliding-window step (raise to 0.7 for more speed) |
| `BRATS_ENABLE_PP` | `1` | enable confidence post-processing |
| `BRATS_PP_SIZE_CUTOFF` | `50` | component size gate |
| `BRATS_PP_T_NETC/SNFH/ET/RC` | `0.6/0.6/0.6/0.5` | per-class confidence thresholds |

### Spatial correctness

All I/O goes through nnU-Net's `SimpleITKIO`, so spacing, origin and direction
are carried over from the input image rather than reconstructed. Every written
segmentation is then re-opened and compared against its input `t1c`; any
mismatch in size, spacing, origin or direction aborts the run instead of
producing a silently misaligned submission. The pipeline never writes to
`/input` and asserts that `/output` stays flat.

This matters because an earlier version of the offline ensembling code fed
already-resampled probability maps to `export_prediction_from_logits`, which
expects *pre*-crop/*pre*-resample logits and re-applies the stored cropping
itself. The result was catastrophically misaligned (IoU ≈ 0.075 against the
same model's own native output) while still producing plausible-looking files.
Routing everything through `SimpleITKIO` plus the post-write geometry assertion
removes that entire class of failure.

**Validation.** Running this entrypoint on a case that had also been produced by
the offline reference pipeline gives 3 differing voxels out of ~37 000
foreground voxels (foreground Dice 0.999973); the residual is float summation
order. With TTA disabled the agreement is Dice ≈ 0.996, the expected TTA
difference.

---

## 6. Submitting

> Verify every step against the current challenge instructions on Synapse —
> registry paths and submission mechanics change between editions.

### Docker image → Synapse registry

The challenge uses Synapse's own Docker registry (`docker.synapse.org`), not
Docker Hub. You need a Synapse account, a Synapse **project**, and a Synapse
**personal access token** (Account Settings → Personal Access Tokens) with the
`view`, `download` and `modify` scopes.

```bash
# 1. build (needs weights/ staged first)
bash scripts/prepare_weights.sh
docker build -t brats-mets-task1:latest .

# 2. smoke-test locally with the official run command BEFORE pushing
docker run --rm --network none --gpus=all \
  --volume $PWD/Validation/:/input:ro \
  --volume $PWD/results/:/output:rw \
  --memory=48G --shm-size=16G \
  brats-mets-task1:latest

# 3. log in (username = Synapse username, password = the access token)
docker login docker.synapse.org

# 4. tag with YOUR Synapse project id, then push
docker tag brats-mets-task1:latest docker.synapse.org/synXXXXXXX/brats-mets-task1:v1
docker push docker.synapse.org/synXXXXXXX/brats-mets-task1:v1
```

After the push the image appears under the **Docker** tab of your Synapse
project. Submit it from there (Docker Repository → *Submit to Challenge* →
pick the BraTS-METS Task 1 evaluation queue), or through whatever submission
form the organisers link that year.

Practical notes:

- The image is large (~10 GB: ~7 GB base + 2.9 GB weights). Allow time for the
  push, and push from a machine with a fast uplink.
- Always run step 2 first. `--network none` is the single most common cause of
  a container that works locally and fails on the grader.
- Tag deliberately (`v1`, `v2`, …). `latest` makes it impossible to tell which
  image a given submission actually ran.

### Source code → GitHub

The short paper must link a public repository. This repo is ready to push:

```bash
git remote add origin https://github.com/<user>/<repo>.git
git branch -M main
git push -u origin main
```

`weights/` is git-ignored — publish the checkpoints separately (HuggingFace or
Google Drive) and put that link in §7 below.

## 7. Model weights

Not included in this repository.

**Download:** [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model_Weights-blue)](https://huggingface.co/keramik05/MICCAI-METS_challenge_bi-intel_model)

Expected layout after extraction (also produced by `scripts/prepare_weights.sh`):

```
weights/
├── nnUNet_results/Dataset001_BraTS/
│   ├── nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3_FT900__nnUNetPlans__3d_fullres/
│   │   ├── plans.json, dataset.json
│   │   └── fold_{0,1,2,3,4}/checkpoint_final.pth
│   └── nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__3__nnUNetPlans__3d_fullres/
│       ├── plans.json, dataset.json
│       └── fold_{1,3}/checkpoint_final.pth
└── foundation_encoders/
    ├── BrainIAC.ckpt                    # required at inference to rebuild the ViT
    └── Triad-PlainConvUNet-MAE.pth      # training only
```

`BrainIAC.ckpt` is needed at **inference** time: the wrapper reconstructs its
ViT backbone inside `build_network_architecture` before the trained fold weights
are loaded over it. The Triad checkpoint is only read during training.

---

## 8. Preprint

[To be added — required if top-ranked before MICCAI announcement]

---

## 9. Citation

<!-- TODO: add BraTS-MET 2024 citation when released -->

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

Supporting method references (nnU-Net, Triad, BrainIAC) are listed in the short
paper.

---

## 10. Acknowledgements

Data used in this publication were obtained as part of the Challenge project
through Synapse ID (syn74274097).
