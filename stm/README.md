# STM modality

Scanning tunnelling microscopy z-stacks, paired with the molecular structures that produced
them, trained under a contrastive objective.

## Problem

A cube can be compressed with near-zero reconstruction error while its embedding carries no
information about which molecule produced it -- reconstruction and identification are not the
same objective. This modality trains for the second: whether a cube's embedding identifies its
molecule, via InfoNCE between the cube embedding and a 3-D structure embedding of the same
molecule, with the rest of the batch as negatives.

## Encoders

Selected by `model.cube_encoder`. All return a LayerNorm-scaled vector of width `latent_dim`,
so differences between them are attributable to the encoder alone.

| name | description |
| --- | --- |
| `stm3d` | 3-D convolutions along the ordered z axis, followed by a 2-D ladder. The recommended arm. |
| `stm2d` | A pretrained 2-D vision backbone applied to a single slice, following the ASD-STM protocol. Thirteen of fourteen slices are discarded by construction. |
| `sgpt` | A SpectralGPT-shaped 3-D vision transformer, optionally initialised from the published checkpoint. |
| `ters` | The TERS ladder applied to STM cubes. A control: its 1x1 stem treats the slices as unordered. |

## Pretrained weights

The structure encoder checkpoint required by the contrastive arm is described in the
[top-level README](../README.md#pretrained-models). Two encoders take optional
pretrained weights of their own:

| encoder | weights | source |
| --- | --- | --- |
| `sgpt` | `SpectralGPT+.pth` (1.18 GB) | [Zenodo 8412455](https://zenodo.org/records/8412455) |
| `stm2d` | per backbone, e.g. `timm/vit_base_patch16_clip_224.openai` for `clip_vit_b16` | Hugging Face Hub, via timm |

```bash
mkdir -p checkpoints
wget -O checkpoints/SpectralGPT+.pth \
  "https://zenodo.org/api/records/8412455/files/SpectralGPT+.pth/content"
```

`stm2d` downloads its backbone through timm on first use. Compute nodes may have no internet
access, so fetch it on a login node first:

```bash
python -c "from huggingface_hub import hf_hub_download; \
print(hf_hub_download('timm/vit_base_patch16_clip_224.openai', 'pytorch_model.bin'))"
```

These weights are not config keys; pass them through the trainer's flag interface:

```bash
python -m shared.trainers.contrastive --cube-encoder sgpt  --sgpt-ckpt  checkpoints/SpectralGPT+.pth
python -m shared.trainers.contrastive --cube-encoder stm2d --stm2d-ckpt <path printed above>
```

Without weights, `sgpt` and `stm2d` train from random initialisation, which is the
from-scratch control rather than the transfer arm.

## Two objectives over one encoder

Both entry points build the same cube encoder at the same 256-number budget, differing only in
what the vector is scored on -- which is what makes the two results comparable.

| | entry point | config | scored on |
| --- | --- | --- | --- |
| contrastive | `stm/train_clip.py` | `configs/stm_clip.yaml` | top-1 retrieval of the molecule |
| reconstruction | `stm/train.py` | `configs/stm_ae.yaml` | MSE of the rebuilt z-stack |

## Results

`configs/stm_clip.yaml`, trained on 184,180 z-stacks and evaluated on a held-out file:

| metric | value | reference |
| --- | --- | --- |
| size-controlled top-1 retrieval | 0.667 | chance 0.118 (batch 256) |
| normalised rank | 0.087 | 0.5 = chance, 0.0 = perfect |
| linear probe on the exported vector | 2.57x chance | random-init control 1.01x |

A reconstruction-trained encoder reaches MSE 0.0065 against a 0.4809 baseline, yet retrieves
at chance. Retrieval is reported against a size-controlled baseline because atom count alone
identifies a molecule well above chance (0.403 top-1 at batch 64, against 1/B = 0.016).

## Usage

```bash
module load python-pytorch/2.10
export PYTHONPATH="$PWD"

python -m stm.train_clip --config configs/stm_clip.yaml   # contrastive: retrieval
python -m stm.train      --config configs/stm_ae.yaml     # reconstruction: MSE
python -m stm.tests.test_stm                              # 200 assertions
```

`--synthetic` runs either training command against a generated corpus with the real schema --
no cluster, GPU, or corpus needed. It's a functional check, not an experiment; real training
goes through SLURM.

## Layout

```text
stm/
  train_clip.py     entry point: contrastive
  train.py          entry point: reconstruction
  builders.py       registers this modality's readers, encoders and models
  datasets/         the HDF5 readers and their collate
  models/           conv3d.py, timm2d.py, spectral_gpt.py, autoencoder.py
  transforms/       normalisation and the z-slice draw
  tests/            owned by this modality; imports no other modality
```

The training loops live in `shared/trainers/`, which names no modality -- it reaches this
folder's readers and models through the registry, so deleting `ters/` leaves both arms above
working.

## References

ASD-STM (Kurki, Oinonen & Foster, arXiv:2312.08854) -- automated structure discovery for
scanning tunnelling microscopy. MolSpectra (arXiv:2502.16284) -- the contrastive stage is
reimplemented from its published equation.
