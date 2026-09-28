# STM modality

Scanning tunnelling microscopy z-stacks, paired with the molecular structures that produced
them, trained under a contrastive objective.

## Problem

A cube can be compressed with near-zero reconstruction error while its embedding carries no
information about which molecule produced it -- reconstruction and identification are not the
same objective. This modality trains for the second: whether a cube's embedding identifies its
molecule, via InfoNCE between the cube embedding and a 3-D structure embedding of the same
molecule, with the rest of the batch as negatives.

The dataset itself -- schema, splits, and known quirks -- is documented in
[`docs/04`](../docs/04-datasets.md), not here. What the retrieval metrics mean and how to read
a result: [`docs/08`](../docs/08-metrics-explained.md).

## Encoders

Selected by `model.cube_encoder`. All return a LayerNorm-scaled vector of width `latent_dim`,
so differences between them are attributable to the encoder alone.

| name | description |
| --- | --- |
| `stm3d` | 3-D convolutions along the ordered z axis, followed by a 2-D ladder. The recommended arm. |
| `stm2d` | A pretrained 2-D vision backbone applied to a single slice, following the ASD-STM protocol. Thirteen of fourteen slices are discarded by construction. |
| `sgpt` | A SpectralGPT-shaped 3-D vision transformer, optionally initialised from the published checkpoint. |
| `ters` | The TERS ladder applied to STM cubes. A control: its 1x1 stem treats the slices as unordered. |

## Two objectives over one encoder

Both entry points build the same cube encoder at the same 256-number budget, differing only in
what the vector is scored on -- which is what makes the two results comparable.

| | entry point | config | scored on |
| --- | --- | --- | --- |
| contrastive | `stm/train_clip.py` | `configs/stm_clip.yaml` | top-1 retrieval of the molecule |
| reconstruction | `stm/train.py` | `configs/stm_ae.yaml` | MSE of the rebuilt z-stack |

## Usage

```bash
module load python-pytorch/2.10
export PYTHONPATH="$PWD"

python -m stm.train_clip --config configs/stm_clip.yaml   # contrastive: retrieval
python -m stm.train      --config configs/stm_ae.yaml     # reconstruction: MSE
python -m stm.tests.test_stm                              # 200 assertions
sbatch slurm/train_full.sbatch                            # the experiment
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
working. `tests/test_boundary.py` asserts this by running each modality with the other absent.

## References

ASD-STM (Kurki, Oinonen & Foster, arXiv:2312.08854) -- automated structure discovery for
scanning tunnelling microscopy. MolSpectra (arXiv:2502.16284) -- the contrastive stage is
reimplemented from its published equation.
