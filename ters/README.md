# TERS modality

Tip-enhanced Raman spectroscopy hyperspectral cubes, trained under a reconstruction objective.

## Problem

Each cube is a deterministic function of its molecule -- the spectral channels are its normal
modes. This modality asks how much of a cube survives a fixed-width bottleneck, where that
width is not arbitrary: it is the width of the vector a downstream diffusion model conditions
on. Reconstruction error is therefore a statement about that conditioning vector's sufficiency,
not just about compression.

Two bounds on the width are measured. A molecule needs `3N` coordinates plus `N` types, less
six rigid degrees of freedom -- below that floor, loss is guaranteed for the largest molecules
regardless of model quality. PCA over the corpus finds no low-dimensional structure, so
reconstruction error falls smoothly with width and never shows a knee; the width cannot be
picked from a reconstruction-error sweep alone.

## Models

Selected by `model.name`.

| name | description |
| --- | --- |
| `ters_autoencoder` | The 2-D convolutional ladder, mirrored. The reference arm; at `latent_hw=1` its encoder is the reference implementation unmodified. |
| `hsi_spectral` | A per-pixel 1-D spectral autoencoder, after Kuester et al. |
| `hsi_conv3d` | A 3-D convolutional autoencoder, after Chong et al. |

## Two objectives over one encoder

Both entry points build `TersEncoder` at the same 256-number budget, differing only in what
the vector is scored on. That vector is the one the diffusion repo conditions on, so both
numbers describe the conditioning vector itself, not an autoencoder in isolation.

| | entry point | config | scored on |
| --- | --- | --- | --- |
| reconstruction | `ters/train.py` | `configs/ters_ae.yaml` | MSE of the rebuilt cube |
| contrastive | `ters/train_clip.py` | `configs/ters_clip.yaml` | top-1 retrieval of the molecule |

The contrastive arm is unverified against real data and currently runs `--synthetic` only. It
needs the pretrained structure-encoder checkpoint described in the
[top-level README](../README.md#pretrained-models); the reconstruction arm needs none.

## Usage

```bash
module load python-pytorch/2.10
export PYTHONPATH="$PWD"

python -m ters.train      --config configs/ters_ae.yaml   --synthetic  # reconstruction: MSE
python -m ters.train_clip --config configs/ters_clip.yaml --synthetic  # contrastive: retrieval
python -m ters.tests.test_dataset                                      # reader assertions
python -m ters.tests.test_shapes                                       # model assertions
```

## Layout

```text
ters/
  train.py          entry point: reconstruction
  train_clip.py     entry point: contrastive
  builders.py       registers this modality's readers, encoders and models
  datasets/         the HDF5 readers and their collate
  models/           encoder.py, autoencoder.py, hsi_conv3d.py, hsi_spectral.py
  losses/           the spectral-angle objective
  transforms/       per-channel normalisation
  tests/            owned by this modality; imports no other modality
```

The training loops live in `shared/trainers/`, which names no modality -- it reaches this
folder's readers and models through the registry, so deleting `stm/` leaves both arms above
working.

`ters/models/encoder.py` is an unmodified copy of the reference implementation and must stay
one: the premise of the experiment is that the bottleneck *is* the conditioning vector, not one
resembling it. Its state_dict key names are asserted against the reference repository, and
improvements belong in the decoder, which is ours.

## References

Kuester, Gross & Middelmann, *1D-Convolutional Autoencoder*, ISPRS Archives
XLIII-B1-2021:15-21 -- `hsi_spectral`. Chong, Chen & Pan, *3-D convolutional autoencoder*, J.
Electron. Imaging 30(4):041403 (2021) -- `hsi_conv3d`. Hu, Shen, Albanie, Sun & Wu,
*Squeeze-and-Excitation Networks*, arXiv:1709.01507 -- the spectral gating block in
`hsi_conv3d`.
