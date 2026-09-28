# spm-encoder

Learning molecule-identifying representations of scanning probe microscopy (SPM) data.

## Overview

SPM measurements are three-dimensional arrays: a stack of images over tip height for scanning
tunnelling microscopy (STM), or a Raman spectrum at every pixel for tip-enhanced Raman
spectroscopy (TERS). An encoder trained only to reconstruct such data can reach low
reconstruction error while its embedding carries little information about the molecule that
produced the measurement.

This repository trains each encoder with two objectives over the same fixed-size embedding:

- **Contrastive learning.** The measurement embedding is aligned with the embedding of a
  pretrained 3-D molecular structure encoder using the InfoNCE loss, and evaluated by top-1
  retrieval of the correct molecule.
- **Reconstruction.** The embedding is decoded back to the input, and evaluated by mean squared
  error.

On STM data, the contrastively trained encoder achieves 5.7 times the size-controlled chance
retrieval rate, whereas an encoder trained for reconstruction performs at chance.

## Modalities

| Modality | Data | Documentation |
| --- | --- | --- |
| STM | Constant-height image stacks | [stm/README.md](stm/README.md) |
| TERS | Hyperspectral images | [ters/README.md](ters/README.md) |

## Requirements

Developed on the CSC Roihu cluster (NVIDIA GH200) with the `python-pytorch/2.10` module, which
provides PyTorch, h5py, PyYAML, timm and PyTorch Geometric.

```bash
module load python-pytorch/2.10
export PYTHONPATH="$PWD"
```

## Pretrained models

Contrastive training requires the TorchMD-NET equivariant transformer pretrained on PCQM4Mv2
by denoising (Zaidi et al., 2023):

```bash
mkdir -p checkpoints
wget -O checkpoints/denoised-pcqm4mv2.ckpt \
  https://github.com/shehzaidi/pre-training-via-denoising/raw/2d81667c4daf519a1bfd33f1e1257eefa527db61/checkpoints/denoised-pcqm4mv2.ckpt
```

Set its path with `--set model.structure_ckpt=checkpoints/denoised-pcqm4mv2.ckpt`.
Reconstruction training does not use it. Pretrained weights for individual encoders are listed
in the modality documentation.

## Usage

```bash
python -m stm.train_clip  --config configs/stm_clip.yaml    # STM, contrastive
python -m stm.train       --config configs/stm_ae.yaml      # STM, reconstruction
python -m ters.train_clip --config configs/ters_clip.yaml   # TERS, contrastive
python -m ters.train      --config configs/ters_ae.yaml     # TERS, reconstruction
```

Configuration files inherit from `configs/base.yaml`, and any value can be overridden with
`--set section.key=value`. Adding `--synthetic` runs on a small generated dataset for testing.

## Acknowledgements

- `torchmdnet/` is taken without modification from
  [shehzaidi/pre-training-via-denoising](https://github.com/shehzaidi/pre-training-via-denoising)
  (commit `2d81667`, MIT License).
- The contrastive loss follows MolSpectra (Wang et al., 2025).
- SpectralGPT pretrained weights are supported; no SpectralGPT source code is included.

## References

- Wang et al., *MolSpectra: Pre-training 3D Molecular Representation with Multi-modal Energy
  Spectra*, ICLR 2025. [arXiv:2502.16284](https://arxiv.org/abs/2502.16284)
- Kurki, Oinonen & Foster, *Automated Structure Discovery for Scanning Tunneling Microscopy*,
  ACS Nano 2024. [arXiv:2312.08854](https://arxiv.org/abs/2312.08854)
- Sethi et al., *Automated Structure Discovery for Tip Enhanced Raman Spectroscopy*, 2026.
  [arXiv:2602.19932](https://arxiv.org/abs/2602.19932)
- Zaidi et al., *Pre-training via Denoising for Molecular Property Prediction*, ICLR 2023.
  [arXiv:2206.00133](https://arxiv.org/abs/2206.00133)
