"""STM cube normalisation, and the z-slice draw."""

import torch


def _slice_for(row, epoch, n_z, seed=0):
    """Which z-slice row ``row`` gets in epoch ``epoch``, deterministically.

    Hashed from (row, epoch, seed) rather than drawn from a global RNG, so a DataLoader worker
    needs no shared state and the draw is reproducible under a seeded run.
    """
    x = (int(row) * 0x9E3779B97F4A7C15 + int(epoch) * 0xBF58476D1CE4E5B9
         + int(seed) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    return int(x % int(n_z))


class NormalizeStmCube:
    """Standardise a z-stack over every axis at once.

    Whole-cube, not per-slice: the z decay envelope is real physical structure, and
    standardising each slice separately would flatten it to unit variance. The TERS normaliser
    is per-channel instead, because spectral bins are unordered and carry no such envelope.
    """

    def __init__(self, uint16_max=65535.0, asinh_scale=0.01, eps=1e-6):
        self.uint16_max = uint16_max
        self.asinh_scale = asinh_scale
        self.eps = eps

    def __call__(self, cube):
        x = cube.float()
        if self.uint16_max and self.uint16_max != 1.0:
            x = x / self.uint16_max
        if self.asinh_scale:
            x = torch.asinh(x / self.asinh_scale)
        mean = x.mean()
        std = x.std()
        if std < self.eps:
            return x - mean
        return (x - mean) / std
