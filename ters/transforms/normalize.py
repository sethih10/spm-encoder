"""TERS cube normalisation.

Not interchangeable with the STM normaliser. This standardises per channel, because TERS
channels are unordered spectral bins of which a large fraction are dead -- the zero-std guard
exists for those. Standardising over the whole cube instead would let one live channel set the
scale for all of them.
"""

import torch


class NormalizeVectorized:
    """Per-channel standardisation over the spatial axes of a ``(C, H, W)`` cube.

    Not optional: raw cube values run 0 to ~840, and a network initialised for unit-scale
    inputs diverges to NaN on them.
    """

    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, cube):
        mean = cube.mean(dim=(1, 2), keepdim=True)
        std = cube.std(dim=(1, 2), keepdim=True)
        # Zero-std guard. 60-80% of TERS channels are dead, and dividing by a near-zero std
        # amplifies float32 cancellation noise (a constant channel's mean is only exact to
        # ~1e-9) into O(0.1) garbage. Substituting 1.0 leaves such a channel at ~0 instead.
        std = torch.where(std < self.eps, torch.ones_like(std), std)
        return (cube - mean) / std
