"""STM z-stack -> a 256-d vector -> the z-stack again.

The reconstruction counterpart to the contrastive arm: same cube encoder, same latent budget,
but scored on whether the code rebuilds the stack rather than on whether it identifies the
molecule. Running both against one encoder is what makes the two numbers comparable.

The decoder is reused unchanged: it maps a vector to a (C, H, W) stack and does not care
whether C counts spectral channels or z-slices.
"""

import torch.nn as nn

from shared.models.decoder import TersDecoder

from .conv3d import StmCubeEncoder


class StmAutoencoder(nn.Module):
    """``(B, Z, H, W) -> (B, latent_dim) -> (B, Z, H, W)``."""

    def __init__(self, in_channels=14, conv3d_channels=(8, 16, 32), z_strides=(2, 2, 1),
                 filters=(64, 128, 256), kernel_size=3, latent_dim=256, image_size=128,
                 n_groups=8, zero_init=True):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.image_size = image_size

        self.encoder = StmCubeEncoder(
            in_channels=in_channels, conv3d_channels=tuple(conv3d_channels),
            z_strides=tuple(z_strides), filters=tuple(filters), kernel_size=kernel_size,
            latent_dim=latent_dim, n_groups=n_groups, pool_size=1)

        # The ladder halves H and W once per entry in `filters`, so the decoder must start
        # where the encoder finished or the reconstruction comes out at the wrong resolution.
        base_hw, rungs = image_size, len(filters)
        for _ in range(rungs):
            base_hw //= 2
        if base_hw * 2 ** rungs != image_size:
            raise ValueError(
                f'image_size={image_size} is not divisible by 2**{rungs}={2 ** rungs}; the '
                f'encoder ladder would end on a non-integer grid. Set data.image_size to the '
                f'real cube size, or drop a level from model.filters.')

        self.decoder = TersDecoder(
            out_channels=in_channels, filters=tuple(filters),
            kernel_size=(kernel_size,) * rungs, latent_dim=latent_dim, base_hw=base_hw,
            n_groups=n_groups, zero_init=zero_init)

    @classmethod
    def from_config(cls, config):
        m, d = config.model, config.data
        return cls(in_channels=m.in_channels,
                   conv3d_channels=tuple(getattr(m, 'stem_channels', None) or (8, 16, 32)),
                   filters=tuple(getattr(m, 'filters', None) or (64, 128, 256)),
                   latent_dim=m.latent_dim, image_size=d.image_size,
                   n_groups=getattr(m, 'n_groups', 8),
                   zero_init=getattr(m, 'zero_init', True))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        if recon.shape != x.shape:
            raise ValueError(
                f'reconstruction {tuple(recon.shape)} does not match input {tuple(x.shape)}. '
                f'The decoder is fixed to {self.image_size}x{self.image_size} by '
                f'data.image_size; this batch is {x.shape[-2]}x{x.shape[-1]}.')
        return recon, z            # (recon, latent), the pair the loop unpacks
