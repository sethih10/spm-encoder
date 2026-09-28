"""Encoder + decoder, and the one place their shapes are checked against each other."""

from torch import nn

from shared.models.decoder import TersDecoder
from .encoder import TersEncoder, TersSpatialEncoder


class TersAutoencoder(nn.Module):
    """``(B, C, H, W) -> (B, latent_dim) -> (B, C, H, W)``.

    ``self.encoder`` stays byte-identical to the encoder ters-2-graph builds as its
    conditioning encoder, so ``self.encoder.state_dict()`` loads there unprefixed.
    """

    def __init__(self, in_channels=100, filters=(64, 128, 256, 512),
                 kernel_size=(3, 3, 3, 3), latent_dim=256, image_size=256, n_groups=8,
                 zero_init=True, latent_hw=1, decoder_pad_blocks=0, stem_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.latent_hw = latent_hw

        # --- NEW (sweep): k=1 is the copy, and it is dispatched to explicitly ---
        # TersSpatialEncoder is numerically equivalent at latent_hw=1 but has different
        # state_dict keys, and the ters-2-graph export contract only loads against the real
        # class. A run that sets stem_channels takes TersSpatialEncoder even at k=1, since the
        # copy has no such parameter.
        if latent_hw == 1 and stem_channels is None:
            self.encoder = TersEncoder(
                in_channels=in_channels, filters=filters, kernel_size=kernel_size,
                latent_dim=latent_dim, n_groups=n_groups)
        else:
            self.encoder = TersSpatialEncoder(
                in_channels=in_channels, filters=filters, kernel_size=kernel_size,
                latent_dim=latent_dim, latent_hw=latent_hw, n_groups=n_groups,
                stem_channels=stem_channels)

        # Ask the encoder where its ladder ends rather than recomputing the pooling rule here.
        base_hw = self.encoder.bottleneck_hw(image_size)
        if base_hw * 2 ** len(filters) != image_size:
            # Fail here, in the constructor, rather than on a shape mismatch inside the loss
            # 40 minutes into a job.
            raise ValueError(
                f'image_size={image_size} is not divisible by 2**{len(filters)}='
                f'{2 ** len(filters)}; the encoder ladder would end on {base_hw}x{base_hw} '
                f'and the decoder would reconstruct '
                f'{base_hw * 2 ** len(filters)}x{base_hw * 2 ** len(filters)} instead. '
                f'Set config.data.image_size to the real cube size, or drop a level from '
                f'config.model.filters.')

        self.decoder = TersDecoder(
            out_channels=in_channels, filters=filters, kernel_size=kernel_size,
            latent_dim=latent_dim, base_hw=base_hw, n_groups=n_groups,
            zero_init=zero_init, latent_hw=latent_hw, pad_blocks=decoder_pad_blocks,
            stem_channels=stem_channels)

    @classmethod
    def from_config(cls, config):
        return cls(in_channels=config.model.in_channels,
                   filters=tuple(config.model.filters),
                   kernel_size=tuple(config.model.kernel_size),
                   latent_dim=config.model.latent_dim,
                   image_size=config.data.image_size,
                   n_groups=config.model.n_groups,
                   zero_init=config.model.zero_init,
                   latent_hw=getattr(config.model, 'latent_hw', 1),
                   decoder_pad_blocks=getattr(config.model, 'decoder_pad_blocks', 0),
                   stem_channels=getattr(config.model, 'stem_channels', None))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        if recon.shape != x.shape:
            # Reachable only for a non-square or wrong-sized batch, since the constructor
            # already checked the configured size. Raised explicitly, because torch would
            # otherwise broadcast some mismatches into a silently meaningless loss.
            raise ValueError(
                f'reconstruction {tuple(recon.shape)} does not match input {tuple(x.shape)}. '
                f'The decoder is fixed to {self.image_size}x{self.image_size} by '
                f'config.data.image_size; this batch is '
                f'{x.shape[-2]}x{x.shape[-1]}.')
        return recon, z
