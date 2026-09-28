"""256-d vector -> reconstructed cube.

Sources:
  * zero-init of the residual branch -- pytorch/vision @ ac8d215 resnet.py,
    https://github.com/pytorch/vision ; openai/guided-diffusion @ 22e0df8 unet.py,
    https://github.com/openai/guided-diffusion ; the argument is Goyal et al.,
    https://arxiv.org/abs/1706.02677
"""

from torch import nn

from .blocks import ResBlock, Upsample

class TersDecoder(nn.Module):
    """``(B, latent_dim) -> (B, out_channels, H, W)``.

    The vector is projected to a ``base_hw`` grid, then a ladder of ResBlock+Upsample stages
    doubles it once per entry in ``filters`` until it reaches the cube's size. ``base_hw`` must
    be what the encoder's ladder finished on, or the reconstruction comes out at the wrong
    resolution.

    ``zero_init`` zeroes the output conv and the residual branch, so the first output is zero
    and the initial loss equals the loss of predicting zero everywhere -- the reference
    ``shared.metrics.reconstruction.baseline_losses`` reports as ``zero``.
    """

    def __init__(self, out_channels=100, filters=(64, 128, 256, 512),
                 kernel_size=(3, 3, 3, 3), latent_dim=256, base_hw=16, n_groups=8,
                 zero_init=True, latent_hw=1, pad_blocks=0, stem_channels=None):
        super().__init__()
        if len(kernel_size) != len(filters):
            raise ValueError(
                f'filters and kernel_size must be the same length, '
                f'got {len(filters)} and {len(kernel_size)}')
        if base_hw < 1:
            raise ValueError(
                f'base_hw={base_hw} < 1: the encoder ladder pooled the image away entirely. '
                f'image_size must be at least 2**len(filters) = {2 ** len(filters)}.')
        if latent_hw < 1 or latent_hw > base_hw or base_hw % latent_hw != 0:
            legal = [k for k in range(1, base_hw + 1) if base_hw % k == 0]
            raise ValueError(
                f'latent_hw={latent_hw} cannot be expanded to the ladder base grid '
                f'{base_hw}x{base_hw}: it must divide base_hw so the pixel-shuffle factor '
                f'base_hw//latent_hw is exact. Legal latent_hw here: {legal}.')
        if latent_dim % (latent_hw ** 2) != 0:
            raise ValueError(
                f'latent_hw={latent_hw} does not divide the budget latent_dim={latent_dim}; '
                f'see TersSpatialEncoder.__init__ for the legal values.')
        if pad_blocks < 0:
            raise ValueError(f'pad_blocks must be >= 0, got {pad_blocks}')

        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.base_hw = base_hw
        self.base_ch = filters[-1]
        self.zero_init = zero_init
        self.latent_hw = latent_hw
        self.latent_ch = latent_dim // (latent_hw ** 2)
        self.shuffle_r = base_hw // latent_hw

        if latent_hw == 1:
            self.fc = nn.Linear(latent_dim, self.base_ch * base_hw * base_hw)
            self.expand = None
        else:
            self.fc = None
            self.expand = nn.Sequential(
                nn.Conv2d(self.latent_ch, self.base_ch * self.shuffle_r ** 2, kernel_size=1),
                nn.PixelShuffle(self.shuffle_r),
            )

        self.pad = nn.ModuleList([
            ResBlock(self.base_ch, self.base_ch, kernel_size[-1], n_groups=n_groups)
            for _ in range(pad_blocks)
        ])

        stem_ch = filters[0] if stem_channels is None else int(stem_channels)
        self.stem_channels = stem_ch
        widths = list(reversed(filters)) + [stem_ch]         # e.g. 512,256,128,64,64
        kernels = list(reversed(kernel_size))
        self.ups = nn.ModuleList([Upsample(widths[i]) for i in range(len(filters))])
        self.blocks = nn.ModuleList([
            ResBlock(widths[i], widths[i + 1], kernels[i], n_groups=n_groups)
            for i in range(len(filters))
        ])

        self.conv_out = nn.Conv2d(stem_ch, out_channels, kernel_size=1)

        self.reset_parameters()

    def reset_parameters(self):
        """Kaiming/relu everywhere, then zero-init the output conv and the residual branch.

        Applied here only. With both zero-inits the decoder's first output is exactly zero.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        if not self.zero_init:
            return

        # --- ADAPT: pytorch/vision resnet.py, `zero_init_residual` ---
        # Zero the last norm in each residual branch so the block starts as an identity and
        # residual variance stops compounding. torchvision zeroes `bn2` in a BasicBlock; this
        # one is `norm2`, the GroupNorm at the end of ResBlock's branch. Same position, same
        # effect.
        for block in list(self.blocks) + list(self.pad):
            nn.init.zeros_(block.norm2.weight)

        # --- ADAPT: openai/guided-diffusion unet.py, `zero_module` on the final output conv ---
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, z):
        if z.dim() != 2:
            raise ValueError(
                f'TersDecoder expects (B, latent_dim), got {tuple(z.shape)}.')
        if z.size(1) != self.latent_dim:
            raise ValueError(
                f'TersDecoder built for latent_dim={self.latent_dim} but got {z.size(1)}.')

        if self.fc is not None:
            h = self.fc(z).view(-1, self.base_ch, self.base_hw, self.base_hw)
        else:
            # The code arrives flat, (B, latent_dim), from either encoder.  C-order here is
            # the inverse of nn.Flatten in TersSpatialEncoder.head.
            h = z.view(-1, self.latent_ch, self.latent_hw, self.latent_hw)
            h = self.expand(h)
        for block in self.pad:
            h = block(h)
        for up, block in zip(self.ups, self.blocks):
            h = up(h)
            h = block(h)
        return self.conv_out(h)
