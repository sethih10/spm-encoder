"""STM z-stack -> one 256-d vector, with 3D convolutions over the height axis."""

import torch.nn as nn

from shared.models.blocks import ResBlock
from shared.models.blocks3d import ResBlock3d


class StmCubeEncoder(nn.Module):
    """(B, Z, H, W) -> (B, latent_dim), via a 3-D stem over the z axis then a 2-D ladder.

    The stem is what distinguishes this from the TERS encoder: z-slices are an ordered axis, so a
    3-D convolution has the right inductive bias where a 1x1 channel mix does not.

    Parameters
    ----------
    conv3d_channels, z_strides : tuple of int
        Widths and z-axis strides of the 3-D stem, one entry per stage. Gated so a 1-slice input
        does not reach a pooling layer with no depth left.
    filters : tuple of int
        Widths of the 2-D ResBlock ladder; each entry halves H and W.
    pool_size : int
        Spatial grid kept before flattening. 1 is a global average pool.
    """

    def __init__(self, in_channels=14, conv3d_channels=(8, 16, 32), z_strides=(2, 2, 1),
                 filters=(64, 128, 256), kernel_size=3, latent_dim=256, n_groups=8,
                 pool_size=1):
        super().__init__()
        if len(conv3d_channels) != len(z_strides):
            raise ValueError(f'conv3d_channels and z_strides must be the same length, got '
                             f'{len(conv3d_channels)} and {len(z_strides)}')
        if pool_size < 1:
            raise ValueError(f'pool_size must be >= 1, got {pool_size}')

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.output_dim = latent_dim
        self.pool_size = pool_size
        self.z_strides = tuple(int(z) for z in z_strides)

        # --- 3D stem: convolves along z ---
        self.stem3d = nn.ModuleList()
        self.pool3d = nn.ModuleList()
        c_in = 1
        z = in_channels
        for c_out, zs in zip(conv3d_channels, z_strides):
            self.stem3d.append(ResBlock3d(c_in, c_out, kernel_size, n_groups=n_groups))
            self.pool3d.append(nn.MaxPool3d((zs, 2, 2)))
            c_in = c_out
            z = max(1, z // zs)
        self.z_out = z
        self.c3d_out = c_in

        # --- collapse z into channels, then the 2D ladder ---
        # Reshape rather than mean-pool: a mean over z would throw away the envelope the 3D
        # stem just spent three stages reading.  The 1x1 conv that follows is then free to
        # weight the surviving slices, which is a different operation from averaging them.
        self.to_2d = nn.Conv2d(self.c3d_out * self.z_out, filters[0], kernel_size=1)
        widths = [filters[0]] + list(filters)
        self.blocks = nn.ModuleList([
            ResBlock(widths[i], widths[i + 1], kernel_size, n_groups=n_groups)
            for i in range(len(filters))
        ])
        self.pool = nn.MaxPool2d(2, 2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(),
            nn.Linear(filters[-1] * pool_size * pool_size, latent_dim),
        )
        # The contrastive loss is cosine, so an unnormalised code would let the model reduce
        # the loss by moving magnitude rather than direction.
        self.out_norm = nn.LayerNorm(latent_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f'StmCubeEncoder expects (B, Z, H, W), got {tuple(x.shape)}')
        if x.size(1) != self.in_channels:
            raise ValueError(f'StmCubeEncoder built for in_channels={self.in_channels} '
                             f'z-slices but got {x.size(1)}')
        b = x.shape[0]
        x = x.unsqueeze(1)                                   # (B, 1, Z, H, W)
        for block, pool, zs in zip(self.stem3d, self.pool3d, self.z_strides):
            x = block(x)
            # Gate on the z axis too, not just H and W.  A 1-slice input -- which is what
            # StmPairDataset(z_slice=...) emits, ASD-STM's protocol -- would otherwise reach
            # MaxPool3d((2,2,2)) on a depth-1 volume and raise "Output size is too small".
            if x.shape[-1] >= 2 and x.shape[-2] >= 2 and x.shape[2] >= zs:
                x = pool(x)
        x = x.reshape(b, x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        x = self.to_2d(x)
        for block in self.blocks:
            x = block(x)
            if min(x.shape[-2:]) >= 2:
                x = self.pool(x)
        return self.out_norm(self.head(x))
