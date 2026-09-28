"""TERS cube -> a single 256-d vector.

``TersEncoder`` is a copy of the conditioning encoder in ters-2-graph, this group's diffusion
repo, and must stay one. The premise of this project is that the latent is the same vector the
diffusion model is conditioned on: ters-2-graph builds this class at ``output_dim=256`` and
feeds its output to ``cond_lin``. Change a layer here and the reconstruction number stops
being a statement about that vector.

Deviations from the original, none of which move a tensor shape or a parameter count:
  * ``ResBlock`` is imported from ``shared.models.blocks`` rather than defined here, because
    the decoder needs it too;
  * ``latent_dim`` replaces the parameter name ``output_dim``, since here it is a bottleneck
    rather than an output. ``output_dim`` is accepted as an alias and ``self.output_dim`` is
    kept, so a config or a state_dict written against ters-2-graph still means what it says;
  * ``self.filters`` is stored and ``bottleneck_hw()`` added, for the decoder to size itself
    against;
  * ``pool_size`` carries a longer explanation, the original having grown the parameter without
    one.

``reset_parameters``, the ``out_norm`` LayerNorm and every layer are unchanged. ``out_norm``
and ``reset_parameters`` mirror the interface ters-2-graph's SpecFormer exposes, so either
encoder can be dropped into the same slot.

.. warning::
   The trunk derives from the AttentionUNet of the Multimodal-ters work, keeping the 1x1 stem
   and the ResBlock+MaxPool ladder and dropping the decoder ladder, the attention gates, the
   final resblock and the output conv. That repo was unavailable when the trunk was written, so
   the correspondence was reconstructed from a written specification rather than from source:
   shapes and training behaviour are verified, exact layer-for-layer correspondence is not.
"""

from torch import nn

from shared.models.blocks import ResBlock


class TersEncoder(nn.Module):
    """TERS cube -> a single conditioning vector.

    Parameters
    ----------
    in_channels : int
        Spectral channels in the cube. Comes from ``TersCubeDataset.in_channels``; with the
        ``channels_100`` key that is 100.
    latent_dim : int
        Width of the bottleneck. 256 to match ``config.model.nf`` in ters-2-graph.
    pool_size : int
        Spatial grid kept by the head before flattening. ``1`` is the original global average
        pool. ``k > 1`` keeps a ``k x k`` map, so coarse spatial layout survives into the
        conditioning vector instead of being averaged away. Output width is unchanged either
        way; only the head's Linear grows, by ``filters[-1] * (pool_size^2 - 1) * latent_dim``
        parameters.

        Exists to keep the copy a copy: ters-2-graph grew the same parameter and its config
        sets it to 4, making its head ``Linear(512*16 = 8192, 256)``. An encoder exported from
        here has to be built at the pool_size that repo uses, or the head weight will not load
        into it.
    """

    def __init__(self, in_channels=100, filters=(64, 128, 256, 512),
                 kernel_size=(3, 3, 3, 3), latent_dim=256, n_groups=8, output_dim=None,
                 pool_size=1):
        super().__init__()
        if output_dim is not None:                  # ters-2-graph spelling; same meaning
            latent_dim = output_dim
        if len(kernel_size) != len(filters):
            raise ValueError(
                f'filters and kernel_size must be the same length, '
                f'got {len(filters)} and {len(kernel_size)}')

        if pool_size < 1:
            raise ValueError(f'pool_size must be >= 1, got {pool_size}')

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.output_dim = latent_dim                # alias, for state_dict-level parity
        self.filters = tuple(filters)
        self.pool_size = pool_size

        # 1x1 stem: mixes the spectral channels at full spatial resolution before any
        # spatial convolution.  With 100 input channels this is where most of the input
        # dimensionality is absorbed.
        self.stem = nn.Conv2d(in_channels, filters[0], kernel_size=1)

        widths = [filters[0]] + list(filters)
        self.blocks = nn.ModuleList([
            ResBlock(widths[i], widths[i + 1], kernel_size[i], n_groups=n_groups)
            for i in range(len(filters))
        ])
        self.pool = nn.MaxPool2d(2, 2)

        # AdaptiveAvgPool2d fixes the output grid, so any H,W still works and the cube size
        # need not be known -- that property is what pool_size inherits.  At pool_size=1 this
        # is the original global average pool.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(),
            nn.Linear(filters[-1] * pool_size * pool_size, latent_dim),
        )
        # Mirrors SpecFormer's out_norm. Keep it: ters-2-graph's cond_lin was initialised for a
        # LayerNorm-scaled input. It also makes the 256 numbers the decoder sees exactly the ones
        # the denoiser sees, so no magnitude leaks into reconstruction.
        self.out_norm = nn.LayerNorm(latent_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Matches the interface ters-2-graph's SpecFormer exposes."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # --- NEW (mirror): tell the decoder where the ladder ends ---
    # The decoder has to start from a spatial grid, and it must be the one this ladder finishes
    # on, or the reconstruction comes out at the wrong resolution. Deriving it in the decoder
    # from `image_size // 2 ** len(filters)` would duplicate the pooling rule in a second place,
    # and the rule is not quite `//` because of the small-input guard in `forward`. Asking the
    # encoder cannot drift. Rejected: running a dummy tensor through and reading the shape,
    # which needs an allocation and a device at construction time.
    def bottleneck_hw(self, image_size):
        """Spatial size entering ``head`` for a square ``image_size`` input."""
        hw = image_size
        for _ in self.blocks:
            if hw >= 2:
                hw //= 2
        return hw

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                f'TersEncoder expects (B, C, H, W), got {tuple(x.shape)}. The cube is '
                f'permuted to channels-first in TersCubeDataset.__getitem__.')
        if x.size(1) != self.in_channels:
            raise ValueError(
                f'TersEncoder built for in_channels={self.in_channels} but got '
                f'{x.size(1)}. Set config.model.in_channels to match '
                f'TersCubeDataset.in_channels.')

        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
            # Four unconditional 2x2 pools would take anything below 16x16 down to a 0-sized
            # dimension.  Stop pooling once there is nothing left to halve; AdaptiveAvgPool2d
            # in the head handles whatever spatial size survives.
            if min(x.shape[-2:]) >= 2:
                x = self.pool(x)
        return self.out_norm(self.head(x))


# --- NEW (sweep): the same 256 numbers, arranged on a grid ---
class TersSpatialEncoder(nn.Module):
    """TERS cube -> ``latent_dim`` numbers laid out as ``latent_hw x latent_hw``.

    The geometry arm: same trunk and same number budget as :class:`TersEncoder`, but the head
    keeps a spatial grid instead of collapsing it, which measures what that collapse costs. It is
    not the class ters-2-graph builds, so no result here is a statement about DiffSpectra's
    conditioning vector.

    ``latent_ch * latent_hw**2 == latent_dim`` is enforced in ``__init__``, so both encoders emit
    exactly ``latent_dim`` numbers. ``latent_hw=1`` is numerically equivalent to TersEncoder but
    has different state_dict keys, so ``TersAutoencoder`` dispatches k=1 to the copy, never here.

    Source: the spatial head is an ADAPT of CompVis/latent-diffusion @ a506df5
    ldm/modules/diffusionmodules/model.py:485-492, plus an ``AdaptiveAvgPool2d(latent_hw)`` so the
    grid is a configured size.

    The sweep this arm belongs to is laid out in docs/03-latent-dimension.md.
    """

    def __init__(self, in_channels=100, filters=(64, 128, 256, 512),
                 kernel_size=(3, 3, 3, 3), latent_dim=256, latent_hw=4, n_groups=8,
                 stem_channels=None):
        super().__init__()
        if len(kernel_size) != len(filters):
            raise ValueError(
                f'filters and kernel_size must be the same length, '
                f'got {len(filters)} and {len(kernel_size)}')
        if latent_hw < 1:
            raise ValueError(f'latent_hw must be >= 1, got {latent_hw}')
        if latent_dim % (latent_hw ** 2) != 0:
            # Fail in the constructor with the legal values in the message, rather than on a
            # reshape 40 minutes into a job.
            legal = [k for k in range(1, latent_dim + 1) if latent_dim % (k * k) == 0]
            raise ValueError(
                f'latent_hw={latent_hw} does not divide the budget: latent_dim={latent_dim} '
                f'is not a multiple of {latent_hw}**2={latent_hw ** 2}, so the code could not '
                f'be exactly {latent_dim} numbers. Legal latent_hw for this latent_dim: '
                f'{legal}.')

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.output_dim = latent_dim
        self.latent_hw = latent_hw
        self.latent_ch = latent_dim // (latent_hw ** 2)
        self.filters = tuple(filters)

        # --- NEW (sweep): the stem width, decoupled from filters[0] ---
        # TersEncoder's 1x1 stem compresses 100 spectral channels to filters[0]=64 at full
        # spatial resolution, before any spatial conv runs -- the first place spectral
        # information can be destroyed. `stem_channels >= in_channels` makes it non-compressive
        # so that can be tested. None = filters[0], i.e. unchanged.
        self.stem_channels = filters[0] if stem_channels is None else int(stem_channels)
        self.stem = nn.Conv2d(in_channels, self.stem_channels, kernel_size=1)
        widths = [self.stem_channels] + list(filters)
        self.blocks = nn.ModuleList([
            ResBlock(widths[i], widths[i + 1], kernel_size[i], n_groups=n_groups)
            for i in range(len(filters))
        ])
        self.pool = nn.MaxPool2d(2, 2)

        # --- ADAPT: Encoder.conv_out, with the grid size made explicit ---
        # AdaptiveAvgPool2d(k) rather than a strided conv so that k == the ladder's own final
        # grid is a true no-op and k=1 is exactly TersEncoder's global average. A strided conv
        # would put learnable parameters in the resampling and make the two ends of the sweep
        # incomparable to the middle.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(latent_hw),
            nn.Conv2d(filters[-1], self.latent_ch, kernel_size=1),
            nn.Flatten(),
        )
        # Over all latent_dim numbers, not per position: TersEncoder's LayerNorm removes the
        # mean and scale of the whole code, and a per-position norm would remove strictly more
        # (the relative magnitude between grid positions), which is information the global
        # vector never had to give up. Normalising the same quantity keeps the comparison fair.
        self.out_norm = nn.LayerNorm(latent_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Identical to :meth:`TersEncoder.reset_parameters`."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def bottleneck_hw(self, image_size):
        """Spatial size entering ``head``; identical rule to :meth:`TersEncoder.bottleneck_hw`."""
        hw = image_size
        for _ in self.blocks:
            if hw >= 2:
                hw //= 2
        return hw

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                f'TersSpatialEncoder expects (B, C, H, W), got {tuple(x.shape)}. The cube is '
                f'permuted to channels-first in TersCubeDataset.__getitem__.')
        if x.size(1) != self.in_channels:
            raise ValueError(
                f'TersSpatialEncoder built for in_channels={self.in_channels} but got '
                f'{x.size(1)}. Set config.model.in_channels to match '
                f'TersCubeDataset.in_channels.')

        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
            if min(x.shape[-2:]) >= 2:
                x = self.pool(x)
        # (B, latent_ch, k, k) -> flattened in C-order by nn.Flatten, which is the order
        # TersDecoder's view() undoes.  The code leaves here as (B, latent_dim) so that both
        # encoders present the identical interface and "the latent is 256 numbers" stays
        # literally true at the boundary.
        return self.out_norm(self.head(x))
