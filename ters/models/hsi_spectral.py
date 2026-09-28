"""Per-pixel spectral autoencoder: compress the spectrum, leave the image alone.

Sources:
  * ``SpectralAutoencoder`` -- ADAPT of the 1D-Convolutional Autoencoder of Kuester,
    Gross & Middelmann, ISPRS Archives XLIII-B1-2021:15-21,
    https://doi.org/10.5194/isprs-archives-XLIII-B1-2021-15-2021
  * zero-init of the output layer -- openai/guided-diffusion @ 22e0df8 nn.py,
    https://github.com/openai/guided-diffusion
"""

from torch import nn
from torch.nn import functional as F


def _strided_lengths(length, n_stages, kernel, stride=2):
    """Spectral lengths after each strided stage, front to back.

    Computed rather than assumed because the decoder has to hit them exactly on the way back
    up -- 100 -> 50 -> 25 -> 13 is not a power of two, and an off-by-one at the top of the
    ladder would silently reconstruct 96 or 104 channels and be caught only by the loss's
    shape check.  ``F.interpolate(size=...)`` in the decoder consumes this list.
    """
    padding = kernel // 2
    lengths = [length]
    for _ in range(n_stages):
        length = (length + 2 * padding - kernel) // stride + 1
        if length < 1:
            raise ValueError(
                f'the spectral ladder collapses below 1 band after {len(lengths)} stages; '
                f'in_channels={lengths[0]} is too few for n_stages={n_stages}. Use fewer '
                f'stages.')
        lengths.append(length)
    return lengths


class SpectralAutoencoder(nn.Module):
    """Per-pixel 1-D spectral autoencoder: every pixel's spectrum compressed independently.

    No spatial mixing anywhere, which is the point -- it isolates how much of a TERS cube is
    spectral redundancy.

    Parameters
    ----------
    in_channels : int
        Spectral channels per pixel.
    latent_per_pixel : int
        Code width per pixel. The compression ratio is ``in_channels / latent_per_pixel``.
    base_width : int
        A memory ceiling rather than a taste choice: 32 does not fit the training batch, and
        lowering the batch would break comparability with the other sweeps.
    n_stages : int
        Encoder stages; each halves the spectral axis.
    """

    def __init__(self, in_channels=100, latent_per_pixel=1, base_width=16, n_stages=3,
                 kernel_size=5, n_groups=8, zero_init=True):
        super().__init__()
        if latent_per_pixel < 1:
            raise ValueError(f'latent_per_pixel must be >= 1, got {latent_per_pixel}')
        if latent_per_pixel >= in_channels:
            # Not a bottleneck at all -- the identity would be optimal.  Fail here rather than
            # produce an excellent-looking SNR that means nothing.
            raise ValueError(
                f'latent_per_pixel={latent_per_pixel} >= in_channels={in_channels}: that is '
                f'not a compression, the identity map would solve it. Use a value below '
                f'{in_channels}; the arms this repo runs are 1, 2 and 4 '
                f'(c_R = {in_channels}, {in_channels / 2:.0f}, {in_channels / 4:.0f}).')
        if kernel_size % 2 == 0:
            raise ValueError(f'kernel_size must be odd for "same" padding, got {kernel_size}')

        self.in_channels = in_channels
        self.latent_per_pixel = latent_per_pixel
        self.base_width = base_width
        self.n_stages = n_stages

        self.lengths = _strided_lengths(in_channels, n_stages, kernel_size)
        widths = [base_width * 2 ** i for i in range(n_stages)]
        self.widths = widths
        pad = kernel_size // 2

        # --- encoder: strided 1D convolutions down the band axis ---
        enc = []
        c_in = 1
        for c_out in widths:
            enc.append(nn.Conv1d(c_in, c_out, kernel_size, stride=2, padding=pad))
            enc.append(nn.GroupNorm(n_groups, c_out))
            enc.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.enc = nn.Sequential(*enc)

        flat = widths[-1] * self.lengths[-1]
        self.to_latent = nn.Linear(flat, latent_per_pixel)
        self.from_latent = nn.Linear(latent_per_pixel, flat)

        # No LayerNorm over the code, unlike TersEncoder.out_norm. That norm exists there because
        # ters-2-graph's `cond_lin` was initialised for a LayerNorm-scaled input; there is no such
        # consumer here, and at latent_per_pixel=1 a LayerNorm over one number maps every pixel's
        # code to exactly 0 and the arm would measure nothing. Stated rather than left implicit,
        # because copying the norm across would look like consistency and would silently destroy
        # the L=1 arm.

        # --- decoder: interpolate to the recorded length, then convolve ---
        # Upsample-then-convolve, not ConvTranspose1d (Odena, Dumoulin & Olah, Distill 2016):
        # transposed convolution at stride 2 leaves periodic artefacts, and here they would sit
        # at the spacing of the vibrational peaks, which is the signal. `size=` rather than
        # `scale_factor=` because the ladder 13 -> 25 -> 50 -> 100 is not a power of two.
        dec = []
        rev = list(reversed(widths))
        for i, c_out in enumerate(rev[1:] + [base_width]):
            dec.append(nn.Conv1d(rev[i], c_out, kernel_size, padding=pad))
            dec.append(nn.GroupNorm(n_groups, c_out))
            dec.append(nn.ReLU(inplace=True))
        self.dec = nn.ModuleList(dec)
        self.conv_out = nn.Conv1d(base_width, 1, kernel_size, padding=pad)

        self.reset_parameters(zero_init=zero_init)

    def reset_parameters(self, zero_init=True):
        """Kaiming as in TersEncoder, plus the zero-init of the output convolution.

        The zero-init is the one that matters. With it the model starts exactly at the
        do-nothing baseline, so this arm's curve is read on the same axis as every other run
        here; without it the initial MSE opens far above that.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init:
            # --- ADAPT: guided-diffusion's zero_module ---
            nn.init.zeros_(self.conv_out.weight)
            nn.init.zeros_(self.conv_out.bias)

    def _check(self, x):
        if x.dim() != 4:
            raise ValueError(
                f'SpectralAutoencoder expects (B, C, H, W), got {tuple(x.shape)}. The cube is '
                f'permuted to channels-first in TersCubeDataset.__getitem__.')
        if x.size(1) != self.in_channels:
            raise ValueError(
                f'SpectralAutoencoder built for in_channels={self.in_channels} but got '
                f'{x.size(1)}. With configs/channel_mask_80.json the cube has 80 bands, not '
                f'100 -- pass --in-channels to match the mask.')

    def encode(self, x):
        """``(B, C, H, W) -> (B, latent_per_pixel, H, W)``."""
        self._check(x)
        b, c, h, w = x.shape
        # Every pixel is an independent example, which is exactly the ISPRS 2021 formulation:
        # "individual spectral signatures ... are supplied to the 1D-CAE". The permute puts the
        # band axis last so the reshape does not interleave pixels.
        s = x.permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
        s = self.enc(s)
        z = self.to_latent(s.flatten(1))
        return z.view(b, h, w, self.latent_per_pixel).permute(0, 3, 1, 2).contiguous()

    def decode(self, z):
        """``(B, latent_per_pixel, H, W) -> (B, C, H, W)``."""
        b, l, h, w = z.shape
        if l != self.latent_per_pixel:
            raise ValueError(
                f'latent has {l} channels, model was built for {self.latent_per_pixel}')
        s = z.permute(0, 2, 3, 1).reshape(b * h * w, l)
        s = self.from_latent(s).view(-1, self.widths[-1], self.lengths[-1])

        # Walk the recorded lengths back up. self.dec is [conv, norm, act] x n_stages.
        for i in range(self.n_stages):
            s = F.interpolate(s, size=self.lengths[self.n_stages - 1 - i], mode='nearest')
            for m in self.dec[3 * i:3 * i + 3]:
                s = m(s)
        s = self.conv_out(s)
        return s.view(b, h, w, self.in_channels).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        if recon.shape != x.shape:
            raise ValueError(
                f'reconstruction {tuple(recon.shape)} does not match input {tuple(x.shape)}')
        return recon, z

    def latent_numbers(self, image_size):
        """Total numbers in the code for one ``image_size`` x ``image_size`` cube."""
        return self.latent_per_pixel * image_size * image_size

    def describe(self, image_size=256):
        n = sum(p.numel() for p in self.parameters())
        lat = self.latent_numbers(image_size)
        total = self.in_channels * image_size * image_size
        return (f'SpectralAutoencoder(L={self.latent_per_pixel}, width={self.base_width}, '
                f'stages={self.n_stages}) lengths={self.lengths}  {n / 1e6:.3f}M params  '
                f'latent {lat} numbers ({total / lat:.1f}x, spectral c_R '
                f'{self.in_channels / self.latent_per_pixel:.1f})')
