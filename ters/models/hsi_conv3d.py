"""3D convolutional autoencoder: the spectrum as an axis, not as unstructured channels.

Sources:
  * ``Conv3dAutoencoder`` -- ADAPT of the 3-D convolutional autoencoder of Chong, Chen
    & Pan, J. Electron. Imaging 30(4):041403 (2021),
    https://doi.org/10.1117/1.JEI.30.4.041403
  * ``SpectralSEBlock`` -- ADAPT of Squeeze-and-Excitation Networks, Hu, Shen, Albanie,
    Sun & Wu, https://arxiv.org/abs/1709.01507
  * resize-then-convolve -- Odena, Dumoulin & Olah,
    https://distill.pub/2016/deconv-checkerboard/
"""

from torch import nn
from torch.nn import functional as F

# ResBlock3d lives in shared/models/blocks3d.py because the STM plugin needs it too.
from shared.models.blocks3d import ResBlock3d

# --- ADAPT: SENet, gated per band ---
class SpectralSEBlock(nn.Module):
    """Squeeze-and-excitation over the band axis of a 3-D feature volume.

    The gate is computed per band rather than per channel, so it can attenuate a dead spectral
    region instead of a feature map. SpecResNet credits much of its spectral fidelity to this,
    which is why it is an arm switch and not a default.
    """

    def __init__(self, channels, depth, reduction=4):
        super().__init__()
        c_hidden = max(channels // reduction, 2)
        d_hidden = max(depth // reduction, 2)
        self.chan = nn.Sequential(nn.Linear(channels, c_hidden), nn.ReLU(inplace=True),
                                  nn.Linear(c_hidden, channels), nn.Sigmoid())
        self.spec = nn.Sequential(nn.Linear(depth, d_hidden), nn.ReLU(inplace=True),
                                  nn.Linear(d_hidden, depth), nn.Sigmoid())
        self.depth = depth

    def forward(self, x):
        b, f, d, h, w = x.shape
        if d != self.depth:
            raise ValueError(
                f'SpectralSEBlock built for depth={self.depth} but got {d}. The depth is fixed '
                f'by in_channels and the spectral strides; rebuild the model rather than '
                f'feeding it a differently-banded cube.')
        gc = self.chan(x.mean(dim=(2, 3, 4))).view(b, f, 1, 1, 1)
        gs = self.spec(x.mean(dim=(1, 3, 4))).view(b, 1, d, 1, 1)
        return x * gc * gs


def _stage_shapes(depth, hw, spectral_strides, kernel=3):
    """(depth, spatial) after each encoder stage, front to back.

    Recorded rather than recomputed in the decoder: duplicating the downsampling rule in a second
    place is how a decoder silently reconstructs the wrong size. 100 -> 50 -> 25 is not a power
    of two, so the decoder interpolates to these exact numbers.
    """
    pad = kernel // 2
    shapes = [(depth, hw)]
    for s in spectral_strides:
        depth = (depth + 2 * pad - kernel) // s + 1
        hw = (hw + 2 * pad - kernel) // 2 + 1
        if depth < 1 or hw < 1:
            raise ValueError(
                f'the 3D ladder collapses to {depth}x{hw} after {len(shapes)} stages; '
                f'fewer stages or fewer spectral strides of 2.')
        shapes.append((depth, hw))
    return shapes


class Conv3dAutoencoder(nn.Module):
    """``(B, C, H, W) -> (B, latent_ch, D', H', W') -> (B, C, H, W)``.

    Parameters
    ----------
    in_channels : int
        Spectral bands. Enters as the depth of a single-channel volume: ``(B, 1, C, H, W)``.
    latent_ch : int
        Feature channels of the latent volume; ``describe()`` prints the total number count.
    widths : tuple of int
        Encoder stage widths, kept small: a 3-D conv at 100x128x128 costs what a 2-D one
        at 128x128 does with 100x the elements.
    spectral_strides : tuple of int
        Per-stage stride down the band axis; the spatial stride is always 2. The default keeps 4x
        more spectral than spatial resolution in the latent, per HyCASS (arXiv:2507.23447).
    n_res, res_from_stage : int
        Residual blocks per stage, and the first stage that gets them. Stage 0 is skipped by
        default because its feature map dominates the ladder's cost.
    use_se : bool
        Insert a :class:`SpectralSEBlock` per stage. An arm switch, measured rather than assumed.
    """

    def __init__(self, in_channels=100, image_size=256, latent_ch=10,
                 widths=(8, 16, 32, 64), spectral_strides=(1, 2, 2, 1), kernel_size=3,
                 n_res=1, res_from_stage=1, use_se=False, n_groups=8, zero_init=True):
        super().__init__()
        if len(widths) != len(spectral_strides):
            raise ValueError(
                f'widths and spectral_strides must be the same length, got {len(widths)} and '
                f'{len(spectral_strides)}')
        if latent_ch < 1:
            raise ValueError(f'latent_ch must be >= 1, got {latent_ch}')

        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_ch = latent_ch
        self.widths = tuple(widths)
        self.use_se = use_se
        pad = kernel_size // 2

        self.shapes = _stage_shapes(in_channels, image_size, spectral_strides, kernel_size)
        d_out, hw_out = self.shapes[-1]
        self.latent_shape = (latent_ch, d_out, hw_out, hw_out)

        n_lat = latent_ch * d_out * hw_out * hw_out
        n_in = in_channels * image_size * image_size
        if n_lat >= n_in:
            raise ValueError(
                f'the latent is {n_lat} numbers against an input of {n_in}: no compression. '
                f'Lower latent_ch (currently {latent_ch}) or add a downsampling stage.')

        # --- encoder ---
        self.enc_down = nn.ModuleList()
        self.enc_body = nn.ModuleList()
        c_in = 1
        for i, c_out in enumerate(widths):
            self.enc_down.append(nn.Sequential(
                nn.Conv3d(c_in, c_out, kernel_size,
                          stride=(spectral_strides[i], 2, 2), padding=pad),
                nn.GroupNorm(n_groups, c_out),
                nn.ReLU(inplace=True)))
            body = [ResBlock3d(c_out, c_out, kernel_size, n_groups=n_groups)
                    for _ in range(n_res if i >= res_from_stage else 0)]
            if use_se:
                body.append(SpectralSEBlock(c_out, self.shapes[i + 1][0]))
            self.enc_body.append(nn.Sequential(*body) if body else nn.Identity())
            c_in = c_out
        # 1x1x1 to the latent width, as latent-diffusion's Encoder.conv_out does: the ladder
        # decides the grid, one pointwise convolution decides how many numbers sit on it, so the
        # budget is a single readable multiplication.
        self.to_latent = nn.Conv3d(widths[-1], latent_ch, kernel_size=1)

        # --- decoder ---
        # Mirror of the encoder. Upsample to the recorded size then convolve: transposed
        # convolution leaves checkerboard artefacts, and 25 -> 50 -> 100 is not a power of two,
        # so scale_factor would land on 26 and 52.
        self.from_latent = nn.Conv3d(latent_ch, widths[-1], kernel_size=1)
        rev = list(reversed(widths))
        self.dec_up = nn.ModuleList()
        self.dec_body = nn.ModuleList()
        for i, c_in_stage in enumerate(rev):
            c_out = rev[i + 1] if i + 1 < len(rev) else widths[0]
            self.dec_up.append(nn.Sequential(
                nn.Conv3d(c_in_stage, c_out, kernel_size, padding=pad),
                nn.GroupNorm(n_groups, c_out),
                nn.ReLU(inplace=True)))
            stage_from_front = len(widths) - 1 - i
            body = [ResBlock3d(c_out, c_out, kernel_size, n_groups=n_groups)
                    for _ in range(n_res if stage_from_front >= res_from_stage else 0)]
            if use_se:
                body.append(SpectralSEBlock(c_out, self.shapes[stage_from_front][0]))
            self.dec_body.append(nn.Sequential(*body) if body else nn.Identity())
        self.conv_out = nn.Conv3d(widths[0], 1, kernel_size, padding=pad)

        self.reset_parameters(zero_init=zero_init)

    def reset_parameters(self, zero_init=True):
        """Kaiming, plus two zero-inits.

        The output zero-init makes the first prediction exactly zero, so the initial loss equals
        the loss of predicting zero everywhere. Without it the model opens far above that and
        spends its first epochs shrinking its own output.

        The residual zero-init (torchvision ``zero_init_residual``, Goyal et al.
        arXiv:1706.02677) zeroes ``norm2`` of every ResBlock3d. Its effect is small; it is kept
        for consistency with the 2-D decoder.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init:
            for m in self.modules():
                if isinstance(m, ResBlock3d):
                    nn.init.zeros_(m.norm2.weight)
                    nn.init.zeros_(m.norm2.bias)
            nn.init.zeros_(self.conv_out.weight)
            nn.init.zeros_(self.conv_out.bias)

    def encode(self, x):
        if x.dim() != 4:
            raise ValueError(
                f'Conv3dAutoencoder expects (B, C, H, W), got {tuple(x.shape)}')
        if x.size(1) != self.in_channels:
            raise ValueError(
                f'Conv3dAutoencoder built for in_channels={self.in_channels} but got '
                f'{x.size(1)}')
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f'this model is fixed to {self.image_size}x{self.image_size} by the recorded '
                f'stage shapes; got {tuple(x.shape[-2:])}')
        h = x.unsqueeze(1)                                  # (B, 1, C, H, W)
        for down, body in zip(self.enc_down, self.enc_body):
            h = body(down(h))
        return self.to_latent(h)

    def decode(self, z):
        h = self.from_latent(z)
        for i, (up, body) in enumerate(zip(self.dec_up, self.dec_body)):
            d, hw = self.shapes[len(self.dec_up) - 1 - i]
            h = F.interpolate(h, size=(d, hw, hw), mode='nearest')
            h = body(up(h))
        return self.conv_out(h).squeeze(1)                  # (B, C, H, W)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        if recon.shape != x.shape:
            raise ValueError(
                f'reconstruction {tuple(recon.shape)} does not match input {tuple(x.shape)}')
        return recon, z

    def latent_numbers(self, image_size=None):
        c, d, h, w = self.latent_shape
        return c * d * h * w

    def describe(self, image_size=None):
        n = sum(p.numel() for p in self.parameters())
        lat = self.latent_numbers()
        total = self.in_channels * self.image_size ** 2
        return (f'Conv3dAutoencoder(latent_ch={self.latent_ch}, widths={self.widths}, '
                f'se={self.use_se}) shapes={self.shapes}  {n / 1e6:.3f}M params  '
                f'latent {self.latent_shape} = {lat} numbers ({total / lat:.1f}x)')
