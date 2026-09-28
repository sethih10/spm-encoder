"""Convolutional primitives shared by the encoder and the decoder.

Sources:
  * ``Upsample`` -- ADAPT of CompVis/latent-diffusion @ a506df5
    ldm/modules/diffusionmodules/model.py, https://github.com/CompVis/latent-diffusion
  * resize-then-convolve rather than transposed convolution -- Odena, Dumoulin & Olah,
    https://distill.pub/2016/deconv-checkerboard/
"""

import torch
from torch import nn


class ResBlock(nn.Module):
    """Two convs and a residual shortcut.

    GroupNorm rather than BatchNorm2d: it does not depend on the batch axis.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=8):
        super().__init__()
        padding = kernel_size // 2                      # 'same' -- the ladder resamples
                                                        # via MaxPool/interpolate, not convs
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act = nn.ReLU(inplace=True)
        # Projection on the skip path only when the width changes, so a same-width block
        # stays a true identity shortcut.
        self.skip = (nn.Conv2d(in_channels, out_channels, kernel_size=1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class Upsample(nn.Module):
    """2x nearest-neighbour upsample followed by a 3x3 conv.

    Nearest+conv rather than a transposed conv, which produces checkerboard artefacts.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)
