"""Shared 3-D conv primitive."""

from torch import nn

class ResBlock3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act = nn.ReLU(inplace=True)
        self.skip = (nn.Conv3d(in_channels, out_channels, kernel_size=1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))
