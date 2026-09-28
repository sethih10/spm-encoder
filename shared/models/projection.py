import math

import torch
from torch import nn
from torch.nn import functional as F

class ProjectionHead(nn.Module):
    """The 2-layer mlp that maps an encoder embedding into the space the loss acts on.

    Kept separate from both encoders: SimCLR section 4.2 measured that contrasting on
    ``g(h)`` rather than on ``h`` leaves more information in ``h``, and ``h`` -- not ``g(h)`` --
    is what gets exported to ters-2-graph as the conditioning vector.  Discarding the head at
    export is the point of having one.
    """

    def __init__(self, in_dim, hidden_dim=None, out_dim=256, use_norm=True):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers += [nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim, bias=False)]
        self.net = nn.Sequential(*layers)
        self.in_dim, self.out_dim = in_dim, out_dim

    def forward(self, x):
        return self.net(x)
