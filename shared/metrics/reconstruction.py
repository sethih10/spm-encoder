"""Reconstruction metrics, and the baseline losses a model's loss is read against.

Sources:
  * the metric set HSI compression papers report -- Kuester et al., IEEE TGRS 2023,
    https://doi.org/10.1109/TGRS.2023.3328222 ; Saeed et al., Remote Sensing 18:1074 (2026),
    https://doi.org/10.3390/rs18071074
"""

import math

import torch
from torch.nn import functional as F

@torch.no_grad()
def reconstruction_metrics(recon, target, eps=1e-8):
    """SNR, SAM and the unexplained fraction, over the whole batch rather than per cube.

    These are what the HSI compression literature reports, so they are what makes this
    project's numbers comparable to it; MSE alone is not. PSNR is not among them: it needs a
    peak value, and a per-channel standardised cube has no natural one.
    """
    mse = F.mse_loss(recon, target).item()
    zero = target.pow(2).mean().item()
    snr = 10.0 * math.log10(zero / mse) if mse > 0 and zero > 0 else float('nan')

    # (B, C, H, W) -> (B*H*W, C): one spectrum per pixel, which is what SAM is defined over.
    b, c, h, w = target.shape
    t = target.permute(0, 2, 3, 1).reshape(-1, c)
    r = recon.permute(0, 2, 3, 1).reshape(-1, c)
    tn = t.norm(dim=1)
    rn = r.norm(dim=1)
    live = (tn > eps) & (rn > eps)
    if live.any():
        cos = (t[live] * r[live]).sum(dim=1) / (tn[live] * rn[live])
        sam = torch.rad2deg(cos.clamp(-1.0, 1.0).arccos()).mean().item()
    else:
        sam = float('nan')

    return {'snr_db': snr, 'sam_deg': sam,
            'frac_unexplained': mse / zero if zero > 0 else float('nan')}


@torch.no_grad()
def baseline_losses(target):
    """Baseline MSEs for a batch of standardised cubes, to put a model's loss in context.

    ``zero``: always output 0. ``mean``: always output this batch's own average cube. A model's
    loss only means something next to these two numbers. ``zero`` is not ~1.0 the way a
    normalised dataset suggests, because a dead channel is exactly 0 and guessing 0 gets it for
    free.
    """
    zero = target.pow(2).mean()                                    # (target - 0)^2, averaged
    b = target.shape[0]
    if b < 2:
        return {'zero': zero.item(), 'mean': float('nan')}         # no 'this batch's average'
                                                                   # with only one cube in it
    mean_cube = target.mean(dim=0, keepdim=True)                   # this batch's own average,
                                                                   # shape (1, C, H, W)
    mean = (target - mean_cube).pow(2).mean() * (b / (b - 1))      # x B/(B-1): mean_cube was
                                                                   # fit on the same b cubes it
                                                                   # is now scored against
    return {'zero': zero.item(), 'mean': mean.item()}
