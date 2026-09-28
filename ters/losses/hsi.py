"""Spectral-angle loss: train on the shape of a spectrum, not only on its magnitude.

Sources:
  * ``spectral_angle_loss`` -- ADAPT of the SAD objective of Palsson, Sigurdsson,
    Sveinsson & Ulfarsson, IEEE Access 6:25646-25656 (2018),
    https://doi.org/10.1109/ACCESS.2018.2818280. Delta: pixels with no direction are dropped
    rather than given an arbitrary angle, and the default returns ``1 - cos``, not radians.
"""

import torch


def spectral_angle_loss(recon, target, form='cosine', eps=1e-6, norm_floor=1e-3,
                        return_live=False):
    """Mean angle between predicted and target spectra, per pixel.

    Parameters
    ----------
    form : {'cosine', 'angle'}
        ``'cosine'`` (default) returns ``1 - cos``. ``'angle'`` returns radians, whose gradient
        blows up exactly as the reconstruction gets good, so a run fails late rather than at the
        start where it would be cheap to see.
    norm_floor : float
        Pixels whose norm falls below this are dropped, not clamped: an all-zero spectrum has no
        direction, and 60-80% of this corpus is dead channels.
    """
    if recon.shape != target.shape:
        raise ValueError(
            f'spectral_angle_loss got recon {tuple(recon.shape)} and target '
            f'{tuple(target.shape)}; they must match')
    if recon.dim() != 4:
        raise ValueError(
            f'spectral_angle_loss expects (B, C, H, W) so the spectrum is axis 1, got '
            f'{tuple(recon.shape)}')
    if form not in ('cosine', 'angle'):
        raise ValueError(f"form must be 'cosine' or 'angle', got {form!r}")

    b, c, h, w = target.shape
    t = target.permute(0, 2, 3, 1).reshape(-1, c)
    r = recon.permute(0, 2, 3, 1).reshape(-1, c)
    tn = t.norm(dim=1)
    rn = r.norm(dim=1)
    live = (tn > norm_floor) & (rn > norm_floor)
    n_live = int(live.sum())
    if n_live == 0:
        # Exactly the zero-init state, and the only correct answer is "no opinion": a constant
        # zero keeps the graph intact so `.backward()` still works, and contributes no gradient.
        loss = (recon * 0.0).sum()
        return (loss, 0.0) if return_live else loss

    cos = (t[live] * r[live]).sum(dim=1) / (tn[live] * rn[live] + eps)
    if form == 'cosine':
        loss = (1.0 - cos).mean()
    else:
        # The clamp is load-bearing here rather than defensive: float32 cancellation puts cos
        # slightly outside [-1, 1] on a near-perfect pixel, and arccos of 1.0000001 is nan.
        loss = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7).arccos().mean()
    frac = n_live / live.numel()
    return (loss, frac) if return_live else loss


def mse_sad_loss(recon, target, sad_weight=0.5, form='cosine'):
    """``mse + sad_weight * spectral_angle``, the combined objective.

    ``sad_weight=0.5`` is set from the measured magnitudes of the two terms, so that neither
    dominates -- chosen, not tuned.
    """
    if sad_weight < 0:
        raise ValueError(f'sad_weight must be >= 0, got {sad_weight}')
    mse = torch.nn.functional.mse_loss(recon, target)
    sad, live = spectral_angle_loss(recon, target, form=form, return_live=True)
    total = mse + sad_weight * sad
    return total, {'mse': float(mse.detach()), 'sad': float(sad.detach()), 'sad_live': live}
