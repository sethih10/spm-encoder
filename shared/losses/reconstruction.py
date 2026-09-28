"""Reconstruction losses. Arithmetic over ``(B, C, H, W)`` cubes.

Sources:
  * the MSE loss shape -- ADAPT of pytorch/examples @ acc295d vae/main.py,
    https://github.com/pytorch/examples
  * the spectral-angle objective -- Palsson, Sigurdsson, Sveinsson & Ulfarsson,
    IEEE Access 6:25646-25656 (2018), https://doi.org/10.1109/ACCESS.2018.2818280
"""


from torch.nn import functional as F

# --- ADAPT: pytorch/examples vae/main.py, loss_function ---
def reconstruction_loss(recon, target, reduction='mean'):
    """MSE between the reconstruction and the standardised cube.

    Per-element reduction, so the loss does not scale with cube size.

    Do not read the number against 1.0. Standardising gives the live channels unit variance, but
    most of a TERS cube is dead channels, which standardise to exactly 0 and cost nothing to
    predict, so a model that outputs nothing at all already scores well below 1.0. The reference
    points are ``shared.metrics.reconstruction.baseline_losses``.
    """
    return F.mse_loss(recon, target, reduction=reduction)


# --- ADAPT: Palsson et al., spectral angle ---
# They use it as the whole objective; here it is a weighted addition to MSE, since a pure angle
# loss is scale-invariant and would leave the magnitude unconstrained. `1 - cos` rather than
# `arccos`: same minimum, but arccos's gradient diverges exactly as the reconstruction becomes
# correct.
def spectral_angle_loss(recon, target, eps=1e-8):
    """Mean ``1 - cos(angle)`` between true and reconstructed spectra, over live pixels.

    Pixels whose true spectrum is identically zero are excluded: their angle is undefined, and
    including them would hand the model a free zero for predicting nothing outside the
    molecule's footprint.  Returns 0 if a batch somehow has no live pixel, so a degenerate
    batch cannot produce a NaN that poisons the optimiser state.
    """
    b, c, h, w = target.shape
    t = target.permute(0, 2, 3, 1).reshape(-1, c)
    r = recon.permute(0, 2, 3, 1).reshape(-1, c)
    tn = t.norm(dim=1)
    live = tn > eps
    if not bool(live.any()):
        return recon.sum() * 0.0                  # keeps the graph, contributes nothing
    t, r = t[live], r[live]
    cos = F.cosine_similarity(t, r, dim=1, eps=eps)
    return (1.0 - cos).mean()


def combined_loss(recon, target, sam_weight=0.0):
    """``mse + sam_weight * spectral_angle_loss``; returns ``(total, parts)``.

    ``sam_weight=0.0`` is exactly :func:`reconstruction_loss`, so the default arm of a sweep is
    bit-for-bit plain MSE and the SAM term cannot silently change a baseline.
    """
    mse = reconstruction_loss(recon, target)
    if not sam_weight:
        return mse, {'mse': mse.item(), 'sam': 0.0}
    sam = spectral_angle_loss(recon, target)
    return mse + sam_weight * sam, {'mse': mse.item(), 'sam': sam.item()}
