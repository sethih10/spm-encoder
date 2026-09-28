"""InfoNCE, the contrastive objective.

Sources:
  * the objective -- reimplemented from the published equation in MolSpectra, Wang et al.,
    ICLR 2025, https://arxiv.org/abs/2502.16284
  * InfoNCE -- van den Oord et al., https://arxiv.org/abs/1807.03748
"""


import torch
from torch.nn import functional as F

def info_nce(z_a, z_b, temperature=0.07, symmetric=True):
    """Symmetric InfoNCE over cosine similarity.

    Parameters
    ----------
    z_a, z_b : (B, D) tensors
        Paired embeddings; row ``i`` of each is the same molecule seen two ways.  Need not be
        normalised -- this function normalises, and doing it here rather than at the call site
        means a caller cannot silently skip it.
    temperature : float
        0.07, MolSpectra's value, never tuned for this corpus.
    symmetric : bool
        True -> CLIP's both-directions form (default); False -> MolSpectra's one-directional.

    Returns
    -------
    scalar loss. Collapsed onto one direction at ``temperature=1`` it is exactly ``log(B)``.
    """
    if z_a.shape != z_b.shape:
        raise ValueError(f'z_a {tuple(z_a.shape)} and z_b {tuple(z_b.shape)} must match; '
                         f'row i of each must be the same molecule')
    if z_a.dim() != 2:
        raise ValueError(f'expected (batch, dim), got {tuple(z_a.shape)}')
    if z_a.shape[0] < 2:
        raise ValueError(f'InfoNCE needs at least 2 samples to have a negative, '
                         f'got batch={z_a.shape[0]}')
    if temperature <= 0:
        raise ValueError(f'temperature must be > 0, got {temperature}')

    a = F.normalize(z_a, dim=-1)
    b = F.normalize(z_b, dim=-1)
    logits = a @ b.t() / temperature                      # (B, B), [i, j] = sim(a_i, b_j)
    target = torch.arange(logits.shape[0], device=logits.device)

    loss = F.cross_entropy(logits, target)                # a -> b
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(logits.t(), target))   # + b -> a
    return loss
