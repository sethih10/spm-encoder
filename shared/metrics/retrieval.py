"""Retrieval metrics, and the baselines that make them readable."""

import math

import torch
from torch.nn import functional as F

@torch.no_grad()
def contrastive_metrics(z_a, z_b):
    """Cone geometry diagnostics, reported alongside the loss.

    ``neg_cos`` is the mean pairwise cosine between conditioning embeddings, far higher than it
    is between raw cubes. A falling loss with ``neg_cos`` unchanged means the cone is unchanged.
    """
    a = F.normalize(z_a, dim=-1)
    b = F.normalize(z_b, dim=-1)
    sim = a @ b.t()
    n = sim.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=sim.device)

    pos = sim[eye]
    neg = sim[~eye]
    tgt = torch.arange(n, device=sim.device)
    r_ab = (sim.argmax(dim=1) == tgt).float().mean()
    r_ba = (sim.argmax(dim=0) == tgt).float().mean()

    # Mean pairwise cosine within each modality: how tight each cone is on its own.
    aa = a @ a.t()
    bb = b @ b.t()
    return {
        'pos_cos': pos.mean().item(),
        'neg_cos': neg.mean().item(),
        'gap': (pos.mean() - neg.mean()).item(),
        'top1_a2b': r_ab.item(),
        'top1_b2a': r_ba.item(),
        'chance': 1.0 / n,
        'cone_a': aa[~eye].mean().item(),
        'cone_b': bb[~eye].mean().item(),
    }


def collapsed_loss(batch_size, temperature=1.0, symmetric=True):
    """The value ``info_nce`` takes when every embedding points the same way.

    The contrastive analogue of ``baseline_losses``: a collapsed model does not score 0, it
    scores ``log(B)``, and the loss curve is unreadable without that line on the same axis.
    """
    del temperature, symmetric   # collapse makes every logit equal, so both drop out
    return math.log(batch_size)


@torch.no_grad()
def size_confound_metrics(z_a, z_b, n_atoms):
    """Separates molecular identity from atom count in a retrieval score.

    Returns ``{}`` for a batch too small or degenerate to support the statistic:

    ``top1_size_only``
        Score of an atom-count-only predictor on this batch. A model at or below it shows no
        identity signal beyond size.
    ``top1_size_controlled``
        Retrieval among same-atom-count candidates only, so size carries no information. Read
        with ``n_size_controlled`` (rows having such a distractor) and
        ``chance_size_controlled``: same-size groups are small, so chance here is well above 1/B.
    ``sim_vs_size_r``
        Pearson r between pair similarity and ``-|n_i - n_j|``. Near 0 indicates similarity is not
        tracking size.
    """
    a = F.normalize(z_a, dim=-1)
    b = F.normalize(z_b, dim=-1)
    sim = a @ b.t()
    n = sim.shape[0]
    if n < 3:
        return {}
    n_atoms = n_atoms.to(sim.device).float()
    size_sim = -(n_atoms[:, None] - n_atoms[None, :]).abs()      # higher = closer in size
    tgt = torch.arange(n, device=sim.device)

    # --- the size-only baseline, with ties resolved by expectation ---
    best = size_sim.max(dim=1, keepdim=True).values
    tied = (size_sim == best)
    credit = torch.where(tied[tgt, tgt], 1.0 / tied.sum(dim=1).float(),
                         torch.zeros(n, device=sim.device))
    top1_size_only = credit.mean().item()

    # --- retrieval where size cannot help: same-atom-count candidates only ---
    same = (n_atoms[:, None] == n_atoms[None, :])
    rows = (same.sum(dim=1) > 1).nonzero(as_tuple=True)[0]        # needs >=1 distractor
    if rows.numel():
        masked = sim.masked_fill(~same, float('-inf'))
        hit = (masked[rows].argmax(dim=1) == rows).float().mean().item()
        # Its own chance level, reported with it. Same-size groups hold only a few candidates,
        # so a size-controlled top-1 must be read against this rather than against 1.0.
        chance_ctl = (1.0 / same.sum(dim=1).float()[rows]).mean().item()

        # --- normalised rank: the one statistic here that is comparable across batch sizes ---
        # `top1_size_controlled / chance` is not -- its ceiling is 1/chance, which grows with
        # batch size, so a gain at batch 256 looks better than the same gain at 64.
        # Rank the true partner among its same-size candidates, mapped to (rank-1)/(group-1):
        # 0.0 = first, 0.5 = chance at every group size, 1.0 = last. Lower is better, unlike
        # every other number in this dict. Groups of size 1 carry no information and are
        # dropped.
        r_sim = sim[rows]
        r_same = same[rows]
        tgt_sim = r_sim.gather(1, rows.view(-1, 1))
        # strictly-greater count = how many same-size candidates beat the true partner
        better = ((r_sim > tgt_sim) & r_same).sum(dim=1).float()
        grp = r_same.sum(dim=1).float()
        norm_rank = (better / (grp - 1).clamp(min=1)).mean().item()
    else:
        # No row had a same-size distractor -- every molecule in the batch was a unique size.
        # nan rather than a fabricated 0: an undefined statistic that prints as a number is
        # indistinguishable from a measured one.
        hit = float('nan')
        chance_ctl = float('nan')
        norm_rank = float('nan')

    # --- does similarity track size at all? ---
    off = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    x = sim[off]
    y = size_sim[off]
    if x.std() < 1e-8 or y.std() < 1e-8:
        r = float('nan')
    else:
        r = (((x - x.mean()) * (y - y.mean())).mean() / (x.std(unbiased=False)
             * y.std(unbiased=False))).item()

    return {'top1_size_only': top1_size_only,
            'top1_size_controlled': hit,
            'chance_size_controlled': chance_ctl,
            'norm_rank_size_controlled': norm_rank,
            'n_size_controlled': float(rows.numel()),
            'sim_vs_size_r': r}
