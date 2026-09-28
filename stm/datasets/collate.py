"""Collate for STM batches."""

import torch


def collate_stm(items):
    """Stack cubes; flatten molecules into the ``(z, pos, batch)`` the ET expects.

    Molecules have different atom counts, so they cannot be stacked -- they are concatenated
    with a ``batch`` vector marking which atoms belong to which molecule. This is PyG's
    convention, written out so a dataset stays a plain ``torch.utils.data.Dataset`` and the
    collate carries no PyG object.
    """
    missing = [k for k in ('cube', 'row', 'z', 'pos') if k not in items[0]]
    if missing:
        raise ValueError(
            f'an STM sample is missing {missing}; every STM reader is paired, so the '
            f'molecule half is not optional. Got keys {sorted(items[0])}.')
    out = {
        'cube': torch.stack([it['cube'] for it in items]),
        'rows': torch.tensor([it['row'] for it in items], dtype=torch.long),
        'z': torch.cat([it['z'] for it in items]),
        'pos': torch.cat([it['pos'] for it in items]),
        'batch': torch.cat([torch.full((it['z'].numel(),), i, dtype=torch.long)
                            for i, it in enumerate(items)]),
        'n_atoms': torch.tensor([it['z'].numel() for it in items], dtype=torch.long),
    }
    return out
