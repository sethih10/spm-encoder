"""Collate for TERS batches. Owned by this plugin.

The molecule half is conditional here, unlike the STM collate: ``TersCubeDataset`` is a
reconstruction corpus with no molecule, and it goes through this same function.
"""

import torch


def collate_ters(items):
    """Stack cubes; flatten molecules into the ``(z, pos, batch)`` the ET expects.

    Molecules have different atom counts, so they cannot be stacked -- they are concatenated
    with a ``batch`` vector marking which atoms belong to which molecule. That is PyG's
    convention, and ``batch`` is the vector the structure encoder's ``scatter`` pools over.
    Written out rather than using ``torch_geometric.data.Batch`` so a dataset stays a plain
    ``torch.utils.data.Dataset`` and the collate carries no PyG object.
    """
    out = {
        'cube': torch.stack([it['cube'] for it in items]),
        'rows': torch.tensor([it['row'] for it in items], dtype=torch.long),
    }

    # --- NEW (contract): mixed is a raise, not a silent drop ---
    # If some samples carry a molecule and some do not, quietly keeping the cubes would train a
    # contrastive objective against whatever subset happened to have one, and the loss would
    # still go down. A reader is paired or it is not.
    paired = [('z' in it and 'pos' in it) for it in items]
    if any(paired) and not all(paired):
        raise ValueError(
            f'{sum(paired)} of {len(items)} samples carry a molecule and the rest do not. '
            f'A batch is paired or unpaired, never both -- check the dataset is not mixing '
            f'two readers.')
    if not any(paired):
        return out

    out['z'] = torch.cat([it['z'] for it in items])
    out['pos'] = torch.cat([it['pos'] for it in items])
    out['batch'] = torch.cat([torch.full((it['z'].numel(),), i, dtype=torch.long)
                              for i, it in enumerate(items)])
    # Atom count per molecule, carried explicitly because it is the size confound: a predictor
    # given nothing but the count already retrieves far above chance, so a retrieval number is
    # only readable next to that baseline.
    out['n_atoms'] = torch.tensor([it['z'].numel() for it in items], dtype=torch.long)
    return out
