"""The 3D-structure side of the contrastive pair: molecules in, one vector out.

Sources:
  * ``TorchMD_ET`` -- REUSE, vendored verbatim under torchmdnet/ from
    https://github.com/shehzaidi/pre-training-via-denoising @ 2d81667c (MIT).
    Upstream lineage: https://github.com/TorchMD/TorchMD-net
"""

import torch
import torch.nn as nn
from torch_scatter import scatter

from torchmdnet.models.torchmd_et import TorchMD_ET

# The architecture keys a TorchMD-net Lightning checkpoint stores, mapped to TorchMD_ET's
# constructor.  Read off the checkpoint rather than hardcoded, so a different checkpoint
# cannot silently be loaded into the wrong architecture.
_HP_TO_KWARG = {
    'embedding_dimension': 'hidden_channels',
    'num_layers': 'num_layers',
    'num_rbf': 'num_rbf',
    'rbf_type': 'rbf_type',
    'trainable_rbf': 'trainable_rbf',
    'activation': 'activation',
    'attn_activation': 'attn_activation',
    'neighbor_embedding': 'neighbor_embedding',
    'num_heads': 'num_heads',
    'distance_influence': 'distance_influence',
    'cutoff_lower': 'cutoff_lower',
    'cutoff_upper': 'cutoff_upper',
    'max_z': 'max_z',
    'max_num_neighbors': 'max_num_neighbors',
    'layernorm_on_vec': 'layernorm_on_vec',
}

CKPT_PREFIX = 'model.representation_model.'

class StructureEncoder(nn.Module):
    """Wraps the vendored ET and returns one ``hidden_channels``-d vector per molecule.

    Parameters
    ----------
    pool : {'mean', 'sum'}
        How per-atom features are reduced to one vector per molecule.
    freeze : bool
        Default False: MolSpectra trains its 3-D encoder jointly, from pretrained weights with
        light finetuning. ``freeze=True`` is the ablation arm.
    """

    def __init__(self, pool='mean', freeze=False, **et_kwargs):
        super().__init__()
        if pool not in ('mean', 'sum'):
            raise ValueError(f"pool must be 'mean' or 'sum', got {pool!r}")
        self.net = TorchMD_ET(**et_kwargs)
        self.pool = pool
        self.out_dim = et_kwargs.get('hidden_channels', 256)
        self.frozen = bool(freeze)
        if self.frozen:
            for p in self.net.parameters():
                p.requires_grad_(False)

    @classmethod
    def from_checkpoint(cls, path, pool='mean', freeze=False, map_location='cpu',
                        strict=True, verbose=True):
        """Build the architecture the checkpoint describes, then load it and say what happened.

        ``strict=True`` raises on any missing or unexpected key. A loader that raises only when
        zero keys match drops a shape-drifted head silently and still prints a reassuring
        "Loaded N/M tensors"; a partial load of a pretrained encoder is not a warning, it is a
        different experiment.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if 'hyper_parameters' not in ckpt or 'state_dict' not in ckpt:
            raise ValueError(
                f'{path} is not a TorchMD-net Lightning checkpoint: expected top-level '
                f'"hyper_parameters" and "state_dict", found {sorted(ckpt)[:10]}')
        hp = ckpt['hyper_parameters']

        missing_hp = [k for k in _HP_TO_KWARG if k not in hp]
        if missing_hp:
            raise ValueError(f'{path} hyper_parameters lack {missing_hp}; cannot rebuild the '
                             f'architecture without guessing, and guessing here loads real '
                             f'weights into the wrong model')
        kwargs = {kw: hp[k] for k, kw in _HP_TO_KWARG.items()}

        model = cls(pool=pool, freeze=freeze, **kwargs)
        sub = {k[len(CKPT_PREFIX):]: v for k, v in ckpt['state_dict'].items()
               if k.startswith(CKPT_PREFIX)}
        if not sub:
            raise ValueError(f'no keys under {CKPT_PREFIX!r} in {path}; found prefixes '
                             f'{sorted({k.split(".")[0] for k in ckpt["state_dict"]})}')

        res = model.net.load_state_dict(sub, strict=False)
        if verbose:
            print(f'[StructureEncoder] {path}')
            print(f'[StructureEncoder]   loaded {len(sub) - len(res.missing_keys)}/{len(sub)} '
                  f'tensors, {len(res.missing_keys)} missing, '
                  f'{len(res.unexpected_keys)} unexpected')
        if strict and (res.missing_keys or res.unexpected_keys):
            raise RuntimeError(
                f'partial load from {path}: {len(res.missing_keys)} missing, '
                f'{len(res.unexpected_keys)} unexpected. '
                f'missing={res.missing_keys[:5]} unexpected={res.unexpected_keys[:5]}. '
                f'Pass strict=False only if you intend to train the rest from scratch.')
        return model

    def forward(self, z, pos, batch):
        """(atoms,) atomic numbers, (atoms, 3) coordinates, (atoms,) molecule index."""
        if z.dim() != 1 or pos.dim() != 2 or pos.shape[1] != 3:
            raise ValueError(f'expected z (atoms,) and pos (atoms, 3), got {tuple(z.shape)} '
                             f'and {tuple(pos.shape)}')
        if z.shape[0] != pos.shape[0] or z.shape[0] != batch.shape[0]:
            raise ValueError(f'z, pos and batch must agree on atom count: '
                             f'{z.shape[0]}, {pos.shape[0]}, {batch.shape[0]}')
        x, _v, _z, _pos, _b = self.net(z, pos, batch)
        n_graphs = int(batch.max().item()) + 1
        return scatter(x, batch, dim=0, dim_size=n_graphs, reduce=self.pool)

    def train(self, mode=True):
        """A frozen encoder stays in eval mode, so its dropout/norms cannot drift."""
        super().train(mode)
        if self.frozen:
            self.net.eval()
        return self
