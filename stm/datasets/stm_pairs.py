"""STM z-stacks paired with the molecule that produced them -- the contrastive corpus."""

import numpy as np
import torch
from torch.utils.data import Dataset

from ..transforms.normalize import NormalizeStmCube, _slice_for

CUBE_KEY = 'X'
XYZ_KEY = 'xyz'
Z_COL = 3           # xyz[..., 3] is the atomic number; 0 marks a padding row

# Modes accepted by ``StmPairDataset(z_slice=...)`` alongside ``None`` and a plain int.
Z_SLICE_MODES = ('random', 'mean')


def resolve_split(h5_path, split=None):
    """The one group name inside an STM HDF5, discovered rather than assumed.

    Done in the parent process: a wrong key inside a DataLoader worker surfaces as a mangled
    traceback from a subprocess.
    """
    import h5py

    with h5py.File(h5_path, 'r') as f:
        groups = [k for k in f.keys() if hasattr(f[k], 'keys')]
        if split is None:
            if CUBE_KEY in f:
                return None                       # flat layout, no split group
            if len(groups) != 1:
                raise KeyError(f'{h5_path}: cannot infer the split group, found {groups}. '
                               f'Pass split= explicitly.')
            split = groups[0]
        if split not in f:
            raise KeyError(f'{h5_path} has no group {split!r}. Top-level keys: {list(f.keys())}')
        for k in (CUBE_KEY, XYZ_KEY):
            if k not in f[split]:
                raise KeyError(f'{h5_path}[{split!r}] has no {k!r}. Members: '
                               f'{list(f[split].keys())}. Without {XYZ_KEY!r} there is no '
                               f'molecule to contrast the cube against.')
        return split


class StmPairDataset(Dataset):
    """STM z-stacks paired with the molecule that produced them.

    Emits ``{'cube', 'z', 'pos', 'row'}`` per sample; ``batch`` and ``n_atoms`` are added by
    the collate. The HDF5 handle is opened lazily, inside the worker, because an h5py handle
    inherited across fork() is not safe to read from.

    Parameters
    ----------
    h5_path : str
        An HDF5 with ``<split>/X`` (N, H, W, Z) and ``<split>/xyz``.
    z_slice : None | int | 'random' | 'mean'
        Which slice reaches the encoder. ``None`` keeps all 14; an int fixes one; ``'random'``
        redraws per epoch via ``set_epoch``; ``'mean'`` averages the stack. One slice loses
        little on this corpus: the 14 are near rank-1 (PC1 = 99.98% of variance, MEASURED),
        which is what makes ASD-STM's single-slice protocol defensible here.
    normalize : bool
        Whole-cube standardisation, which keeps the z decay envelope.
    indices, limit : sequence | int
        Row subset, and a cap applied before it.
    centre : bool
        Centre atom positions on their centroid.
    """

    def __init__(self, h5_path, split=None, normalize=True, indices=None, limit=None,
                 centre=True, z_slice=None, seed=0):
        import h5py

        self.h5_path = str(h5_path)
        self.split = resolve_split(self.h5_path, split)
        self.cube_key = CUBE_KEY if self.split is None else f'{self.split}/{CUBE_KEY}'
        self.xyz_key = XYZ_KEY if self.split is None else f'{self.split}/{XYZ_KEY}'
        self.centre = bool(centre)
        self.normalize = NormalizeStmCube() if normalize is True else (normalize or None)
        self._hf = None

        with h5py.File(self.h5_path, 'r') as f:
            cube_shape = f[self.cube_key].shape
            xyz_shape = f[self.xyz_key].shape
            if len(cube_shape) != 4:
                raise ValueError(f'{self.cube_key} should be (N, H, W, Z), got {cube_shape}')
            if len(xyz_shape) != 3 or xyz_shape[2] < 4:
                raise ValueError(f'{self.xyz_key} should be (N, max_atoms, >=4) with column '
                                 f'{Z_COL} the atomic number, got {xyz_shape}')
            if cube_shape[0] != xyz_shape[0]:
                raise ValueError(f'{self.cube_key} holds {cube_shape[0]} cubes but '
                                 f'{self.xyz_key} holds {xyz_shape[0]} molecules; the pairing '
                                 f'is positional, so they must agree')
            self.n_total = int(cube_shape[0])
            self.image_size = int(cube_shape[1])
            self.stored_channels = int(cube_shape[3])       # z-slices as stored: 14 here
            self.max_atoms = int(xyz_shape[1])

        # --- NEW (stm2d): slice selection ---
        # Validated here in the parent, because a bad index inside a DataLoader worker
        # surfaces as a mangled subprocess traceback.
        self.seed = int(seed)
        self.epoch = 0
        if z_slice is None:
            self.z_slice = None
        elif isinstance(z_slice, str):
            mode = z_slice.strip().lower()
            if mode in ('', 'none', 'all'):
                mode = None
            elif mode.lstrip('+-').isdigit():
                mode = int(mode)                            # '6' from argparse is an index
            elif mode not in Z_SLICE_MODES:
                raise ValueError(
                    f'z_slice={z_slice!r} is not understood. Use None/"none" for all '
                    f'{self.stored_channels} slices, an int index, or one of '
                    f'{list(Z_SLICE_MODES)}.')
            self.z_slice = mode
        else:
            self.z_slice = int(z_slice)
        if isinstance(self.z_slice, int):
            if not 0 <= self.z_slice < self.stored_channels:
                raise IndexError(
                    f'z_slice={self.z_slice} but {self.cube_key} stores '
                    f'{self.stored_channels} slices (valid 0..{self.stored_channels - 1}). '
                    f'Note these are indices, not the z heights in attrs["z_slices"], which '
                    f'run 6..19 on this corpus.')
        # in_channels is what the encoder is built for, so it must be the post-selection count.
        self.in_channels = self.stored_channels if self.z_slice is None else 1

        rows = range(self.n_total) if limit is None else range(min(int(limit), self.n_total))
        rows = list(rows)
        if indices is not None:
            pick = [int(i) for i in indices]
            worst = max(pick) if pick else -1
            if worst >= len(rows):
                raise IndexError(f'indices reference row {worst} of {len(rows)} available')
            rows = [rows[i] for i in pick]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = int(self.rows[idx])

        # Opened on first touch so the handle is created inside the worker: an h5py handle
        # inherited across fork() is not safe to read from.
        if self._hf is None:
            import h5py
            self._hf = h5py.File(self.h5_path, 'r')

        if self.z_slice is None:
            cube = self._hf[self.cube_key][row]                      # (H, W, Z) uint16
            cube = torch.from_numpy(np.asarray(cube)).float().permute(2, 0, 1)
        elif self.z_slice == 'mean':
            cube = self._hf[self.cube_key][row]                      # (H, W, Z)
            cube = torch.from_numpy(np.asarray(cube)).float().permute(2, 0, 1)
            cube = cube.mean(dim=0, keepdim=True)                    # (1, H, W)
        else:
            i = (self.z_slice if isinstance(self.z_slice, int)
                 else _slice_for(row, self.epoch, self.stored_channels, self.seed))
            # Index the slice, not the cube: chunks are (1, H, W, 1), so this decompresses one
            # chunk instead of fourteen. Reading `[row]` and slicing after would be correct
            # and MEASURED 11x slower.
            sl = self._hf[self.cube_key][row, :, :, i]               # (H, W)
            cube = torch.from_numpy(np.asarray(sl)).float().unsqueeze(0)   # (1, H, W)
        cube = cube.contiguous()
        # With one slice this standardises that slice, not the stack: the normaliser reduces
        # over every axis it is handed, and one slice alone has no z envelope left to keep.
        if self.normalize is not None:
            cube = self.normalize(cube)

        raw = np.asarray(self._hf[self.xyz_key][row])                # (max_atoms, >=4)
        keep = raw[:, Z_COL] > 0                                     # 0 marks a padding row
        atoms = raw[keep]
        if atoms.shape[0] == 0:
            raise ValueError(f'{self.h5_path} row {row} has no atoms with Z > 0; the cube has '
                             f'no molecule to contrast against')
        pos = torch.from_numpy(atoms[:, :3].astype(np.float32))
        z = torch.from_numpy(atoms[:, Z_COL].astype(np.int64))
        if self.centre:
            pos = pos - pos.mean(dim=0, keepdim=True)
        return {'cube': cube, 'z': z, 'pos': pos, 'row': row}

    def set_epoch(self, epoch):
        """Advance the epoch that ``z_slice='random'`` mixes into its per-row choice.

        Train dataset only, once per epoch, before iterating. Validation must keep ``epoch``
        fixed: checkpoint selection reads ``top1_size_controlled`` on val, and resampling val
        slices every epoch would make that criterion noisy for a reason unrelated to the model.

        Workers pick the new value up because this loader does not pass ``persistent_workers``,
        so they are re-forked each epoch. A no-op for every other ``z_slice`` mode, which is
        why the training loop can call it unconditionally.
        """
        self.epoch = int(epoch)
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_hf'] = None                    # never pickle a live HDF5 handle to a worker
        return state

    def close(self):
        if self._hf is not None:
            self._hf.close()
            self._hf = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def describe(self):
        sliced = ('' if self.z_slice is None
                  else f'  (z_slice={self.z_slice!r} of {self.stored_channels})')
        return (f'StmPairDataset({self.h5_path}, split={self.split!r}): {len(self)} of '
                f'{self.n_total} pairs, cube {self.in_channels}x{self.image_size}'
                f'x{self.image_size}, <= {self.max_atoms} atoms{sliced}')


def collate_pairs(items):
    """Alias for ``collate_stm``, whose body lives in this plugin's own collate.py.

    An alias rather than a second implementation. The name is load-bearing:
    ``tools/rescore.py``, ``tools/linear_probe.py`` and the STM tests all import it.
    """
    from .collate import collate_stm
    return collate_stm(items)
