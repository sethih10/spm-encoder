"""TERS hyperspectral cubes straight out of the HDF5, alone or paired with their molecule.

Run as a script, reads a few samples from a file and prints what came back:

    python -m ters.datasets.ters_cubes --h5 CUBES.h5 [--paired]
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from ..transforms.normalize import NormalizeVectorized


class TersCubeDataset(Dataset):
    """TERS hyperspectral cubes from HDF5, emitting ``{'cube': (C, H, W)}``.

    Opens its HDF5 handle lazily, in the worker that reads from it. ``keep_channels`` applies
    the channel mask; ``indices`` selects rows.
    """

    def __init__(self, h5_path, channels_key='channels_100', normalize=True, indices=None,
                 keep_channels=None):
        import h5py

        self.h5_path = h5_path
        self.channels_key = channels_key
        self.normalize = NormalizeVectorized() if normalize else None
        self._hf = None

        # Fail in the parent, not in a DataLoader worker where the traceback is unreadable.
        with h5py.File(h5_path, 'r') as f:
            if channels_key not in f:
                raise KeyError(
                    f'{h5_path} has no dataset {channels_key!r}. '
                    f'Available top-level keys: {list(f.keys())}')
            shape = f[channels_key].shape
            dtype = f[channels_key].dtype

        if len(shape) != 4:
            raise ValueError(
                f'{channels_key} has shape {shape}; expected 4 dims (N, H, W, C)')

        self.n_total = int(shape[0])
        self.cube_shape = tuple(int(s) for s in shape[1:])   # (H, W, C) as stored
        self.height, self.width = self.cube_shape[0], self.cube_shape[1]
        # Every reader reports its spatial size under one name, whatever its own axes are
        # called, so a trainer can size a model without knowing which modality it has.
        self.image_size = self.height
        self.epoch = 0
        self.stored_channels = int(shape[3])
        self.dtype = dtype

        # --- NEW (channel mask): drop spectral bins that are empty in the whole corpus ---
        # ch49-71 is the Raman silent region and is exactly zero in every cube, so keeping those
        # channels lowers both the zero baseline and the MSE without reconstructing anything.
        # It does not fix per-molecule emptiness: most dead channels are dead only for their own
        # molecule. Channel occupancy is measured in docs/09-corpus-structure.md.
        if keep_channels is None:
            self.keep_channels = None
            self.in_channels = self.stored_channels
        else:
            keep = [int(c) for c in keep_channels]
            if not keep:
                raise ValueError('keep_channels is empty; nothing would be reconstructed')
            worst = max(keep)
            if worst >= self.stored_channels or min(keep) < 0:
                raise ValueError(
                    f'keep_channels references channel {worst} but {channels_key} stores '
                    f'{self.stored_channels}. A mask measured on a different corpus or a '
                    f'different channels_key will not transfer -- re-run tools.channel_stats.')
            if len(set(keep)) != len(keep):
                raise ValueError('keep_channels contains duplicates')
            # Kept as a tensor so __getitem__ indexes without rebuilding it per sample, and
            # sorted so the channel order is the spectral order regardless of how the mask
            # file listed them.
            self.keep_channels = torch.as_tensor(sorted(keep), dtype=torch.long)
            self.in_channels = len(keep)                     # feeds TersEncoder(in_channels=)

        if indices is None:
            self.indices = None
        else:
            self.indices = [int(i) for i in indices]
            worst = max(self.indices) if self.indices else -1
            if worst >= self.n_total:
                raise IndexError(
                    f'indices reference row {worst} but {channels_key} holds only '
                    f'{self.n_total} cubes')

    def __len__(self):
        return self.n_total if self.indices is None else len(self.indices)

    def __getitem__(self, idx):
        row = idx if self.indices is None else self.indices[idx]

        # Opened on first touch, so the handle is created inside the worker.  An h5py handle
        # inherited across fork() is not safe to read from.
        if self._hf is None:
            import h5py
            self._hf = h5py.File(self.h5_path, 'r')

        cube = self._hf[self.channels_key][int(row)]                     # (H, W, C)
        cube = torch.from_numpy(np.asarray(cube)).float().permute(2, 0, 1)
        if self.keep_channels is not None:
            # Before `contiguous()` and before normalising: standardisation is per-channel and
            # independent, so selecting first is the same result for less work.
            cube = cube.index_select(0, self.keep_channels)
        cube = cube.contiguous()
        cube = cube if self.normalize is None else self.normalize(cube)
        # A dict, like every other reader, so one collate and one trainer serve both modalities.
        # `row` is the HDF5 row and not the index into this dataset, so a metric traces back to
        # a molecule even under `indices`/`limit`.
        return {'cube': cube, 'row': int(row)}

    def __getstate__(self):
        # A live HDF5 handle is not pickled to a worker (spawn start method, or a parent that
        # already read a sample before the loader forked).
        state = self.__dict__.copy()
        state['_hf'] = None
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

    def set_epoch(self, epoch):
        """No-op. STM redraws its z-slice each epoch; a TERS cube has no z axis to redraw, so
        every epoch sees the same cubes. Defined so the loop can call it without asking which
        modality it has."""
        self.epoch = int(epoch)

    def describe(self):
        masked = ('' if self.keep_channels is None
                  else f'  (masked {self.stored_channels}->{self.in_channels} channels)')
        return (f'{len(self)} cubes (of {self.n_total})  (H,W,C)={self.cube_shape}  '
                f'{self.dtype}  in_channels={self.in_channels}{masked}')


def _self_test(h5_path, channels_key, n=3):
    """Exercise the dataset against a real file and report what came out."""
    ds = TersCubeDataset(h5_path, channels_key=channels_key)
    print(ds.describe())
    for i in range(min(n, len(ds))):
        c = ds[i]['cube']
        live = int((c.std(dim=(1, 2)) > 1e-3).sum())
        print(f'  sample {i}: {tuple(c.shape)}  mean={c.mean():+.3e}  std={c.std():.3f}  '
              f'live_channels={live}/{c.shape[0]}  '
              f'zero-baseline MSE={c.pow(2).mean():.3f}')
    ds.close()
    return ds


class TersPairDataset(TersCubeDataset):
    """TERS cubes paired with the molecule that produced them.

    Emits the same dict ``StmPairDataset`` does, so one collate and one trainer serve both
    modalities. Subclasses ``TersCubeDataset`` with two deltas: ``__init__`` additionally
    requires the geometry groups and cross-checks their length, and ``__getitem__`` adds ``z``
    and ``pos``.

    Indexed by ``str(row)``, never by ``.keys()`` order: the geometry groups are keyed by
    string and h5py iterates them lexicographically, which would bind nearly every molecule to
    the wrong cube silently. The schema is unverified against a real corpus -- the checks in
    ``__init__`` are what make the first real file either work or say why not.
    """

    POS_KEY = 'atom_positions'
    Z_KEY = 'atomic_numbers'

    def __init__(self, h5_path, channels_key='channels_100', normalize=True, indices=None,
                 keep_channels=None, centre=True, pos_key=POS_KEY, z_key=Z_KEY):
        super().__init__(h5_path, channels_key=channels_key, normalize=normalize,
                         indices=indices, keep_channels=keep_channels)
        import h5py

        self.centre = bool(centre)
        self.pos_key, self.z_key = pos_key, z_key
        # Fail in the parent, before fork.
        with h5py.File(h5_path, 'r') as f:
            for k in (self.z_key, self.pos_key):
                if k not in f:
                    raise KeyError(
                        f'{h5_path} has no {k!r}, so it carries cubes but no molecules. '
                        f'Top-level keys: {list(f.keys())}. Use TersCubeDataset for the '
                        f'reconstruction arm, or regenerate the corpus with geometry groups.')
                if len(f[k]) != self.n_total:
                    raise ValueError(
                        f'{h5_path}: {k!r} holds {len(f[k])} molecules but {channels_key} '
                        f'holds {self.n_total} cubes. The pairing is by row index, so these '
                        f'must agree -- this file is not a single consistent split.')

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample['z'], sample['pos'] = self._molecule(sample['row'])
        return sample

    def _molecule(self, row):
        key = str(row)                      # not .keys() order -- see the class docstring
        z_raw = np.asarray(self._hf[self.z_key][key])
        p_raw = np.asarray(self._hf[self.pos_key][key])
        z = torch.from_numpy(z_raw).long().reshape(-1)
        pos = torch.from_numpy(p_raw).float()
        if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] != z.shape[0]:
            raise ValueError(
                f'{self.h5_path}: {self.pos_key}[{key!r}] has shape {tuple(pos.shape)}; '
                f'expected ({int(z.shape[0])}, 3) to match {self.z_key}[{key!r}]. If the '
                f'corpus stores (3, n) or includes substrate atoms, that is a schema question '
                f'to settle before training, not something to transpose away here.')
        if self.centre:
            pos = pos - pos.mean(dim=0, keepdim=True)
        return z, pos


# The CLI is last so every class it can reach is already defined.
if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--channels-key', default='channels_100')
    ap.add_argument('-n', type=int, default=3)
    ap.add_argument('--paired', action='store_true',
                    help='read the geometry groups too, i.e. exercise TersPairDataset')
    a = ap.parse_args()
    if a.paired:
        ds = TersPairDataset(a.h5, channels_key=a.channels_key)
        print(ds.describe())
        for i in range(min(a.n, len(ds))):
            s = ds[i]
            print(f'  sample {i}: cube {tuple(s["cube"].shape)}  '
                  f'{int(s["z"].numel())} atoms  centroid '
                  f'{float(s["pos"].mean(0).abs().max()):.2e}')
        ds.close()
    else:
        _self_test(a.h5, a.channels_key, a.n)
