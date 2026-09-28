"""Tiny HDF5 files with the real schemas, for tests and --synthetic runs."""

import os

import numpy as np


def make_synthetic_h5(path, n=8, hw=32, channels=6, dead_frac=0.5, seed=0):
    """Write ``path`` and return it.  Idempotent -- an existing file is overwritten."""
    import h5py

    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)

    cubes = np.zeros((n, hw, hw, channels), dtype=np.float32)
    n_dead = int(channels * dead_frac)
    for i in range(n):
        # Which channels are live varies per sample, as it does in the real file, where no
        # channel was live in every sample probed.
        live = rng.permutation(channels)[n_dead:]
        for c in live:
            cubes[i, :, :, c] = np.abs(rng.normal(1.0, 0.5, size=(hw, hw))).astype(np.float32)
        if len(live):
            cubes[i, 0, 0, live[0]] = 500.0                       # spike
        cubes[i, :, :, 0] = 3.0 if i % 2 == 0 else cubes[i, :, :, 0]   # constant non-zero ch
    if n >= 2:
        cubes[1] = cubes[0]                                       # duplicated cube

    with h5py.File(path, 'w') as f:
        f.create_dataset('channels_100', data=cubes, chunks=(1, hw, hw, channels))
    return path


def make_synthetic_stm_h5(path, n=24, hw=32, z_slices=14, max_atoms=54, seed=0):
    """Build a small STM HDF5 with the real schema, for tests and --synthetic runs.

        <split>/X    (N, hw, hw, z_slices) uint16, chunked one slice per chunk
        <split>/xyz  (N, max_atoms, 4)     float32, Z in column 0, 0 marks padding

    Includes four edge cases: a 2-atom molecule, a max_atoms molecule, duplicated atom
    counts, and a flat cube.

    Returns (path, split, counts).
    """
    import h5py

    rng = np.random.default_rng(seed)
    split = 'train'
    X = rng.integers(0, 65535, size=(n, hw, hw, z_slices), dtype=np.uint16)
    X[0] = 1234                                   # flat cube: zero variance

    # A decay envelope along z, so the axis carries real ordering for a 3D stem to find.
    env = np.linspace(1.0, 0.25, z_slices, dtype=np.float32)
    X = (X.astype(np.float32) * env).astype(np.uint16)

    xyz = np.zeros((n, max_atoms, 5), dtype=np.float32)
    counts = rng.integers(4, 20, size=n)
    counts[0] = 2                                 # corpus minimum
    counts[1] = max_atoms                         # no padding at all
    counts[2:6] = 11                              # shared size -> same-size distractors exist
    for i, c in enumerate(counts):
        xyz[i, :c, :3] = rng.normal(0, 2.0, size=(c, 3))
        xyz[i, :c, 3] = rng.choice([1, 6, 7, 8, 16, 17], size=c)   # real elements only
        xyz[i, :c, 4] = rng.normal(0, 0.2, size=c)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with h5py.File(path, 'w') as f:
        g = f.create_group(split)
        g.create_dataset('X', data=X, chunks=(1, hw, hw, 1))
        g.create_dataset('xyz', data=xyz, chunks=(1, max_atoms, 5))
        g.create_dataset('scale', data=rng.random(n).astype(np.float32) * 1e-11)
        g.create_dataset('cid', data=np.array([str(i).encode() for i in range(n)]))
    return path, split, counts


def make_synthetic_ters_pair_h5(path, n=12, hw=16, channels=6, max_atoms=20, seed=0):
    """Build a small TERS HDF5 with cubes and the geometry groups, for --synthetic runs.

        channels_<C>     (N, hw, hw, C) float32
        atomic_numbers   group keyed '0'..'N-1' as strings
        atom_positions   group keyed the same way

    The string keying mirrors the real schema, including the lexicographic-iteration hazard the
    reader guards against.

    Returns (path, counts).
    """
    import h5py

    rng = np.random.default_rng(seed)
    cubes = rng.random((n, hw, hw, channels)).astype('float32')
    counts = [2, max_atoms] + [7, 7, 7] + list(rng.integers(3, max_atoms, size=max(0, n - 5)))
    counts = [int(c) for c in counts[:n]]

    with h5py.File(path, 'w') as f:
        f.create_dataset('channels_100', data=cubes,
                         chunks=(1, hw, hw, channels), compression='gzip')
        gz = f.create_group('atomic_numbers')
        gp = f.create_group('atom_positions')
        for i, c in enumerate(counts):
            gz.create_dataset(str(i), data=rng.choice([1, 6, 7, 8], size=c).astype('int64'))
            # +5.0 so the centroid is far from the origin and centre=True is measurable.
            gp.create_dataset(str(i), data=(rng.random((c, 3)) + 5.0).astype('float64'))
    return path, counts


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='synthetic.h5')
    ap.add_argument('-n', type=int, default=8)
    ap.add_argument('--hw', type=int, default=32)
    ap.add_argument('--channels', type=int, default=6)
    a = ap.parse_args()
    print(make_synthetic_h5(a.out, n=a.n, hw=a.hw, channels=a.channels))
