"""The TERS HDF5 readers: what they emit, and the traps in reading them."""

import os
import tempfile
import traceback

import torch

from ters.datasets.collate import collate_ters
from ters.datasets.ters_cubes import TersCubeDataset, TersPairDataset
from tests.make_synthetic import make_synthetic_h5, make_synthetic_ters_pair_h5

#: The TERS schema, declared here because this file is what enforces it.
#: TERS has two readers and, unlike STM, one of them has no molecule at all, so the molecule
#: half is genuinely optional here.
SAMPLE_REQUIRED = ('cube', 'row')
SAMPLE_MOLECULE = ('z', 'pos')
BATCH_REQUIRED = ('cube', 'rows')
BATCH_MOLECULE = ('z', 'pos', 'batch', 'n_atoms')

FAILURES = []

def check(name, cond, detail=''):
    ok = bool(cond)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not ok:
        FAILURES.append(name)
    return ok


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


def _paired(n=12, hw=16):
    p, counts = make_synthetic_ters_pair_h5(
        os.path.join(tempfile.mkdtemp(), 'tp.h5'), n=n, hw=hw)
    return TersPairDataset(p), counts


def _plain(n=8, hw=16, channels=6):
    p = make_synthetic_h5(os.path.join(tempfile.mkdtemp(), 't.h5'),
                          n=n, hw=hw, channels=channels)
    return TersCubeDataset(p)


def test_paired_sample_is_exhaustive():
    print('test_paired_sample_is_exhaustive')
    ds, _ = _paired()
    s = ds[0]
    want = set(SAMPLE_REQUIRED) | set(SAMPLE_MOLECULE)
    check('a paired sample is exactly cube/row/z/pos', set(s) == want,
          f'{sorted(set(s) ^ want)} differ')
    check('cube is (C,H,W) float32', s['cube'].ndim == 3 and s['cube'].dtype is torch.float32)
    check('z is 1-D int64', s['z'].dtype is torch.int64 and s['z'].ndim == 1)
    check('pos is (N,3) float32',
          s['pos'].shape == (s['z'].numel(), 3) and s['pos'].dtype is torch.float32)


def test_unpaired_reader_emits_the_required_half_only():
    """The asymmetry with STM: a reconstruction corpus has no molecule, and the collate must
    not invent one."""
    print('test_unpaired_reader_emits_the_required_half_only')
    ds = _plain()
    check('the cube-only reader emits exactly cube/row',
          set(ds[0]) == set(SAMPLE_REQUIRED), str(sorted(ds[0])))
    b = collate_ters([ds[i] for i in range(4)])
    check('its batch carries no invented molecule keys', set(b) == set(BATCH_REQUIRED),
          str(sorted(b)))


def test_paired_batch_is_exhaustive():
    print('test_paired_batch_is_exhaustive')
    ds, counts = _paired()
    b = collate_ters([ds[i] for i in range(len(ds))])
    check('a paired batch is exactly cube/rows/z/pos/batch/n_atoms',
          set(b) == set(BATCH_REQUIRED) | set(BATCH_MOLECULE), str(sorted(b)))
    check('n_atoms matches the planted counts', b['n_atoms'].tolist() == counts)
    check('segments sum to the atom total',
          int(b['batch'].numel()) == sum(counts) == int(b['z'].numel()))


def test_mixing_paired_and_unpaired_raises():
    """A batch is paired or unpaired, never both -- silently dropping the molecules would
    still train, against whatever subset happened to be paired."""
    print('test_mixing_paired_and_unpaired_raises')
    ds, _ = _paired()
    check('a mixed batch raises', raises(lambda: collate_ters([ds[0], _plain()[0]]), ValueError))


def test_string_keyed_groups_are_indexed_by_str_not_iteration_order():
    """h5py iterates '0','1','10','100' -- key order would mispair nearly every molecule."""
    print('test_string_keyed_groups_are_indexed_by_str_not_iteration_order')
    ds, counts = _paired()
    check('row i pairs with molecule str(i), for all 12 rows',
          all(ds[i]['z'].numel() == counts[i] for i in range(len(ds))),
          f'lexicographic order would put row 10 at position 2; counts={counts}')
    check('the planted 2-atom molecule survives', ds[0]['z'].numel() == 2)
    check('the planted max-atom molecule is read whole', ds[1]['z'].numel() == counts[1])


def test_pair_reader_centres_and_validates():
    print('test_pair_reader_centres_and_validates')
    p, _ = make_synthetic_ters_pair_h5(os.path.join(tempfile.mkdtemp(), 'tp.h5'), n=12, hw=16)
    check('centre=True subtracts the centroid',
          float(TersPairDataset(p, centre=True)[0]['pos'].mean(0).abs().max()) < 1e-5)
    check('centre=False leaves it where the file put it',
          float(TersPairDataset(p, centre=False)[0]['pos'].mean(0).abs().max()) > 1.0)
    q = make_synthetic_h5(os.path.join(tempfile.mkdtemp(), 'plain.h5'), n=8, hw=16, channels=6)
    check('a geometry-less file raises, naming the alternative',
          raises(lambda: TersPairDataset(q), KeyError))


def test_pair_reader_inherits_the_parent_behaviour():
    print('test_pair_reader_inherits_the_parent_behaviour')
    p, _ = make_synthetic_ters_pair_h5(os.path.join(tempfile.mkdtemp(), 'tp.h5'),
                                       n=12, hw=16, channels=6)
    masked = TersPairDataset(p, keep_channels=[0, 2, 4])
    check('the channel mask still applies',
          masked.in_channels == 3 and tuple(masked[0]['cube'].shape) == (3, 16, 16))
    sub = TersPairDataset(p, indices=[0, 2])
    check('indices subset, and row follows the FILE row not the index',
          len(sub) == 2 and sub[1]['row'] == 2)
    check('normalisation is the parent NormalizeVectorized',
          type(TersPairDataset(p).normalize).__name__ == 'NormalizeVectorized')


def main():
    for t in (test_paired_sample_is_exhaustive,
              test_unpaired_reader_emits_the_required_half_only,
              test_paired_batch_is_exhaustive,
              test_mixing_paired_and_unpaired_raises,
              test_string_keyed_groups_are_indexed_by_str_not_iteration_order,
              test_pair_reader_centres_and_validates,
              test_pair_reader_inherits_the_parent_behaviour):
        try:
            t()
        except Exception:
            FAILURES.append(t.__name__ + ' (raised)')
            traceback.print_exc()
    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILED: {FAILURES}')
        raise SystemExit(1)
    print('ters dataset checks passed')


if __name__ == '__main__':
    main()
