#!/usr/bin/env python3
"""Contrastive training on TERS cubes.

Aligns the embedding of a TERS cube with the embedding of the molecule that produced it
(InfoNCE), the same objective the STM arm runs. The counterpart to ters/train.py: the same
encoder, scored on whether its latent identifies the molecule rather than on whether it
rebuilds the cube. The loop is shared/trainers/contrastive.py; this file selects the TERS reader.

    python -m ters.train_clip --config configs/ters_clip.yaml
    python -m ters.train_clip --config configs/ters_clip.yaml --synthetic

The paired reader is unverified against a real corpus -- no TERS file with geometry groups
exists on /scratch. --synthetic exercises the wiring; the real arm waits on a corpus.
"""

import argparse
import sys

DEFAULT_CONFIG = 'configs/ters_clip.yaml'
DATASET = 'ters_pairs'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', default=DEFAULT_CONFIG,
                    help=f'experiment configuration (default: {DEFAULT_CONFIG})')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                    help='dotted override, e.g. model.latent_dim=512. An unknown key raises.')
    ap.add_argument('--workdir', default='', help='overrides training.workdir')
    ap.add_argument('--synthetic', action='store_true',
                    help='train on a generated corpus carrying the real schema')
    a = ap.parse_args(argv)

    from shared.config import load_config
    from shared.trainers.contrastive import args_from_config, run

    cfg = load_config(a.config, a.set)
    if a.workdir:
        cfg.training.workdir = a.workdir
    if a.synthetic:
        cfg = _synthetic(cfg)
    return run(args_from_config(cfg, dataset=DATASET))


def _synthetic(cfg):
    """Point the config at a freshly built TERS pair corpus carrying the real schema."""
    import os
    import tempfile

    from tests.make_synthetic import make_synthetic_ters_pair_h5
    tmp = tempfile.mkdtemp(prefix='ters-synthetic-')
    path, _ = make_synthetic_ters_pair_h5(
        os.path.join(tmp, 'train.h5'), n=max(int(cfg.data.max_train) or 12, 8),
        hw=int(cfg.data.image_size), channels=int(cfg.model.in_channels))
    cfg.data.root = tmp
    cfg.data.train_split = cfg.data.val_split = 'train'
    cfg.training.workdir = cfg.training.workdir or os.path.join(tmp, 'run')
    print(f'[ters.train_clip] --synthetic: built {path}')
    return cfg


if __name__ == '__main__':
    sys.exit(main())
