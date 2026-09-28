#!/usr/bin/env python3
"""Contrastive training on STM z-stacks.

Aligns the embedding of an STM z-stack with the embedding of the molecule that produced it
(InfoNCE), so top-1 retrieval between the two measures whether the cube identifies the
structure. This file selects the STM reader and this modality's defaults; the loop is shared.

    python -m stm.train_clip --config configs/stm_clip.yaml
    python -m stm.train_clip --config configs/stm_clip.yaml --set model.cube_encoder=stm2d
    python -m stm.train_clip --config configs/smoke_synthetic.yaml --synthetic

Runs go through SLURM; see slurm/train_full.sbatch. --synthetic is a functional check, not an
experiment.
"""

import argparse
import sys

DEFAULT_CONFIG = 'configs/stm_clip.yaml'
DATASET = 'stm_pairs'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', default=DEFAULT_CONFIG,
                    help=f'experiment configuration (default: {DEFAULT_CONFIG})')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                    help='dotted override, e.g. model.cube_encoder=stm2d. An unknown key '
                         'raises rather than being silently ignored.')
    ap.add_argument('--workdir', default='', help='overrides training.workdir')
    ap.add_argument('--synthetic', action='store_true',
                    help='train on a generated corpus carrying the real schema: no cluster, '
                         'no GPU and no corpus required')
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
    """Point the config at a freshly built STM corpus carrying the real schema."""
    import os
    import tempfile

    from shared.synthetic import make_synthetic_stm_h5
    tmp = tempfile.mkdtemp(prefix='stm-synthetic-')
    path, split, _ = make_synthetic_stm_h5(os.path.join(tmp, 'train.h5'),
                                           n=max(int(cfg.data.max_train) or 24, 8),
                                           hw=int(cfg.data.image_size))
    cfg.data.root = tmp
    cfg.data.train_split = cfg.data.val_split = split
    cfg.training.workdir = cfg.training.workdir or os.path.join(tmp, 'run')
    print(f'[stm.train_clip] --synthetic: built {path}')
    return cfg


if __name__ == '__main__':
    sys.exit(main())
