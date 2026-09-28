#!/usr/bin/env python3
"""Reconstruction training on STM z-stacks.

Compresses a z-stack to a fixed-width vector and rebuilds it, so the reported MSE measures how
much of the stack that vector retains. The counterpart to stm/train_clip.py, which scores the
same encoder on retrieval instead; running both is what makes the two numbers comparable.

    python -m stm.train --config configs/stm_ae.yaml
    python -m stm.train --config configs/stm_ae.yaml --set model.latent_dim=512
    python -m stm.train --config configs/stm_ae.yaml --synthetic

Runs go through SLURM; --synthetic is a functional check, not an experiment.
"""

import argparse
import sys

DEFAULT_CONFIG = 'configs/stm_ae.yaml'


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
    from shared.trainers import reconstruction

    cfg = load_config(a.config, a.set)
    if a.workdir:
        cfg.training.workdir = a.workdir
    if a.synthetic:
        return reconstruction.run(cfg, synthetic=True, **_synthetic(cfg))
    return reconstruction.run(cfg)


def _synthetic(cfg):
    """Build a small STM corpus carrying the real schema, and size the config to it."""
    import os
    import tempfile

    from tests.make_synthetic import make_synthetic_stm_h5
    tmp = tempfile.mkdtemp(prefix='stm-ae-synthetic-')
    train_h5, split, _ = make_synthetic_stm_h5(os.path.join(tmp, 'train.h5'), n=8, hw=32)
    val_h5, _, _ = make_synthetic_stm_h5(os.path.join(tmp, 'val.h5'), n=4, hw=32, seed=1)
    cfg.data.train_split = cfg.data.val_split = split
    cfg.data.image_size = 32
    cfg.model.filters = (8, 16)          # 32 -> 16 -> 8 at the bottleneck
    cfg.model.latent_dim = 16            # a real bottleneck at this size, not 256
    cfg.training.workdir = cfg.training.workdir or os.path.join(tmp, 'run')
    print(f'[stm.train] --synthetic: built {train_h5}')
    return dict(train_h5=train_h5, val_h5=val_h5)


if __name__ == '__main__':
    sys.exit(main())
