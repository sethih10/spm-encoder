#!/usr/bin/env python3
"""Reconstruction training on TERS cubes.

Compresses a cube to a fixed-width vector and rebuilds it, so the reported MSE measures how
much of the cube that vector retains. That vector is the one ters-2-graph conditions its
diffusion model on, so the number describes that pipeline and not an autoencoder in isolation.
ters/train_clip.py scores the same encoder on retrieval instead; the loop is
shared/trainers/reconstruction.py.

    python -m ters.train --config configs/ters_ae.yaml
    python -m ters.train --config configs/ters_ae.yaml --set model.name=hsi_conv3d
    python -m ters.train --config configs/ters_ae.yaml --synthetic

Runs go through SLURM; --synthetic is a functional check, not an experiment.
"""

import argparse
import sys

DEFAULT_CONFIG = 'configs/ters_ae.yaml'


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
    return reconstruction.run(cfg, synthetic=a.synthetic)


if __name__ == '__main__':
    sys.exit(main())
