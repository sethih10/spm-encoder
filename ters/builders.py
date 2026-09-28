"""Cube-encoder and reconstruction-model builders the TERS plugin contributes.

Each builder is a thin adapter: it takes a config, imports its model lazily so an optional
dependency cannot break ``import shared.registry``, and returns the model. Selection is by
config string through the registry rather than an if/elif on an ``--arch`` flag.
"""

from shared.registry import DATASETS, ENCODERS, MODELS, PAIRS


@ENCODERS.register('ters')
def build_ters(args, in_channels, image_size):
    from .models.encoder import TersEncoder
    cube = TersEncoder(in_channels=in_channels, latent_dim=args.latent_dim)
    print('[build] NOTE: TersEncoder opens with a 1x1 mix over channels, which has no '
          'inductive bias for the ordered z axis. Control arm, not the recommended one.')
    return cube


# --- the TERS/HSI reconstruction models ---

@MODELS.register('ters_autoencoder')
def build_ters_autoencoder(cfg):
    from .models.autoencoder import TersAutoencoder
    return TersAutoencoder.from_config(cfg)


@MODELS.register('hsi_spectral')
def build_hsi_spectral(cfg):
    from .models.hsi_spectral import SpectralAutoencoder
    m = cfg.model
    return SpectralAutoencoder(in_channels=m.in_channels,
                               latent_per_pixel=getattr(m, 'latent_per_pixel', 1),
                               base_width=getattr(m, 'base_width', 16),
                               n_stages=getattr(m, 'n_stages', 3))


@MODELS.register('hsi_conv3d')
def build_hsi_conv3d(cfg):
    from .models.hsi_conv3d import Conv3dAutoencoder
    m = cfg.model
    return Conv3dAutoencoder(in_channels=m.in_channels, image_size=cfg.data.image_size,
                             latent_ch=getattr(m, 'latent_ch', 10),
                             widths=tuple(getattr(m, 'widths', (8, 16, 32, 64))),
                             spectral_strides=tuple(getattr(m, 'spectral_strides', (1, 2, 2, 1))),
                             n_res=getattr(m, 'n_res', 1),
                             use_se=getattr(m, 'use_se', False))


# --- the readers this plugin provides ---
# Registered here so a trainer or a tool names a dataset instead of importing one. A reader
# imported at module level from core code would make the STM arm depend on this plugin being
# installed.

@DATASETS.register('ters_cubes')
def build_ters_cubes(*a, **kw):
    from .datasets.ters_cubes import TersCubeDataset
    return TersCubeDataset(*a, **kw)


@DATASETS.register('ters_pairs')
def build_ters_pairs(*a, **kw):
    from .datasets.ters_cubes import TersPairDataset
    return TersPairDataset(*a, **kw)


@PAIRS.register('ters_pairs')
def ters_pairs(h5, args, indices=None, limit=None):
    from .datasets.collate import collate_ters
    from .datasets.ters_cubes import TersPairDataset
    # channels_key and the channel mask are TERS's alone; `limit` becomes a range because the
    # reader takes indices rather than a count.
    if limit and indices is None:
        indices = range(limit)
    ds = TersPairDataset(h5, indices=indices,
                         channels_key=getattr(args, 'channels_key', 'channels_100'),
                         keep_channels=getattr(args, 'keep_channels', None))
    ds.collate = collate_ters
    return ds
