"""Encoder, model and reader builders the STM plugin registers."""

from shared.registry import DATASETS, ENCODERS, MODELS, PAIRS, pick


@ENCODERS.register('stm3d')
def build_stm3d(args, in_channels, image_size):
    from .models.conv3d import StmCubeEncoder
    return StmCubeEncoder(in_channels=in_channels, latent_dim=args.latent_dim,
                          conv3d_channels=tuple(args.stem_channels),
                          filters=tuple(args.filters),
                          pool_size=args.cube_pool_size)


@ENCODERS.register('sgpt')
def build_sgpt(args, in_channels, image_size):
    from .models.spectral_gpt import (SGPT_SIZES, SpectralGptCubeEncoder,
                               load_spectralgpt_weights)
    if args.sgpt_ckpt and args.sgpt_size != 'base':
        raise ValueError(
            f'--sgpt-ckpt needs --sgpt-size base: the released checkpoint is ViT-Base, and '
            f'{args.sgpt_size!r} would load almost nothing. Drop the checkpoint to train '
            f'{args.sgpt_size!r} from scratch, or use base to transfer.')
    cube = SpectralGptCubeEncoder(in_channels=in_channels, image_size=image_size,
                                  latent_dim=args.latent_dim,
                                  band_mode=args.sgpt_band_mode, pool=args.sgpt_pool,
                                  trainable_blocks=args.sgpt_trainable_blocks,
                                  **pick(SGPT_SIZES, args.sgpt_size,
                                         'sgpt size', '--sgpt-size'))
    print(f'[build] sgpt: {cube.num_patches} tokens '
          f'(T={cube.t_grid} x {cube.hw_grid} x {cube.hw_grid}), '
          f'band_mode={cube.band_mode} -> z_in={cube.z_in}, '
          f't_patch={cube.t_patch_size}, pool={cube.pool}')
    if args.sgpt_ckpt:
        load_spectralgpt_weights(cube, args.sgpt_ckpt, min_match=args.sgpt_min_match)
    else:
        print('[build] NOTE: --sgpt-ckpt not given, so this ViT is RANDOM INIT. That is '
              'the from-scratch control, not the SpectralGPT transfer arm.')
    n_frozen = sum(p.numel() for p in cube.parameters() if not p.requires_grad)
    print(f'[build] sgpt trunk: trainable_blocks={cube.trainable_blocks}, '
          f'{n_frozen / 1e6:.2f} M frozen')
    return cube


@ENCODERS.register('stm2d')
def build_stm2d(args, in_channels, image_size):
    from .models.timm2d import Stm2dCubeEncoder
    cube = Stm2dCubeEncoder(
        in_channels=in_channels, image_size=image_size, latent_dim=args.latent_dim,
        backbone=args.stm2d_backbone, timm_name=args.stm2d_timm_name,
        in_chans=args.stm2d_in_chans, resize=args.stm2d_resize,
        slice_mode=args.stm2d_slice_mode, slice_index=args.stm2d_slice_index,
        trainable_blocks=args.stm2d_trainable_blocks,
        pretrained=not args.stm2d_random_init, ckpt=args.stm2d_ckpt,
        min_match=args.stm2d_min_match, seed=args.seed)
    n_head = sum(p.numel() for p in list(cube.head.parameters())
                 + list(cube.out_norm.parameters()))
    print(f'[build] stm2d: {cube.timm_name}  in_chans={cube.in_chans} '
          f'resize={cube.resize} -> feat {cube.feat_dim} -> latent {cube.latent_dim}; '
          f'slice_mode={cube.slice_mode} (index {cube.slice_index} at eval)')
    print(f'[build] stm2d: trainable stages {cube.trainable_stages}/{cube.n_stages}, '
          f'head {n_head / 1e6:.2f} M. NOTE the head is not budget-matched across '
          f'backbones -- num_features differs (768 for CLIP ViT-B, 2048 for ResNet-50).')
    if in_channels > 1:
        print(f'[build] stm2d: in_channels={in_channels}, so the ENCODER selects the '
              f'slice. Prefer --z-slice on the loader: chunks are (1,H,W,1), so reading '
              f'one slice is 11x cheaper (MEASURED 0.60 ms vs 6.58 ms).')
    else:
        print('[build] stm2d: 13 of 14 z-slices are discarded by construction. PC1 of the '
              'stack is 99.98% of variance (MEASURED), so that is the trade this arm '
              'exists to test -- not an oversight.')
    return cube


# --- the reconstruction model this plugin provides ---
@MODELS.register('stm_autoencoder')
def build_stm_autoencoder(cfg):
    from .models.autoencoder import StmAutoencoder
    return StmAutoencoder.from_config(cfg)


# --- the readers this plugin provides ---
@DATASETS.register('stm_pairs')
def build_stm_pairs(*a, **kw):
    from .datasets.stm_pairs import StmPairDataset
    return StmPairDataset(*a, **kw)


@DATASETS.register('stm_cubes')
def build_stm_cubes(h5_path, channels_key=None, normalize=True, indices=None,
                    keep_channels=None, **kw):
    # The reconstruction loop calls every reader with the TERS reader's keywords. channels_key
    # and keep_channels are spectral-channel concepts and have no STM meaning -- a z-stack's
    # 14 slices are an axis, not a selectable channel set -- so they are accepted and ignored
    # rather than raising on a config that carries them from base.yaml.
    from .datasets.stm_cubes import StmCubeDataset
    return StmCubeDataset(h5_path, normalize=normalize, indices=indices, **kw)


@PAIRS.register('stm_pairs')
def stm_pairs(h5, args, indices=None, limit=None):
    from .datasets.collate import collate_stm
    from .datasets.stm_pairs import StmPairDataset
    # z_slice and seed are STM's alone -- a z-stack has a z axis to draw from and a TERS cube
    # does not, which is why the contrastive loop cannot build this dataset itself.
    ds = StmPairDataset(h5, indices=indices, limit=limit,
                        z_slice=getattr(args, 'z_slice', None),
                        seed=getattr(args, 'seed', 0))
    ds.collate = collate_stm
    return ds
