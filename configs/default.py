"""Default configuration."""

from types import SimpleNamespace

def get_config():
    config = SimpleNamespace()

    # ---- data ---
    config.data = SimpleNamespace()
    config.data.root = '/scratch/project_2001912/planar_hdf5_0.5'
    # Disjoint {train,val,test}.h5. Only `channels_100` is stored, so this key is not really
    # a choice.
    config.data.channels_key = 'channels_100'
    config.data.train_split = 'train'
    config.data.val_split = 'val'
    config.data.image_size = 256         # MEASURED; the fixed-size decoder raises on a mismatch
    config.data.normalize = True         # raw values run 0 to ~2556; not optional in practice
    config.data.channel_mask = ''        # '' keeps all 100; else a configs/masks/*.json

    # MEASURED peak 221.6 cube/s, job 984449; loader-bound, so this sets the wall clock.
    # Coupled to batch_size -- 72 workers OOM'd.
    config.data.num_workers = 32
    # Cap the training set for a quick run.  0 = use all of it.
    config.data.max_train = 0
    config.data.max_val = 0

    # ---- model ---
    config.model = SimpleNamespace()
    config.model.in_channels = 100       # must equal TersCubeDataset.in_channels
    config.model.latent_dim = 256        # = ters-2-graph config.model.nf, the conditioning width
    config.model.latent_hw = 1           # 1 = the global vector; k>1 selects TersSpatialEncoder
    config.model.decoder_pad_blocks = 0  # capacity control for the latent_hw sweep
    config.model.filters = (64, 128, 256, 512)      # = ters-2-graph config.model.ters_filters
    config.model.kernel_size = (3, 3, 3, 3)
    config.model.n_groups = 8
    config.model.stem_channels = None    # None = filters[0]; setting it selects TersSpatialEncoder
    config.model.zero_init = True

    # ---- training ---
    config.training = SimpleNamespace()
    # MEASURED 40.0 s/epoch at 16 against 58.3 at 64, job 984517: smaller is faster, because
    # the run is loader-bound. Coupled to warmup_steps.
    config.training.batch_size = 16
    config.training.eval_batch_size = 16
    # MEASURED as the wrong budget, job 991661: val_mse bottoms at epoch 25 and degrades to
    # 0.2718 by 99. Left at 100 -- best.pt tracks val, and the curve is the overfitting evidence.
    config.training.n_epochs = 100
    config.training.lr = 1e-4            # not measured; carried-over default.
    # MEASURED false, job 991661, that the bottleneck regularises. Never swept.
    config.training.weight_decay = 0.0
    config.training.sam_weight = 0.0     # 0 = plain MSE
    config.training.grad_clip = 1.0
    # A fixed step count, so its share moves with batch_size: 1.3% at bs=16, 21% at bs=256.
    config.training.warmup_steps = 500
    # bf16 on GH200: no loss scaler needed, and unlike fp16 its exponent range covers the
    # ~1e-9 tails a standardised dead channel can produce.
    config.training.amp_dtype = 'bf16'   # 'bf16' | 'fp16' | 'off'
    config.training.log_every = 20
    config.training.eval_every = 1       # epochs
    config.training.ckpt_every = 5       # epochs
    config.training.seed = 0
    config.training.workdir = 'exp/default'
    config.training.max_steps = 0        # 0 = no cap; >0 stops early (smoke tests)

    return config


# --- NEW (cli): dotted-path overrides ---
# So a sweep is a flag, not an edited file: an edited config that gets committed makes every
# later run's provenance ambiguous. Rejected absl.flags, which wants to own argv parsing at
# import time.
def apply_overrides(config, assignments):
    """``apply_overrides(cfg, ['model.latent_dim=512'])``.

    Values are parsed with ``ast.literal_eval`` and fall back to the raw string.  An unknown
    key raises -- a typo that silently sets nothing is the failure mode worth preventing.
    """
    import ast

    for item in assignments or []:
        if '=' not in item:
            raise ValueError(f'--set expects key=value, got {item!r}')
        key, raw = item.split('=', 1)
        node = config
        parts = key.split('.')
        for p in parts[:-1]:
            if not hasattr(node, p):
                raise KeyError(f'unknown config section {p!r} in {key!r}')
            node = getattr(node, p)
        if not hasattr(node, parts[-1]):
            raise KeyError(
                f'unknown config key {key!r}. Section has: '
                f'{sorted(vars(node))}')
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        setattr(node, parts[-1], value)
    return config


def as_dict(config):
    """Nested ``SimpleNamespace`` -> nested ``dict``, for logging and checkpoints."""
    if isinstance(config, SimpleNamespace):
        return {k: as_dict(v) for k, v in vars(config).items()}
    if isinstance(config, tuple):
        return list(config)
    return config
