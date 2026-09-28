class Registry:

    def __init__(self, what, doc):
        self.what = what          # 'cube encoder' -- appears in the error messages
        self.doc = doc            # where to read about adding one
        self._fns = {}

    def register(self, name, fn=None):
        """Decorator, or ``ENCODERS.register('stm3d', fn)`` from outside.

        The plain-call form is for a class that must stay byte-identical to its export and so
        cannot carry a decorator.
        """
        def _add(f):
            if name in self._fns:
                prev = self._fns[name]
                raise ValueError(
                    f'{self.what} {name!r} is already registered to '
                    f'{prev.__module__}.{prev.__qualname__}; two builders under one name means '
                    f'the config silently selects whichever module imported last')
            self._fns[name] = f
            return f
        return _add if fn is None else _add(fn)

    def build(self, name, *a, **kw):
        if name not in self._fns:
            raise ValueError(
                f'unknown {self.what} {name!r}. Registered: {sorted(self._fns)}. '
                f'To add one, see {self.doc}.')
        return self._fns[name](*a, **kw)

    def names(self):
        """Insertion-ordered, so argparse ``choices=`` and ``--help`` stay stable."""
        return tuple(self._fns)

    def __contains__(self, name):
        return name in self._fns


#: Plugin packages, imported by name in load_plugins().
PLUGINS = ('stm', 'ters')

#: Cube -> vector. Entries satisfy ENCODER_CONTRACT.
ENCODERS = Registry('cube encoder', '')

#: Whole models. Kept apart from ENCODERS so the wrong kind fails at lookup.
MODELS = Registry('model', '')

#: Readers. Entries take the reader's own arguments.
DATASETS = Registry('dataset', '')

#: Paired readers. Fixed signature, each returns a dataset carrying .collate.
PAIRS = Registry('pair dataset', '')


#: Checked by tests/test_registry.py. See docs/15-adding-a-model.md.
ENCODER_CONTRACT = {
    'latent_dim': 'int -- width of the exported vector; ClipModel sizes its head from it',
    'output_dim': 'int, == latent_dim -- the name ters-2-graph reads off the exported encoder',
    'out_norm': 'nn.LayerNorm(latent_dim), identical across all four arms -- the contrastive '
                'loss is cosine, so an unnormalised code lowers it by moving magnitude '
                'rather than direction',
    'forward': '(B, C, H, W) -> (B, latent_dim)',
    'reset_parameters': 'callable, and it must drop pretrained weights too -- '
                        'tools/linear_probe.py builds its random-init control with it, and a '
                        'reset that kept the trunk would report a control that is not one',
}


def pick(table, key, what, where):
    """Look up an option table, with the error argparse ``choices=`` used to give.

    STM2D_BACKBONES and SGPT_SIZES are ``name -> measured kwargs`` (num_features, params_m,
    grid_tied, native_size). They stay beside their measurements instead of being copied into
    a config schema, and every read comes through here.
    """
    if key not in table:
        raise ValueError(f'{where}: unknown {what} {key!r}. Available: {sorted(table)}')
    return table[key]


def batch_requires(batch, keys, *, consumer, dataset_name=''):
    """Raise unless ``batch`` carries every key in ``keys``.

    Each modality declares its own batch schema and nothing forces the two to agree, so the
    consumer states what it reads and checks the first batch. ``consumer`` and
    ``dataset_name`` are named in the error: a bare KeyError four frames into a loop does not
    identify the mis-wired arm.
    """
    missing = [k for k in keys if k not in batch]
    if not missing:
        return batch
    src = f' produced by dataset {dataset_name!r}' if dataset_name else ''
    raise ValueError(
        f'{consumer} needs {sorted(missing)} in each batch, and the batch{src} has '
        f'{sorted(batch)}.\n'
        f'Each modality declares its own schema and nothing forces the two to agree, so this '
        f'is what a drift between them looks like. Either the reader should emit '
        f'{sorted(missing)}, or this arm should not be run against that dataset.')


def load_plugins():
    """Import every plugin so its registrations run. Idempotent.

    Registration lives in the plugin, not here: adding a modality must not mean editing a
    central list. The heavy third-party imports (timm, torch_scatter) sit inside the builder
    bodies, so this stays cheap and ``import shared.registry`` survives a node missing an
    optional dependency.
    """
    import importlib
    missing = []
    for name in PLUGINS:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as e:
            # An absent modality is not an error -- either folder can be deleted or left
            # unshipped. A plugin that is present but raises on its own dependency re-raises:
            # that is a broken install, and registering nothing would resurface later as
            # "unknown encoder".
            if e.name and not e.name.split('.')[0] == name:
                raise
            missing.append(name)
    if missing:
        print(f'[registry] plugins not present, skipped: {sorted(missing)}')
    return tuple(n for n in PLUGINS if n not in missing)
