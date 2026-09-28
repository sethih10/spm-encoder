import ast
import copy
import os
from types import SimpleNamespace

import yaml

def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


_MAX_BASE_DEPTH = 4


def load_yaml(path, _seen=None):
    path = os.path.abspath(path)
    # A cycle would otherwise recurse until the stack dies, with a traceback naming neither
    # file. Report the ring instead.
    _seen = list(_seen or [])
    if path in _seen:
        ring = ' -> '.join(os.path.basename(p) for p in _seen + [path])
        raise ValueError(f'base: forms a cycle: {ring}')
    if len(_seen) >= _MAX_BASE_DEPTH:
        chain = ' -> '.join(os.path.basename(p) for p in _seen + [path])
        raise ValueError(
            f'base: is {len(_seen) + 1} deep, past the limit of {_MAX_BASE_DEPTH}: {chain}. '
            f'The base chain exceeds the depth this loader accepts.')
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f'{path}: a config must be a mapping, got {type(raw).__name__}')
    parent = raw.pop('base', None)
    if parent is None:
        return raw
    parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(path), parent)
    if not os.path.exists(parent_path):
        raise FileNotFoundError(f'{path}: base: {parent!r} does not exist ({parent_path})')
    return _merge(load_yaml(parent_path, _seen + [path]), raw)


def to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [to_namespace(v) for v in d]
    return d


def as_dict(config):
    if isinstance(config, SimpleNamespace):
        return {k: as_dict(v) for k, v in vars(config).items()}
    if isinstance(config, (list, tuple)):
        return [as_dict(v) for v in config]
    return config


def apply_overrides(config, assignments):
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
            raise KeyError(f'unknown config key {key!r}. Section has: {sorted(vars(node))}')
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        setattr(node, parts[-1], value)
    return config


def load_config(path, overrides=()):
    return apply_overrides(to_namespace(load_yaml(path)), overrides)


def dump_resolved(config, workdir, name='resolved_config.yaml'):
    if not workdir:
        return None
    os.makedirs(workdir, exist_ok=True)
    out = os.path.join(workdir, name)
    with open(out, 'w') as f:
        yaml.safe_dump(as_dict(config), f, sort_keys=True, default_flow_style=False)
    return out
