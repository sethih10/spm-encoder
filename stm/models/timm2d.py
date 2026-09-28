"""One STM z-slice -> one 256-d vector, through a pretrained 2-D vision backbone.

Sources:
  * the trunk -- REUSE of timm (Apache-2.0) through its public factory,
    https://github.com/huggingface/pytorch-image-models
  * the single-random-slice protocol -- ADAPT of ASD-STM, Kurki, Oinonen & Foster,
    ACS Nano 18:11130-11138 (2024), https://arxiv.org/abs/2312.08854 ; random while
    training, fixed at eval
  * the backbone choice -- ADAPT of the ranking measured in UniMR, Cao et al.,
    Adv. Sci. 13(11) 2026, https://doi.org/10.1002/advs.202516428 ; their best and worst
    STM backbones are the transfer arm and the control here
  * frozen-trunk / trainable-block convention -- LiT, Zhai et al.,
    https://arxiv.org/abs/2111.07991
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Supported backbones. Numbers MEASURED against timm 1.0.27 at 128x128, num_classes=0.
# ``grid_tied``: VisionTransformer.__init__ takes img_size, ResNet.__init__ raises on it.
# The ``quickgelu`` in the CLIP tag is load-bearing -- the plain tag resolves to the same
# weights but runs standard GELU, with no crash and nothing min_match can see.
STM2D_BACKBONES = {
    'clip_vit_b16': dict(timm_name='vit_base_patch16_clip_quickgelu_224.openai',
                         num_features=768, params_m=85.30, grid_tied=True, native_size=224),
    'vit_small':    dict(timm_name='vit_small_patch16_224.augreg_in21k_ft_in1k',
                         num_features=384, params_m=21.42, grid_tied=True, native_size=224),
    # the control: UniMR measured this at F1 0.72 against CLIP's 0.93 on STM.
    'resnet50':     dict(timm_name='resnet50.a1_in1k',
                         num_features=2048, params_m=23.50, grid_tied=False, native_size=224),
    'resnet18':     dict(timm_name='resnet18.a1_in1k',
                         num_features=512, params_m=11.17, grid_tied=False, native_size=224),
}


def load_timm_backbone(timm_name, in_chans=3, img_size=None, num_classes=0,
                       pretrained=True, ckpt='', min_match=0.98, seed=0, verbose=True):
    """Build a timm backbone and verify the pretrained weights actually loaded.

    timm reports a failed load only at ``INFO`` and silently falls back to ``strict=False``, so a
    typo'd tag or a node with no egress yields a randomly-initialised model that trains happily.
    Verified by diffing against a ``pretrained=False`` twin built under the same seed; only
    ``ndim >= 2`` tensors count. Raises below ``min_match``. See docs/02-code-practices.md 5.1.

    Returns
    -------
    (model, report) where report carries the fraction, the still-random keys, and the source.
    """
    import timm                    # lazy: importing this module must not require timm

    kwargs = dict(num_classes=num_classes, in_chans=in_chans)
    if img_size is not None:
        kwargs['img_size'] = img_size          # ResNet.__init__ has no such parameter

    # train_clip.py seeds once before build(), and every arm must see the same data order at the
    # same --seed or the arms are not comparable.  So borrow the stream and put it back.
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        ref = timm.create_model(timm_name, pretrained=False, **kwargs)
        if not pretrained:
            if verbose:
                print('[stm2d] NOTE: pretrained=False, so this backbone is RANDOM INIT. That '
                      'is the from-scratch control, not the transfer arm.')
            return ref, dict(fraction=0.0, matched=0, trunk=0, missed=[],
                             source='random-init', pretrained=False)
        torch.manual_seed(seed)
        overlay = dict(file=ckpt) if ckpt else None
        try:
            model = timm.create_model(timm_name, pretrained=True,
                                      pretrained_cfg_overlay=overlay, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f'{timm_name}: could not obtain pretrained weights '
                f'({type(e).__name__}: {e}).\n'
                f'This arm is named for a pretrained backbone and will not fall back to random '
                f'init.\n'
                f'MEASURED 2026-09-16: ~/.cache/huggingface did not exist, and compute nodes '
                f'here may have no egress. Fix, on the login node:\n'
                f'    python -m tools.prefetch_stm2d --backbone <arm>\n'
                f'then pass the path it prints as --stm2d-ckpt.') from e
    finally:
        torch.set_rng_state(rng_state)

    r, m = ref.state_dict(), model.state_dict()
    cand = [k for k, v in r.items() if v.ndim >= 2 and v.numel() > 1]
    loaded = [k for k in cand if not torch.equal(r[k].float(), m[k].float())]
    missed = [k for k in cand if k not in set(loaded)]
    frac = len(loaded) / max(1, len(cand))
    cfg = getattr(model, 'pretrained_cfg', {}) or {}
    src = ckpt or cfg.get('hf_hub_id') or cfg.get('url') or '?'

    if verbose:
        print(f'[stm2d] {timm_name}  <- {src}')
        print(f'[stm2d] {len(loaded)}/{len(cand)} weight tensors changed by the load '
              f'({frac:.1%}); our head/out_norm are separate and stay random')
        if missed:
            print(f'[stm2d]   UNCHANGED, i.e. still at random init ({len(missed)}): '
                  f'{missed[:6]}{" ..." if len(missed) > 6 else ""}')
        if hasattr(model, 'blocks') and hasattr(model.blocks[0].mlp, 'act'):
            act = type(model.blocks[0].mlp.act).__name__
            print(f'[stm2d]   activation={act}'
                  + ('' if 'clip' not in timm_name else
                     '  (the *_clip_*.openai tags REQUIRE QuickGELU; plain GELU here would be '
                     'the wrong non-linearity on the right weights, and min_match cannot see '
                     'it)'))
    if frac < min_match:
        raise RuntimeError(
            f'{timm_name}: only {len(loaded)}/{len(cand)} ({frac:.1%}) of the weight tensors '
            f'were changed by the pretrained load, below min_match={min_match:.0%}.\n'
            f'The rest are still at random init, and this arm would be reported as '
            f'"pretrained {timm_name}".\n'
            f'Still random: {missed[:10]}\n'
            f'Run `python -m tools.inspect_stm2d_backbone {timm_name} --in-chans {in_chans}` '
            f'to see which keys timm dropped, and why.')
    del ref
    return model, dict(fraction=frac, matched=len(loaded), trunk=len(cand), missed=missed,
                       source=str(src), pretrained=True)


class Stm2dCubeEncoder(nn.Module):
    """(B, Z, H, W) -> (B, latent_dim), through a pretrained 2-D backbone on one slice.

    Interface-identical to the other cube encoders -- same ``latent_dim`` attribute, same
    ``Linear -> LayerNorm`` head -- so a difference between arms is attributable to the trunk.

    Parameters
    ----------
    in_channels : int
        z-slices arriving from the loader. 1 when the reader already sliced (the normal path);
        14 when it did not, in which case ``slice_mode`` selects here instead.
    resize : int
        Spatial size fed to the trunk. The backbone's native 224 uses every pretrained tensor
        verbatim; the cube's own 128 makes timm resample the position grid. Neither is free, so
        it is exposed rather than picked silently.
    in_chans : int
        1 or 3. At 1, timm sums the pretrained RGB stem, which is identical to replicating a grey
        image to 3 channels -- a third of the stem FLOPs, not a modelling trade.
    trainable_blocks : int
        0 = head only (the LiT convention), -1 = everything,
        N = the last N. N counts ViT blocks but ResNet stages, so it means different things across
        backbones; ``build()`` prints the resolved count.
    """

    def __init__(self, in_channels=14, image_size=128, latent_dim=256,
                 backbone='clip_vit_b16', timm_name='', in_chans=3, resize=0,
                 slice_mode='random', slice_index=0, trainable_blocks=0,
                 pretrained=True, ckpt='', min_match=0.98, seed=0, verbose=True):
        super().__init__()
        if backbone in STM2D_BACKBONES:
            cfg = dict(STM2D_BACKBONES[backbone])
        elif timm_name:
            cfg = dict(timm_name=timm_name, num_features=None, grid_tied=None, native_size=224,
                       params_m=None)
            if verbose:
                print(f'[stm2d] NOTE: {timm_name!r} is not in STM2D_BACKBONES, so its shape '
                      f'and size are unmeasured here. grid_tied is probed at build time.')
        else:
            raise ValueError(f'unknown backbone {backbone!r}; known: '
                             f'{sorted(STM2D_BACKBONES)}. Or pass timm_name= explicitly.')
        if timm_name and backbone in STM2D_BACKBONES and timm_name != cfg['timm_name']:
            raise ValueError(
                f'backbone={backbone!r} means {cfg["timm_name"]!r} but timm_name='
                f'{timm_name!r} was also given. Two names for one trunk -- pass one.')
        if in_chans not in (1, 3):
            raise ValueError(f'in_chans must be 1 or 3, got {in_chans}')
        if slice_mode not in ('random', 'fixed', 'passthrough'):
            raise ValueError(f'slice_mode must be random|fixed|passthrough, got {slice_mode!r}')
        if slice_index < 0:
            raise ValueError(f'slice_index must be >= 0, got {slice_index}')

        resize = int(resize) or int(cfg['native_size'])
        if resize not in (int(image_size), int(cfg['native_size'])):
            raise ValueError(
                f'resize must be {image_size} (the cube size, position prior resampled) or '
                f'{cfg["native_size"]} (the backbone native size, every pretrained tensor '
                f'verbatim); got {resize}')

        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.latent_dim = int(latent_dim)
        self.output_dim = int(latent_dim)        # the attribute build()/downstream code reads
        self.backbone_name = backbone
        self.timm_name = cfg['timm_name']
        self.in_chans = int(in_chans)
        self.resize = resize
        self.slice_mode = slice_mode
        self.slice_index = int(slice_index)

        grid_tied = cfg['grid_tied']
        if grid_tied is None:                    # unregistered name: probed rather than assumed
            import timm
            try:
                timm.create_model(self.timm_name, pretrained=False, num_classes=0,
                                  in_chans=self.in_chans, img_size=resize)
                grid_tied = True
            except TypeError:
                grid_tied = False
            if verbose:
                print(f'[stm2d] probed {self.timm_name}: grid_tied={grid_tied}')
        self.grid_tied = bool(grid_tied)

        self.backbone, self.load_report = load_timm_backbone(
            self.timm_name, in_chans=self.in_chans,
            img_size=(resize if self.grid_tied else None), num_classes=0,
            pretrained=pretrained, ckpt=ckpt, min_match=min_match, seed=seed, verbose=verbose)

        feat = int(self.backbone.num_features)
        if cfg['num_features'] and feat != cfg['num_features']:
            raise RuntimeError(
                f'{self.timm_name} reports num_features={feat} but STM2D_BACKBONES records '
                f'{cfg["num_features"]}. timm changed under us -- re-measure before trusting '
                f'this arm.')
        self.feat_dim = feat

        # The contrastive loss is cosine, so an unnormalised code lets the model lower the loss
        # by moving magnitude rather than direction.  Identical across all four arms.
        self.head = nn.Linear(feat, latent_dim)
        self.out_norm = nn.LayerNorm(latent_dim)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.ones_(self.out_norm.weight)
        nn.init.zeros_(self.out_norm.bias)

        # --- NEW (stm2d): 2 parameters, identity init, always trainable. ---
        # At trainable_blocks=0 the stem is frozen, so nothing in the trunk can adapt to this
        # input scale.  Rejected: the backbone's own ImageNet/CLIP mean/std -- a cube that has
        # been through NormalizeStmCube is asinh'd and standardised, not in [0,1], so those
        # constants are not the right affine.  Identity init means step 0 is off-the-shelf.
        self.in_gain = nn.Parameter(torch.ones(1))
        self.in_bias = nn.Parameter(torch.zeros(1))

        self.seed = int(seed)                    # recorded for provenance; see load_report
        self.set_trainable_blocks(trainable_blocks)

    # --- NEW (control): the random-init control arm ---
    def reset_parameters(self):
        """Discard all learned weights, pretrained trunk included. Same architecture, untrained.

        The trunk is re-randomised too: resetting only the head would answer a different
        question and report a much higher control.

        Draws from the global RNG stream, as the other encoders' resets do, so two calls give
        different weights and a run seeded once at the top is still fully reproducible.

        Rejected: rebuilding the trunk at the fixed ``self.seed``. It looks more deterministic
        but is worse on both counts -- it makes this the only encoder whose reset ignores the
        stream, and on an arm built with ``pretrained=False`` it reproduces the current weights
        bit-for-bit, i.e. the "control" silently becomes a no-op.
        """
        device = self.head.weight.device
        # load_timm_backbone() restores the stream around its own build, so this one draw is
        # the only trunk-side RNG this method spends.
        trunk_seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        print(f'[stm2d] reset_parameters: DISCARDING the {self.load_report.get("source")} '
              f'trunk and re-initialising {self.timm_name} at random (seed {trunk_seed}). '
              f'This is the from-scratch control, not the transfer arm.')
        self.backbone, self.load_report = load_timm_backbone(
            self.timm_name, in_chans=self.in_chans,
            img_size=(self.resize if self.grid_tied else None), num_classes=0,
            pretrained=False, seed=trunk_seed, verbose=False)
        self.backbone.to(device)

        # Must stay identical to the head/out_norm/affine init in __init__.
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.ones_(self.out_norm.weight)
        nn.init.zeros_(self.out_norm.bias)
        with torch.no_grad():
            self.in_gain.fill_(1.0)
            self.in_bias.zero_()

        # Freezing is a property of the arm, not of the weights, so it is re-applied -- and it
        # must be, because the new trunk's parameters all default to requires_grad=True.
        self.set_trainable_blocks(self.trainable_blocks)
        return self

    # --- freezing ---
    def _stage_modules(self):
        """The trunk's stages, outermost last, so "the last N" is well defined.

        ``group_matcher(coarse=True)`` on timm 1.0.27 groups ViT as ``^blocks\\.(\\d+)`` and
        ResNet as ``^layer(\\d+)``; hence the two lookups below.
        """
        if hasattr(self.backbone, 'blocks'):
            return list(self.backbone.blocks)                      # ViT: 12
        stages = [getattr(self.backbone, f'layer{i}', None) for i in (1, 2, 3, 4)]
        stages = [s for s in stages if s is not None]
        if not stages:
            raise RuntimeError(
                f'{type(self.backbone).__name__}: cannot find stages to freeze. '
                f'group_matcher says {self.backbone.group_matcher(coarse=True)}')
        return stages                                              # ResNet: 4

    def _tail_modules(self):
        return [m for m in (getattr(self.backbone, 'norm', None),
                            getattr(self.backbone, 'fc_norm', None)) if m is not None]

    def set_trainable_blocks(self, n):
        """Freeze the trunk except the last ``n`` stages.  0 = head only, -1 = all.

        Rejected: ``timm.utils.model.freeze``, which is otherwise the right tool because it
        also stops BatchNorm running stats drifting on the ResNet arms.  It converts BN ->
        ``FrozenBatchNorm2d`` in place, which makes the state_dict key set a function of the
        freeze setting -- and tools/rescore.py loads state dicts strict.  ``train()`` below
        gets the same BN protection at no state_dict cost.
        """
        n = int(n)
        trainable_all = (n < 0)
        for p in self.parameters():
            p.requires_grad = trainable_all
        for p in list(self.head.parameters()) + list(self.out_norm.parameters()):
            p.requires_grad = True
        self.in_gain.requires_grad = True
        self.in_bias.requires_grad = True
        stages = self._stage_modules()
        if not trainable_all and n > 0:
            if n > len(stages):
                raise ValueError(
                    f'trainable_blocks={n} but {self.timm_name} exposes only {len(stages)} '
                    f'stages to unfreeze; -1 trains everything.')
            for mod in list(stages[-n:]) + self._tail_modules():
                for p in mod.parameters():
                    p.requires_grad = True
        self.trainable_blocks = n
        self.trainable_stages = len(stages) if trainable_all else n
        self.n_stages = len(stages)
        return self

    def train(self, mode=True):
        """A frozen trunk stays in eval mode, so its norms and dropout cannot drift.

        ``requires_grad=False`` is not enough on its own -- BatchNorm running statistics
        update in train mode regardless.
        """
        super().train(mode)
        if self.trainable_blocks == 0:
            self.backbone.eval()
        return self

    # --- forward ---
    def _select_slice(self, x):
        z = x.shape[1]
        if z == 1:
            return x                                       # the loader already sliced
        if self.slice_mode == 'passthrough':
            raise ValueError(
                f'slice_mode="passthrough" needs Z=1 but got Z={z}. Either set '
                f'--z-slice on the loader, or use --stm2d-slice-mode random|fixed.')
        if self.slice_mode == 'fixed' or not self.training:
            # Evaluation is always deterministic: train_clip.score selects the exported
            # checkpoint on top1_size_controlled, and a resampled val slice would make that
            # criterion noisy for a reason unrelated to the encoder.
            return x[:, min(self.slice_index, z - 1):min(self.slice_index, z - 1) + 1]
        idx = torch.randint(0, z, (x.shape[0], 1, 1, 1), device=x.device)
        return x.gather(1, idx.expand(-1, 1, x.shape[2], x.shape[3]))

    def _prepare(self, x):
        """Slice -> 1 or 3 channels -> the trunk's resolution -> the input affine."""
        x = self._select_slice(x)
        if self.in_chans == 3:
            x = x.expand(-1, 3, -1, -1)                    # stem used verbatim, no surgery
        if x.shape[-1] != self.resize or x.shape[-2] != self.resize:
            x = F.interpolate(x, size=(self.resize, self.resize), mode='bicubic',
                              align_corners=False, antialias=True)
        return x * self.in_gain + self.in_bias

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f'Stm2dCubeEncoder expects (B, Z, H, W), got {tuple(x.shape)}')
        if x.size(1) != self.in_channels:
            raise ValueError(
                f'Stm2dCubeEncoder built for in_channels={self.in_channels} z-slices but got '
                f'{x.size(1)}; the loader and the encoder must agree. Rebuild the encoder with '
                f'in_channels={x.size(1)}, or change --z-slice.')
        return self.out_norm(self.head(self.backbone(self._prepare(x))))


def _self_test():
    """Shapes, freezing and determinism, with no network and no data."""
    torch.manual_seed(0)
    cfg = dict(in_channels=14, image_size=32, latent_dim=16, backbone='resnet18',
               resize=32, pretrained=False, verbose=False)
    enc = Stm2dCubeEncoder(**cfg)
    x = torch.randn(3, 14, 32, 32)
    out = enc(x)
    print(f'  (3,14,32,32) -> {tuple(out.shape)}  feat_dim={enc.feat_dim}  '
          f'stages={enc.n_stages}  row-mean={float(out.detach().mean(dim=-1).abs().max()):.2e}')
    enc.eval()
    assert torch.equal(enc(x), enc(x)), 'eval must be deterministic'
    trainable = {n for n, p in enc.named_parameters() if p.requires_grad}
    print(f'  trainable at trainable_blocks=0: {sorted(trainable)}')
    one = Stm2dCubeEncoder(**{**cfg, 'in_channels': 1})
    print(f'  (2,1,32,32) -> {tuple(one(torch.randn(2, 1, 32, 32)).shape)}')
    print('  OK')


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list or not a.self_test:
        print(f'{"arm":16s} {"timm name":48s} {"feat":>6s} {"params":>9s}  grid_tied')
        for k, v in STM2D_BACKBONES.items():
            print(f'{k:16s} {v["timm_name"]:48s} {v["num_features"]:6d} '
                  f'{v["params_m"]:8.2f}M  {v["grid_tied"]}')
    if a.self_test:
        _self_test()
