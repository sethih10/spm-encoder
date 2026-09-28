"""STM z-stack -> one 256-d vector, through a SpectralGPT-shaped 3-D ViT.

Sources:
  * ``SpectralGptCubeEncoder`` -- ADAPT of the architecture in SpectralGPT, Hong et al.,
    IEEE TPAMI 2024, https://arxiv.org/abs/2311.07113 . The published weights are
    loaded from https://zenodo.org/records/8412455 ; that repository is GPLv3, so no
    code of it is used -- the blocks are written from the paper and the checkpoint.
  * trunk widths -- DeiT, Touvron et al., https://arxiv.org/abs/2012.12877 Table 1
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _largest_divisor_at_most(n, k):
    """The largest divisor of ``n`` that is ``<= k``; 1 if there is none.

    Used by ``band_mode='native'``, which keeps every band and adapts the tubelet depth to one
    that divides the band count.
    """
    for d in range(min(k, n), 0, -1):
        if n % d == 0:
            return d
    return 1


# SpectralGPT+, read from the authors' released configuration.  Named rather than inlined so
# that a run that deviates from the pretrained shape is visible in the diff.
SGPT_BASE = dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 patch_size=8, t_patch_size=3, n_bands=12, image_size=128)

# --- NEW (sgpt): trunk widths for the from-scratch arms ---
# Only ``base`` matches the released checkpoint's shape, and only ``base`` may be combined
# with --sgpt-ckpt; the others train the same architecture from scratch on this corpus.
# ``small`` and ``tiny`` are DeiT's standard widths.
SGPT_SIZES = {
    'base':  dict(embed_dim=768, depth=12, num_heads=12),   # 85.6 M -- SpectralGPT+
    'small': dict(embed_dim=384, depth=12, num_heads=6),    # ~21.8 M
    'tiny':  dict(embed_dim=192, depth=12, num_heads=3),    # ~5.6 M -- nearest to stm3d's 1.31 M
}


class Mlp(nn.Module):
    """The ViT feed-forward block.  ``fc1``/``fc2`` are named to match the checkpoint."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """Multi-head self-attention with separate ``q``, ``k``, ``v`` projections.

    Separate because the checkpoint stores them that way: SpectralGPT+.pth has
    ``blocks.N.attn.{q,k,v}.{weight,bias}`` as three tensors each, not the fused ``qkv`` that
    timm and MAE use. Assuming the fused layout leaves every attention block at random init,
    with no crash and no warning.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'embed_dim {dim} must be divisible by num_heads {num_heads}')
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape

        def to_heads(t):
            return t.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled_dot_product_attention rather than an explicit softmax: it is the memory-
        # efficient kernel, and at 1024 tokens x batch 256 the explicit (N, N) matrix is the
        # difference between fitting on the GH200 and not.
        x = F.scaled_dot_product_attention(to_heads(self.q(x)), to_heads(self.k(x)),
                                           to_heads(self.v(x)))
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class Block(nn.Module):
    """Pre-norm transformer block.  Pre-norm, not post-norm: that is what ViT and MAE use."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# --- NEW (sgpt): tubelet embedding ---
# Rejected: a 2-D PatchEmbed over the 14 slices as channels.  That is a 1x1 mix over the
# ordered axis in layer 1 -- the early crush this encoder exists to avoid.
class PatchEmbed3d(nn.Module):
    """(B, 1, Z, H, W) -> (B, T*h*w, D), with ``T = Z/t``, ``h = H/p``, ``w = W/p``."""

    def __init__(self, embed_dim, patch_size, t_patch_size, in_chans=1):
        super().__init__()
        self.patch_size, self.t_patch_size = patch_size, t_patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(t_patch_size, patch_size, patch_size),
                              stride=(t_patch_size, patch_size, patch_size))

    def forward(self, x):
        x = self.proj(x)                      # (B, D, T, h, w)
        self.grid = x.shape[2:]               # kept so the caller can shape position embeddings
        return x.flatten(2).transpose(1, 2)   # (B, T*h*w, D)


class SpectralGptCubeEncoder(nn.Module):
    """(B, Z, H, W) -> (B, latent_dim). Interface-identical to ``StmCubeEncoder``.

    Parameters
    ----------
    in_channels : int
        z-slices in the data. 14 for this corpus.
    image_size : int
        Spatial size of the cube. 128 here, which is exactly SpectralGPT+'s.
    latent_dim : int
        Output width. 256, the width ``cond_lin`` expects.
    band_mode : {'interpolate', 'pad', 'native'}
        How 14 z-slices are made to fit a ``t_patch_size`` tubelet. None is free; the choice
        is the experiment.

        interpolate  resample 14 -> 12; the only mode using every pretrained tensor verbatim
        pad          14 -> 15; one tubelet in five sees a z-step of zero
        native       keeps 14, resamples the pretrained tubelet kernel; ~2x the compute
    pool : {'mean', 'cls'}
        ``mean`` in practice: the released checkpoint has no class token, so ``'cls'`` means a
        randomly-initialised one.
    cls_embed : bool
        False, to match the checkpoint -- SpectralGPT does not inherit MAE's class token.
    trainable_blocks : int
        0 freezes the trunk and trains only the head (the LiT argument, Zhai et al.,
        https://arxiv.org/abs/2111.07991); -1 trains everything; N trains the last N blocks.
    """

    def __init__(self, in_channels=14, image_size=128, latent_dim=256,
                 embed_dim=SGPT_BASE['embed_dim'], depth=SGPT_BASE['depth'],
                 num_heads=SGPT_BASE['num_heads'], mlp_ratio=SGPT_BASE['mlp_ratio'],
                 patch_size=SGPT_BASE['patch_size'], t_patch_size=SGPT_BASE['t_patch_size'],
                 n_bands=SGPT_BASE['n_bands'], band_mode='interpolate', pool='mean',
                 sep_pos_embed=True, cls_embed=False, trainable_blocks=0):
        super().__init__()
        if band_mode not in ('interpolate', 'pad', 'native'):
            raise ValueError(f"band_mode must be 'interpolate', 'pad' or 'native', "
                             f'got {band_mode!r}')
        if pool not in ('mean', 'cls'):
            raise ValueError(f"pool must be 'mean' or 'cls', got {pool!r}")
        if pool == 'cls' and not cls_embed:
            raise ValueError("pool='cls' needs cls_embed=True")
        if image_size % patch_size != 0:
            raise ValueError(f'image_size {image_size} must be divisible by patch_size '
                             f'{patch_size}')

        # --- resolve the band axis, which is the whole design problem here ---
        if band_mode == 'interpolate':
            z_in = n_bands
        elif band_mode == 'pad':
            z_in = int(math.ceil(in_channels / t_patch_size) * t_patch_size)
        else:                                                     # 'native'
            z_in = in_channels
            t_patch_size = _largest_divisor_at_most(in_channels, t_patch_size)
        if z_in % t_patch_size != 0:
            raise ValueError(f'band_mode={band_mode!r} produced z={z_in}, which '
                             f't_patch_size={t_patch_size} does not divide')

        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.output_dim = latent_dim          # the attribute train_clip.py's build() reads
        self.band_mode = band_mode
        self.z_in = z_in
        self.t_patch_size = t_patch_size
        self.patch_size = patch_size
        self.pool = pool
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed

        self.t_grid = z_in // t_patch_size
        self.hw_grid = image_size // patch_size
        self.num_patches = self.t_grid * self.hw_grid * self.hw_grid

        self.patch_embed = PatchEmbed3d(embed_dim, patch_size, t_patch_size)

        # --- position embeddings, named to match the checkpoint ---
        # SpectralGPT inherits MAE-st's separable form: a spatial grid plus a per-band-group
        # vector, added by broadcast.  That is lucky here -- changing the number of band groups
        # only ever touches a (1, T, D) tensor, not the full 1024-token grid.
        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.hw_grid * self.hw_grid, embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.t_grid, embed_dim))
            if cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            n = self.num_patches + (1 if cls_embed else 0)
            self.pos_embed = nn.Parameter(torch.zeros(1, n, embed_dim))
        if cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # The same ``Linear -> LayerNorm`` head as the other cube encoders, so a difference
        # between arms is attributable to the trunk.  The contrastive loss is cosine, so an
        # unnormalised code lets the model reduce the loss by moving magnitude, not direction.
        self.head = nn.Linear(embed_dim, latent_dim)
        self.out_norm = nn.LayerNorm(latent_dim)

        self.reset_parameters()
        self.set_trainable_blocks(trainable_blocks)

    def reset_parameters(self):
        for p in (getattr(self, n, None) for n in
                  ('pos_embed', 'pos_embed_spatial', 'pos_embed_temporal',
                   'pos_embed_class', 'cls_token')):
            if p is not None:
                nn.init.trunc_normal_(p, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def set_trainable_blocks(self, n):
        """Freeze the trunk except the last ``n`` blocks.  ``0`` = head only, ``-1`` = all."""
        trainable_all = (n < 0)
        for p in self.parameters():
            p.requires_grad = trainable_all
        for p in list(self.head.parameters()) + list(self.out_norm.parameters()):
            p.requires_grad = True
        if not trainable_all and n > 0:
            for blk in self.blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.norm.parameters():
                p.requires_grad = True
        self.trainable_blocks = n

    def _positions(self, x):
        """Add the position embedding, prepending the class token if there is one."""
        b = x.shape[0]
        if self.sep_pos_embed:
            # spatial (1, hw, D) tiled over T; temporal (1, T, D) repeated over hw.
            pos = (self.pos_embed_spatial.repeat(1, self.t_grid, 1)
                   + torch.repeat_interleave(self.pos_embed_temporal,
                                             self.hw_grid * self.hw_grid, dim=1))
            x = x + pos
            if self.cls_embed:
                cls = (self.cls_token + self.pos_embed_class).expand(b, -1, -1)
                x = torch.cat([cls, x], dim=1)
        else:
            if self.cls_embed:
                x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)
            x = x + self.pos_embed
        return x

    def _fit_bands(self, x):
        """(B, Z, H, W) -> (B, z_in, H, W) under the chosen ``band_mode``."""
        z = x.shape[1]
        if z == self.z_in:
            return x
        if self.band_mode == 'interpolate':
            # Linear along z only.  align_corners=False maps output i to input coordinate
            # (i+0.5)*scale-0.5, so with H and W unchanged those axes are an exact identity.
            return F.interpolate(x.unsqueeze(1),
                                 size=(self.z_in, x.shape[2], x.shape[3]),
                                 mode='trilinear', align_corners=False).squeeze(1)
        if self.band_mode == 'pad':
            return F.pad(x, (0, 0, 0, 0, 0, self.z_in - z), mode='replicate')
        raise ValueError(f'band_mode={self.band_mode!r} expects {self.z_in} slices, got {z}')

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f'SpectralGptCubeEncoder expects (B, Z, H, W), '
                             f'got {tuple(x.shape)}')
        if x.size(1) != self.in_channels:
            raise ValueError(f'SpectralGptCubeEncoder built for in_channels='
                             f'{self.in_channels} z-slices but got {x.size(1)}')
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            raise ValueError(f'built for {self.image_size}x{self.image_size} cubes, got '
                             f'{tuple(x.shape[-2:])}; the position embedding is tied to that '
                             f'grid, so this is not resizable at call time')

        x = self._fit_bands(x).unsqueeze(1)          # (B, 1, z_in, H, W)
        x = self.patch_embed(x)                      # (B, num_patches, D)
        x = self._positions(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pooled = x[:, 0] if self.pool == 'cls' else x[:, (1 if self.cls_embed else 0):].mean(1)
        return self.out_norm(self.head(pooled))


def strip_prefix(k):
    """Drop the wrapper prefixes checkpoints pick up (DDP, MAE, timm-style nesting).

    Shared with ``tools.inspect_sgpt_ckpt``: when the inspector and the loader
    normalise names differently, the inspector reports ABSENT for tensors the loader loads
    fine, and the diagnostic that exists to prevent a silent failure becomes one.
    """
    for prefix in ('module.', 'encoder.', 'backbone.', 'model.'):
        if k.startswith(prefix):
            k = k[len(prefix):]
    return k


# --- NEW (sgpt): the loader ---
def load_spectralgpt_weights(model, path, min_match=0.9, verbose=True):
    """Load SpectralGPT+ weights into ``model``, and verify enough of them landed.

    Resamples the tubelet kernel and the temporal position embedding when the band mode changed
    their shape, then raises if the matched fraction falls below ``min_match`` -- a partially
    loaded trunk otherwise trains and gets reported as pretrained.
    See docs/02-code-practices.md 5.1.
    """
    obj = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('model', 'state_dict', 'module'):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict):
        raise ValueError(f'{path}: expected a state dict, got {type(obj).__name__}')

    src = {}
    for k, v in obj.items():
        if not torch.is_tensor(v):
            continue
        k = strip_prefix(k)
        if k.startswith('decoder') or k.startswith('mask_token'):
            continue                      # the MAE decoder, not the encoder
        src[k] = v

    tgt = model.state_dict()
    adapted = []
    for k, v in list(src.items()):
        if k not in tgt:
            continue
        want = tgt[k].shape
        if v.shape == want:
            continue
        if k == 'patch_embed.proj.weight' and v.dim() == 5:
            # (D, 1, t, p, p): resample the tubelet depth when band_mode changed it.
            v = F.interpolate(v.permute(0, 1, 3, 4, 2).reshape(-1, 1, v.shape[2]),
                              size=want[2], mode='linear', align_corners=False)
            v = v.reshape(want[0], want[1], want[3], want[4], want[2]).permute(0, 1, 4, 2, 3)
            src[k], adapted = v.contiguous(), adapted + [f'{k} depth->{want[2]}']
        elif k == 'pos_embed_temporal' and v.dim() == 3:
            v = F.interpolate(v.transpose(1, 2), size=want[1], mode='linear',
                              align_corners=False).transpose(1, 2)
            src[k], adapted = v.contiguous(), adapted + [f'{k} T->{want[1]}']
        elif k in ('pos_embed_spatial', 'pos_embed') and v.dim() == 3:
            g_src, g_tgt = int(round(v.shape[1] ** 0.5)), int(round(want[1] ** 0.5))
            if g_src * g_src == v.shape[1] and g_tgt * g_tgt == want[1]:
                v = F.interpolate(
                    v.reshape(1, g_src, g_src, -1).permute(0, 3, 1, 2),
                    size=(g_tgt, g_tgt), mode='bicubic', align_corners=False)
                src[k] = v.permute(0, 2, 3, 1).reshape(1, want[1], -1).contiguous()
                adapted.append(f'{k} grid {g_src}->{g_tgt}')

    src = {k: v for k, v in src.items() if k in tgt and v.shape == tgt[k].shape}
    missing, unexpected = model.load_state_dict(src, strict=False)

    ours = {'head.weight', 'head.bias', 'out_norm.weight', 'out_norm.bias'}
    trunk = [k for k in tgt if k not in ours]
    matched = [k for k in trunk if k in src]
    frac = len(matched) / max(1, len(trunk))

    if verbose:
        print(f'[sgpt] {path}')
        print(f'[sgpt] loaded {len(matched)}/{len(trunk)} trunk tensors ({frac:.1%}), '
              f'{len(ours)} head tensors are ours and stay random')
        for a in adapted:
            print(f'[sgpt]   ADAPTED {a}')
        skipped = [k for k in trunk if k not in src]
        if skipped:
            print(f'[sgpt]   NOT LOADED ({len(skipped)}): {skipped[:6]}'
                  f'{" ..." if len(skipped) > 6 else ""}')
        extra = [k for k in src if k not in tgt][:6]
        if extra:
            print(f'[sgpt]   checkpoint keys with no home here: {extra} ...')
    if frac < min_match:
        raise RuntimeError(
            f'{path}: only {len(matched)}/{len(trunk)} ({frac:.1%}) of the trunk loaded, below '
            f'min_match={min_match:.0%}. This would train a mostly-random ViT-Base and report '
            f'it as pretrained. Run `python -m tools.inspect_sgpt_ckpt {path}` to see the '
            f'checkpoint\'s actual key names and shapes, then fix the mapping.')
    return dict(matched=len(matched), trunk=len(trunk), fraction=frac, adapted=adapted,
                missing=list(missing), unexpected=list(unexpected))
