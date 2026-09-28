"""Contrastive training: aligns a cube embedding with that of the molecule that produced it.

InfoNCE, so retrieval between the two measures whether the cube identifies the structure.

Modality-agnostic. The cube encoder comes from ENCODERS and the paired reader from PAIRS,
both filled by whichever plugins are installed, so this file names neither stm nor ters.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.registry import ENCODERS, PAIRS, batch_requires, load_plugins
from shared.losses.contrastive import info_nce
from shared.models.projection import ProjectionHead
from shared.metrics.retrieval import contrastive_metrics, size_confound_metrics
from shared.losses.reconstruction import reconstruction_loss
from shared.metrics.reconstruction import baseline_losses
from shared.models.decoder import TersDecoder
from shared.models.structure import StructureEncoder

DEFAULT_CKPT = '/scratch/project_2001912/sethih1/checkpoints/denoised-pcqm4mv2.ckpt'

class ClipModel(nn.Module):
    """Cube encoder + structure encoder + a projection head on each side."""

    def __init__(self, cube_encoder, structure_encoder, proj_dim=128, decoder=None):
        super().__init__()
        self.cube = cube_encoder
        self.structure = structure_encoder
        self.cube_proj = ProjectionHead(cube_encoder.latent_dim, out_dim=proj_dim)
        self.struct_proj = ProjectionHead(structure_encoder.out_dim, out_dim=proj_dim)
        # Contrastive plus full reconstruction, with the contrastive weight a config value so
        # weight=0 reproduces ters-ae exactly -- one codebase, both arms. The decoder is built
        # only when it will be used, so the pure-contrastive arm carries no unused parameters
        # into the optimiser.
        self.decoder = decoder

    def forward(self, batch):
        h_cube = self.cube(batch['cube'])
        h_struct = self.structure(batch['z'], batch['pos'], batch['batch'])
        recon = self.decoder(h_cube) if self.decoder is not None else None
        return self.cube_proj(h_cube), self.struct_proj(h_struct), h_cube, recon


# --- NEW (registry): --help order is pinned, membership is not ---
# argparse validates membership, but the order is what a reader sees and what every archived
# sbatch command was written against, so the four original arms keep their positions and
# anything newly registered is appended. Rejected: sorting, or raw registration order -- both
# reshuffle --help for no reason the day a plugin is added.
_HISTORICAL = ('stm3d', 'ters', 'sgpt', 'stm2d')


def _encoder_choices():
    load_plugins()
    known = set(ENCODERS.names())
    return tuple([n for n in _HISTORICAL if n in known]
                 + [n for n in ENCODERS.names() if n not in _HISTORICAL])


def _pair_choices():
    load_plugins()
    return PAIRS.names()


def _choices(module, table):
    """argparse ``choices`` from a plugin's option table, or () if that plugin is absent.

    A missing modality must not break the other one, so the import is deferred into this
    function: an absent plugin leaves the flag unconstrained instead of making the whole
    module unimportable. Selecting a value from a plugin that is not installed still fails,
    in the builder, naming what is registered.
    """
    import importlib
    try:
        return tuple(getattr(importlib.import_module(module), table))
    except (ImportError, AttributeError):
        return ()


def build(args, in_channels, image_size, device):
    """The cube encoder is the arm under test; everything else is held fixed.

    ``stm3d`` convolves along the ordered z axis; ``ters`` opens with a 1x1 mix that treats the
    slices as unordered, which is right for TERS spectral bins and wrong for STM heights;
    ``sgpt`` is a SpectralGPT-shaped 3-D ViT, optionally with the authors' pretrained weights.
    All three return a LayerNorm-scaled ``latent_dim`` vector, so a difference between the arms
    is attributable to the encoder alone.
    """
    # The prints are an interface: slurm/logs/ and tools/read_arms.py parse them.
    # Statement order is load-bearing: torch is seeded once before this runs and
    # Stm2dCubeEncoder re-seeds inside its constructor, so building the towers in a different
    # order changes every initial weight for the same --seed.
    load_plugins()
    cube = ENCODERS.build(args.cube_encoder, args, in_channels, image_size)
    n_cube = sum(p.numel() for p in cube.parameters())
    print(f'[build] cube encoder: {type(cube).__name__} ({n_cube / 1e6:.2f} M)')
    struct = StructureEncoder.from_checkpoint(
        args.ckpt, freeze=args.freeze_structure, strict=True)
    decoder = None
    if args.recon_weight > 0:
        # base_hw=16 with three stages reaches 16*2**3 = 128, the STM cube size, independently
        # of which encoder produced the latent -- so the two cube encoders share one decoder
        # and the arms stay comparable.
        decoder = TersDecoder(out_channels=in_channels, filters=(64, 128, 256),
                              kernel_size=(3, 3, 3), latent_dim=args.latent_dim, base_hw=16)
        print(f'[build] decoder ON (recon_weight={args.recon_weight}, '
              f'contrastive_weight={args.contrastive_weight})')
    model = ClipModel(cube, struct, proj_dim=args.proj_dim, decoder=decoder).to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[build] trainable parameters: {n_tr / 1e6:.2f} M '
          f'(structure encoder {"FROZEN" if args.freeze_structure else "finetuned"})')
    return model


# --- NEW (wandb): experiment tracking ---
# Not a dependency of the model code: every function below is a no-op when --wandb is absent
# or the package is missing, so the import stays optional and an offline cluster node cannot
# fail a run because of a logging library.
def load_dotenv(path='.env'):
    """Read KEY=VALUE lines into os.environ without overwriting what is already set.

    A 3-line parser rather than python-dotenv: that package is not in the CSC module, and the
    file this reads holds exactly one key. Values are never printed -- .env holds a credential
    and is in .gitignore.
    """
    if not os.path.exists(path):
        return False
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"\''))
    return True


def init_wandb(args, model):
    """Return a live wandb run, or None. Does not raise: logging must not stop training."""
    if not args.wandb:
        return None
    load_dotenv(args.dotenv)
    if not os.environ.get('WANDB_API_KEY') and args.wandb_mode == 'online':
        print('[wandb] no WANDB_API_KEY in env or .env -- falling back to offline mode')
        args.wandb_mode = 'offline'
    try:
        import wandb
    except Exception as exc:                                  # noqa: BLE001
        print(f'[wandb] import failed ({type(exc).__name__}: {exc}); continuing WITHOUT '
              f'tracking. curve.json is unaffected.')
        return None
    try:
        run = wandb.init(project=args.wandb_project, name=args.wandb_name or None,
                         mode=args.wandb_mode, config=vars(args), dir=args.workdir or None)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        run.summary['params_total'] = sum(p.numel() for p in model.parameters())
        run.summary['params_trainable'] = n_tr
        run.summary['cube_encoder'] = type(model.cube).__name__
        print(f'[wandb] {args.wandb_mode}: {run.url if args.wandb_mode == "online" else run.dir}')
        return run
    except Exception as exc:                                  # noqa: BLE001
        print(f'[wandb] init failed ({type(exc).__name__}: {exc}); continuing WITHOUT tracking.')
        return None


def wandb_log(run, ep, tr, va, score):
    """Flatten the two metric dicts the epoch loop already builds. No new measurement here."""
    if run is None:
        return
    row = {'epoch': ep, 'score_size_ctl_over_chance': score}
    for prefix, m in (('train', tr), ('val', va)):
        for k, v in m.items():
            if isinstance(v, (int, float)):
                row[f'{prefix}/{k}'] = v
    try:
        run.log(row, step=ep)
    except Exception as exc:                                  # noqa: BLE001
        print(f'[wandb] log failed at epoch {ep} ({type(exc).__name__}); continuing.')


def move(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


#: What the contrastive loop reads out of a batch. Checked on the first batch of a run.
CONTRASTIVE_NEEDS = ('cube', 'z', 'pos', 'batch', 'n_atoms')


def run_epoch(model, loader, device, args, optimiser=None, scaler=None, step0=0):
    train = optimiser is not None
    model.train(train)
    totals, n_batches, step, n_nonfinite = {}, 0, step0, 0
    checked = False
    for batch in loader:
        batch = move(batch, device)
        if not checked:
            # Once per pass, not per step: a batch's keys do not change between steps, so a
            # per-step check would be pure overhead on the inner loop.
            batch_requires(batch, CONTRASTIVE_NEEDS, consumer='the contrastive loop',
                           dataset_name=getattr(args, 'dataset', 'stm_pairs'))
            checked = True
        b = batch['cube'].shape[0]
        if b < 2:
            continue                       # InfoNCE has no negative in a batch of 1
        _dt = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(
            getattr(args, 'amp_dtype', None) or ('fp16' if args.amp else 'off'))
        with torch.set_grad_enabled(train), torch.autocast(
                'cuda', dtype=_dt or torch.bfloat16,
                enabled=_dt is not None and device.type == 'cuda'):
            z_cube, z_struct, _, recon = model(batch)
            # --- the null control ---
            # Destroy the cube<->molecule pairing by permuting the structure side within the
            # batch.  There is then no learnable relationship, so every retrieval number must
            # fall to its baseline.  If it does not, the metric is measuring something other
            # than what it claims and no positive result from it can be trusted.  Re-drawn
            # every batch so nothing consistent can be memorised.
            if args.shuffle_pairs:
                z_struct = z_struct[torch.randperm(z_struct.shape[0], device=z_struct.device)]
            c_loss = info_nce(z_cube.float(), z_struct.float(),
                              temperature=args.temperature, symmetric=not args.one_directional)
            loss = args.contrastive_weight * c_loss
            r_loss = None
            if recon is not None:
                r_loss = reconstruction_loss(recon.float(), batch['cube'].float())
                loss = loss + args.recon_weight * r_loss
        # A single non-finite batch must not poison the epoch average. The overflow happens
        # inside the encoder, so the `.float()` above does not catch it; guarding at the loss
        # does. Counted and printed rather than dropped: a run with 300 skipped batches is a
        # different run.
        if not torch.isfinite(loss):
            n_nonfinite += 1
            if train:
                optimiser.zero_grad(set_to_none=True)
            continue

        if train:
            # Warmup, then cosine decay. Inlined rather than CosineAnnealingLR: the scheduler
            # owns .step() and would fight the manual warmup over param_groups[0].
            if step + 1 <= args.warmup_steps:
                lr = args.lr * (step + 1) / max(args.warmup_steps, 1)
            else:
                prog = (step + 1 - args.warmup_steps) / max(args.total_steps - args.warmup_steps, 1)
                prog = min(max(prog, 0.0), 1.0)
                lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * prog))
            for g in optimiser.param_groups:
                g['lr'] = lr
            optimiser.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimiser.step()
            step += 1

        m = contrastive_metrics(z_cube.float(), z_struct.float())
        m.update(size_confound_metrics(z_cube.float(), z_struct.float(), batch['n_atoms']))
        m['loss'] = loss.item()
        m['c_loss'] = c_loss.item()
        m['collapsed'] = math.log(b)
        if r_loss is not None:
            m['recon'] = r_loss.item()
            m['recon_mean_baseline'] = baseline_losses(batch['cube'].float())['mean']
        for k, v in m.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1
    out = {k: v / max(n_batches, 1) for k, v in totals.items()}
    out['nonfinite'] = n_nonfinite
    out['batches'] = n_batches
    return (out, step)


def fmt(tag, ep, m, dt):
    if not m.get('batches'):
        return (f'{tag} ep{ep:03d}  ALL {m.get("nonfinite", 0)} batches non-finite '
                f'-- nothing to report. Re-run without --amp.')
    return (f'{tag} ep{ep:03d}  loss {m["loss"]:.4f} (collapsed {m["collapsed"]:.4f})  '
            f'top1 {m["top1_a2b"]:.3f}/{m["top1_b2a"]:.3f} vs chance {m["chance"]:.3f}  '
            f'pos {m["pos_cos"]:+.3f} neg {m["neg_cos"]:+.3f} gap {m["gap"]:+.3f}  '
            f'cone_cube {m["cone_a"]:.3f}'
            + (f'  | SIZE-ONLY {m["top1_size_only"]:.3f}'
               f'  size-ctl {m["top1_size_controlled"]:.3f}'
               f'/ch {m["chance_size_controlled"]:.3f}(n={m["n_size_controlled"]:.0f})'
               f'  nrank {m["norm_rank_size_controlled"]:.3f}(0.5=chance)'
               f'  r={m["sim_vs_size_r"]:+.2f}' if 'top1_size_only' in m else '')
            + (f'  | recon {m["recon"]:.4f} (mean-baseline {m["recon_mean_baseline"]:.4f})'
               if 'recon' in m else '')
            + (f'  !! {m["nonfinite"]}/{m["nonfinite"] + m["batches"]} NON-FINITE'
               if m.get('nonfinite') else '')
            + f'  {dt:.1f}s')


def _parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dataset', default='stm_pairs', choices=_pair_choices() or None,
                    help='which modality\'s paired reader builds the cubes. Defaults to '
                         'stm_pairs, which is what every archived invocation of this loop '
                         'assumed before the reader became selectable.')
    ap.add_argument('--channels-key', default='channels_100',
                    help='TERS only: the HDF5 dataset holding the cubes. Ignored by readers '
                         'that do not take it.')
    ap.add_argument('--h5', default='/scratch/project_2001912/sethih1/stm-data/train.h5',
                    help='TRAINING cubes')
    ap.add_argument('--val-h5', default='/scratch/project_2001912/sethih1/stm-data/val.h5',
                    help="validation file. Pass '' to split --h5 instead (smoke runs only)")
    ap.add_argument('--val-limit', type=int, default=4000,
                    help='molecules from --val-h5 to validate on')
    ap.add_argument('--lr-min', type=float, default=1e-6)
    ap.add_argument('--ckpt', default=DEFAULT_CKPT, help='pretrained TorchMD-NET ET')
    ap.add_argument('--limit', type=int, default=0,
                    help='molecules from --h5 to train on; 0 = ALL of them (the default). '
                         'A small value is for smoke runs.')
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=16,
                    help='NOT MEASURED for this corpus. MolSpectra used 128; InfoNCE quality '
                         'scales with the negative count.')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup-steps', type=int, default=100)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--one-directional', action='store_true',
                    help="use MolSpectra's implemented form instead of the paper's symmetric one")
    ap.add_argument('--contrastive-weight', type=float, default=1.0,
                    help='weight on the InfoNCE term')
    ap.add_argument('--recon-weight', type=float, default=0.0,
                    help='weight on the reconstruction term. S4: with '
                         '--contrastive-weight 0 --recon-weight 1 this reproduces ters-ae, '
                         'which is the control arm for "did contrastive cause the change?"')
    ap.add_argument('--shuffle-pairs', action='store_true',
                    help='NULL CONTROL: permute the molecule side inside each batch so no '
                         'cube<->molecule relationship exists. Every retrieval number must '
                         'fall to its baseline; if it does not, the metric is broken.')
    ap.add_argument('--cube-encoder', choices=_encoder_choices(),
                    default='stm3d',
                    help="'stm3d' convolves along the ordered z axis (recommended for "
                         "STM); 'ters' is the 1x1-stem control arm; 'sgpt' is a "
                         "SpectralGPT-shaped 3-D ViT (see --sgpt-*); 'stm2d' is a pretrained "
                         '2-D vision backbone on a single z-slice, which is ASD-STM\'s own '
                         'protocol (see --stm2d-* and --z-slice)')
    ap.add_argument('--z-slice', default='none',
                    help="what the loader hands the cube encoder along z. 'none' (default) = "
                         'all 14 ordered slices, i.e. every run before this flag existed, '
                         "unchanged. 'random' = ONE slice per sample per epoch, which is "
                         'ASD-STM\'s protocol (arXiv:2312.08854: a single randomly chosen '
                         "slice into a 2-D attention U-Net). 'mean' = the z-average, the "
                         'control for "does collapsing z cost anything". An int 0-13 = that '
                         'fixed slice (0 is z=6 A, nearest the molecule, and carries 22.0%% of '
                         'stack energy; 13 is z=19 A and carries 1.0%%). Anything but none '
                         'makes in_channels 1, and is 11x cheaper to read.')
    ap.add_argument('--stm2d-backbone', default='clip_vit_b16',
                    choices=_choices('stm.models.timm2d', 'STM2D_BACKBONES') or None,
                    help="timm trunk for the stm2d arm. 'clip_vit_b16' is the recommendation: "
                         'UniMR measured off-the-shelf CLIP/ViT at F1 0.93 clustering '
                         'same-molecule STM images against 0.72 for ResNet-50. '
                         "'resnet50' is kept as the named "
                         'control so "is it the transformer, or the pretraining?" is '
                         'answerable rather than assumed.')
    ap.add_argument('--stm2d-timm-name', default='',
                    help='an unregistered timm name, bypassing --stm2d-backbone. Its shape and '
                         'size are then unmeasured here; grid_tied is probed at build time.')
    ap.add_argument('--stm2d-ckpt', default='',
                    help='a local weights file, used instead of fetching. MEASURED '
                         '2026-09-16: ~/.cache/huggingface did not exist, and compute nodes '
                         'here may have no egress, so run `python -m tools.prefetch_stm2d '
                         '--backbone <arm>` on the login node and pass the path it prints.')
    ap.add_argument('--stm2d-random-init', action='store_true',
                    help='do NOT load pretrained weights. This is the from-scratch control '
                         'that separates "the pretraining" from "a 2-D backbone on one '
                         'slice"; it is not the transfer arm. Pair it with '
                         '--stm2d-trainable-blocks -1, or the run is a frozen random trunk '
                         'that trains only the head, and would be reported as the '
                         'from-scratch control while never having trained from scratch.')
    ap.add_argument('--stm2d-min-match', type=float, default=0.98,
                    help='refuse to start unless this fraction of the trunk was changed by '
                         'the pretrained load. Tighter than --sgpt-min-match because at '
                         '--stm2d-in-chans 3 and the native resize there is no legitimately '
                         'unloaded weight tensor. Guards the same silent-partial-load failure '
                         'as --sgpt-min-match.')
    ap.add_argument('--stm2d-in-chans', type=int, default=3, choices=(1, 3),
                    help='3 replicates the slice to RGB and keeps the pretrained stem '
                         'byte-verbatim. 1 is mathematically identical -- timm sums the RGB '
                         'stem over its input axis (verified in adapt_input_conv), and '
                         'sum_c W[:,c]*x is what feeding (x,x,x) computes -- at a third of the '
                         'stem FLOPs. Not a modelling trade.')
    ap.add_argument('--stm2d-resize', type=int, default=0,
                    help="spatial size fed to the trunk; 0 means the backbone's native 224, "
                         'which uses every pretrained tensor including pos_embed verbatim. '
                         'Pass 128 to feed the cube at native resolution instead, which makes '
                         'timm resample the position grid 197->65. Neither is free; this is '
                         'the same kind of choice --sgpt-band-mode is.')
    ap.add_argument('--stm2d-slice-mode', default='random',
                    choices=('random', 'fixed', 'passthrough'),
                    help='how the ENCODER picks a slice when the loader did not (--z-slice '
                         'none). Irrelevant when --z-slice already emits one channel. '
                         'Evaluation is always deterministic regardless of this.')
    ap.add_argument('--stm2d-slice-index', type=int, default=0,
                    help='the slice used at eval, and by --stm2d-slice-mode fixed. 0 = z=6 A.')
    ap.add_argument('--stm2d-trainable-blocks', type=int, default=0,
                    help='0 freezes the trunk and trains only the head (the LiT argument, and '
                         'the default because P1 is already overfitting); -1 trains '
                         'everything; N>0 trains the last N. WARNING: N counts ViT BLOCKS (12 '
                         'available) but ResNet STAGES (4), so the same number means very '
                         'different things across backbones -- [build] prints the resolved '
                         'count.')
    ap.add_argument('--sgpt-ckpt', default='',
                    help='SpectralGPT+.pth from zenodo.org/records/8412455. WEIGHTS ONLY - '
                         'their repo is GPLv3 and no code of theirs is vendored here. Without '
                         'this the sgpt arm is a random-init ViT-Base, which is the '
                         'from-scratch control rather than the transfer arm. Run '
                         '`python -m tools.inspect_sgpt_ckpt <path>` first.')
    ap.add_argument('--wandb', action='store_true',
                    help='log to Weights & Biases. Off by default: a logging library must '
                         'never be able to fail a training run, so every wandb path here is '
                         'a no-op when this is absent or the import fails.')
    ap.add_argument('--wandb-project', default='ters-clip')
    ap.add_argument('--wandb-name', default='', help='run name; defaults to wandb\'s own')
    ap.add_argument('--wandb-mode', default='online', choices=('online', 'offline', 'disabled'))
    ap.add_argument('--dotenv', default='.env',
                    help='file holding WANDB_API_KEY. In .gitignore; never echoed.')
    ap.add_argument('--sgpt-size', default='base', choices=_choices('stm.models.spectral_gpt', 'SGPT_SIZES') or None,
                    help="ViT trunk width. 'base' (85.6 M) is SpectralGPT+'s own shape and the "
                         "ONLY one --sgpt-ckpt can load. 'small' (21.8 M) and 'tiny' (5.6 M) "
                         'are for training a transformer from scratch on this corpus, where '
                         'ViT-Base is sized for 1M satellite scenes, not ~950 molecules.')
    ap.add_argument('--sgpt-band-mode', default='interpolate',
                    choices=('interpolate', 'pad', 'native'),
                    help="how 14 z-slices are fitted to a 3-deep tubelet. 'interpolate' "
                         'resamples 14->12 and is the only mode that uses every pretrained '
                         "tensor verbatim; 'pad' replicates 14->15; 'native' keeps 14 with a "
                         '2-deep tubelet and throws away the pretrained depth weights, which '
                         'makes it the control for "is the spectral prior worth anything".')
    ap.add_argument('--sgpt-pool', default='mean', choices=('mean', 'cls'),
                    help='mean-pool the patch tokens (MAE finetuning convention) or read the '
                         'class token')
    ap.add_argument('--sgpt-trainable-blocks', type=int, default=0,
                    help='0 freezes the trunk and trains only the head (the LiT argument, and '
                         'the default because P1 is already overfitting at 1.31 M params); '
                         '-1 trains everything; N>0 trains the last N blocks.')
    ap.add_argument('--sgpt-min-match', type=float, default=0.9,
                    help='refuse to start if less than this fraction of the ViT trunk loaded '
                         'from the checkpoint. Guards a partial load that prints a number '
                         'and carries on instead of raising.')
    ap.add_argument('--stem-channels', type=int, nargs='+', default=[8, 16, 32],
                    help='widths of the 3D stem. The default gives the stem 1.1%% of the '
                         'model - which is odd, since reading the z axis is its whole reason '
                         'for existing. not MEASURED.')
    ap.add_argument('--filters', type=int, nargs='+', default=[64, 128, 256],
                    help='widths of the 2D ladder after the z-collapse')
    ap.add_argument('--cube-pool-size', type=int, default=1,
                    help='spatial grid kept by the head. 1 is a global average pool, which '
                         'deletes where a feature sits; ters-2-graph sets 4 for that reason. '
                         'The ladder ends on 2x2 here, so 2 keeps everything. not MEASURED.')
    ap.add_argument('--latent-dim', type=int, default=256)
    ap.add_argument('--proj-dim', type=int, default=128)
    ap.add_argument('--freeze-structure', action='store_true')
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--amp', action='store_true',
                    help='enable autocast. KEPT for compatibility: with no --amp-dtype this '
                         'still means fp16, which is what every run before 2026-09-17 got.')
    ap.add_argument('--amp-dtype', default=None, choices=('off', 'bf16', 'fp16'),
                    help='autocast precision. Unset '
                         'means fp16 if --amp was given, else off, so existing sbatch scripts '
                         'behave exactly as before. A bare torch.autocast(cuda) is FP16, and '
                         'it discarded 3,808/35,950 train and 492/1,519 val batches (10.6%%) '
                         'in job 1264302 -- with AMP off measuring better top-1, 0.193 vs '
                         '0.166. bf16 keeps the speed and has fp32\'s exponent range, so it '
                         'needs no GradScaler. Prefer bf16 on GH200.')
    ap.add_argument('--weight-decay', type=float, default=0.01,
                    help='AdamW weight decay. The AdamW call '
                         'used to pass only lr, so every run before 2026-09-17 sat on this '
                         '0.01 default; the CLIP family uses ~0.1. Left at 0.01 so no existing '
                         'command changes meaning -- pass 0.1 explicitly.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workdir', default='')
    return ap


def main(argv=None):
    """Flag entry point, for the archived sbatch invocations and for --help."""
    return run(_parser().parse_args(argv))


def args_from_config(cfg, **overrides):
    """Config object -> the namespace ``run`` takes, without a flag string in between.

    Starts from the parser's own defaults, so every default has exactly one definition and a
    config route cannot silently disagree with a flag route. An unknown key raises here rather
    than being dropped, which is the failure mode a dict update would hide.
    """
    args = _parser().parse_args([])
    for key, value in {**_config_to_flat(cfg), **overrides}.items():
        if not hasattr(args, key):
            raise ValueError(
                f'{key!r} is not an argument of the contrastive loop. Known: '
                f'{sorted(vars(args))}')
        setattr(args, key, value)
    return args


def _config_to_flat(cfg):
    """The config keys this loop reads, mapped to its own argument names.

    A key the config does not set is left at the parser's default rather than raising, so a
    modality config states only what it changes -- there is still exactly one definition of
    every default, in the parser.
    """
    # (argument name, config section, key in that section)
    TABLE = (
        ('cube_encoder', 'model', 'cube_encoder'), ('latent_dim', 'model', 'latent_dim'),
        ('proj_dim', 'model', 'proj_dim'), ('ckpt', 'model', 'structure_ckpt'),
        ('stem_channels', 'model', 'stem_channels'), ('filters', 'model', 'filters'),
        ('cube_pool_size', 'model', 'cube_pool_size'),
        ('freeze_structure', 'model', 'freeze_structure'),
        ('z_slice', 'data', 'z_slice'), ('num_workers', 'data', 'num_workers'),
        ('val_limit', 'data', 'max_val'), ('limit', 'data', 'max_train'),
        ('channels_key', 'data', 'channels_key'),
        ('epochs', 'training', 'n_epochs'), ('batch_size', 'training', 'batch_size'),
        ('lr', 'training', 'lr'), ('lr_min', 'training', 'lr_min'),
        ('warmup_steps', 'training', 'warmup_steps'),
        ('weight_decay', 'training', 'weight_decay'),
        ('amp_dtype', 'training', 'amp_dtype'), ('seed', 'training', 'seed'),
        ('workdir', 'training', 'workdir'),
        ('temperature', 'objective', 'temperature'),
        ('contrastive_weight', 'objective', 'contrastive_weight'),
        ('recon_weight', 'objective', 'recon_weight'),
    )
    _MISSING = object()
    flat = {}
    for arg, section, key in table:
        value = getattr(getattr(cfg, section, None), key, _MISSING)
        if value is not _MISSING and value is not None:
            flat[arg] = value

    d, o, t = cfg.data, cfg.objective, cfg.training
    root = str(d.root).rstrip('/')
    flat['h5'] = f'{root}/{d.train_split}.h5'
    flat['val_h5'] = f'{root}/{d.val_split}.h5'
    flat['one_directional'] = not getattr(o, 'symmetric', True)
    flat['shuffle_pairs'] = bool(getattr(o, 'null_control', False))
    if getattr(t, 'max_steps', 0):
        raise ValueError(
            f'training.max_steps={t.max_steps} is not supported by the contrastive loop; it '
            f'bounds work with training.n_epochs and data.max_train. Set max_steps: 0 and use '
            f'n_epochs/max_train instead (configs/smoke_synthetic.yaml does exactly that).')
    return flat


def run(args):
    # Resolve before anything stores vars(args), so the checkpoint records what actually ran.
    args.amp_dtype = args.amp_dtype or ('fp16' if args.amp else 'off')
    args.amp = args.amp_dtype != 'off'          # keep the existing `args.amp and ...` guards
    if args.amp_dtype == 'fp16':
        print('[env] WARNING: fp16 autocast. MEASURED: this discards 10.6% of '
              'train and 32.4% of val batches on this exact model, and AMP-off scoring better. '
              'Pass --amp-dtype bf16.')

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[env] device={device} torch={torch.__version__}')

    # Two files by default: train.h5 to train on, val.h5 to validate on. Passing --val-h5 ''
    # falls back to a single-file split, which is what the small smoke runs want.
    # 0 means 'all' for both limits below.
    args.limit = None if not args.limit or args.limit <= 0 else args.limit
    args.val_limit = None if not args.val_limit or args.val_limit <= 0 else args.val_limit
    # One definition of the z settings, so train and val cannot drift apart by an edit.  The
    # validation set keeps epoch 0 forever: `score` selects the exported checkpoint on
    # top1_size_controlled, and a val set that resampled its slices each epoch would make that
    # criterion noisy for a reason unrelated to the model.  Only the train dataset gets
    # set_epoch, in the loop below.
    def pairs(h5, **kw):
        return PAIRS.build(args.dataset, h5, args, **kw)

    full = pairs(args.h5, limit=args.limit)
    print(f'[data] train file: {full.describe()}')
    if args.val_h5:
        train_ds = pairs(args.h5, limit=args.limit)
        val_ds = pairs(args.val_h5, limit=args.val_limit)
        print(f'[data] val file:   {val_ds.describe()}')
    else:
        n_val = max(2, int(len(full) * args.val_frac))
        train_ds = pairs(args.h5, limit=args.limit, indices=list(range(n_val, len(full))))
        val_ds = pairs(args.h5, limit=args.limit, indices=list(range(n_val)))
        print('[data] SINGLE-FILE split (--val-h5 empty): train and val are disjoint slices '
              'of the same file. Fine for a smoke run, wrong for a real model.')
    print(f'[data] train {len(train_ds)}  val {len(val_ds)}  batch {args.batch_size}  '
          f'collapsed baseline log(B) = {math.log(args.batch_size):.4f}')

    common = dict(batch_size=args.batch_size, collate_fn=train_ds.collate,
                  num_workers=args.num_workers, drop_last=True,
                  pin_memory=(device.type == 'cuda'))
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    # Check both loaders here, in the parent, before anything is built. drop_last=True is
    # required -- a trailing batch of 1 has no negative and InfoNCE is undefined on it -- but
    # that means a split smaller than one batch silently yields zero batches, and the first
    # symptom is a KeyError deep in the logging. Raise with the arithmetic and the flag that
    # fixes it instead.
    for name, ds, loader in (('train', train_ds, train_loader), ('val', val_ds, val_loader)):
        if len(loader) == 0:
            raise SystemExit(
                f'--- no full {name} batches ---\n'
                f'{len(ds)} {name} pairs at --batch-size {args.batch_size} with drop_last=True '
                f'gives 0 batches.\n'
                f'(--limit {args.limit} split at --val-frac {args.val_frac} -> '
                f'{len(train_ds)} train / {len(val_ds)} val)\n'
                f'Fix: raise --limit to at least '
                f'{int(math.ceil(args.batch_size / max(args.val_frac, 1e-9)))}, '
                f'or lower --batch-size to {len(ds)} or less.')

    # Both checks belong here, in the parent, before the workdir exists and before any worker
    # forks.
    # A random slice is incoherent with reconstruction: the encoder sees slice k but nothing
    # conveys k to the decoder, so the target is a fresh draw and the Bayes-optimal output is
    # the mean slice. A fixed slice or the z-mean is fine.
    if args.recon_weight > 0 and args.z_slice == 'random':
        raise SystemExit(
            '--- --recon-weight > 0 with --z-slice random ---\n'
            'The encoder is handed a different slice every epoch and the decoder is never told '
            'which one, so the reconstruction target is a random draw from 14 and the optimum '
            'is the mean slice. The term measures nothing.\n'
            'Fix: --z-slice <int> (e.g. 6) or --z-slice mean for a well-posed single-slice '
            'autoencoder, or --recon-weight 0 for the pure contrastive arm.')
    if args.recon_weight > 0 and args.z_slice not in (None, 'none') and args.cube_encoder \
            == 'stm2d':
        print(f'[build] NOTE: joint recon arm on {full.in_channels} channel(s). Encoder input '
              f'and decoder output match, so the objective is well posed -- but its recon '
              f'baseline is not comparable to a 14-channel run, because a 1-channel target has '
              f'a different variance. Compare only against another single-slice run.')
    if (args.cube_encoder == 'stm2d' and args.stm2d_random_init
            and args.stm2d_trainable_blocks == 0):
        raise SystemExit(
            '--- --stm2d-random-init with --stm2d-trainable-blocks 0 ---\n'
            'That is a frozen random trunk: only the head would train, against random '
            'features, and the arm would still be reported as the from-scratch control.\n'
            'Fix: add --stm2d-trainable-blocks -1 for the real from-scratch control.')

    model = build(args, full.in_channels, full.image_size, device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    print(f'[opt] AdamW lr={args.lr} weight_decay={args.weight_decay} '
          f'precision={args.amp_dtype}')
    scaler = torch.amp.GradScaler('cuda',
                                  enabled=args.amp_dtype == 'fp16'
                                  and device.type == 'cuda')

    # The cosine schedule needs to know where the end is.
    args.total_steps = args.epochs * len(train_loader)
    print(f'[sched] {len(train_loader)} steps/epoch x {args.epochs} = {args.total_steps} steps; '
          f'warmup {args.warmup_steps}, lr {args.lr} -> {args.lr_min}')

    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)

    def save(tag, ep, metrics):
        if not args.workdir:
            return
        # The cube encoder alone is the deliverable -- the projection heads are discarded at
        # export by design (SimCLR 4.2: contrasting on g(h) leaves more in h).  The full model
        # goes alongside it so a run can be resumed or re-scored without retraining.
        torch.save({'state_dict': model.cube.state_dict(), 'epoch': ep,
                    'val_metrics': metrics, 'args': vars(args),
                    'cube_encoder_class': type(model.cube).__name__},
                   os.path.join(args.workdir, f'cube_encoder_{tag}.pt'))
        torch.save({'model': model.state_dict(), 'epoch': ep, 'args': vars(args)},
                   os.path.join(args.workdir, f'full_{tag}.pt'))

    def score(m):
        """The model-selection criterion: size-controlled retrieval over its own chance level.

        Not val loss, which bottoms out well before top-1 stops climbing, so selecting on it
        would discard most of the retrieval gain. Not raw top-1 either: an atom-count oracle
        alone scores far above chance on this corpus, so top-1 partly rewards guessing
        molecule size. `size-ctl / chance` is the only number here that is about molecular
        identity. Returns -inf when the metric is absent, so a degenerate epoch can never be
        selected as best.
        """
        sc, ch = m.get('top1_size_controlled'), m.get('chance_size_controlled')
        if not m.get('batches') or sc is None or ch in (None, 0) or sc != sc or ch != ch:
            return float('-inf')
        return sc / ch

    wb = init_wandb(args, model)
    hist, step, best = [], 0, float('-inf')
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        # Re-draw the training slices for this epoch. A no-op unless --z-slice random, and not
        # called on val_ds, whose slices stay fixed so the selection criterion is comparable
        # across epochs.
        train_ds.set_epoch(ep)
        tr, step = run_epoch(model, train_loader, device, args, opt, scaler, step)
        print(fmt('[train]', ep, tr, time.time() - t0))
        t0 = time.time()
        va, _ = run_epoch(model, val_loader, device, args)
        print(fmt('[val]  ', ep, va, time.time() - t0), flush=True)
        hist.append({'epoch': ep, 'train': tr, 'val': va})

        s = score(va)
        wandb_log(wb, ep, tr, va, s if s > float('-inf') else None)
        if s > best:
            best = s
            save('best', ep, va)
            print(f'[best]  ep{ep:03d}  size-ctl/chance = {s:.3f}  -> checkpoint saved',
                  flush=True)
        if args.workdir:
            with open(os.path.join(args.workdir, 'curve.json'), 'w') as fh:
                json.dump({'args': vars(args), 'history': hist,
                           'best_score': best if best > float('-inf') else None}, fh, indent=2)

    if wb is not None:
        wb.summary['best_size_ctl_over_chance'] = best if best > float('-inf') else None
        wb.finish()
    save('last', args.epochs, hist[-1]['val'] if hist else {})
    if args.workdir:
        print(f'[done] {args.workdir}: cube_encoder_best.pt (size-ctl/chance {best:.3f}), '
              f'cube_encoder_last.pt, curve.json')


if __name__ == '__main__':
    main()
