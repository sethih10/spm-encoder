"""Train the TERS autoencoder.

Sources:
  * the epoch/eval loop shape -- ADAPT of pytorch/examples @ acc295d vae/main.py,
    https://github.com/pytorch/examples. Reconstruction only, so no KL term; AMP, grad
    clipping and the LR schedule are ours.
  * warmup-then-cosine -- SGDR, Loshchilov & Hutter, https://arxiv.org/abs/1608.03983
"""

import argparse
import csv
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from configs.default import apply_overrides, as_dict, get_config
from shared.losses.reconstruction import combined_loss, reconstruction_loss
from shared.metrics.reconstruction import baseline_losses, reconstruction_metrics
from shared.registry import DATASETS, MODELS, batch_requires, load_plugins

# Dataset named here, not imported: core-side code must not hard-depend on one modality, or
# deleting the other modality would break this one too.

#: What the reconstruction loop reads, checked on the first batch.
RECON_NEEDS = ('cube',)

#: Fallbacks for archived invocations that predate the reader and model becoming selectable,
#: from when both always meant TERS. A config names its own through data.name and model.name.
_DATASET = 'ters_cubes'
_MODEL = 'ters_autoencoder'


def _dataset_of(config):
    return getattr(config.data, 'name', None) or _DATASET


def _model_of(config):
    return getattr(config.model, 'name', None) or _MODEL


def load_channel_mask(config):
    """``config.data.channel_mask`` -> the ``keep`` list, or None.

    The mask is applied to train and val identically; it is a property of the training corpus
    (tools.channel_stats fits it on train only) and must not differ between the two, or the
    model would be scored on channels it never saw.
    """
    path = getattr(config.data, 'channel_mask', '')
    if not path:
        return None
    with open(path) as fh:
        spec = json.load(fh)
    keep = spec['keep']
    print(f"channel mask {path}: keep {len(keep)} of {spec['n_channels']} channels "
          f"(threshold {spec.get('threshold')}, fitted on {spec.get('h5')})")
    return keep


def build_loaders(config, train_h5, val_h5):
    keep = load_channel_mask(config)
    train_ds = DATASETS.build(
        _dataset_of(config),
        train_h5, channels_key=config.data.channels_key, normalize=config.data.normalize,
        indices=range(config.data.max_train) if config.data.max_train else None,
        keep_channels=keep)
    val_ds = DATASETS.build(
        _dataset_of(config),
        val_h5, channels_key=config.data.channels_key, normalize=config.data.normalize,
        indices=range(config.data.max_val) if config.data.max_val else None,
        keep_channels=keep)

    # Checked before the first forward pass, in the parent: an in_channels or image_size that
    # disagrees with the file produces a clear message here instead of a shape error inside
    # autocast.
    if train_ds.in_channels != config.model.in_channels:
        raise ValueError(
            f'{train_h5} has {train_ds.in_channels} channels but config.model.in_channels='
            f'{config.model.in_channels}. Set --set model.in_channels={train_ds.in_channels}.')
    if (train_ds.height, train_ds.width) != (config.data.image_size,) * 2:
        raise ValueError(
            f'{train_h5} cubes are {train_ds.height}x{train_ds.width} but '
            f'config.data.image_size={config.data.image_size}. The decoder is fixed-size; '
            f'set --set data.image_size={train_ds.height} (must be square).')
    if val_ds.in_channels != train_ds.in_channels:
        raise ValueError(
            f'train has {train_ds.in_channels} channels, val has {val_ds.in_channels}')

    common = dict(num_workers=config.data.num_workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=config.data.num_workers > 0)
    train_dl = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True,
                          drop_last=True, **common)
    val_dl = DataLoader(val_ds, batch_size=config.training.eval_batch_size, shuffle=False,
                        drop_last=False, **common)
    return train_ds, val_ds, train_dl, val_dl


def amp_context(config, device):
    """``(autocast_ctx_factory, GradScaler_or_None)`` for the configured precision."""
    mode = str(config.training.amp_dtype).lower()
    if device.type != 'cuda' or mode == 'off':
        # Explicitly disabled autocast rather than a silent no-op: on CPU autocast(bf16) is
        # legal but slow and changes numerics, and the CPU path here only exists for tests.
        return (lambda: torch.autocast('cpu', enabled=False)), None
    if mode == 'bf16':
        return (lambda: torch.autocast('cuda', dtype=torch.bfloat16)), None
    if mode == 'fp16':
        return (lambda: torch.autocast('cuda', dtype=torch.float16)), torch.amp.GradScaler('cuda')
    raise ValueError(f"training.amp_dtype must be 'bf16', 'fp16' or 'off', got {mode!r}")


# --- NEW (preflight): refuse to start a long run into a filesystem that cannot hold it ---
# Checked up front rather than left to the first checkpoint write: a full Lustre quota blocks
# in an uninterruptible write instead of raising, so the run hangs to its walltime with no
# traceback.
def check_workdir_space(workdir, config, safety=3.0):
    """Raise unless ``workdir`` can hold the checkpoints this run will write."""
    import shutil

    n_param = sum(
        p.numel() for p in MODELS.build(_model_of(config), config).parameters())
    # best.pt and last.pt each carry params + AdamW exp_avg + exp_avg_sq, all fp32.
    per_ckpt = n_param * 4 * 3
    need = per_ckpt * 2 * safety
    free = shutil.disk_usage(workdir).free
    if free < need:
        raise OSError(
            f'{workdir} has {free/2**30:.1f} GiB free but this run needs about '
            f'{need/2**30:.1f} GiB ({per_ckpt/2**30:.1f} GiB x2 checkpoints x{safety:g} '
            f'safety).\n'
            f'  Home (/users) is a 15 GB quota on this cluster -- check with '
            f'`lfs quota -h -u $USER /users/$USER`, and note the quota can be exhausted '
            f'while the mount still reports free space.\n'
            f'  Put the run on scratch instead: '
            f'--set training.workdir=/scratch/project_2001912/ters-ae-runs/<name>')
    print(f'workdir {workdir}: {free/2**30:.1f} GiB free, '
          f'need ~{need/2**30:.1f} GiB for checkpoints -- ok')


def set_lr(optimizer, step, total_steps, config):
    """Linear warmup then cosine decay to zero; returns the LR actually applied."""
    warmup = config.training.warmup_steps
    base = config.training.lr
    if warmup > 0 and step < warmup:
        lr = base * (step + 1) / warmup
    else:
        # total_steps can be 0 if the loader is empty, or <= warmup on a smoke run; both would
        # make the denominator zero or negative, so clamp rather than emit a nan LR.
        span = max(total_steps - warmup, 1)
        progress = min(max(step - warmup, 0) / span, 1.0)
        lr = base * 0.5 * (1.0 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


@torch.no_grad()
def evaluate(model, loader, device, autocast, max_batches=0):
    model.eval()
    tot, tot_zero, tot_mean, n, n_mean = 0.0, 0.0, 0.0, 0, 0
    # SNR and SAM alongside the MSE: they are what the HSI compression literature reports, and
    # without them this run's number cannot be placed against any published result. SAM is
    # averaged over batches rather than over pixels globally; the batches are equal-sized
    # except the last, which is close enough for a reporting metric and avoids holding every
    # pixel's angle in memory.
    tot_snr, tot_sam, n_sam = 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        cube = batch['cube']   # readers emit a dict
        if max_batches and i >= max_batches:
            break
        cube = cube.to(device, non_blocking=True)
        with autocast():
            recon, _ = model(cube)
            loss = reconstruction_loss(recon.float(), cube)
        base = baseline_losses(cube)
        met = reconstruction_metrics(recon.float(), cube)
        bs = cube.size(0)
        if not math.isnan(met['snr_db']):
            tot_snr += met['snr_db'] * bs
        if not math.isnan(met['sam_deg']):
            tot_sam += met['sam_deg'] * bs
            n_sam += bs
        tot += loss.item() * bs
        tot_zero += base['zero'] * bs
        # baseline_losses returns nan for a batch of 1 (it is undefined there) -- a trailing
        # odd-sized eval batch must not turn the whole column into nan.
        if not math.isnan(base['mean']):
            tot_mean += base['mean'] * bs
            n_mean += bs
        n += bs
    model.train()
    if n == 0:
        return {'val_mse': float('nan'), 'val_zero': float('nan'), 'val_mean': float('nan'),
                'val_snr_db': float('nan'), 'val_sam_deg': float('nan')}
    return {'val_mse': tot / n, 'val_zero': tot_zero / n,
            'val_mean': tot_mean / n_mean if n_mean else float('nan'),
            'val_snr_db': tot_snr / n, 'val_sam_deg': tot_sam / n_sam if n_sam else float('nan')}


# --- NEW (interop): checkpoint layout ---
# Two state_dicts are written, not one. `model` is the whole autoencoder, for resuming;
# `state_dict` is the encoder alone, under the key the diffusion repo loads from.
def save_checkpoint(path, model, optimizer, epoch, step, config, metrics):
    torch.save({
        'model': model.state_dict(),
        'state_dict': model.encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'config': as_dict(config),
        'metrics': metrics,
    }, path)


def main(argv=None):
    # Once, here: every later DATASETS/MODELS lookup needs the registry already populated.
    load_plugins()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', nargs='*', default=[], metavar='key=value',
                    help='dotted config overrides, e.g. model.latent_dim=512')
    ap.add_argument('--synthetic', action='store_true',
                    help='train on a generated HDF5 instead of the real one (no cluster needed)')
    ap.add_argument('--resume', default='', help='checkpoint to resume from')
    args = ap.parse_args(argv)
    return run(apply_overrides(get_config(), args.set), synthetic=args.synthetic,
               resume=args.resume)


def run(config, synthetic=False, resume='', train_h5=None, val_h5=None):
    """Train from a config object. The entry points call this; main() is the flag route.

    ``train_h5``/``val_h5`` let a modality supply its own corpus -- a generated one, or a pair
    of paths the config does not describe. Given them, ``synthetic`` only relaxes the run size;
    the loop never builds a corpus it would have to know the modality to build.
    """
    load_plugins()
    args = argparse.Namespace(synthetic=synthetic, resume=resume)
    supplied = train_h5 is not None

    if args.synthetic and supplied:
        config.data.num_workers = 0
        config.training.batch_size = 2
        config.training.n_epochs = min(config.training.n_epochs, 3)
        config.training.warmup_steps = 2
    elif args.synthetic:
        # The code must run before the data does.
        from tests.make_synthetic import make_synthetic_h5
        tmp = os.path.join(config.training.workdir, 'synthetic')
        train_h5 = make_synthetic_h5(os.path.join(tmp, 'train.h5'), n=8, hw=32, channels=6)
        val_h5 = make_synthetic_h5(os.path.join(tmp, 'val.h5'), n=4, hw=32, channels=6, seed=1)
        config.data.image_size = 32
        config.model.in_channels = 6
        config.model.filters = (8, 16)
        config.model.kernel_size = (3, 3)
        # A real bottleneck, not the default 256: at 6x32x32 = 6144 values, 256 would only be
        # 24x compression, well below what a real run tests. latent_dim=16 keeps the smoke run
        # in the same compression regime, so its loss curve is representative rather than easy.
        config.model.latent_dim = 16
        config.data.num_workers = 0
        config.training.batch_size = 2
        config.training.n_epochs = min(config.training.n_epochs, 3)
        config.training.warmup_steps = 2
    elif not supplied:
        train_h5 = os.path.join(config.data.root, f'{config.data.train_split}.h5')
        val_h5 = os.path.join(config.data.root, f'{config.data.val_split}.h5')

    torch.manual_seed(config.training.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    workdir = config.training.workdir
    os.makedirs(workdir, exist_ok=True)
    check_workdir_space(workdir, config)

    train_ds, val_ds, train_dl, val_dl = build_loaders(config, train_h5, val_h5)
    model = MODELS.build(_model_of(config), config).to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f'train: {train_ds.describe()}')
    print(f'val  : {val_ds.describe()}')
    # The geometry is printed because it is the sweep axis: `latent_dim` alone does not
    # identify a run.
    k = config.model.latent_hw
    geom = f'{k}x{k}x{config.model.latent_dim // (k * k)}'
    print(f'model: encoder {n_enc/1e6:.2f}M + decoder {n_dec/1e6:.2f}M = '
          f'{(n_enc+n_dec)/1e6:.2f}M params   latent_dim={config.model.latent_dim}   '
          f'device={device}')
    print(f'latent: {geom} = {config.model.latent_dim} numbers   '
          f'encoder={type(model.encoder).__name__}   '
          f'decoder_pad_blocks={config.model.decoder_pad_blocks}')
    print(f'compression: {config.model.in_channels*config.data.image_size**2} values -> '
          f'{config.model.latent_dim} '
          f'({config.model.in_channels*config.data.image_size**2/config.model.latent_dim:.0f}x)')

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr,
                                  weight_decay=config.training.weight_decay)
    autocast, scaler = amp_context(config, device)

    start_epoch, step = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch, step = ck['epoch'] + 1, ck['step']
        print(f'resumed from {args.resume} at epoch {start_epoch}, step {step}')

    with open(os.path.join(workdir, 'config.json'), 'w') as fh:
        json.dump(as_dict(config), fh, indent=2)
    csv_path = os.path.join(workdir, 'curve.csv')
    write_header = not os.path.exists(csv_path)
    csv_fh = open(csv_path, 'a', newline='')
    csv_w = csv.writer(csv_fh)
    if write_header:
        csv_w.writerow(['epoch', 'step', 'train_mse', 'train_zero', 'train_mean',
                        'val_mse', 'val_zero', 'val_mean', 'val_snr_db', 'val_sam_deg',
                        'lr', 'secs'])

    total_steps = config.training.max_steps or config.training.n_epochs * max(len(train_dl), 1)

    best = float('inf')
    stop = False
    lr = config.training.lr   # referenced by the CSV row even if an epoch yields no batches
    for epoch in range(start_epoch, config.training.n_epochs):
        t0 = time.time()
        run, run_zero, run_mean, nb, nb_mean = 0.0, 0.0, 0.0, 0, 0

        for batch in train_dl:
            if step == 0:
                batch_requires(batch, RECON_NEEDS, consumer='the reconstruction loop',
                               dataset_name=_dataset_of(config))
            cube = batch['cube']
            cube = cube.to(device, non_blocking=True)
            lr = set_lr(optimizer, step, total_steps, config)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                recon, _ = model(cube)
                # The SAM term is added to the training objective only. Validation keeps
                # reporting plain MSE (and SNR/SAM as metrics), so a run with sam_weight>0
                # stays comparable to every run without it -- changing the training loss and
                # the reported score together would make the sweep unreadable.
                loss, _parts = combined_loss(recon.float(), cube,
                                             config.training.sam_weight)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
                optimizer.step()

            base = baseline_losses(cube)
            run += loss.item(); run_zero += base['zero']; nb += 1
            if not math.isnan(base['mean']):
                run_mean += base['mean']; nb_mean += 1
            step += 1

            if step % config.training.log_every == 0:
                print(f'  e{epoch} s{step:6d}  mse={run/nb:.4f}  '
                      f'[zero={run_zero/nb:.4f} '
                      f'mean={run_mean/nb_mean if nb_mean else float("nan"):.4f}]  lr={lr:.2e}')
            if config.training.max_steps and step >= config.training.max_steps:
                stop = True
                break

        train_mse = run / max(nb, 1)
        metrics = {'train_mse': train_mse,
                   'train_zero': run_zero / max(nb, 1),
                   'train_mean': run_mean / nb_mean if nb_mean else float('nan')}

        if epoch % config.training.eval_every == 0 or stop:
            metrics.update(evaluate(model, val_dl, device, autocast))
        else:
            metrics.update({'val_mse': '', 'val_zero': '', 'val_mean': '',
                            'val_snr_db': '', 'val_sam_deg': ''})

        secs = time.time() - t0
        # The two baselines are printed alongside the loss on every line: an MSE that fell is
        # only progress if it is below `mean`, which is what the decoder reaches from its bias
        # alone.
        print(f'epoch {epoch}: train_mse={train_mse:.4f} '
              f'val_mse={metrics["val_mse"] if metrics["val_mse"] != "" else float("nan"):} '
              f'| baselines val zero={metrics.get("val_zero", "")} mean={metrics.get("val_mean", "")} '
              f'| snr={metrics.get("val_snr_db", "")} dB sam={metrics.get("val_sam_deg", "")} deg '
              f'| {secs:.1f}s')
        csv_w.writerow([epoch, step, f'{train_mse:.6f}', f'{metrics["train_zero"]:.6f}',
                        f'{metrics["train_mean"]:.6f}', metrics['val_mse'], metrics['val_zero'],
                        metrics['val_mean'], metrics['val_snr_db'], metrics['val_sam_deg'],
                        f'{lr:.3e}', f'{secs:.1f}'])
        csv_fh.flush()

        if isinstance(metrics['val_mse'], float) and metrics['val_mse'] < best:
            best = metrics['val_mse']
            save_checkpoint(os.path.join(workdir, 'best.pt'), model, optimizer, epoch, step,
                            config, metrics)
        if (epoch + 1) % config.training.ckpt_every == 0 or stop:
            save_checkpoint(os.path.join(workdir, 'last.pt'), model, optimizer, epoch, step,
                            config, metrics)
        if stop:
            print(f'stopped at max_steps={config.training.max_steps}')
            break

    csv_fh.close()
    train_ds.close(); val_ds.close()
    print(f'done. best val_mse={best:.4f}  workdir={workdir}')


if __name__ == '__main__':
    main()
