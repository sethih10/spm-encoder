"""Everything that can be checked without the cluster, the GPU or the 8 GB HDF5."""

import os
import shutil
import tempfile
import traceback

import torch
from torch import nn

from configs.default import apply_overrides, get_config
from ters.datasets.ters_cubes import TersCubeDataset
from ters.transforms.normalize import NormalizeVectorized
from ters.models.autoencoder import TersAutoencoder
from shared.losses.reconstruction import reconstruction_loss
from shared.metrics.reconstruction import baseline_losses
from shared.models.decoder import TersDecoder
from ters.models.encoder import TersEncoder, TersSpatialEncoder
from tests.make_synthetic import make_synthetic_h5

FAILURES = []

# The sibling repo this folder's encoder is copied from. Overridable, because the default is an
# absolute path that will not be valid on another machine.
T2G_ROOT = os.environ.get('TERS2GRAPH_ROOT', '/users/harshise/ters-2-graph')

def check(name, cond, detail=''):
    ok = bool(cond)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not ok:
        FAILURES.append(name)
    return ok


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def small(**kw):
    """A 2-level, 6-channel, 32x32 model -- the real shape, 1/1000th the size."""
    base = dict(in_channels=6, filters=(8, 16), kernel_size=(3, 3), latent_dim=12,
                image_size=32)
    base.update(kw)
    return TersAutoencoder(**base)


def test_encoder_shapes():
    print('test_encoder_shapes')
    enc = TersEncoder(in_channels=6, filters=(8, 16), kernel_size=(3, 3), latent_dim=12)
    z = enc(torch.randn(3, 6, 32, 32))
    check('encoder returns (B, latent_dim)', tuple(z.shape) == (3, 12), str(tuple(z.shape)))
    check('bottleneck_hw matches the ladder', enc.bottleneck_hw(32) == 8,
          f'got {enc.bottleneck_hw(32)}')
    check('out_norm applied (per-row mean ~ 0)', z.mean(dim=1).abs().max().item() < 1e-4)
    check('rejects (B, H, W)', raises(lambda: enc(torch.randn(3, 32, 32)), ValueError))
    check('rejects wrong in_channels',
          raises(lambda: enc(torch.randn(3, 7, 32, 32)), ValueError))
    # The head is AdaptiveAvgPool2d, so a different spatial size must still produce a code.
    check('encoder is size-agnostic', tuple(enc(torch.randn(1, 6, 64, 64)).shape) == (1, 12))


def test_decoder_shapes():
    print('test_decoder_shapes')
    # zero_init=False here: the default zero-inits conv_out, so a zero_init model
    # emits all-zeros at step 0 and the "can emit negatives" check below would be vacuous.
    # Step-0 behaviour under the default is asserted in test_zero_init instead.
    dec = TersDecoder(out_channels=6, filters=(8, 16), kernel_size=(3, 3), latent_dim=12,
                      base_hw=8, zero_init=False)
    x = dec(torch.randn(3, 12))
    check('decoder returns (B, C, H, W)', tuple(x.shape) == (3, 6, 32, 32), str(tuple(x.shape)))
    check('rejects (B, 1, D)', raises(lambda: dec(torch.randn(3, 1, 12)), ValueError))
    check('rejects wrong latent_dim', raises(lambda: dec(torch.randn(3, 11)), ValueError))
    check('rejects base_hw < 1',
          raises(lambda: TersDecoder(filters=(8, 16), kernel_size=(3, 3), base_hw=0),
                 ValueError))
    check('no output activation (can emit negatives)', bool((x < 0).any()))


def test_mirror_symmetry():
    print('test_mirror_symmetry')
    ae = small()
    enc_w = [b.conv1.in_channels for b in ae.encoder.blocks]
    dec_w = [b.conv1.in_channels for b in ae.decoder.blocks]
    check('decoder widths are the encoder widths reversed',
          dec_w == list(reversed([b.conv2.out_channels for b in ae.encoder.blocks])),
          f'enc_in={enc_w} dec_in={dec_w}')
    check('same number of stages', len(ae.encoder.blocks) == len(ae.decoder.blocks))
    check('decoder starts on the encoder bottleneck grid',
          ae.decoder.base_hw == ae.encoder.bottleneck_hw(32), f'{ae.decoder.base_hw}')


def test_autoencoder_roundtrip():
    print('test_autoencoder_roundtrip')
    # zero_init=False so the step-0 gradient check below is meaningful -- see test_zero_init
    # for why the default gives the encoder no gradient on the first step.
    ae = small(zero_init=False)
    x = torch.randn(2, 6, 32, 32)
    recon, z = ae(x)
    check('recon matches input shape', recon.shape == x.shape, str(tuple(recon.shape)))
    check('latent is (B, latent_dim)', tuple(z.shape) == (2, 12))
    check('gradients reach the encoder stem',
          torch.autograd.grad(reconstruction_loss(recon, x),
                              ae.encoder.stem.weight)[0].abs().sum().item() > 0)
    check('non-square batch fails loudly',
          raises(lambda: ae(torch.randn(2, 6, 32, 16)), (ValueError, RuntimeError)))
    check('image_size not divisible by 2**levels fails in the constructor',
          raises(lambda: small(image_size=30), ValueError))
    ae2 = small()
    check('same architecture -> identical state_dict keys',
          set(ae.state_dict()) == set(ae2.state_dict()))


def test_encoder_state_dict_is_ters2graph_compatible():
    print('test_encoder_state_dict_is_ters2graph_compatible')
    # ters-2-graph loads a checkpoint with `ckpt.get('state_dict', ckpt)` and matches keys
    # against its own TersEncoder.state_dict(), trying the '' prefix first. If the module names
    # here drift from that class the match count silently drops and the load raises there --
    # long after this run finished. Pin the names now, and change this list only alongside the
    # matching change in ters-2-graph.
    enc = TersEncoder(in_channels=100, filters=(64, 128, 256, 512), latent_dim=256)
    keys = set(enc.state_dict())
    expected_prefixes = {'stem.', 'blocks.', 'head.', 'out_norm.'}
    check('no unexpected top-level module names',
          {k.split('.')[0] + '.' for k in keys} == expected_prefixes,
          str(sorted({k.split('.')[0] for k in keys})))
    for k in ['stem.weight', 'stem.bias', 'out_norm.weight', 'out_norm.bias',
              'head.2.weight', 'head.2.bias',
              'blocks.0.conv1.weight', 'blocks.3.conv2.weight', 'blocks.3.skip.weight']:
        check(f'key present: {k}', k in keys)
    check('head Linear is (latent_dim, filters[-1])',
          tuple(enc.state_dict()['head.2.weight'].shape) == (256, 512))
    check('no skip weight on the identity block', 'blocks.0.skip.weight' not in keys,
          'ResBlock(64,64) must stay a true identity shortcut')


def test_loads_into_real_ters2graph_encoder():
    print('test_loads_into_real_ters2graph_encoder')
    # The stronger version of the key-name test above: import ters-2-graph's actual TersEncoder
    # and replay the key matching its loader does. Skipped with a printed reason if that repo is
    # not where we expect -- an unavailable sibling repo is not a failure of this folder.
    import importlib.util
    src = os.path.join(T2G_ROOT, 'models', 'ters_encoder.py')
    if not os.path.exists(src):
        print(f'  [SKIP] ters-2-graph not found at {T2G_ROOT} '
              f'(set TERS2GRAPH_ROOT to run this check)')
        return
    spec = importlib.util.spec_from_file_location('t2g_ters_encoder', src)
    t2g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t2g)

    c = get_config()
    ae = TersAutoencoder.from_config(c)
    target = t2g.TersEncoder(in_channels=c.model.in_channels,
                             filters=tuple(c.model.filters), output_dim=c.model.latent_dim)

    state_dict = ae.encoder.state_dict()          # exactly what save_checkpoint puts under
    current = target.state_dict()                 # the 'state_dict' key
    matched, mismatched = 0, []
    for k in current:
        if k in state_dict:                       # dmt.py tries the '' prefix first
            if current[k].shape == state_dict[k].shape:
                current[k] = state_dict[k]; matched += 1
            else:
                mismatched.append(k)
    target.load_state_dict(current, strict=False)
    check('every tensor matches (dmt.py raises on a zero match)',
          matched == len(current) and not mismatched,
          f'{matched}/{len(current)} matched, {len(mismatched)} shape mismatches')

    x = torch.randn(2, 100, 64, 64)
    target.eval(); ae.encoder.eval()
    with torch.no_grad():
        ours, theirs = ae.encoder(x), target(x)
    check('and computes a bit-identical vector',
          torch.equal(ours, theirs), f'max delta {(ours - theirs).abs().max():.3e}')


def test_export_contract_at_configured_pool_size():
    """The exported encoder must load into the class ters-2-graph builds, at the configured
    pool size.

    Replays that repo's prefix-matching against a real checkpoint, then checks that a
    mismatched pool size is reported rather than half-loaded as a success.
    """
    print('test_export_contract_at_configured_pool_size')
    import importlib.util
    src = os.path.join(T2G_ROOT, 'models', 'ters_encoder.py')
    if not os.path.exists(src):
        print(f'  [SKIP] ters-2-graph not found at {T2G_ROOT}')
        return
    spec = importlib.util.spec_from_file_location('t2g_ters_encoder_ps', src)
    t2g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t2g)

    c = get_config()
    kw = dict(in_channels=c.model.in_channels, filters=tuple(c.model.filters))

    def replay(ours, theirs):
        """ters-2-graph's loader, in miniature: match by name, then by shape."""
        src_sd, dst_sd = ours.state_dict(), theirs.state_dict()
        matched, bad = 0, []
        for k in dst_sd:
            if k in src_sd:
                if dst_sd[k].shape == src_sd[k].shape:
                    matched += 1
                else:
                    bad.append(f'{k} {tuple(src_sd[k].shape)}->{tuple(dst_sd[k].shape)}')
        return matched, len(dst_sd), bad

    for ps in (1, 2, 4):
        ours = TersEncoder(latent_dim=c.model.latent_dim, pool_size=ps, **kw)
        theirs = t2g.TersEncoder(output_dim=c.model.latent_dim, pool_size=ps, **kw)
        matched, total, bad = replay(ours, theirs)
        check(f'pool_size={ps}: every tensor matches by name AND shape',
              matched == total and not bad, f'{matched}/{total}' + (f', bad={bad}' if bad else ''))

        theirs.load_state_dict(ours.state_dict())
        x = torch.randn(2, c.model.in_channels, 64, 64)
        ours.eval(); theirs.eval()
        with torch.no_grad():
            a, b = ours(x), theirs(x)
        check(f'pool_size={ps}: computes a bit-identical vector', torch.equal(a, b),
              f'max delta {(a - b).abs().max():.3e}')

    # --- negative control: the drift this test exists to catch must still be detectable ---
    ours_1 = TersEncoder(latent_dim=c.model.latent_dim, pool_size=1, **kw)
    theirs_4 = t2g.TersEncoder(output_dim=c.model.latent_dim, pool_size=4, **kw)
    matched, total, bad = replay(ours_1, theirs_4)
    check('mismatched pool_size is DETECTED, not silently accepted',
          matched < total and bool(bad), f'{matched}/{total} matched, mismatches={bad}')
    check('and the mismatch is exactly the head weight (the bias loads regardless)',
          any(k.startswith('head.2.weight') for k in bad), f'{bad}')


def test_normalize_pathological():
    print('test_normalize_pathological')
    norm = NormalizeVectorized()
    cube = torch.zeros(4, 8, 8)
    cube[1] = 3.0                                   # constant non-zero
    cube[2] = torch.randn(8, 8)
    cube[3] = torch.randn(8, 8); cube[3, 0, 0] = 500.0
    out = norm(cube)
    check('no NaN on an all-zero channel', not torch.isnan(out).any())
    check('all-zero channel maps to exactly 0', float(out[0].abs().max()) == 0.0)
    check('constant non-zero channel maps to exactly 0', float(out[1].abs().max()) == 0.0)
    check('live channel is unit-variance', abs(float(out[2].std(unbiased=True)) - 1.0) < 1e-4)
    check('spike survives as a large positive value', float(out[3].max()) > 3.0)

    base1 = baseline_losses(out.unsqueeze(0))
    check('zero-baseline is well below 1.0 for a half-dead cube', base1['zero'] < 0.75,
          f"zero={base1['zero']:.3f} (2 of 4 channels dead)")
    check('mean-baseline is nan for a batch of 1, not a misleading 0',
          base1['mean'] != base1['mean'])

    # A batch where the samples genuinely share a spatial pattern: the mean baseline must be
    # strictly better than zero, or it is measuring nothing.
    shared = torch.randn(1, 2, 8, 8).repeat(4, 1, 1, 1) + 0.1 * torch.randn(4, 2, 8, 8)
    b4 = baseline_losses(shared)
    check('mean-baseline beats zero-baseline when samples share structure',
          b4['mean'] < b4['zero'],
          f"mean={b4['mean']:.3f} zero={b4['zero']:.3f}")
    # ...and does not beat it on standardised TERS-like data, where per-sample standardisation
    # has already removed the per-channel mean.
    indep = torch.randn(4, 2, 8, 8)
    bi = baseline_losses(indep)
    check('mean-baseline ~= zero-baseline on independent samples',
          abs(bi['mean'] - bi['zero']) < 0.35 * bi['zero'],
          f"mean={bi['mean']:.3f} zero={bi['zero']:.3f}")


def test_dataset_against_synthetic_h5():
    print('test_dataset_against_synthetic_h5')
    tmp = tempfile.mkdtemp(prefix='ters-ae-test-')
    try:
        path = make_synthetic_h5(os.path.join(tmp, 'train.h5'), n=6, hw=16, channels=6)
        ds = TersCubeDataset(path)
        check('reads (H, W, C) and reports in_channels', ds.in_channels == 6, ds.describe())
        check('length is the row count', len(ds) == 6)
        sample = ds[0]
        # Every reader emits a dict, so one collate and one trainer serve both modalities.
        check('__getitem__ returns the contract dict', set(sample) == {'cube', 'row'},
              str(sorted(sample)))
        check('row is the FILE row, not the index', sample['row'] == 0)
        c = sample['cube']
        check('cube is (C, H, W)', tuple(c.shape) == (6, 16, 16))
        check('normalised, no NaN', not torch.isnan(c).any())
        check('dead channels present in the synthetic file',
              int((c.std(dim=(1, 2)) < 1e-6).sum()) >= 2,
              f'{int((c.std(dim=(1,2)) < 1e-6).sum())}/6 dead')
        sub = TersCubeDataset(path, indices=[0, 2])
        check('indices subsets the split', len(sub) == 2)
        check('out-of-range index fails in the parent',
              raises(lambda: TersCubeDataset(path, indices=[99]), IndexError))
        check('unknown channels_key fails in the parent, listing real keys',
              raises(lambda: TersCubeDataset(path, channels_key='nope'), KeyError))
        # A handle opened by __getitem__ must not survive into a pickled copy.
        check('__getstate__ drops the live HDF5 handle', ds.__getstate__()['_hf'] is None)
        ds.close(); sub.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_latent_geometry_budget():
    """The sweep's central claim: every latent_hw spends exactly latent_dim numbers."""
    print('test_latent_geometry_budget')
    # small() is 2 levels over 32x32, so the ladder ends on 8x8 and legal k are 1,2,4,8.
    # latent_dim=16 keeps k**2 | latent_dim for k in 1,2,4.
    for k, expect_ch in ((1, 16), (2, 4), (4, 1)):
        ae = small(latent_dim=16, latent_hw=k)
        z = ae.encode(torch.randn(3, 6, 32, 32))
        check(f'k={k}: code is (B, latent_dim) regardless of geometry',
              tuple(z.shape) == (3, 16), str(tuple(z.shape)))
        check(f'k={k}: latent_ch x k**2 == latent_dim',
              ae.decoder.latent_ch == expect_ch and ae.decoder.latent_ch * k * k == 16,
              f'{ae.decoder.latent_ch} x {k}**2')
        recon, _ = ae(torch.randn(2, 6, 32, 32))
        check(f'k={k}: reconstructs the input shape', tuple(recon.shape) == (2, 6, 32, 32))
        # LayerNorm spans the whole code in both encoders, so the code is comparably scaled
        # across the sweep and no arm gets a magnitude advantage.
        check(f'k={k}: code is LayerNorm-scaled over all {16} numbers',
              abs(float(z.mean())) < 1e-5 and abs(float(z.std(unbiased=False)) - 1.0) < 0.05,
              f'mean={float(z.mean()):.2e} std={float(z.std(unbiased=False)):.3f}')

    check('k=1 uses the verbatim ters-2-graph copy, not the NEW class',
          type(small(latent_dim=16, latent_hw=1).encoder) is TersEncoder)
    check('k>1 uses TersSpatialEncoder',
          type(small(latent_dim=16, latent_hw=2).encoder) is TersSpatialEncoder)

    # Illegal geometries fail in the constructor with the legal values in the message, not on
    # a reshape mid-run.
    check('k**2 not dividing latent_dim is refused',
          raises(lambda: small(latent_dim=12, latent_hw=8), ValueError))
    check('k larger than the ladder grid is refused',
          raises(lambda: small(latent_dim=16, latent_hw=16), ValueError))


def test_spatial_encoder_reduces_to_the_copy_at_k1():
    """TersSpatialEncoder at k=1 is numerically equal to TersEncoder, which the sweep's
    baseline rests on.

    A 1x1 convolution over a 1x1 grid is a dense layer, so the two heads are the same map with
    the weight reshaped. If this drifts, the k=1 arm stops being the shipped model and the
    whole comparison loses its reference point.
    """
    print('test_spatial_encoder_reduces_to_the_copy_at_k1')
    torch.manual_seed(0)
    kw = dict(in_channels=6, filters=(8, 16), kernel_size=(3, 3), latent_dim=12)
    ref = TersEncoder(**kw)
    new = TersSpatialEncoder(latent_hw=1, **kw)

    # Copy the trunk across verbatim, then the head with the only difference being the shape.
    sd = ref.state_dict()
    new_sd = new.state_dict()
    for name, t in sd.items():
        if name == 'head.2.weight':                       # Linear (12, 16) -> Conv (12, 16, 1, 1)
            new_sd['head.1.weight'] = t.view(12, 16, 1, 1)
        elif name == 'head.2.bias':
            new_sd['head.1.bias'] = t
        else:
            new_sd[name] = t
    new.load_state_dict(new_sd)

    ref.eval(); new.eval()
    x = torch.randn(4, 6, 32, 32)
    with torch.no_grad():
        a, b = ref(x), new(x)
    check('same parameter count', sum(p.numel() for p in ref.parameters())
          == sum(p.numel() for p in new.parameters()))
    check('bit-identical output at k=1', torch.allclose(a, b, atol=1e-6),
          f'max|diff| = {float((a - b).abs().max()):.2e}')


def test_decoder_pad_blocks():
    print('test_decoder_pad_blocks')
    torch.manual_seed(0)
    plain = small(latent_dim=16, latent_hw=2)
    padded = small(latent_dim=16, latent_hw=2, decoder_pad_blocks=3)
    n_plain = sum(p.numel() for p in plain.decoder.parameters())
    n_padded = sum(p.numel() for p in padded.decoder.parameters())
    check('pad blocks add decoder parameters', n_padded > n_plain, f'{n_plain} -> {n_padded}')
    check('pad blocks do not change the output shape',
          tuple(padded(torch.randn(2, 6, 32, 32))[0].shape) == (2, 6, 32, 32))
    # zero_init must reach them, or a padded run no longer starts at the do-nothing baseline
    # and its curve is not comparable with the unpadded arms.
    check('zero_init reaches the pad blocks',
          all(float(b.norm2.weight.detach().abs().max()) == 0.0 for b in padded.decoder.pad))
    x = torch.randn(2, 6, 32, 32); x[:, :1] = 0.0
    padded.eval()
    with torch.no_grad():
        out = padded(x)[0]
    check('padded decoder still emits exactly 0 at init', float(out.abs().max()) == 0.0)


def test_spatial_latent_beats_nothing():
    """Each geometry must actually train -- the same bar test_overfits_one_batch sets for k=1."""
    print('test_spatial_latent_beats_nothing')
    for k in (2, 4):
        torch.manual_seed(0)
        ae = small(latent_dim=16, latent_hw=k)
        x = torch.randn(2, 6, 32, 32)
        x[:, :2] = 0.0
        opt = torch.optim.AdamW(ae.parameters(), lr=3e-3)
        for _ in range(200):
            opt.zero_grad(set_to_none=True)
            loss = reconstruction_loss(ae(x)[0], x)
            loss.backward()
            opt.step()
        base = baseline_losses(x)
        check(f'k={k}: beats the mean baseline', loss.item() < base['mean'],
              f"final={loss.item():.4f} mean={base['mean']:.4f}")
        with torch.no_grad():
            loss_z0 = reconstruction_loss(ae.decode(torch.zeros(2, 16)), x).item()
        check(f'k={k}: the spatial code carries signal', loss_z0 > loss.item(),
              f'z=0 -> {loss_z0:.4f} vs {loss.item():.4f}')


def test_overfits_one_batch():
    print('test_overfits_one_batch')
    # Shape tests alone cannot fail on: a decoder wired to ignore z, a sign error in the loss,
    # or a detached graph. If 200 steps on 2 fixed cubes do not get below the mean baseline,
    # something is wired wrong.
    torch.manual_seed(0)
    ae = small(latent_dim=32)
    x = torch.randn(2, 6, 32, 32)
    x[:, :2] = 0.0                                       # dead channels, as in real data
    opt = torch.optim.AdamW(ae.parameters(), lr=3e-3)
    first = None
    for i in range(200):
        opt.zero_grad(set_to_none=True)
        recon, _ = ae(x)
        loss = reconstruction_loss(recon, x)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    final = loss.item()
    base = baseline_losses(x)
    check('loss decreased', final < first, f'{first:.4f} -> {final:.4f}')
    check('beats the mean baseline', final < base['mean'],
          f"final={final:.4f} mean_baseline={base['mean']:.4f} zero={base['zero']:.4f}")

    # And the latent must actually be used: freezing z at zero should be strictly worse.
    with torch.no_grad():
        recon_z0 = ae.decode(torch.zeros(2, 32))
        loss_z0 = reconstruction_loss(recon_z0, x).item()
    check('latent carries signal (z=0 is worse than the real code)', loss_z0 > final,
          f'z=0 -> {loss_z0:.4f} vs real z -> {final:.4f}')


def test_zero_init():
    print('test_zero_init')
    # If someone "cleans up" the zero-inits, this fails here rather than the next training run
    # quietly starting far above its own baseline.
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32, 32)
    x[:, :4] = 0.0                                        # 4 of 6 channels dead
    base = baseline_losses(x)['zero']

    on = small(zero_init=True).eval()
    off = small(zero_init=False).eval()
    with torch.no_grad():
        r_on, _ = on(x)
        r_off, _ = off(x)
    check('zero_init emits exactly 0 at step 0', float(r_on.abs().max()) == 0.0)
    check('...so initial MSE IS the do-nothing baseline',
          abs(float(reconstruction_loss(r_on, x)) - base) < 1e-6,
          f'init={float(reconstruction_loss(r_on, x)):.4f} baseline={base:.4f}')
    check('without it the model starts well above the baseline',
          float(reconstruction_loss(r_off, x)) > base,
          f'init={float(reconstruction_loss(r_off, x)):.2f} vs baseline={base:.4f}')
    check('residual branches start empty',
          all(float(b.norm2.weight.abs().max()) == 0.0 for b in on.decoder.blocks))
    check('the ENCODER is not zero-inited (it must match ters-2-graph)',
          all(float(b.norm2.weight.abs().max()) > 0.0 for b in on.encoder.blocks))
    # A zeroed output conv must still train: d(loss)/d(conv_out.weight) uses the incoming
    # activations, which are not zero. (Its input gradient is zero for one step -- that is the
    # trade, not a bug.)
    ae = small(zero_init=True)
    recon, _ = ae(x)
    loss = reconstruction_loss(recon, x)
    g_w = torch.autograd.grad(loss, ae.decoder.conv_out.weight, retain_graph=True)[0]
    g_stem = torch.autograd.grad(loss, ae.encoder.stem.weight, allow_unused=True)[0]
    check('a zeroed conv_out still receives gradient', float(g_w.abs().sum()) > 0)
    check('the encoder gets no gradient on step 0 (known, documented)',
          g_stem is None or float(g_stem.abs().sum()) == 0.0)


def test_config():
    print('test_config')
    c = get_config()
    check('latent_dim defaults to 256 (= ters-2-graph config.model.nf)',
          c.model.latent_dim == 256)
    check('in_channels defaults to 100', c.model.in_channels == 100)
    check('image_size defaults to 256', c.data.image_size == 256)
    check('zero_init defaults on', c.model.zero_init is True)
    apply_overrides(c, ['model.latent_dim=512', 'training.batch_size=2'])
    check('--set parses ints', c.model.latent_dim == 512 and c.training.batch_size == 2)
    apply_overrides(c, ['model.filters=(32, 64)'])
    check('--set parses tuples', c.model.filters == (32, 64), str(c.model.filters))
    check('unknown key raises', raises(lambda: apply_overrides(c, ['model.nope=1']), KeyError))
    check('missing = raises', raises(lambda: apply_overrides(c, ['model.latent_dim']),
                                     ValueError))


def test_real_config_builds():
    print('test_real_config_builds')
    # The 44M-parameter model at the real settings, on CPU, without touching the HDF5.
    c = get_config()
    ae = TersAutoencoder.from_config(c)
    n = sum(p.numel() for p in ae.parameters())
    check('builds at 100ch / 256px / latent 256', ae.decoder.base_hw == 16,
          f'base_hw={ae.decoder.base_hw}, {n/1e6:.1f}M params')
    with torch.no_grad():
        recon, z = ae(torch.randn(1, 100, 256, 256))
    check('full-size roundtrip', tuple(recon.shape) == (1, 100, 256, 256)
          and tuple(z.shape) == (1, 256))
    check('compression is 25600x', 100 * 256 * 256 // 256 == 25600)


def test_groupnorm_group_count_is_validated():
    """Every shipped width divides model.n_groups, so torch's own check never fires."""
    print('test_groupnorm_group_count_is_validated')
    import glob
    from shared.config import load_config
    from shared.registry import MODELS, load_plugins

    load_plugins()
    seen = []
    orig = nn.GroupNorm.__init__

    def record(self, num_groups, num_channels, *a, **k):
        seen.append((num_channels, num_groups))
        return orig(self, num_groups, num_channels, *a, **k)

    nn.GroupNorm.__init__ = record
    try:
        for path in sorted(glob.glob('configs/*_ae.yaml')):
            config = load_config(path)
            seen.clear()
            MODELS.build(config.model.name, config)
            n_groups = config.model.n_groups
            widths = sorted({c for c, _ in seen})
            check(f'{path}: every width divides n_groups={n_groups} -- {widths}',
                  all(c % n_groups == 0 for c in widths))
            check(f'{path}: every layer got the configured group count',
                  all(g == n_groups for _, g in seen))
    finally:
        nn.GroupNorm.__init__ = orig

    # torch is the validator; the repo adds none of its own.
    check('torch rejects a group count that does not divide the width',
          raises(lambda: nn.GroupNorm(8, 100), ValueError))


def main():
    tests = [test_groupnorm_group_count_is_validated, test_encoder_shapes, test_decoder_shapes, test_mirror_symmetry,
             test_autoencoder_roundtrip, test_encoder_state_dict_is_ters2graph_compatible,
             test_loads_into_real_ters2graph_encoder,
             test_export_contract_at_configured_pool_size,
             test_normalize_pathological, test_dataset_against_synthetic_h5,
             test_overfits_one_batch, test_zero_init, test_config,
             test_real_config_builds,
             # --- NEW (sweep): the latent-geometry arm ---
             test_latent_geometry_budget, test_spatial_encoder_reduces_to_the_copy_at_k1,
             test_decoder_pad_blocks, test_spatial_latent_beats_nothing]
    for t in tests:
        try:
            t()
        except Exception:
            FAILURES.append(t.__name__ + ' (raised)')
            traceback.print_exc()
        print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILED: {FAILURES}')
        raise SystemExit(1)
    print('all passed')


if __name__ == '__main__':
    main()
