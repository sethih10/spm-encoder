"""The STM contrastive path: dataset, cube encoder, structure encoder."""

import math
import os
import shutil
import sys
import tempfile
import traceback

import torch

from stm.datasets.stm_pairs import StmPairDataset, collate_pairs
from stm.transforms.normalize import NormalizeStmCube, _slice_for
from shared.metrics.retrieval import size_confound_metrics
from stm.models.spectral_gpt import SpectralGptCubeEncoder, load_spectralgpt_weights
from stm.models.timm2d import STM2D_BACKBONES, Stm2dCubeEncoder, load_timm_backbone
from stm.models.conv3d import StmCubeEncoder
from tests.make_synthetic import make_synthetic_stm_h5

FAILURES = []
CKPT = os.environ.get('TORCHMD_CKPT',
                      '/scratch/project_2001912/sethih1/checkpoints/denoised-pcqm4mv2.ckpt')
SGPT_CKPT = os.environ.get(
    'SGPT_CKPT', '/scratch/project_2001912/sethih1/checkpoints/SpectralGPT+.pth')
STM2D_CKPT = os.environ.get(
    'STM2D_CKPT',
    '/scratch/project_2001912/sethih1/checkpoints/stm2d/clip_vit_b16.bin')

# resnet18 at random init: the smallest registered trunk, and it needs no network to build.
STM2D_TINY = dict(in_channels=14, image_size=32, latent_dim=16, backbone='resnet18',
                  resize=32, pretrained=False, verbose=False)

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


def test_normalize_stm_cube():
    """NormalizeStmCube preserves the decay envelope along z."""
    print('test_normalize_stm_cube')
    norm = NormalizeStmCube()
    z = 14
    # A cube with a deliberate monotone decay along z.
    base = torch.rand(z, 16, 16) * 1000
    cube = base * torch.linspace(1.0, 0.2, z)[:, None, None]
    out = norm(cube)
    means = out.mean(dim=(1, 2))
    check('whole-cube standardised (mean ~0, std ~1)',
          abs(out.mean().item()) < 1e-5 and abs(out.std().item() - 1) < 1e-3,
          f'mean={out.mean():.2e} std={out.std():.4f}')
    check('the z envelope survives (slice means still ordered)',
          means[0].item() > means[-1].item(), f'{means[0]:+.3f} -> {means[-1]:+.3f}')
    check('no NaN', bool(torch.isfinite(out).all()))

    # Pathological: a flat cube has zero std and must not become NaN.
    flat = norm(torch.full((z, 8, 8), 7.0))
    check('a flat cube yields zeros, not NaN',
          bool(torch.isfinite(flat).all()) and flat.abs().max().item() < 1e-6)

    # Contrast: the TERS normaliser, which standardises each channel alone, would destroy it.
    per_slice = (cube - cube.mean(dim=(1, 2), keepdim=True)) / cube.std(dim=(1, 2),
                                                                       keepdim=True)
    ps_means = per_slice.mean(dim=(1, 2))
    check('per-slice standardisation WOULD destroy it (the rejected alternative)',
          abs(ps_means[0].item() - ps_means[-1].item()) < 1e-5,
          f'{ps_means[0]:+.2e} -> {ps_means[-1]:+.2e}')


def test_pair_dataset_and_collate():
    print('test_pair_dataset_and_collate')
    tmp = tempfile.mkdtemp()
    try:
        path, split, counts = make_synthetic_stm_h5(os.path.join(tmp, 'stm.h5'), n=24, hw=32)
        ds = StmPairDataset(path)
        check('split discovered without being told', ds.split == split, str(ds.split))
        check('z-slice count read from the file', ds.in_channels == 14, str(ds.in_channels))
        check('all rows present', len(ds) == 24, str(len(ds)))

        it = ds[0]
        check('cube is (Z, H, W) channels-first', tuple(it['cube'].shape) == (14, 32, 32),
              str(tuple(it['cube'].shape)))
        check('the 2-atom molecule survives the padding mask', it['z'].numel() == 2,
              f"{it['z'].numel()} atoms")
        it1 = ds[1]
        check('a molecule with NO padding rows is read whole', it1['z'].numel() == 54,
              f"{it1['z'].numel()} atoms")
        check('atomic numbers are real elements',
              set(it1['z'].tolist()) <= {1, 6, 7, 8, 16, 17}, str(sorted(set(it1['z'].tolist()))))
        check('positions are centred', it1['pos'].mean(dim=0).abs().max().item() < 1e-5)

        b = collate_pairs([ds[i] for i in range(4)])
        check('cubes stack', tuple(b['cube'].shape) == (4, 14, 32, 32))
        check('atoms concatenate, not stack', b['z'].dim() == 1)
        check('batch vector length == atom count', b['batch'].numel() == b['z'].numel())
        check('n_atoms matches the batch vector',
              b['n_atoms'].tolist() == [int((b['batch'] == i).sum()) for i in range(4)],
              str(b['n_atoms'].tolist()))
        check('limit= truncates', len(StmPairDataset(path, limit=5)) == 5)

        # A file with no xyz is not a pair corpus, and must fail in the parent with a reason.
        import h5py
        bad = os.path.join(tmp, 'noxyz.h5')
        with h5py.File(path, 'r') as src, h5py.File(bad, 'w') as dst:
            dst.create_group(split).create_dataset('X', data=src[f'{split}/X'][:])
        check('a file without xyz raises, naming the missing key',
              raises(lambda: StmPairDataset(bad), KeyError))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stm_cube_encoder():
    print('test_stm_cube_encoder')
    enc = StmCubeEncoder(in_channels=14, latent_dim=32, filters=(8, 16), conv3d_channels=(4, 8), n_groups=4,
                         z_strides=(2, 1))
    out = enc(torch.randn(3, 14, 32, 32))
    check('returns (B, latent_dim)', tuple(out.shape) == (3, 32), str(tuple(out.shape)))
    check('out_norm applied (row mean ~0)', out.mean(dim=1).abs().max().item() < 1e-4)
    check('z is reduced but never crushed to 1', enc.z_out > 1, f'z_out={enc.z_out}')
    check('rejects (B, H, W)', raises(lambda: enc(torch.randn(3, 32, 32)), ValueError))
    check('rejects the wrong slice count',
          raises(lambda: enc(torch.randn(3, 10, 32, 32)), ValueError))
    check('rejects mismatched stem/stride lengths',
          raises(lambda: StmCubeEncoder(conv3d_channels=(4, 8), z_strides=(2,)), ValueError))
    check('size-agnostic (AdaptiveAvgPool head)',
          tuple(enc(torch.randn(1, 14, 64, 64)).shape) == (1, 32))

    # The operator must actually read the ordering along z: a reversed stack is a different
    # physical object and must not map to the same code.
    x = torch.randn(2, 14, 32, 32)
    enc.eval()
    with torch.no_grad():
        a, b = enc(x), enc(torch.flip(x, dims=[1]))
    # A 1-slice input is what StmPairDataset(z_slice=...) emits, so the 3D pooling has to be
    # gated on depth as well as on H and W: a depth-1 volume reaching MaxPool3d((2,2,2))
    # raises "Output size is too small".
    for zc in (1, 3):
        e1 = StmCubeEncoder(in_channels=zc, latent_dim=32, filters=(8, 16))
        check(f'stm3d accepts a {zc}-slice input (the --z-slice path)',
              tuple(e1(torch.randn(2, zc, 64, 64)).shape) == (2, 32),
              f'z_out={e1.z_out}')
    check('and the 14-slice z schedule is unchanged by that guard',
          StmCubeEncoder(in_channels=14, latent_dim=32).z_out == 3)

    check('reversing the z order changes the code', not torch.allclose(a, b, atol=1e-5),
          f'max delta {(a - b).abs().max():.3e}')

    # not `enc(x).sum()`: the encoder ends in a LayerNorm, which forces every row to zero mean,
    # so that sum is identically ~0 regardless of the weights and its gradient is mathematically
    # zero. A squared objective is not invariant to the LayerNorm and gives a real gradient.
    grad = torch.autograd.grad((enc(x) ** 2).sum(), enc.stem3d[0].conv1.weight)[0]
    check('gradients reach the 3D stem', grad.abs().sum().item() > 1e-8,
          f'{grad.abs().sum().item():.3e}')
    flat = torch.autograd.grad(enc(x).sum(), enc.stem3d[0].conv1.weight)[0]
    check('and sum() is degenerate here, which is why it is not used',
          flat.abs().sum().item() < 1e-4, f'{flat.abs().sum().item():.3e}')


def test_size_confound_metrics():
    """The metric that decides whether a retrieval result means anything."""
    print('test_size_confound_metrics')
    n = 32
    # All molecules the same size -> size carries no information, so the oracle is at chance.
    same = torch.full((n,), 20)
    z = torch.eye(n, 64)
    m = size_confound_metrics(z, z.clone(), same)
    check('equal sizes -> size-only oracle collapses to chance',
          abs(m['top1_size_only'] - 1.0 / n) < 1e-6,
          f"{m['top1_size_only']:.4f} vs {1 / n:.4f}")
    check('and every row has same-size distractors',
          m['n_size_controlled'] == n, str(m['n_size_controlled']))
    check('perfect embeddings still retrieve when size cannot help',
          m['top1_size_controlled'] == 1.0)

    # All distinct sizes -> size alone identifies every molecule.
    distinct = torch.arange(n)
    m2 = size_confound_metrics(z, z.clone(), distinct)
    check('distinct sizes -> size-only oracle is perfect', m2['top1_size_only'] == 1.0,
          f"{m2['top1_size_only']:.4f}")
    check('and no row has a same-size distractor, reported as such',
          m2['n_size_controlled'] == 0, str(m2['n_size_controlled']))
    check('undefined size-controlled retrieval is nan, not a fabricated 0',
          math.isnan(m2['top1_size_controlled']))

    check('too small a batch returns nothing rather than a meaningless number',
          size_confound_metrics(z[:2], z[:2].clone(), distinct[:2]) == {})

    # --- normalised rank must be batch-invariant, which `gain` is not ---
    # top1_size_controlled / chance has a ceiling of 1/chance, and chance falls as the batch
    # grows, so that ratio silently rewards a bigger batch. Normalised rank is anchored at
    # both ends at every batch size, and those two anchors are exact, so they are asserted
    # exactly.
    torch.manual_seed(0)
    rnds = []
    for b in (32, 64, 128, 256):
        na_b = torch.randint(2, 52, (b,))
        eye = torch.eye(b, 300)
        m_p = size_confound_metrics(eye, eye.clone(), na_b)
        m_w = size_confound_metrics(eye, -eye.clone(), na_b)
        check(f'batch {b}: a perfect ranking is exactly 0.0',
              abs(m_p['norm_rank_size_controlled']) < 1e-9)
        check(f'batch {b}: the worst ranking is exactly 1.0',
              abs(m_w['norm_rank_size_controlled'] - 1.0) < 1e-9)
        # average several draws: one draw over small groups is very noisy
        vals = [size_confound_metrics(torch.randn(b, 300), torch.randn(b, 300), na_b)
                ['norm_rank_size_controlled'] for _ in range(12)]
        rnds.append(sum(vals) / len(vals))
    for b, r in zip((32, 64, 128, 256), rnds):
        check(f'batch {b}: random ranking sits at chance (0.5)', abs(r - 0.5) < 0.08,
              f'{r:.3f}')
    check('and chance does NOT drift with batch size (unlike gain)',
          max(rnds) - min(rnds) < 0.08, f'spread {max(rnds) - min(rnds):.3f}')


def test_structure_encoder_loads_pretrained():
    print('test_structure_encoder_loads_pretrained')
    if not os.path.exists(CKPT):
        print(f'  [SKIP] no checkpoint at {CKPT} (set TORCHMD_CKPT to run this check)')
        return
    from shared.models.structure import StructureEncoder
    enc = StructureEncoder.from_checkpoint(CKPT, strict=True, verbose=False)
    check('out_dim is the ET width', enc.out_dim == 256, str(enc.out_dim))
    check('not frozen by default (MolSpectra trains it jointly)', not enc.frozen)

    z = torch.tensor([6, 1, 1, 1, 1, 8, 1])
    pos = torch.tensor([[0., 0., 0.], [.63, .63, .63], [-.63, -.63, .63],
                        [-.63, .63, -.63], [.63, -.63, -.63], [0., 0., 0.], [.96, 0., 0.]])
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1])
    enc.eval()
    with torch.no_grad():
        out = enc(z, pos, batch)
    check('one vector per molecule', tuple(out.shape) == (2, 256), str(tuple(out.shape)))
    check('finite', bool(torch.isfinite(out).all()))
    check('distinct molecules get distinct embeddings',
          not torch.allclose(out[0], out[1], atol=1e-4))
    check('mismatched atom/batch lengths raise',
          raises(lambda: enc(z, pos, batch[:-1]), ValueError))

    frozen = StructureEncoder.from_checkpoint(CKPT, freeze=True, verbose=False)
    check('freeze=True removes every gradient',
          not any(p.requires_grad for p in frozen.parameters()))
    frozen.train(True)
    check('and a frozen encoder stays in eval mode', not frozen.net.training)


SGPT_TINY = dict(embed_dim=32, depth=2, num_heads=4, image_size=32, patch_size=8,
                 latent_dim=16, in_channels=14)


def test_sgpt_cube_encoder():
    print('test_sgpt_cube_encoder')
    enc = SpectralGptCubeEncoder(**SGPT_TINY)
    x = torch.randn(3, 14, 32, 32)
    out = enc(x)
    check('returns (B, latent_dim)', tuple(out.shape) == (3, 16), str(tuple(out.shape)))
    check('out_norm applied (row mean ~0)', out.mean(dim=1).abs().max().item() < 1e-4)
    check('exposes the attributes build() reads',
          enc.latent_dim == 16 and enc.output_dim == 16)

    # The token arithmetic is the band_mode decision; if it drifts, the pretrained position
    # embeddings stop applying and nothing else would say so.
    for mode, z_in, t, tg in (('interpolate', 12, 3, 4), ('pad', 15, 3, 5), ('native', 14, 2, 7)):
        e = SpectralGptCubeEncoder(**{**SGPT_TINY, 'band_mode': mode})
        check(f'band_mode={mode!r}: z_in={z_in}, t_patch={t}, T={tg}',
              (e.z_in, e.t_patch_size, e.t_grid) == (z_in, t, tg),
              f'got {(e.z_in, e.t_patch_size, e.t_grid)}')
        check(f'band_mode={mode!r}: {tg * 16} tokens', e.num_patches == tg * 16,
              str(e.num_patches))
        check(f'band_mode={mode!r}: forward runs', tuple(e(x).shape) == (3, 16))

    # Resampling 14->12 may move the samples, but it must not flatten the z decay envelope --
    # the failure NormalizeStmCube exists to avoid, restated for the interpolation.
    env = torch.linspace(1.0, 0.1, 14).view(1, 14, 1, 1) * torch.ones(1, 14, 32, 32)
    z12 = enc._fit_bands(env)
    means = z12.mean(dim=(0, 2, 3))
    check('interpolate 14->12 keeps the z envelope monotone',
          bool((means[:-1] > means[1:]).all()), f'{means[0]:.3f} -> {means[-1]:.3f}')
    check('and it is a real resample, not a truncation', z12.shape[1] == 12)

    a = enc(env)
    b = enc(torch.flip(env, dims=[1]))
    check('reversing the z order changes the code', not torch.allclose(a, b, atol=1e-5),
          'a tubelet embedding that ignored z would give the same vector')

    # --- the checkpoint layout, pinned ---
    names = dict(enc.blocks[0].attn.named_parameters())
    check('attn stores q/k/v separately, as SpectralGPT+ does',
          {'q.weight', 'k.weight', 'v.weight'} <= set(names) and 'qkv.weight' not in names,
          'a fused qkv loads zero attention tensors from the real checkpoint')
    check('cls_embed defaults False, as SpectralGPT+ has no class token',
          not enc.cls_embed and not hasattr(enc, 'cls_token'))
    check("pool='cls' is refused when there is no class token",
          raises(lambda: SpectralGptCubeEncoder(**SGPT_TINY, pool='cls'), ValueError))

    check('rejects (B, H, W)', raises(lambda: enc(torch.randn(3, 32, 32)), ValueError))
    check('rejects the wrong slice count',
          raises(lambda: enc(torch.randn(3, 10, 32, 32)), ValueError))
    check('rejects the wrong spatial size (pos embed is tied to the grid)',
          raises(lambda: enc(torch.randn(3, 14, 64, 64)), ValueError))
    check('rejects a bad band_mode',
          raises(lambda: SpectralGptCubeEncoder(band_mode='nope'), ValueError))
    check('rejects indivisible embed_dim/num_heads',
          raises(lambda: SpectralGptCubeEncoder(**{**SGPT_TINY, 'num_heads': 5}), ValueError))

    # Freezing: the default trains the head only, which is what makes the arm affordable.
    frozen = SpectralGptCubeEncoder(**SGPT_TINY, trainable_blocks=0)
    trainable = {n for n, p in frozen.named_parameters() if p.requires_grad}
    check('trainable_blocks=0 trains only the head',
          trainable == {'head.weight', 'head.bias', 'out_norm.weight', 'out_norm.bias'},
          str(sorted(trainable)))
    check('trainable_blocks=-1 trains everything',
          all(p.requires_grad for p in
              SpectralGptCubeEncoder(**SGPT_TINY, trainable_blocks=-1).parameters()))
    last = SpectralGptCubeEncoder(**SGPT_TINY, trainable_blocks=1)
    check('trainable_blocks=1 trains the last block but not the first',
          all(p.requires_grad for p in last.blocks[-1].parameters())
          and not any(p.requires_grad for p in last.blocks[0].parameters()))
    # A frozen trunk must still pass gradient through to the head, or the arm trains nothing.
    frozen(x).pow(2).mean().backward()
    check('gradients still reach the head under a frozen trunk',
          frozen.head.weight.grad is not None
          and frozen.head.weight.grad.abs().sum().item() > 1e-8)


def test_sgpt_weight_loading():
    """The anti-silent-failure machinery, tested without the real 400 MB checkpoint.

    The failure mode guarded against is a partial load that prints a number and carries on.
    """
    print('test_sgpt_weight_loading')
    tmp = tempfile.mkdtemp()
    try:
        ref = SpectralGptCubeEncoder(**SGPT_TINY)
        trunk = {k: v for k, v in ref.state_dict().items()
                 if not k.startswith(('head.', 'out_norm.'))}

        # 1. A checkpoint with the expected names loads completely, and the head stays ours.
        path = os.path.join(tmp, 'ok.pth')
        torch.save({'model': trunk}, path)
        fresh = SpectralGptCubeEncoder(**SGPT_TINY)
        r = load_spectralgpt_weights(fresh, path, verbose=False)
        check('a well-named checkpoint loads 100% of the trunk', r['fraction'] == 1.0,
              f"{r['matched']}/{r['trunk']}")
        check('and the trunk really changed',
              torch.allclose(fresh.blocks[0].attn.q.weight, ref.blocks[0].attn.q.weight))
        check('the head is NOT taken from the checkpoint (it is ours)',
              not torch.allclose(fresh.head.weight, ref.head.weight))

        # 2. A 'module.' prefix and an MAE decoder must not break it -- both are in the wild.
        path2 = os.path.join(tmp, 'wrapped.pth')
        torch.save({'model': {**{'module.' + k: v for k, v in trunk.items()},
                              'decoder_blocks.0.norm1.weight': torch.zeros(8),
                              'mask_token': torch.zeros(1, 1, 32)}}, path2)
        r2 = load_spectralgpt_weights(SpectralGptCubeEncoder(**SGPT_TINY), path2, verbose=False)
        check("a 'module.' prefix and MAE decoder keys are handled", r2['fraction'] == 1.0,
              f"{r2['matched']}/{r2['trunk']}")

        # 3. Shape adaptation: the same checkpoint into a different band_mode. The tubelet
        #    depth (3->2) and the temporal position embedding (T 4->7) must be resampled.
        native = SpectralGptCubeEncoder(**{**SGPT_TINY, 'band_mode': 'native'})
        r3 = load_spectralgpt_weights(native, path, verbose=False)
        check('band_mode=native still loads the trunk by adapting shapes',
              r3['fraction'] == 1.0, f"{r3['matched']}/{r3['trunk']}")
        check('and it says WHICH tensors it adapted', len(r3['adapted']) == 2,
              str(r3['adapted']))
        check('the adapted patch embed has the new tubelet depth',
              native.patch_embed.proj.weight.shape[2] == 2)
        check('a native-mode forward still runs', tuple(native(torch.randn(2, 14, 32, 32)).shape)
              == (2, 16))

        # 4. A checkpoint that does not match must raise, not warn.
        bad = os.path.join(tmp, 'bad.pth')
        torch.save({'model': {'backbone.stage1.conv.weight': torch.zeros(4, 4, 3, 3)}}, bad)
        check('a checkpoint that does not match RAISES rather than training random weights',
              raises(lambda: load_spectralgpt_weights(
                  SpectralGptCubeEncoder(**SGPT_TINY), bad, verbose=False), RuntimeError))
        # ...and the escape hatch used by tools/inspect_sgpt_ckpt.py still reports.
        r4 = load_spectralgpt_weights(SpectralGptCubeEncoder(**SGPT_TINY), bad,
                                      min_match=0.0, verbose=False)
        check('min_match=0 reports instead of raising, for the inspector tool',
              r4['fraction'] == 0.0, f"{r4['matched']}/{r4['trunk']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_sgpt_loads_the_real_checkpoint():
    """The regression test the synthetic ones cannot be: the actual released file.

    The synthetic checkpoints above are built from this module's own state_dict, so they would
    pass even if every name were wrong -- they only test the adapter logic. This one tests the
    architecture guess itself.
    """
    print('test_sgpt_loads_the_real_checkpoint')
    if not os.path.exists(SGPT_CKPT):
        print(f'  [SKIP] no checkpoint at {SGPT_CKPT} (set SGPT_CKPT to run this check)')
        return
    enc = SpectralGptCubeEncoder(in_channels=14, image_size=128)
    r = load_spectralgpt_weights(enc, SGPT_CKPT, verbose=False)
    check('100% of the ViT-Base trunk loads from the released SpectralGPT+',
          r['fraction'] == 1.0, f"{r['matched']}/{r['trunk']}")
    check('198 trunk tensors, which is the checkpoint\'s own encoder-side count',
          r['trunk'] == 198, str(r['trunk']))
    check('band_mode=interpolate needs NO shape adaptation (1024 tokens either way)',
          r['adapted'] == [], str(r['adapted']))
    check('the pretrained tubelet really is 3 deep over 4 band groups',
          enc.patch_embed.proj.weight.shape[2] == 3 and enc.t_grid == 4)
    out = enc(torch.randn(2, 14, 128, 128))
    check('and a real-shape forward runs on the loaded weights',
          tuple(out.shape) == (2, 256) and bool(torch.isfinite(out).all()))

    # native mode is the control arm; it must load too, by adapting the two spectral tensors.
    nat = SpectralGptCubeEncoder(in_channels=14, image_size=128, band_mode='native')
    rn = load_spectralgpt_weights(nat, SGPT_CKPT, verbose=False)
    check('band_mode=native also loads 100%, having adapted the spectral tensors',
          rn['fraction'] == 1.0 and len(rn['adapted']) == 2,
          f"{rn['fraction']:.0%}, adapted={rn['adapted']}")

    # The inspector must not crash: a diagnostic that raises while describing a healthy
    # checkpoint reads as "the checkpoint is broken", which is worse than no diagnostic.
    import subprocess
    p = subprocess.run([sys.executable, '-m', 'tools.inspect_sgpt_ckpt', SGPT_CKPT,
                        '--max-print', '0'], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    check('tools.inspect_sgpt_ckpt runs clean on the real checkpoint',
          p.returncode == 0, (p.stderr or '').strip().splitlines()[-1:] or 'exit 0')
    check('and it reports 100.0%', '100.0% of the trunk loaded' in p.stdout,
          (p.stdout or '').strip().splitlines()[-1:])


def test_z_slice_modes():
    """Every --z-slice mode's shape, and in_channels agreeing with what is emitted."""
    print('test_z_slice_modes')
    tmp = tempfile.mkdtemp()
    try:
        path, split, _ = make_synthetic_stm_h5(os.path.join(tmp, 'stm.h5'), n=24, hw=32)

        # --- the 14-channel default must be unchanged; this is the regression guard ---
        base = StmPairDataset(path)
        check('z_slice defaults to None, so nothing existing moves', base.z_slice is None)
        check('default in_channels is still the stored z count', base.in_channels == 14,
              str(base.in_channels))
        check('default cube is still (14, H, W)', tuple(base[0]['cube'].shape) == (14, 32, 32),
              str(tuple(base[0]['cube'].shape)))

        for mode in (0, 13, 'mean', 'random'):
            ds = StmPairDataset(path, z_slice=mode)
            c = ds[5]['cube']
            check(f'z_slice={mode!r}: cube is (1, H, W)', tuple(c.shape) == (1, 32, 32),
                  str(tuple(c.shape)))
            check(f'z_slice={mode!r}: in_channels == emitted channels',
                  ds.in_channels == c.shape[0] == 1, str(ds.in_channels))
            check(f'z_slice={mode!r}: finite', bool(torch.isfinite(c).all()))
            # One slice standardised alone has unit std, which is what makes 'random' nearly
            # a no-op on a rank-1 stack.
            if mode != 'mean':
                check(f'z_slice={mode!r}: the SELECTED SLICE is standardised (std ~ 1)',
                      abs(float(c.std()) - 1.0) < 0.02, f'{float(c.std()):.4f}')

        # A fixed int must read that slice.  Checked against a whole-cube read, so the narrowed
        # hyperslab cannot silently return slice 0.
        whole = StmPairDataset(path, normalize=None)[7]['cube']
        for k in (0, 6, 13):
            got = StmPairDataset(path, z_slice=k, normalize=None)[7]['cube']
            check(f'z_slice={k}: the one-chunk read equals read-then-index',
                  torch.equal(got[0], whole[k]),
                  f'max delta {float((got[0] - whole[k]).abs().max()):.3e}')
        check("z_slice='mean' is the z-average, not slice 0",
              torch.allclose(StmPairDataset(path, z_slice='mean', normalize=None)[7]['cube'][0],
                             whole.mean(dim=0), atol=1e-4))

        check('an out-of-range slice raises in the PARENT',
              raises(lambda: StmPairDataset(path, z_slice=14), IndexError))
        check('a negative slice raises',
              raises(lambda: StmPairDataset(path, z_slice=-1), IndexError))
        check('an unknown mode raises, naming the legal set',
              raises(lambda: StmPairDataset(path, z_slice='middle'), ValueError))
        check("'none' is accepted as the all-slices spelling argparse passes",
              StmPairDataset(path, z_slice='none').in_channels == 14)

        b = collate_pairs([StmPairDataset(path, z_slice='random')[i] for i in range(4)])
        check('single-slice cubes collate to (B, 1, H, W)',
              tuple(b['cube'].shape) == (4, 1, 32, 32), str(tuple(b['cube'].shape)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_z_slice_determinism():
    """Train re-draws every epoch; validation must not.

    If validation resampled its slices, `train_clip.score` -- which selects the exported
    checkpoint on top1_size_controlled -- would be selecting partly on input noise.
    """
    print('test_z_slice_determinism')
    tmp = tempfile.mkdtemp()
    try:
        path, _, _ = make_synthetic_stm_h5(os.path.join(tmp, 'stm.h5'), n=24, hw=32)
        ds = StmPairDataset(path, z_slice='random')

        check('a fixed epoch gives the same sample twice, whatever the global RNG state',
              torch.equal(ds[3]['cube'], (torch.manual_seed(999), ds[3]['cube'])[1]))
        e0 = [_slice_for(r, 0, 14, 0) for r in ds.rows]
        e1 = [_slice_for(r, 1, 14, 0) for r in ds.rows]
        check('a new epoch re-draws', e0 != e1, f'{e0[:6]} -> {e1[:6]}')
        check('set_epoch actually changes what __getitem__ returns',
              not torch.equal(ds.set_epoch(0)[3]['cube'].clone(),
                              ds.set_epoch(1)[3]['cube']))
        check('a different seed gives a different assignment',
              [_slice_for(r, 0, 14, 1) for r in ds.rows] != e0)

        # "Fixed" must not silently mean "always slice 0", and the mix must not be lumpy: a
        # plain (row * C) % 14 would be, because 14 shares a factor with a power of two.
        spread = {_slice_for(r, 0, 14, 0) for r in range(4000)}
        check('the assignment covers every slice', spread == set(range(14)),
              f'{len(spread)} of 14')
        cnt = [sum(_slice_for(r, 0, 14, 0) == k for r in range(14000)) for k in range(14)]
        check('and is roughly uniform', max(cnt) < 1.3 * min(cnt),
              f'{min(cnt)}..{max(cnt)}, expected ~1000')

        # It must not touch the global stream, or this arm sees a different data order than
        # stm3d at the same --seed.
        torch.manual_seed(7)
        before = torch.randn(4)
        torch.manual_seed(7)
        _ = [ds[i]['cube'] for i in range(4)]
        check('drawing samples leaves the global RNG stream untouched',
              torch.equal(before, torch.randn(4)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stm2d_cube_encoder():
    """The interface contract, freezing, and z-order blindness.  Offline, random init."""
    print('test_stm2d_cube_encoder')
    torch.manual_seed(0)
    enc = Stm2dCubeEncoder(**STM2D_TINY)
    x = torch.randn(3, 14, 32, 32)
    out = enc(x)
    check('returns (B, latent_dim)', tuple(out.shape) == (3, 16), str(tuple(out.shape)))
    check('out_norm applied (row mean ~ 0)',
          float(out.detach().mean(dim=-1).abs().max()) < 1e-4)
    check('exposes the attributes build() and the downstream repo read',
          enc.latent_dim == enc.output_dim == 16)
    check('rejects (B, H, W)', raises(lambda: enc(torch.randn(2, 32, 32)), ValueError))
    check('rejects the wrong slice count, naming the flag',
          raises(lambda: enc(torch.randn(2, 7, 32, 32)), ValueError))

    # --- NEW (control): reset_parameters is the random-init control arm ---
    # tools/linear_probe.py calls reset_parameters on whichever encoder it is given, so every
    # encoder must have one.
    check('has reset_parameters, like every other encoder',
          callable(getattr(enc, 'reset_parameters', None)))
    before_head = enc.head.weight.detach().clone()
    before_trunk = next(iter(enc.backbone.parameters())).detach().clone()
    enc.reset_parameters()
    check('reset_parameters re-initialises the head',
          not torch.equal(before_head, enc.head.weight.detach()))
    check('reset_parameters re-initialises the TRUNK too (the honest from-scratch control)',
          not torch.equal(before_trunk,
                          next(iter(enc.backbone.parameters())).detach()))
    check('reset_parameters restores the identity affine',
          float(enc.in_gain) == 1.0 and float(enc.in_bias) == 0.0)
    check('reset_parameters re-applies freezing, so the arm is unchanged',
          enc.trainable_blocks == STM2D_TINY.get('trainable_blocks', 0))
    check('the encoder still runs afterwards',
          tuple(enc(x).shape) == (3, 16))
    # It draws from the global stream, like every other encoder's reset_parameters, so it is
    # reproducible under a seeded run rather than invariant. Both halves matter: a reset that
    # ignored the stream would be the only one in the repo that did, and on a pretrained=False
    # arm a fixed-seed rebuild reproduces the current weights and the control becomes a no-op.
    torch.manual_seed(7)
    enc.reset_parameters()
    once = enc.head.weight.detach().clone()
    once_trunk = next(iter(enc.backbone.parameters())).detach().clone()
    torch.manual_seed(7)
    enc.reset_parameters()
    check('reset_parameters replays exactly under the same global seed',
          torch.equal(once, enc.head.weight.detach())
          and torch.equal(once_trunk, next(iter(enc.backbone.parameters())).detach()))
    enc.reset_parameters()
    check('and gives DIFFERENT weights on a second call, so the control is never a no-op',
          not torch.equal(once_trunk, next(iter(enc.backbone.parameters())).detach()))

    one = Stm2dCubeEncoder(**{**STM2D_TINY, 'in_channels': 1})
    check('a Z=1 input from the loader works with no encoder change',
          tuple(one(torch.randn(2, 1, 32, 32)).shape) == (2, 16))
    check('passthrough with Z=14 raises, naming the fix',
          raises(lambda: Stm2dCubeEncoder(
              **{**STM2D_TINY, 'slice_mode': 'passthrough'})(x), ValueError))

    # slice_mode='fixed' must take that slice: build an input whose slices are constants.
    ramp = torch.arange(14, dtype=torch.float32).reshape(1, 14, 1, 1).expand(2, 14, 8, 8)
    for k in (0, 5, 13):
        e = Stm2dCubeEncoder(**{**STM2D_TINY, 'image_size': 8, 'resize': 8,
                                'slice_mode': 'fixed', 'slice_index': k})
        check(f'slice_mode=fixed picks slice {k}',
              float(e._select_slice(ramp).unique()) == float(k))

    # Eval is deterministic even at slice_mode='random' -- the model-selection guard.
    enc.eval()
    check('eval is deterministic under slice_mode=random', torch.equal(enc(x), enc(x)))
    enc.train()
    check('but training does re-draw the slice',
          len({tuple(enc(x).detach().flatten()[:4].tolist()) for _ in range(12)}) > 1)

    # z-order blindness is a design property here: the deliberate inverse of the z-asymmetry
    # the stm3d and sgpt arms assert.
    lo = Stm2dCubeEncoder(**{**STM2D_TINY, 'slice_mode': 'fixed', 'slice_index': 0}).eval()
    hi = Stm2dCubeEncoder(**{**STM2D_TINY, 'slice_mode': 'fixed', 'slice_index': 13}).eval()
    hi.load_state_dict(lo.state_dict())
    check('z-order blind by construction: enc@0(x) == enc@13(flip_z(x))',
          torch.allclose(lo(x), hi(torch.flip(x, dims=[1])), atol=1e-5))

    # in_chans=3 must feed three identical channels, so "stem used verbatim" is a test.
    p3 = Stm2dCubeEncoder(**{**STM2D_TINY, 'in_chans': 3})._prepare(x)
    check('in_chans=3 replicates the slice, it does not invent colour',
          torch.equal(p3[:, 0], p3[:, 1]) and torch.equal(p3[:, 1], p3[:, 2]))

    # Freezing.
    check('trainable_blocks=0 trains exactly head + out_norm + the input affine',
          {n for n, q in enc.named_parameters() if q.requires_grad}
          == {'head.weight', 'head.bias', 'out_norm.weight', 'out_norm.bias',
              'in_gain', 'in_bias'})
    check('a frozen trunk stays in eval mode, so BatchNorm stats cannot drift',
          not enc.train(True).backbone.training and enc.training)
    allp = Stm2dCubeEncoder(**{**STM2D_TINY, 'trainable_blocks': -1})
    check('trainable_blocks=-1 trains everything',
          all(q.requires_grad for q in allp.parameters()))
    two = Stm2dCubeEncoder(**{**STM2D_TINY, 'trainable_blocks': 2})
    check('trainable_blocks=2 unfreezes the last 2 stages but not the first',
          all(q.requires_grad for q in two.backbone.layer4.parameters())
          and not any(q.requires_grad for q in two.backbone.layer1.parameters()))
    check('asking for more stages than exist raises',
          raises(lambda: Stm2dCubeEncoder(**{**STM2D_TINY, 'trainable_blocks': 9}),
                 ValueError))

    # A squared objective, never .sum(): the encoder ends in a LayerNorm, so sum()'s gradient
    # is mathematically zero and the check would pass on float noise.
    enc.zero_grad()
    enc(x).pow(2).mean().backward()
    check('gradients reach the head under a frozen trunk',
          enc.head.weight.grad is not None and float(enc.head.weight.grad.abs().sum()) > 1e-8)

    # The export contract.
    rt = Stm2dCubeEncoder(**STM2D_TINY)
    rt.load_state_dict(enc.state_dict())          # strict
    rt.eval(); enc.eval()
    check('the exported state_dict round-trips strict and reproduces the code',
          torch.equal(rt(x), enc(x)))

    check('an unknown backbone raises, naming the known set',
          raises(lambda: Stm2dCubeEncoder(**{**STM2D_TINY, 'backbone': 'nope'}), ValueError))
    check('in_chans must be 1 or 3',
          raises(lambda: Stm2dCubeEncoder(**{**STM2D_TINY, 'in_chans': 2}), ValueError))
    check('resize must be the cube size or the backbone native size',
          raises(lambda: Stm2dCubeEncoder(**{**STM2D_TINY, 'resize': 77}), ValueError))


def test_stm2d_guard_refuses_a_partial_load():
    """A partial or absent pretrained load must raise, not warn.

    timm will not do it for us: against a junk checkpoint it prints only "Unexpected keys ...
    This may be expected if model is being adapted" while loading none of the weight tensors,
    and load_pretrained returns None, so there is no report to inspect. The counting and the
    raising are this module's own.
    """
    print('test_stm2d_guard_refuses_a_partial_load')
    tmp = tempfile.mkdtemp()
    try:
        junk = os.path.join(tmp, 'junk.pth')
        torch.save({'backbone.stage1.conv.weight': torch.zeros(4, 4, 3, 3)}, junk)
        check('a checkpoint matching nothing RAISES rather than training random weights',
              raises(lambda: Stm2dCubeEncoder(**{**STM2D_TINY, 'ckpt': junk,
                                                 'pretrained': True}), RuntimeError))
        try:
            Stm2dCubeEncoder(**{**STM2D_TINY, 'ckpt': junk, 'pretrained': True})
            msg = ''
        except RuntimeError as e:
            msg = str(e)
        check('and the message carries the fix, not just the complaint',
              'inspect_stm2d_backbone' in msg or 'prefetch_stm2d' in msg,
              msg.splitlines()[0] if msg else 'no message')

        missing = os.path.join(tmp, 'absent.pth')
        check('a nonexistent --stm2d-ckpt raises with the prefetch command named',
              raises(lambda: Stm2dCubeEncoder(**{**STM2D_TINY, 'ckpt': missing,
                                                 'pretrained': True}), RuntimeError))

        # in_chans=3 here: with in_chans=1 timm's own adapt_input_conv raises
        # KeyError('conv1.weight') on a state dict this malformed, before any of this module's
        # counting happens. That is wrapped into the "could not obtain pretrained weights"
        # RuntimeError above, which is the right outcome -- but it means the reporting path
        # (min_match=0, used by tools/inspect_stm2d_backbone) is only exercised at in_chans=3.
        _, rep = load_timm_backbone('resnet18', in_chans=3, ckpt=junk, min_match=0.0,
                                    verbose=False)
        check('min_match=0 reports instead of raising, for the inspector tool',
              rep['fraction'] == 0.0, f'{rep["matched"]}/{rep["trunk"]}')
        check('a malformed checkpoint at in_chans=1 raises from timm and is wrapped clearly',
              raises(lambda: load_timm_backbone('resnet18', in_chans=1, ckpt=junk,
                                                min_match=0.0, verbose=False), RuntimeError))

        check('pretrained=False does NOT raise -- it is the from-scratch control',
              Stm2dCubeEncoder(**STM2D_TINY).load_report['pretrained'] is False)

        # The same-seed probe must not shift the run's random stream, or this arm silently
        # sees a different data order than stm3d at the same --seed.  Two separate invariants:
        # the loader consumes nothing, and the constructor consumes the same amount whether
        # the trunk was pretrained or random (it advances by this module's head init, which is
        # correct -- the head is ours and is meant to be random).
        def stream_after(fn):
            torch.manual_seed(7)
            fn()
            return torch.randn(4)

        torch.manual_seed(7)
        base = torch.randn(4)
        check('load_timm_backbone leaves the global RNG untouched (pretrained=False)',
              torch.equal(base, stream_after(lambda: load_timm_backbone(
                  'resnet18', in_chans=3, pretrained=False, verbose=False))))
        check('load_timm_backbone leaves it untouched on the LOAD path too',
              torch.equal(base, stream_after(lambda: load_timm_backbone(
                  'resnet18', in_chans=3, ckpt=junk, min_match=0.0, verbose=False))))
        rnd = stream_after(lambda: Stm2dCubeEncoder(**STM2D_TINY))
        pre = stream_after(lambda: Stm2dCubeEncoder(
            **{**STM2D_TINY, 'ckpt': junk, 'pretrained': True, 'min_match': 0.0,
               'in_chans': 3}))
        check('the constructor advances the stream only by our own head init',
              not torch.equal(base, rnd))
        check('and advances it IDENTICALLY pretrained vs random -- the comparability property',
              torch.equal(rnd, pre))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stm2d_registry_matches_timm():
    """STM2D_BACKBONES records measured shapes; this catches a timm upgrade moving them."""
    print('test_stm2d_registry_matches_timm')
    import inspect as _inspect
    # --- NEW (skip): loud skip, as test_structure_encoder_loads_pretrained does ---
    # A bare `import timm` would fail the whole suite on a node without timm, for a reason
    # unrelated to the code under test. Every other optional dependency here skips.
    try:
        import timm
    except ImportError:
        print('  [SKIP] timm is not importable (it is in python-pytorch/2.10; this skip is '
              'for nodes or module versions where it is not)')
        return
    for arm, cfg in STM2D_BACKBONES.items():
        if 'vit_base' in cfg['timm_name']:
            continue                       # 85 M at random init is slow; covered by the net test
        kw = dict(pretrained=False, num_classes=0, in_chans=1)
        if cfg['grid_tied']:
            kw['img_size'] = 128
        m = timm.create_model(cfg['timm_name'], **kw)
        check(f'{arm}: num_features matches the recorded value',
              m.num_features == cfg['num_features'],
              f'{m.num_features} vs recorded {cfg["num_features"]}')
        tied = 'img_size' in _inspect.signature(type(m).__init__).parameters
        check(f'{arm}: grid_tied matches timm\'s own signature', tied == cfg['grid_tied'],
              f'{tied} vs recorded {cfg["grid_tied"]}')


def test_stm2d_loads_the_real_backbone():
    """The regression the synthetic guards cannot be: the actual released CLIP weights.

    Skips with a printed reason, as test_structure_encoder_loads_pretrained does -- the suite
    is meant to run with no /scratch and no network.
    """
    print('test_stm2d_loads_the_real_backbone')
    if not os.path.exists(STM2D_CKPT):
        print(f'  [SKIP] no weights at {STM2D_CKPT}.')
        print('  [SKIP] Run `python -m tools.prefetch_stm2d --backbone clip_vit_b16` on the '
              'login node, or set STM2D_CKPT, to run this check.')
        return
    enc = Stm2dCubeEncoder(in_channels=1, image_size=128, latent_dim=256,
                           backbone='clip_vit_b16', ckpt=STM2D_CKPT, verbose=False)
    r = enc.load_report
    check('100% of the CLIP ViT-B/16 weight tensors were changed by the load',
          r['fraction'] == 1.0, f'{r["matched"]}/{r["trunk"]}, still random: {r["missed"][:4]}')
    # A percentage cannot see this: the quickgelu and non-quickgelu tags share one hf_hub_id
    # and both load 100%, but only one runs the non-linearity OpenAI CLIP was trained with.
    check('QuickGELU, not GELU -- min_match is blind to this',
          type(enc.backbone.blocks[0].mlp.act).__name__ == 'QuickGELU',
          type(enc.backbone.blocks[0].mlp.act).__name__)
    check('pos_embed is the pretrained 197-token grid, used verbatim at the native size',
          tuple(enc.backbone.pos_embed.shape) == (1, 197, 768),
          str(tuple(enc.backbone.pos_embed.shape)))
    out = enc(torch.randn(2, 1, 128, 128))
    check('a real-shape forward runs on the loaded weights',
          tuple(out.shape) == (2, 256) and bool(torch.isfinite(out).all()))
    nat = Stm2dCubeEncoder(in_channels=1, image_size=128, latent_dim=256,
                           backbone='clip_vit_b16', resize=128, ckpt=STM2D_CKPT, verbose=False)
    check('resize=128 also loads 100%, having resampled pos_embed 197 -> 65',
          nat.load_report['fraction'] == 1.0
          and tuple(nat.backbone.pos_embed.shape) == (1, 65, 768),
          str(tuple(nat.backbone.pos_embed.shape)))


def test_sample_and_batch_keys_are_exhaustive():
    """The per-key checks above catch a key going missing; only this catches one being added.

    A stray key reaches the trainer, which reads batches by name, and shows up as an
    unexplained tensor on the device rather than as a failure.
    """
    print('test_sample_and_batch_keys_are_exhaustive')
    tmp = tempfile.mkdtemp()
    try:
        path, split, _ = make_synthetic_stm_h5(os.path.join(tmp, 'stm.h5'), n=8, hw=32)
        ds = StmPairDataset(path, split=split)
        s = ds[0]
        check('a sample is exactly cube/row/z/pos', set(s) == {'cube', 'row', 'z', 'pos'},
              str(sorted(s)))
        check('cube is (C,H,W) float32', s['cube'].ndim == 3 and s['cube'].dtype is torch.float32)
        check('z is 1-D int64', s['z'].dtype is torch.int64 and s['z'].ndim == 1)
        check('pos is (N,3) float32',
              s['pos'].shape == (s['z'].numel(), 3) and s['pos'].dtype is torch.float32)
        check('row is a plain int', isinstance(s['row'], int))

        items = [ds[i] for i in range(len(ds))]
        b = collate_pairs(items)
        check('a batch is exactly cube/rows/z/pos/batch/n_atoms',
              set(b) == {'cube', 'rows', 'z', 'pos', 'batch', 'n_atoms'}, str(sorted(b)))

        # Every STM reader is paired, so the collate may assume a molecule.
        check('a sample with no molecule raises rather than being collated',
              raises(lambda: collate_pairs([{'cube': s['cube'], 'row': s['row']}]), ValueError))

        # The name every sbatch and tool imports; it must stay an alias, not a second body.
        from stm.datasets.collate import collate_stm
        a2 = collate_stm(items)
        check('collate_pairs is collate_stm', set(a2) == set(b)
              and all(torch.equal(a2[k], b[k]) for k in b))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tests = [test_normalize_stm_cube, test_pair_dataset_and_collate,
             test_sample_and_batch_keys_are_exhaustive,
             test_z_slice_modes, test_z_slice_determinism, test_stm_cube_encoder,
             test_stm2d_cube_encoder, test_stm2d_guard_refuses_a_partial_load,
             test_stm2d_registry_matches_timm, test_stm2d_loads_the_real_backbone,
             test_sgpt_cube_encoder, test_sgpt_weight_loading,
             test_sgpt_loads_the_real_checkpoint,
             test_size_confound_metrics, test_structure_encoder_loads_pretrained]
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
