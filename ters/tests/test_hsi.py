"""Everything checkable about ters/models/hsi_*.py without the cluster, the GPU or the HDF5."""

import math
import traceback

import torch

from ters.models.hsi_conv3d import Conv3dAutoencoder, ResBlock3d, SpectralSEBlock
from ters.losses.hsi import mse_sad_loss, spectral_angle_loss
from ters.models.hsi_spectral import SpectralAutoencoder

FAILURES = []

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


# --------------------------------------------------------------------------- spectral (1D)

def test_spectral_shapes():
    print('test_spectral_shapes')
    m = SpectralAutoencoder(in_channels=12, latent_per_pixel=2, base_width=4, n_stages=2, n_groups=4)
    x = torch.randn(3, 12, 8, 8)
    recon, z = m(x)
    check('roundtrip keeps the cube shape', tuple(recon.shape) == (3, 12, 8, 8),
          str(tuple(recon.shape)))
    check('latent is (B, L, H, W) -- the spatial axes are UNTOUCHED',
          tuple(z.shape) == (3, 2, 8, 8), str(tuple(z.shape)))
    check('lengths recorded front to back', m.lengths == [12, 6, 3], str(m.lengths))
    check('rejects (B, H, W)', raises(lambda: m(torch.randn(3, 8, 8)), ValueError))
    check('rejects wrong band count', raises(lambda: m(torch.randn(3, 13, 8, 8)), ValueError))


def test_spectral_is_per_pixel():
    print('test_spectral_is_per_pixel')
    # The defining property of this family (Kuester et al., ISPRS 2021: "the spatial dimension
    # is not used in the compression").  If a pixel's code depended on its neighbours this
    # would fail, and the arm would not be the thing the literature describes.
    torch.manual_seed(0)
    m = SpectralAutoencoder(in_channels=12, latent_per_pixel=2, base_width=4, n_stages=2, n_groups=4,
                            zero_init=False).eval()
    x = torch.randn(1, 12, 4, 4)
    with torch.no_grad():
        z_full = m.encode(x)
        z_pixel = m.encode(x[:, :, 1:2, 2:3])
    check('one pixel encodes identically in and out of its neighbourhood',
          torch.allclose(z_full[:, :, 1, 2], z_pixel[:, :, 0, 0], atol=1e-5),
          f'max |d| = {(z_full[:, :, 1, 2] - z_pixel[:, :, 0, 0]).abs().max():.2e}')


def test_spectral_budget_matches_the_compression_sweep():
    print('test_spectral_budget_matches_the_compression_sweep')
    # L=1 at 256px is a like-for-like replacement for the compression sweep's best arm, which
    # spent 65,536 numbers at 100x. This test is what keeps that claim true if anyone changes
    # a default.
    m = SpectralAutoencoder(in_channels=100, latent_per_pixel=1)
    check('L=1 at 256px is exactly 65,536 numbers', m.latent_numbers(256) == 65536,
          str(m.latent_numbers(256)))
    check('that is 100.0x compression',
          abs(100 * 256 * 256 / m.latent_numbers(256) - 100.0) < 1e-9)
    check('L=4 is 262,144 numbers / 25x',
          SpectralAutoencoder(in_channels=100, latent_per_pixel=4).latent_numbers(256) == 262144)
    check('refuses a non-bottleneck',
          raises(lambda: SpectralAutoencoder(in_channels=12, latent_per_pixel=12), ValueError))
    check('refuses L < 1',
          raises(lambda: SpectralAutoencoder(in_channels=12, latent_per_pixel=0), ValueError))


def test_spectral_zero_init():
    print('test_spectral_zero_init')
    m = SpectralAutoencoder(in_channels=12, latent_per_pixel=2, base_width=4, n_stages=2, n_groups=4)
    x = torch.randn(2, 12, 8, 8)
    with torch.no_grad():
        recon, _ = m(x)
    check('starts at exactly the do-nothing baseline', float(recon.abs().max()) == 0.0,
          f'max |recon| = {float(recon.abs().max()):.2e}')
    m2 = SpectralAutoencoder(in_channels=12, latent_per_pixel=2, base_width=4, n_stages=2, n_groups=4,
                             zero_init=False)
    with torch.no_grad():
        recon2, _ = m2(x)
    check('and that is the init, not the architecture', float(recon2.abs().max()) > 0.0)


# --------------------------------------------------------------------------- conv3d

def test_conv3d_shapes():
    print('test_conv3d_shapes')
    m = Conv3dAutoencoder(in_channels=12, image_size=16, latent_ch=2, widths=(4, 8), n_groups=4,
                          spectral_strides=(2, 1), n_res=1, res_from_stage=0)
    x = torch.randn(2, 12, 16, 16)
    recon, z = m(x)
    check('roundtrip keeps the cube shape', tuple(recon.shape) == (2, 12, 16, 16),
          str(tuple(recon.shape)))
    check('latent is a spectral-spatial VOLUME', tuple(z.shape) == (2, 2, 6, 4, 4),
          str(tuple(z.shape)))
    check('stage shapes recorded', m.shapes == [(12, 16), (6, 8), (6, 4)], str(m.shapes))
    check('rejects a wrong image size',
          raises(lambda: m(torch.randn(2, 12, 32, 32)), ValueError))
    check('rejects a latent that is not a bottleneck',
          raises(lambda: Conv3dAutoencoder(in_channels=12, image_size=16, latent_ch=200,
                                           widths=(4, 8), n_groups=4, spectral_strides=(2, 1)), ValueError))


def test_conv3d_budget():
    print('test_conv3d_budget')
    # The real arm.  100 bands / 256 px / strides (1,2,2,1) -> latent (10, 25, 16, 16).
    m = Conv3dAutoencoder(in_channels=100, image_size=256, latent_ch=10)
    check('latent volume is (10, 25, 16, 16)', m.latent_shape == (10, 25, 16, 16),
          str(m.latent_shape))
    check('64,000 numbers', m.latent_numbers() == 64000, str(m.latent_numbers()))
    cr = 100 * 256 * 256 / m.latent_numbers()
    # 2.3% fewer numbers than the compression sweep's best arm, so comparing the two scores is
    # conservative rather than flattering.
    check('102.4x, i.e. tighter than the 100x arm it is compared to', abs(cr - 102.4) < 1e-6,
          f'{cr:.2f}x')


def test_conv3d_zero_init_and_se():
    print('test_conv3d_zero_init_and_se')
    m = Conv3dAutoencoder(in_channels=12, image_size=16, latent_ch=2, widths=(4, 8), n_groups=4,
                          spectral_strides=(2, 1), use_se=True)
    x = torch.randn(2, 12, 16, 16)
    with torch.no_grad():
        recon, _ = m(x)
    check('SE model still starts at the do-nothing baseline',
          float(recon.abs().max()) == 0.0, f'max |recon| = {float(recon.abs().max()):.2e}')
    n_se = sum(1 for mod in m.modules() if isinstance(mod, SpectralSEBlock))
    check('SE blocks are actually inserted (encoder + decoder stages)', n_se == 4, str(n_se))
    check('no SE blocks when the flag is off',
          sum(1 for mod in Conv3dAutoencoder(in_channels=12, image_size=16, latent_ch=2,
                                             widths=(4, 8), n_groups=4, spectral_strides=(2, 1)).modules()
              if isinstance(mod, SpectralSEBlock)) == 0)

    se = SpectralSEBlock(channels=4, depth=6)
    y = se(torch.randn(2, 4, 6, 3, 3))
    check('SE preserves shape', tuple(y.shape) == (2, 4, 6, 3, 3))
    check('SE rejects the wrong spectral depth',
          raises(lambda: se(torch.randn(2, 4, 5, 3, 3)), ValueError))

    rb = ResBlock3d(4, 4, n_groups=4)
    check('ResBlock3d residual branch is zero-init-able and keeps shape',
          tuple(rb(torch.randn(1, 4, 3, 3, 3)).shape) == (1, 4, 3, 3, 3))


# --------------------------------------------------------------------------- SAD loss

def test_spectral_angle_loss():
    print('test_spectral_angle_loss')
    t = torch.randn(2, 8, 4, 4).abs() + 0.1          # every pixel live
    check('identical spectra -> 0', float(spectral_angle_loss(t, t)) < 1e-6,
          f'{float(spectral_angle_loss(t, t)):.2e}')
    # the point of the loss: scale invariance.  Palsson et al. (IEEE Access 2018) choose SAD
    # over MSE precisely because MSE "discriminates between endmembers based on their absolute
    # magnitude".  A 3x-too-large reconstruction is spectrally perfect and MSE hates it.
    scaled = 3.0 * t
    check('a scalar multiple is spectrally perfect', float(spectral_angle_loss(t, scaled)) < 1e-6)
    check('...and MSE is not indifferent to it',
          float(torch.nn.functional.mse_loss(scaled, t)) > 0.1,
          f'{float(torch.nn.functional.mse_loss(scaled, t)):.3f}')

    a = torch.zeros(1, 3, 1, 1); a[0, 0] = 1.0
    b = torch.zeros(1, 3, 1, 1); b[0, 1] = 1.0
    check('orthogonal spectra -> 1 - cos = 1', abs(float(spectral_angle_loss(a, b)) - 1.0) < 1e-6)
    check('...and 90 degrees in the angle form',
          abs(float(spectral_angle_loss(a, b, form='angle')) - math.pi / 2) < 1e-3)
    check("rejects an unknown form", raises(lambda: spectral_angle_loss(a, b, form='deg'),
                                            ValueError))
    check('rejects mismatched shapes',
          raises(lambda: spectral_angle_loss(a, torch.zeros(1, 4, 1, 1)), ValueError))


def test_spectral_angle_loss_at_zero_init():
    print('test_spectral_angle_loss_at_zero_init')
    # The state every run in this folder starts in, and the one that breaks a naive SAD.
    t = torch.randn(2, 8, 4, 4)
    r = torch.zeros_like(t, requires_grad=True)
    loss, live = spectral_angle_loss(r, t, return_live=True)
    check('no live pixel at zero-init', live == 0.0, f'live={live}')
    check('and the loss is 0, not nan', float(loss) == 0.0 and not math.isnan(float(loss)))
    loss.backward()
    check('gradient exists and is exactly zero -- MSE moves first, SAD switches itself on',
          r.grad is not None and float(r.grad.abs().max()) == 0.0)

    # Dead pixels are excluded rather than scored 0 degrees.
    t2 = torch.zeros(1, 4, 2, 1)
    t2[0, :, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])     # one live pixel, one dead
    r2 = torch.zeros_like(t2)
    r2[0, :, 0, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])     # orthogonal on the live one
    loss2, live2 = spectral_angle_loss(r2, t2, return_live=True)
    check('a dead pixel does not earn a free 0 degrees',
          abs(float(loss2) - 1.0) < 1e-6 and abs(live2 - 0.5) < 1e-9,
          f'loss={float(loss2):.4f} live={live2}')


def test_mse_sad_loss():
    print('test_mse_sad_loss')
    t = torch.randn(2, 8, 4, 4).abs() + 0.1
    r = (t + 0.05 * torch.randn_like(t)).requires_grad_(True)
    total, parts = mse_sad_loss(t * 1.0, r, sad_weight=0.5)
    check('reports both terms and the live fraction',
          set(parts) == {'mse', 'sad', 'sad_live'}, str(sorted(parts)))
    check('total = mse + w*sad',
          abs(float(total) - (parts['mse'] + 0.5 * parts['sad'])) < 1e-6)
    total.backward()
    check('and it is differentiable', r.grad is not None and float(r.grad.abs().max()) > 0)
    check('refuses a negative weight',
          raises(lambda: mse_sad_loss(t, t, sad_weight=-1.0), ValueError))


# --------------------------------------------------------------------------- optimisation

def _structured_cube(b, c, hw, rank=2, seed=0):
    """A cube whose spectra live on a low-dimensional manifold, like real ones do.

    Not torch.randn: on iid Gaussian bands a per-pixel autoencoder is rate-limited to MSE
    1 - L/C however long it trains, so a "loss must halve" check would fail on a model that
    had reached the information-theoretic optimum. Real TERS channels are a molecule's normal
    modes, so the fixture is `rank` fixed spectra mixed per pixel, which a rank-sized code can
    represent.
    """
    g = torch.Generator().manual_seed(seed)
    endmembers = torch.randn(rank, c, generator=g)
    weights = torch.randn(b, rank, hw, hw, generator=g)
    return torch.einsum('brhw,rc->bchw', weights, endmembers)


def _overfits(model, x, steps=60, lr=1e-2):
    """Trains the model for a few steps and reports whether the loss falls.

    Shape tests pass on a model wired backwards; this one does not.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    first = last = None
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        recon, _ = model(x)
        loss = torch.nn.functional.mse_loss(recon, x)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss)
        last = float(loss)
    return first, last


def test_spectral_overfits_one_batch():
    print('test_spectral_overfits_one_batch')
    torch.manual_seed(0)
    m = SpectralAutoencoder(in_channels=12, latent_per_pixel=4, base_width=8, n_stages=2)
    x = _structured_cube(2, 12, 8, rank=2)
    first, last = _overfits(m, x, steps=150)
    check('loss falls on a single batch', last < 0.3 * first, f'{first:.4f} -> {last:.4f}')


def test_conv3d_overfits_one_batch():
    print('test_conv3d_overfits_one_batch')
    torch.manual_seed(0)
    m = Conv3dAutoencoder(in_channels=12, image_size=16, latent_ch=4, widths=(4, 8), n_groups=4,
                          spectral_strides=(2, 1), n_res=1, res_from_stage=0)
    x = _structured_cube(1, 12, 16, rank=2, seed=1)
    first, last = _overfits(m, x, steps=60)
    check('loss falls on a single batch', last < 0.5 * first, f'{first:.4f} -> {last:.4f}')


def test_sad_training_step_from_zero_init():
    print('test_sad_training_step_from_zero_init')
    # End to end: a zero-initialised model trained on MSE+SAD must take a finite first step and
    # must not emit nan. This is the case that would otherwise be discovered 40 minutes into a
    # GPU job.
    torch.manual_seed(0)
    m = SpectralAutoencoder(in_channels=12, latent_per_pixel=4, base_width=8, n_stages=2)
    x = _structured_cube(2, 12, 8, rank=2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    lives, totals, mses = [], [], []
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        recon, _ = m(x)
        total, parts = mse_sad_loss(recon, x, sad_weight=0.5)
        total.backward()
        opt.step()
        lives.append(parts['sad_live'])
        totals.append(float(total))
        mses.append(parts['mse'])
    check('first step sees no live pixel', lives[0] == 0.0, f'live={lives[0]}')
    check('the SAD term switches itself on', lives[-1] > 0.9, f'live={lives[-1]:.2f}')
    check('no nan anywhere', all(not math.isnan(v) for v in totals + mses))
    # The total rises at step 1 by construction -- the SAD term is 0 while the reconstruction
    # is 0 and becomes real the moment it is not, so only the MSE component is monotone here.
    # Reading the total as a training curve is the mistake this check exists to document.
    check('MSE falls', mses[-1] < 0.3 * mses[0], f'{mses[0]:.4f} -> {mses[-1]:.4f}')
    check('the total falls once the SAD term is live',
          totals[-1] < totals[1], f'{totals[1]:.4f} -> {totals[-1]:.4f}')


def main():
    tests = [test_spectral_shapes, test_spectral_is_per_pixel,
             test_spectral_budget_matches_the_compression_sweep, test_spectral_zero_init,
             test_conv3d_shapes, test_conv3d_budget, test_conv3d_zero_init_and_se,
             test_spectral_angle_loss, test_spectral_angle_loss_at_zero_init,
             test_mse_sad_loss, test_spectral_overfits_one_batch,
             test_conv3d_overfits_one_batch, test_sad_training_step_from_zero_init]
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
