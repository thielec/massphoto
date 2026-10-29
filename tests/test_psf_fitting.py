import os
import subprocess
import sys

import numpy as np
import pytest

from massphoto.psf_fitting import PSF2D


def test_psf2d_thumbnail_shape_and_values():
    psf = PSF2D(psf_size=11, cache_divider=64, cache_interpolate=True)
    thumb = psf.get_thumbnail()
    assert thumb.shape == (11, 11)
    # centre should be the maximum (PSF is peaked)
    cy = thumb.shape[0] // 2
    cx = thumb.shape[1] // 2
    assert thumb[cy, cx] == np.max(thumb)

    psf = PSF2D(psf_size=11, cache_divider=64, cache_interpolate=False)
    thumb = psf.get_thumbnail()
    assert thumb.shape == (11, 11)
    # centre should be the maximum (PSF is peaked)
    cy = thumb.shape[0] // 2
    cx = thumb.shape[1] // 2
    assert thumb[cy, cx] == np.max(thumb)

    # a new PSF2D with the same parameters should return the same thumbnail
    psf2 = PSF2D(psf_fun=psf)
    thumb2 = psf2.get_thumbnail()
    assert np.allclose(thumb, thumb2)
    assert psf2.psf_cache_divider == psf.psf_cache_divider
    assert psf2.psf_cache_interpolate == psf.psf_cache_interpolate
    lookup2, fit2 = psf2.get_lookup_and_fitfun()
    assert callable(lookup2)
    assert callable(fit2)

    # A experimental PSF thumbnail should be cropped or padded to the correct size
    psf3 = PSF2D(psf_experimental=thumb, psf_size=11, cache_divider=4, cache_interpolate=True)
    for thumb_size in [9, 11, 13]:
        thumb = psf3.get_thumbnail(thumbnail_size=thumb_size)
        assert thumb.shape == (thumb_size, thumb_size)
        # centre should be the maximum (PSF is peaked)
        cy = thumb.shape[0] // 2
        cx = thumb.shape[1] // 2
        assert thumb[cy, cx] == np.max(thumb)


def test_psf2d_fit_recovers_amplitude_and_position_callable():
    rng = np.random.RandomState(42)
    psf = PSF2D(psf_size=11, cache_divider=128, cache_interpolate=True)
    lookup = psf.get_lookupfun()
    fit = psf.get_fitfun()

    amps = [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]
    xs = np.linspace(-2, 2, 5)
    ys = np.linspace(-2, 2, 5)

    for amp in amps:
        for x_true in xs:
            for y_true in ys:
                img = lookup(np.array([amp, x_true, y_true]))
                img_noisy = img + rng.normal(scale=1e-4, size=img.shape)

                alpha, x, y, l2_sqrt, r2, alpha_se = fit(img_noisy, 3)

                # absolute tolerance plus a relative allowance for very small amplitudes
                assert abs(alpha - amp) < max(0.02, 0.2 * abs(amp))
                assert abs(x - x_true) < 0.2
                assert abs(y - y_true) < 0.2
                assert r2 > 0.6


def test_psf2d_fit_recovers_amplitude_and_position_experimental():
    rng = np.random.RandomState(123)
    # use an experimental PSF thumbnail and make a PSF2D from it
    base = PSF2D(psf_size=11, cache_divider=64, cache_interpolate=True)
    # Precompilation of the experimental PSF path is done in conftest.py:
    # Any changes to the input parameters of PSF2D here should be reflected in conftest.py
    thumb = base.get_thumbnail()

    psf = PSF2D(psf_experimental=thumb, psf_size=11, cache_divider=64, cache_interpolate=True)
    lookup, fit = psf.get_lookup_and_fitfun()

    amps = [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]
    xs = np.linspace(-2, 2, 20)
    ys = np.linspace(-2, 2, 20)

    for amp in amps:
        for x_true in xs:
            for y_true in ys:
                img = lookup(np.array([amp, x_true, y_true]))
                img_noisy = img + rng.normal(scale=5e-5, size=img.shape)

                alpha, x, y, l2_sqrt, r2, alpha_se = fit(img_noisy, 3)

                # absolute tolerance plus a relative allowance for very small amplitudes
                assert abs(alpha - amp) < max(0.03, 0.25 * abs(amp))
                assert abs(x - x_true) < 0.22
                assert abs(y - y_true) < 0.22
                assert r2 > 0.5


def test_psf_uncompiled_subprocess():
    if os.environ.get("NUMBA_DISABLE_JIT") == "1":
        pytest.skip("Avoid recursive subprocess when NUMBA_DISABLE_JIT is set")
    # Skip this test if numba is not available, since we already tested the uncompiled version in the previous test.
    pytest.importorskip("numba")

    env = os.environ.copy()
    env["NUMBA_DISABLE_JIT"] = "1"
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    # Run just the specific test(s) in a clean process
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_psf_fitting.py",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_estimate_psf_and_fit_psf_params():
    """Create a stack of shifted PSFs with noise, estimate PSF and fit parameters.

    Shorter implementation using the PSF2D helper.
    """
    rng = np.random.RandomState(1)
    N = 1000
    size = 11

    from massphoto.psf_fitting import estimate_psf, fit_psf_params

    psf_obj = PSF2D(psf_size=size, cache_divider=128, cache_interpolate=True)
    lookup = psf_obj.get_lookupfun()

    # generate stack with random subpixel shifts and additive noise
    stack = np.empty((N, size, size), dtype=float)
    shifts = rng.uniform(-2, 2, size=(N, 2))
    for i, (dx, dy) in enumerate(shifts):
        img = lookup(np.array([1.0, dx, dy]))
        img = img + rng.normal(scale=0.1, size=img.shape)
        stack[i] = img

    # estimate PSF (centre_psf removes phase and centres the OTF) and verify input is not mutated
    stack_orig = stack.copy()
    psf_est = estimate_psf(stack, centre_psf=True, psf_size=size)
    np.testing.assert_array_equal(stack, stack_orig)

    # compare with noise-free centred PSF (from PSF2D)
    psf_ref = psf_obj.get_thumbnail()
    psf_est_n = psf_est / (np.max(psf_est) + np.finfo(float).eps)
    psf_ref_n = psf_ref / np.max(psf_ref)
    corr = np.corrcoef(psf_est_n.flatten(), psf_ref_n.flatten())[0, 1]
    assert corr > 0.95, f"Estimated PSF correlation {corr:.3f} is below threshold"

    # fit parameters to the estimated PSF and compare with defaults
    psf_fit, psf_params_fit = fit_psf_params(psf_est, update_defaults=False)
    corr_fit = np.corrcoef(psf_fit.flatten(), psf_ref_n.flatten())[0, 1]
    assert corr_fit > 0.95, f"Fitted PSF correlation {corr_fit:.3f} is below threshold"


def test_fit_psf_params_rejects_mismatched_p0_length():
    from massphoto.psf_fitting import fit_psf_params

    psf_guess = np.ones((11, 11), dtype=float)
    with pytest.raises(ValueError, match="same length"):
        fit_psf_params(psf_guess, p0=[1.0], fit_params=("a12", "w"))


def test_psf_functions_handle_empty_defaults():
    from massphoto.psf import psf_2d
    from massphoto.psf_fitting import fit_psf_params
    from massphoto.utils import defaults

    psf_ref = PSF2D(psf_size=11).get_thumbnail()
    with defaults.context():
        defaults.set_state({})
        val = psf_2d(0.0, 0.0)
        assert np.isfinite(val)

        psf_obj = PSF2D()
        thumb = psf_obj.get_thumbnail()
        assert thumb.shape == (11, 11)

        psf_fit, params = fit_psf_params(psf_ref, update_defaults=True)
        assert psf_fit.shape == (11, 11)
        assert "a12" in params
        assert defaults.psf_params is not None
