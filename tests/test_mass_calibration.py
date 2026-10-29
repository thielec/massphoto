import numpy as np
import pandas as pd
import pytest
from numpy.polynomial import Polynomial

from massphoto import defaults
from massphoto.distfit import FitResult
from massphoto.mass_calibration import (
    _invert_polynomial_coefficients,
    apply,
    check_coefficients,
    fit,
    get_calibration,
    get_coefficients,
)


def test_mass_calibration_basic_roundtrip():
    contrasts = np.array([0.01, 0.02, 0.03], dtype=float)
    masses = np.array([10.0, 20.0, 30.0], dtype=float)

    poly = fit(contrasts, masses, degree=1, direction="contrast_to_mass")
    assert isinstance(poly, Polynomial)

    calib = get_calibration(poly, direction="contrast_to_mass")
    masses_out = calib(contrasts)
    assert np.allclose(masses_out, masses, atol=1e-6)

    inv = get_calibration(poly, direction="mass_to_contrast")
    contrasts_out = inv(masses)
    assert np.allclose(contrasts_out, contrasts, atol=1e-6)


def test_apply_mass_calibration_array_and_df():
    coeffs = [0.0, 100.0]
    contrasts = np.array([0.1, 0.2], dtype=float)
    masses = apply(contrasts, coeffs)
    assert np.allclose(masses, np.array([10.0, 20.0]))

    df = pd.DataFrame({"contrasts": contrasts, "contrasts_se": [0.01, 0.02]})
    df_out = apply(df.copy(), coeffs)
    assert "masses_kDa" in df_out.columns
    assert np.allclose(df_out["masses_kDa"].to_numpy(), np.array([10.0, 20.0]))
    assert "masses_kDa_se" in df_out.columns


def test_coefficients_inversion_and_validation():
    coeffs = [1.0, 2.0]
    inv = _invert_polynomial_coefficients(coeffs)
    assert np.allclose(inv, [-0.5, 0.5])

    poly = Polynomial(coeffs)
    inv_poly = _invert_polynomial_coefficients(poly)
    assert isinstance(inv_poly, Polynomial)

    out = get_coefficients(poly, direction="mass_to_contrast")
    assert np.allclose(out, inv)

    with pytest.raises(ValueError):
        check_coefficients([0.0, 1.0], direction="contrast_to_mass")


def test_mass_calibration_defaults():
    contrasts = np.array([0.01, 0.02, 0.03], dtype=float)
    masses = np.array([10.0, 20.0, 30.0], dtype=float)

    poly = fit(contrasts, masses, degree=1, direction="contrast_to_mass", update_defaults=True)

    calib_defaults = defaults.get("mass_calibration")
    calib_return = get_coefficients(poly)
    assert np.allclose(calib_defaults, calib_return)
    assert np.allclose(calib_defaults, poly.convert().coef)

    # test if defaults are used in apply
    masses_out = apply(contrasts)
    assert np.allclose(masses_out, masses, rtol=1e-6)


def test_apply_mass_calibration_requires_contrasts_column():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="contrasts"):
        apply(df, [0.0, 100.0])


def test_get_calibration_rejects_invalid_direction():
    with pytest.raises(ValueError, match="direction must be one of"):
        get_calibration([0.0, 100.0], direction="invalid")


def test_invert_polynomial_coefficients_rejects_higher_order():
    with pytest.raises(ValueError, match="Only first order"):
        _invert_polynomial_coefficients([1.0, 2.0, 3.0])


def test_fit_mass_calibration_mass_input_uses_reference_calibration():
    target_masses = np.array([86.0, 172.0, 258.0], dtype=float)
    previous_calibration = Polynomial([-4.0, 30000.0])
    measured_masses = target_masses + np.array([2.0, -1.5, 1.0], dtype=float)

    mass_to_contrast = get_calibration(previous_calibration, direction="mass_to_contrast")
    contrasts_manual = mass_to_contrast(measured_masses)
    fit_manual = fit(contrasts_manual, target_masses)

    fit_from_mass = fit(
        measured_masses,
        target_masses,
        input_unit="mass",
        reference_calibration=previous_calibration,
    )
    assert np.allclose(fit_from_mass.convert().coef, fit_manual.convert().coef)


def test_fit_mass_calibration_auto_detect_mass_logs(caplog):
    target_masses = np.array([86.0, 172.0, 258.0], dtype=float)
    previous_calibration = Polynomial([-4.0, 30000.0])
    measured_masses = target_masses + np.array([2.0, -1.5, 1.0], dtype=float)

    caplog.set_level("DEBUG")
    fit(
        measured_masses,
        target_masses,
        input_unit="auto",
        reference_calibration=previous_calibration,
    )

    assert "received mass-domain peak positions" in caplog.text


def test_fit_mass_calibration_accepts_fitresult_and_peaks_dataframe_sorted():
    target_masses = np.array([86.0, 172.0, 258.0], dtype=float)
    peaks_unsorted = pd.DataFrame(
        {
            "mu": [0.0090, 0.0030, 0.0060],
            "sigma": [0.0010, 0.0010, 0.0010],
            "A": [1.0, 1.0, 1.0],
        }
    )
    fitres = FitResult(
        peaks=peaks_unsorted,
        background=0.0,
        background_err=None,
        counts=np.array([1.0]),
        bin_edges=np.array([0.0, 1.0]),
        model_counts=np.array([1.0]),
        residuals=np.array([0.0]),
        cov=None,
        metrics={},
        warnings=[],
    )

    fit_expected = fit(np.sort(peaks_unsorted["mu"].to_numpy()), target_masses)
    fit_from_peaks = fit(peaks_unsorted, target_masses)
    fit_from_fitres = fit(fitres, target_masses)

    assert np.allclose(fit_from_peaks.convert().coef, fit_expected.convert().coef)
    assert np.allclose(fit_from_fitres.convert().coef, fit_expected.convert().coef)


def test_fit_mass_calibration_mass_input_requires_invertible_calibration():
    measured_masses = np.array([88.0, 170.5, 259.0], dtype=float)
    target_masses = np.array([86.0, 172.0, 258.0], dtype=float)

    with defaults.context({"mass_calibration": None}, copy=True):
        with pytest.raises(ValueError, match="Cannot convert mass-domain peaks"):
            fit(measured_masses, target_masses, input_unit="mass")

    with pytest.raises(ValueError, match="Cannot convert mass-domain peaks"):
        fit(
            measured_masses,
            target_masses,
            input_unit="mass",
            reference_calibration=[1.0, 2.0, 3.0],
        )
