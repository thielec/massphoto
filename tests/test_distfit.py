import numpy as np
import pandas as pd
import pytest

from massphoto.distfit import fit
from massphoto.distfit.model import bin_integrated_gaussian


def test_bin_integrated_gaussian_properties():
    """Validate the core model function independently of the fitter."""
    mu = 15.0
    sigma = 2.0
    amplitude = 1000.0

    # Use a wide range and fine bins to minimize discretization error
    bin_edges = np.linspace(0, 30, 1000)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    counts = bin_integrated_gaussian(bin_edges, centres=[mu], sigmas=[sigma], amplitudes=[amplitude], background=0.0)

    # Check total counts (area under curve)
    total_counts = np.sum(counts)
    np.testing.assert_allclose(total_counts, amplitude, rtol=1e-5)

    # Check empirical mean
    empirical_mean = np.sum(counts * bin_centers) / total_counts
    np.testing.assert_allclose(empirical_mean, mu, rtol=1e-4)

    # Check empirical standard deviation
    empirical_var = np.sum(counts * (bin_centers - empirical_mean) ** 2) / total_counts
    empirical_std = np.sqrt(empirical_var)
    np.testing.assert_allclose(empirical_std, sigma, rtol=1e-3)


def test_fit_histogram_recovery():
    """Test fitting using data generated from numpy's random generator to avoid circular dependency."""
    np.random.seed(42)

    true_centres = [60.0, 120.0]
    true_sigmas = [7.0, 9.0]
    true_amplitudes = [5000, 8000]
    true_background = 10.0

    # expected uncertainties (for non-overlapping peaks):
    exp_centre_se = np.divide(true_sigmas, (np.array(true_amplitudes)) ** 0.5)
    exp_simga_se = np.divide(true_sigmas, (2 * np.array(true_amplitudes) - 2) ** 0.5)
    exp_amplitude_se = np.sqrt(true_amplitudes)

    samples_1 = np.random.normal(loc=true_centres[0], scale=true_sigmas[0], size=true_amplitudes[0])
    samples_2 = np.random.normal(loc=true_centres[1], scale=true_sigmas[1], size=true_amplitudes[1])
    all_samples = np.concatenate([samples_1, samples_2])

    bin_edges = np.linspace(0, 500, 250)
    observed_counts, _ = np.histogram(all_samples, bins=bin_edges)

    observed_counts = np.random.poisson(observed_counts + true_background)

    # Fit with good initial guesses
    result = fit(
        data=(observed_counts, bin_edges),
        initial_peaks=true_centres,
        initial_sigma=true_sigmas,
        initial_amps=true_amplitudes,
        method="hybrid",
        include_background=True,
    )

    fitted_peaks = result.peaks

    np.testing.assert_allclose(fitted_peaks["mu"].values, true_centres, rtol=0.05)
    np.testing.assert_allclose(fitted_peaks["sigma"].values, true_sigmas, rtol=0.1)
    np.testing.assert_allclose(fitted_peaks["A"].values, true_amplitudes, rtol=0.1)
    np.testing.assert_allclose(result.background, true_background, rtol=0.2)
    # Allow higher relative error for the uncertainty estimates
    np.testing.assert_allclose(fitted_peaks["mu_err"].to_numpy(), exp_centre_se, rtol=0.2)
    np.testing.assert_allclose(fitted_peaks["sigma_err"].to_numpy(), exp_simga_se, rtol=0.2)
    np.testing.assert_allclose(fitted_peaks["A_err"].to_numpy(), exp_amplitude_se, rtol=0.2)

    # Fit with method full
    result = fit(
        data=(observed_counts, bin_edges),
        initial_peaks=true_centres,
        initial_sigma=true_sigmas,
        initial_amps=true_amplitudes,
        method="full",
        include_background=True,
    )

    fitted_peaks = result.peaks

    np.testing.assert_allclose(fitted_peaks["mu"].values, true_centres, rtol=0.05)
    np.testing.assert_allclose(fitted_peaks["sigma"].values, true_sigmas, rtol=0.1)
    np.testing.assert_allclose(fitted_peaks["A"].values, true_amplitudes, rtol=0.1)
    np.testing.assert_allclose(result.background, true_background, rtol=0.2)

    # Fit without guess
    result = fit(data=all_samples, bins=bin_edges, include_background=False)

    fitted_peaks = result.peaks

    np.testing.assert_allclose(fitted_peaks["mu"].values, true_centres, rtol=0.05)
    np.testing.assert_allclose(fitted_peaks["sigma"].values, true_sigmas, rtol=0.1)
    np.testing.assert_allclose(fitted_peaks["A"].values, true_amplitudes, rtol=0.1)


@pytest.mark.parametrize(
    "empty_input",
    [
        pd.DataFrame({"contrasts": []}),
        pd.Series([], dtype=float),
        np.array([]),
    ],
)
def test_fit_histogram_empty_data_returns_empty_result(empty_input):
    """fit on empty data should warn and return an empty FitResult without raising."""
    bins = np.arange(-100, 700, 2.5)
    result = fit(empty_input, bins=bins, include_background=True)

    assert len(result.peaks) == 0
    assert result.background == 0.0
    assert result.counts.sum() == 0
    assert result.model_counts.sum() == 0
    assert len(result.warnings) > 0
