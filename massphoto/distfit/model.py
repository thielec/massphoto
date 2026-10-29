"""
Core model for binned, integrated Gaussian peaks.

Provides:
- bin_integrated_gaussian(...) — vectorised sum of bin-integrated Gaussians (returns expected counts per bin).
- jacobian_bin_integrated(...) — analytic Jacobian of the model with respect to parameters
  ordered as [A1, mu1, sigma1, A2, mu2, sigma2, ..., background].

Numerical notes:
- Uses Gaussian CDF via erf for exact bin integrals.
- Derivatives use standard normal PDF for edges.
- Safe-guards against extremely small sigma to avoid numerical issues.
"""

import numpy as np
from scipy.special import erf

# constants
_SQRT2 = np.sqrt(2.0)
_SQRT2PI = np.sqrt(2.0 * np.pi)


def _cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + erf(x / _SQRT2))


def _pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / _SQRT2PI


def bin_integrated_gaussian(
    bin_edges: np.ndarray,
    centres: np.ndarray | list[float],
    sigmas: np.ndarray | list[float],
    amplitudes: np.ndarray | list[float],
    background: float = 0.0,
    min_sigma: float = 1e-12,
) -> np.ndarray:
    """
    Compute expected counts per histogram bin for a sum of bin-integrated Gaussians.

    Uses exact bin integrals via Gaussian CDF (erf).

    Parameters
    ----------
    bin_edges : histogram bin edges (length = n_bins + 1)
    centres : peak centres µ
    sigmas : peak widths sigma (clipped to min_sigma)
    amplitudes : integrated counts (area) for each peak
    background : constant background per bin
    min_sigma : smallest allowed sigma

    Returns
    -------
    model : expected counts per bin (n_bins,)
    """
    bin_edges = np.asarray(bin_edges)
    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise ValueError("bin_edges must be a 1D array with at least two elements")
    centres = np.asarray(centres, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    if not (centres.shape == sigmas.shape == amplitudes.shape):
        raise ValueError("centres, sigmas and amplitudes must have the same shape")

    # n_peaks = centres.size
    sigmas = np.maximum(sigmas, min_sigma)

    # Broadcast bin edges and peak parameters for vectorized computation
    b1 = bin_edges[:-1][:, None]  # (n_bins, 1)
    b2 = bin_edges[1:][:, None]
    mu = centres[None, :]  # (1, n_peaks)
    s = sigmas[None, :]
    A = amplitudes[None, :]

    # Standardized coordinates
    z1 = (b1 - mu) / s
    z2 = (b2 - mu) / s

    # Fraction of each Gaussian in each bin
    frac = 0.5 * (erf(z2 / _SQRT2) - erf(z1 / _SQRT2))  # (n_bins, n_peaks)
    model = (A * frac).sum(axis=1)

    if background != 0.0:
        model = model + float(background)
    return model


def jacobian_bin_integrated(
    bin_edges: np.ndarray,
    centres: np.ndarray | list[float],
    sigmas: np.ndarray | list[float],
    amplitudes: np.ndarray | list[float],
    include_background: bool = True,
    min_sigma: float = 1e-12,
) -> np.ndarray:
    """
    Analytic Jacobian of the bin-integrated Gaussian model.

    Parameter ordering: [A1, mu1, sigma1, A2, mu2, sigma2, ..., background]

    Returns
    -------
    J : shape (n_bins, n_params) where J[i,j] = d(model_i)/d(param_j)
    """
    bin_edges = np.asarray(bin_edges)
    centres = np.asarray(centres, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    if not (centres.shape == sigmas.shape == amplitudes.shape):
        raise ValueError("centres, sigmas and amplitudes must have the same shape")

    n_bins = bin_edges.size - 1
    n_peaks = centres.size
    n_params = n_peaks * 3 + (1 if include_background else 0)
    sigmas = np.maximum(sigmas, min_sigma)

    b1 = bin_edges[:-1][:, None]
    b2 = bin_edges[1:][:, None]
    mu = centres[None, :]
    s = sigmas[None, :]
    A = amplitudes[None, :]

    z1 = (b1 - mu) / s
    z2 = (b2 - mu) / s
    frac = _cdf(z2) - _cdf(z1)

    # Derivative with respect to amplitude
    dA = frac

    # Derivative with respect to centre (mu)
    pdf_z1 = _pdf(z1)
    pdf_z2 = _pdf(z2)
    dmu = -(pdf_z2 - pdf_z1) / s
    dmu = A * dmu

    # Derivative with respect to sigma
    coeff2 = (b2 - mu) / (s * s)
    coeff1 = (b1 - mu) / (s * s)
    dsig = -coeff2 * pdf_z2 + coeff1 * pdf_z1
    dsig = A * dsig

    # Assemble Jacobian
    J = np.zeros((n_bins, n_params), dtype=float)
    col = 0
    for k in range(n_peaks):
        J[:, col] = dA[:, k]
        J[:, col + 1] = dmu[:, k]
        J[:, col + 2] = dsig[:, k]
        col += 3

    if include_background:
        J[:, -1] = 1.0

    return J


def pointwise_gaussian(
    x_pos: np.ndarray,
    centres: np.ndarray | list[float],
    sigmas: np.ndarray | list[float],
    amplitudes: np.ndarray | list[float],
    background: float = 0.0,
    min_sigma: float = 1e-12,
    binwidth: float = 1.0,
) -> np.ndarray:
    """
    Compute expected counts per histogram bin for a sum of *pointwise* Gaussians.

    Notes
    - Parameter meanings are unchanged from `bin_integrated_gaussian`:
      `amplitudes` are treated as integrated counts (area). They are converted to
      peak heights using A_peak = A_area / (sigma * sqrt(2*pi)).
    - This function is designed to calculate the higher-resolution model for plotting.

    Parameters
    ----------
    x_pos : positions at which to evaluate the model
    centres : peak centres µ
    sigmas : peak widths sigma (clipped to min_sigma)
    amplitudes : integrated counts (area) for each peak
    background : constant background per bin
    min_sigma : smallest allowed sigma

    Returns
    -------
    model : expected counts per bin (n_bins,)
    """
    x_pos = np.asarray(x_pos)
    if x_pos.ndim != 1:
        raise ValueError("x_pos must be a 1D array")
    centres = np.asarray(centres, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    if not (centres.shape == sigmas.shape == amplitudes.shape):
        raise ValueError("centres, sigmas and amplitudes must have the same shape")

    sigmas = np.maximum(sigmas, min_sigma)

    # Broadcast shapes: (n_bins, 1) vs (1, n_peaks)
    x = x_pos[:, None]  # (n_bins, 1)
    mu = centres[None, :]  # (1, n_peaks)
    s = sigmas[None, :]  # (1, n_peaks)
    A_area = amplitudes[None, :]  # (1, n_peaks)

    # convert integrated area -> peak height for Gaussian:
    # Gaussian area = height * sigma * sqrt(2*pi)  =>  height = area / (sigma * sqrt(2*pi))
    height = A_area / (s * _SQRT2PI)

    # evaluate Gaussian at x positions (pointwise)
    z = (x - mu) / s
    gauss = height * np.exp(-0.5 * z * z)  # (n_bins, n_peaks)

    model = gauss.sum(axis=1) * binwidth  # scale by bin width to get counts

    if background != 0.0:
        model = model + float(background)
    return model
