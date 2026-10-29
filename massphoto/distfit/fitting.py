"""
Histogram peak fitting.

The main entry point is :func:`fit`, which fits a sum of bin-integrated
Gaussians (optionally including a constant background) to a histogram.

Notes
-----
The default ``hybrid`` method treats amplitudes (and optional background) as
linear parameters solved by NNLS inside the objective, which is often more
stable for overlapping peaks.
"""

import logging
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
import pandas as pd
from numpy.linalg import pinv
from numpy.typing import ArrayLike
from scipy.optimize import least_squares, minimize, nnls

from .detect import estimate_bandwidth_from_fwhm, find_peak_guesses
from .model import bin_integrated_gaussian, jacobian_bin_integrated, pointwise_gaussian
from .utils import initial_amplitude_guess, standardize_histogram_input

logger = logging.getLogger(__name__)


@dataclass
class FitResult:
    """
    Complete fitting result.

    This dataclass stores all fit outputs (peaks, background, model, residuals,
    covariance, and summary metrics) as attributes.
    """

    #: Per-peak fit parameters with columns ``A``, ``mu``, ``sigma``, and ``*_err``.
    peaks: pd.DataFrame
    #: Constant background level (0.0 if not fitted).
    background: float
    #: Uncertainty of background if estimated.
    background_err: float | None
    #: Observed histogram bin counts used for fitting.
    counts: np.ndarray
    #: Histogram bin edges corresponding to counts.
    bin_edges: np.ndarray
    #: Model-predicted counts per bin.
    model_counts: np.ndarray
    #: counts - model_counts for each bin.
    residuals: np.ndarray
    #: Covariance matrix for parameters when available.
    cov: np.ndarray | None
    #: Goodness-of-fit metrics (rss, chi2, red_chi2, aic, bic).
    metrics: dict[str, float]
    #: Warnings emitted during fitting/covariance estimation.
    warnings: list[str]
    # Convenience attributes for simpler plotting
    #: Bin centers corresponding to counts/model_counts.
    bin_centers: np.ndarray | None = None
    #: Bin widths corresponding to counts/model_counts.
    bin_widths: float | None = None
    #: Fitted model evaluated at high-resolution x values when available.
    model_x: np.ndarray | None = None
    #: Fitted model y values at high-resolution x when available.
    model_y: np.ndarray | None = None


def _design_matrix_frac(bin_edges: np.ndarray, centres: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Build design matrix for NNLS (each column = unit-amplitude peak)."""
    n_peaks = centres.size
    n_bins = bin_edges.size - 1
    Y = np.empty((n_bins, n_peaks), dtype=float)
    for k in range(n_peaks):
        amps = np.zeros(n_peaks, dtype=float)
        amps[k] = 1.0
        Y[:, k] = bin_integrated_gaussian(bin_edges, centres, sigmas, amps, background=0.0)
    return Y


def _compute_metrics(counts: np.ndarray, model: np.ndarray, n_params: int) -> dict[str, float]:
    """Compute goodness-of-fit metrics (RSS, chi2, AIC, BIC)."""
    residuals = counts - model
    rss = float(np.sum(residuals**2))
    n = counts.size
    dof = max(1, n - n_params)
    var = counts + 1.0
    chi2 = float(np.sum((residuals**2) / var))
    red_chi2 = chi2 / dof
    ll = -0.5 * np.sum(np.log(2 * np.pi * var) + (residuals**2) / var)
    aic = float(2 * n_params - 2 * ll)
    bic = float(np.log(n) * n_params - 2 * ll)
    return {"rss": rss, "chi2": chi2, "red_chi2": red_chi2, "aic": aic, "bic": bic}


def _fit_histogram_hybrid(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    centres: np.ndarray,
    sigmas: np.ndarray,
    bounds_mu: list[tuple[float, float]] | None = None,
    bounds_sigma: list[tuple[float, float]] | None = None,
    include_background: bool = True,
    verbose: int = 0,
) -> FitResult:
    """
    Hybrid fitting: optimise centres & sigmas, solve amplitudes by NNLS.

    This approach is more stable for overlapping peaks because amplitudes
    (linear parameters) are solved analytically at each step.

    Parameters
    ----------
    counts : histogram bin counts
    bin_edges : histogram bin edges
    centres : initial peak centres (mu values)
    sigmas : initial peak widths (sigma values)
    bounds_mu : bounds for centres (default: data range)
    bounds_sigma : bounds for widths (default: small positive to infinity)
    include_background : whether to fit constant background
    verbose : verbosity level

    Returns
    -------
    FitResult with peaks, background, uncertainties, model, residuals, metrics
    """
    n_bins = counts.size
    centres0 = np.asarray(centres, dtype=float)
    sigmas0 = np.asarray(sigmas, dtype=float)
    n_peaks = centres0.size
    x0 = np.concatenate([centres0, sigmas0])

    if bounds_mu is None:
        bounds_mu = [(-np.inf, np.inf)] * n_peaks
    if bounds_sigma is None:
        bounds_sigma = [(1e-9, np.inf)] * n_peaks

    lb = np.array([b[0] for b in bounds_mu] + [b[0] for b in bounds_sigma], dtype=float)
    ub = np.array([b[1] for b in bounds_mu] + [b[1] for b in bounds_sigma], dtype=float)
    ub = np.maximum(lb, ub)
    x0 = np.clip(x0, lb, ub)
    bounds = [(float(l), float(u)) for l, u in zip(lb, ub)]

    def objective(x: np.ndarray) -> float:
        mu = x[:n_peaks]
        sig = x[n_peaks:]
        if np.any(sig <= 0):
            return 1e100
        Y = _design_matrix_frac(bin_edges, mu, sig)
        Y_ext = np.hstack([Y, np.ones((n_bins, 1), dtype=float)]) if include_background else Y
        p_nnls, _ = nnls(Y_ext, counts)
        model = Y_ext.dot(p_nnls)
        residuals = counts - model
        return float(np.sum(residuals**2))

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    if not res.success and verbose:
        logger.warning("Hybrid optimisation did not converge: %s", res.message)

    x_opt = res.x
    mu_opt = x_opt[:n_peaks]
    sigma_opt = x_opt[n_peaks:]
    Y_final = _design_matrix_frac(bin_edges, mu_opt, sigma_opt)
    Y_ext_final = np.hstack([Y_final, np.ones((n_bins, 1), dtype=float)]) if include_background else Y_final
    amps_bg, _ = nnls(Y_ext_final, counts)
    amplitudes_opt = amps_bg[:-1] if include_background else amps_bg
    background_opt = float(amps_bg[-1]) if include_background else 0.0
    model_counts = Y_ext_final.dot(amps_bg)
    residuals = counts - model_counts
    warnings = []

    # Numerical Jacobian for non-linear parameters
    try:
        eps = 1e-6
        J_num = np.zeros((n_bins, 2 * n_peaks), dtype=float)
        base_model = model_counts.copy()
        for j in range(2 * n_peaks):
            dx = np.zeros_like(x_opt)
            dj = eps * max(1.0, abs(x_opt[j]))
            dx[j] = dj
            x_pert = x_opt + dx
            mu_p = x_pert[:n_peaks]
            sig_p = x_pert[n_peaks:]
            Yp = _design_matrix_frac(bin_edges, mu_p, sig_p)
            Yp_ext = np.hstack([Yp, np.ones((n_bins, 1), dtype=float)]) if include_background else Yp
            pp, _ = nnls(Yp_ext, counts)
            model_p = Yp_ext.dot(pp)
            J_num[:, j] = (model_p - base_model) / dj
        J_res = -J_num
        var = counts + 1.0
        W = 1.0 / np.sqrt(var)
        Jw = J_res * W[:, None]
        r_w = residuals * W
        JTJ = Jw.T.dot(Jw)
        cov_nonlin = pinv(JTJ) * (r_w @ r_w) / max(1, n_bins - Jw.shape[1])
    except Exception:
        cov_nonlin = None
        warnings.append("Numerical Jacobian failed for covariance estimation")

    # Covariance for linear parameters (amplitudes)
    try:
        var = counts + 1.0
        W = 1.0 / np.sqrt(var)
        Yw = Y_final * W[:, None]
        JTJ_lin = Yw.T.dot(Yw)
        cov_A = pinv(JTJ_lin) * (((residuals * W) @ (residuals * W)) / max(1, n_bins - Yw.shape[1]))
        A_errs = np.sqrt(np.abs(np.diag(cov_A)))
    except Exception:
        cov_A = None
        A_errs = np.full(n_peaks, np.nan)
        warnings.append("Amplitude covariance estimation failed")

    bg_err = float(np.sqrt(np.abs(np.diag(cov_A))[-1])) if (include_background and (cov_A is not None)) else None
    mu_errs = np.full(n_peaks, np.nan)
    sigma_errs = np.full(n_peaks, np.nan)
    if cov_nonlin is not None:
        for k in range(n_peaks):
            mu_errs[k] = float(np.sqrt(np.abs(cov_nonlin[k, k])))
            sigma_errs[k] = float(np.sqrt(np.abs(cov_nonlin[n_peaks + k, n_peaks + k])))

    peaks_out = pd.DataFrame(
        {
            "A": amplitudes_opt,
            "mu": mu_opt,
            "sigma": sigma_opt,
            "A_err": A_errs,
            "mu_err": mu_errs,
            "sigma_err": sigma_errs,
        }
    )

    cov_full = None
    if cov_A is not None and cov_nonlin is not None:
        try:
            cov_full = np.zeros((3 * n_peaks + (1 if include_background else 0),) * 2)
            mA = cov_A.shape[0]
            cov_full[:mA, :mA] = cov_A
            offset = mA
            cov_full[offset : offset + cov_nonlin.shape[0], offset : offset + cov_nonlin.shape[1]] = cov_nonlin
        except Exception:
            cov_full = None
            warnings.append("Could not build combined covariance matrix")

    n_params_effective = (n_peaks * 3) + (1 if include_background else 0)
    metrics = _compute_metrics(counts, model_counts, n_params_effective)
    return FitResult(
        peaks=peaks_out,
        background=background_opt,
        background_err=bg_err,
        counts=counts,
        bin_edges=bin_edges,
        model_counts=model_counts,
        residuals=residuals,
        cov=cov_full,
        metrics=metrics,
        warnings=warnings,
    )


def _fit_histogram_full(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    centres: np.ndarray,
    sigmas: np.ndarray,
    amplitudes: np.ndarray,
    include_background: bool = True,
    weights: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    jacobian: bool = True,
    verbose: int = 0,
) -> FitResult:
    """
    Full non-linear least-squares over all parameters [A, mu, sigma, ...].

    Uses analytic Jacobian from model.jacobian_bin_integrated.

    Parameters
    ----------
    counts : histogram bin counts
    bin_edges : histogram bin edges
    centres : initial peak centres (mu values)
    sigmas : initial peak widths (sigma values)
    amplitudes : initial peak amplitudes (A values)
    include_background : whether to fit constant background
    weights : per-bin weights (default: 1/sqrt(counts + 1))
    bounds : parameter bounds as (lb, ub) arrays
    jacobian : whether to use analytic Jacobian
    verbose : verbosity level

    Returns
    -------
    FitResult with peaks, background, uncertainties, model, residuals, metrics
    """
    centres0 = np.asarray(centres, dtype=float)
    sigmas0 = np.asarray(sigmas, dtype=float)
    amps0 = np.asarray(amplitudes, dtype=float)
    n_peaks = centres0.size
    n_bins = counts.size

    # Pack parameters as [A1, mu1, sigma1, A2, mu2, sigma2, ..., background]
    p0 = []
    for k in range(n_peaks):
        p0.extend([float(amps0[k]), float(centres0[k]), float(sigmas0[k])])
    if include_background:
        p0.append(0.0)
    p0 = np.array(p0, dtype=float)

    if weights is None:
        weights = 1.0 / np.sqrt(counts + 1.0)
    weights = np.asarray(weights, dtype=float).flatten()
    if weights.size != n_bins:
        raise ValueError("weights must have same length as counts")

    def residuals_fn(p: np.ndarray) -> np.ndarray:
        amps = p[0::3][:n_peaks]
        mus = p[1::3][:n_peaks]
        sigs = p[2::3][:n_peaks]
        bg = float(p[-1]) if include_background else 0.0
        model = bin_integrated_gaussian(bin_edges, mus, sigs, amps, background=bg)
        return (counts - model) * weights

    def jac_fn(p: np.ndarray) -> np.ndarray:
        amps = p[0::3][:n_peaks]
        mus = p[1::3][:n_peaks]
        sigs = p[2::3][:n_peaks]
        J = jacobian_bin_integrated(bin_edges, mus, sigs, amps, include_background=include_background)
        return -J * weights[:, None]

    if bounds is None:
        lb = []
        ub = []
        for _k in range(n_peaks):
            lb.extend([0.0, -np.inf, 1e-12])
            ub.extend([np.inf, np.inf, np.inf])
        if include_background:
            lb.append(0.0)
            ub.append(np.inf)
        lb = np.array(lb, dtype=float)
        ub = np.array(ub, dtype=float)
    else:
        lb, ub = bounds

    ls_kwargs = {"x0": p0, "bounds": (lb, ub), "ftol": 1e-9, "xtol": 1e-9, "verbose": 2 if verbose else 0}
    if jacobian:
        res = least_squares(residuals_fn, **ls_kwargs, jac=jac_fn, method="trf")
    else:
        res = least_squares(residuals_fn, **ls_kwargs, method="trf")

    if not res.success and verbose:
        logger.warning("Full least-squares did not converge: %s", res.message)

    p_opt = res.x
    amps_opt = p_opt[0::3][:n_peaks]
    mus_opt = p_opt[1::3][:n_peaks]
    sigs_opt = p_opt[2::3][:n_peaks]
    bg_opt = float(p_opt[-1]) if include_background else 0.0
    model_counts = bin_integrated_gaussian(bin_edges, mus_opt, sigs_opt, amps_opt, background=bg_opt)
    residuals = counts - model_counts

    # Covariance estimation
    try:
        J_final = jac_fn(p_opt) if jacobian else res.jac
        r = residuals * weights
        dof = max(1, n_bins - p_opt.size)
        s2 = float((r @ r) / dof)
        JTJ = J_final.T.dot(J_final)
        cov = pinv(JTJ) * s2
        perr = np.sqrt(np.abs(np.diag(cov)))
    except Exception:
        cov = None
        perr = np.full(p_opt.size, np.nan)

    A_errs = np.array([perr[3 * k] if cov is not None else np.nan for k in range(n_peaks)])
    mu_errs = np.array([perr[3 * k + 1] if cov is not None else np.nan for k in range(n_peaks)])
    sigma_errs = np.array([perr[3 * k + 2] if cov is not None else np.nan for k in range(n_peaks)])

    peaks_out = pd.DataFrame(
        {
            "A": amps_opt,
            "mu": mus_opt,
            "sigma": sigs_opt,
            "A_err": A_errs,
            "mu_err": mu_errs,
            "sigma_err": sigma_errs,
        }
    )

    bg_err = float(perr[-1]) if (include_background and cov is not None) else None
    metrics = _compute_metrics(counts, model_counts, p_opt.size)
    warnings = []
    if cov is None:
        warnings.append("Covariance not available (jacobian singular or computation failed)")

    return FitResult(
        peaks=peaks_out,
        background=bg_opt,
        background_err=bg_err,
        counts=counts,
        bin_edges=bin_edges,
        model_counts=model_counts,
        residuals=residuals,
        cov=cov,
        metrics=metrics,
        warnings=warnings,
    )


@overload
def fit(
    data: pd.DataFrame | np.ndarray | tuple[np.ndarray, np.ndarray],
    bins: int | np.ndarray | None = None,
    *,
    initial_peaks: ArrayLike | None = None,
    initial_sigma: float | ArrayLike | None = None,
    initial_amps: float | ArrayLike | None = None,
    binwidth: float | None = None,
    method: Literal["hybrid"] = "hybrid",
    min_prominence: float | None = None,
    max_peaks: int | None = None,
    include_background: bool = True,
    bounds_mu: list[tuple[float, float]] | None = None,
    bounds_sigma: list[tuple[float, float]] | None = None,
    verbose: int = 0,
) -> FitResult: ...


@overload
def fit(
    data: pd.DataFrame | np.ndarray | tuple[np.ndarray, np.ndarray],
    bins: int | np.ndarray | None = None,
    *,
    initial_peaks: ArrayLike | None = None,
    initial_sigma: float | ArrayLike | None = None,
    initial_amps: float | ArrayLike | None = None,
    binwidth: float | None = None,
    method: Literal["full"],
    min_prominence: float | None = None,
    max_peaks: int | None = None,
    include_background: bool = True,
    weights: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    jacobian: bool = True,
    verbose: int = 0,
) -> FitResult: ...


def fit(
    data: pd.DataFrame | np.ndarray | tuple[np.ndarray, np.ndarray],
    bins: int | np.ndarray | None = None,
    *,
    initial_peaks: ArrayLike | None = None,
    initial_sigma: float | ArrayLike | None = None,
    initial_amps: float | ArrayLike | None = None,
    binwidth: float | None = None,
    method: str = "hybrid",
    min_prominence: float | None = None,
    max_peaks: int | None = None,
    include_background: bool = True,
    **kwargs,
) -> FitResult:
    """
    Convenience wrapper to call either hybrid or full fitting method.

    If initial_peaks is not provided, automatically detects peaks and sets bounds.

    Accepts various input formats:

    - data=samples: unbinned data with auto-binning
    - data=samples, bins=N: unbinned data with N bins
    - data=samples, bins=edges: unbinned data with custom edges
    - data=samples, binwidth=w: unbinned data with bin width w
    - data=(counts, edges): pre-computed histogram

    Parameters
    ----------
    data : unbinned samples OR tuple (counts, bin_edges)
    bins : number of bins or bin edges array
    initial_peaks : list of peak positions (mu values) or None for auto-detection
    initial_sigma : single sigma for all peaks, or list of sigmas (None for auto-estimation)
    initial_amps : single amplitude for all peaks, or list of amplitudes (None for auto-estimation)
    binwidth : bin width for automatic binning
    method : 'hybrid' or 'full'
    min_prominence : minimum peak prominence for auto-detection (None for auto-estimation)
    max_peaks : maximum number of peaks to detect (None for unlimited)
    include_background : whether to fit constant background
    **kwargs : passed to the selected fitting engine. The hybrid method accepts
        ``bounds_mu``, ``bounds_sigma``, and ``verbose``; the full method accepts
        ``weights``, ``bounds``, ``jacobian``, and ``verbose``.

    Notes
    --------
        - If initial_peaks is not provided, peaks will be automatically detected using find_peak_guesses. Use
          min_prominence and max_peaks to control detection sensitivity.
        - If initial_sigma is not provided, it will be estimated from the data using estimate_bandwidth_from_fwhm
          based on the highest peak's FWHM. You can also provide a single sigma for all peaks or individual sigmas.
        - In most cases, the 'hybrid' method is recommended for better stability with overlapping peaks, while 'full'
          fits amplitudes as free parameters and allows to constrain them with bounds if needed.


    Examples
    --------
    >>> # Automatic peak detection with custom binning
    >>> result = fit(masses, binwidth=2.5, include_background=True)

    >>> # Manual peak specification with automatic sigma/amplitude
    >>> result = fit(masses, initial_peaks=[100, 200, 300], bins=50)

    >>> # Manual peak specification with shared sigma
    >>> result = fit(masses, initial_peaks=[100, 200], initial_sigma=10.0)

    >>> # Full manual specification
    >>> result = fit(masses, initial_peaks=[100, 200],
    ...                        initial_sigma=[8.0, 12.0], initial_amps=[500, 300])

    Typical output
    --------------
    ::

        FitResult(
            peaks=DataFrame with columns:
                ['A', 'mu', 'sigma', 'A_err', 'mu_err', 'sigma_err'],
            background=...,            # Fitted background value
            background_err=...,        # Uncertainty of background
            counts=array([...]),       # Histogram bin counts
            bin_edges=array([...]),    # Histogram bin edges
            model_counts=array([...]), # Model-predicted counts per bin
            residuals=array([...]),    # Difference: counts - model_counts
            cov=array([...]),          # Covariance matrix (if available)
            metrics={...},             # Goodness-of-fit metrics
            warnings=[...],            # List of warnings (if any)
            # Convenience attributes for plotting
            bin_centers=array([...]),  # Bin centers for plotting
            bin_widths=...,            # Bin width for plotting
            model_x=array([...]),      # High-resolution x values for model plotting
            model_y=array([...]),      # Model values at high-resolution x for plotting
        )
    """

    # Standardize histogram input once for all paths
    counts, bin_edges = standardize_histogram_input(data, bins, binwidth)

    if counts.sum() == 0:
        logger.warning("fit: data is empty or all-zero - returning empty FitResult")
        empty_peaks = pd.DataFrame(columns=["A", "mu", "sigma", "A_err", "mu_err", "sigma_err"])
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = float(np.mean(np.diff(bin_edges)))
        empty_result = FitResult(
            peaks=empty_peaks,
            background=0.0,
            background_err=None,
            counts=counts,
            bin_edges=bin_edges,
            model_counts=np.zeros_like(counts),
            residuals=np.zeros_like(counts),
            cov=None,
            metrics={},
            warnings=["Data is empty or all-zero - no fit performed"],
            bin_centers=bin_centers,
            bin_widths=bin_widths,
            model_x=np.linspace(bin_edges[0], bin_edges[-1], num=10 * counts.size),
            model_y=np.zeros(10 * counts.size),
        )
        return empty_result

    if initial_peaks is None:
        est_sigma = float(np.mean(np.asarray(initial_sigma))) if initial_sigma is not None else None
        initial_peaks, est_sigma = find_peak_guesses(
            (counts, bin_edges), sigma=est_sigma, min_prominence=min_prominence, max_peaks=max_peaks
        )
        initial_sigma = est_sigma if initial_sigma is None else initial_sigma
    elif initial_sigma is None:
        initial_sigma = estimate_bandwidth_from_fwhm((counts, bin_edges))

    initial_peaks = np.asarray(initial_peaks, dtype=float)
    if initial_peaks.size == 0:
        raise ValueError("No initial peaks provided or detected")
    initial_sigma = np.broadcast_to(initial_sigma, len(initial_peaks))

    kwargs["include_background"] = (
        include_background  # Pass to underlying methods, only included in signature for docs/typing clarity
    )

    # Dispatch to appropriate method
    if method == "hybrid":
        fitres = _fit_histogram_hybrid(counts, bin_edges, centres=initial_peaks, sigmas=initial_sigma, **kwargs)
    elif method == "full":
        if initial_amps is None:
            initial_amps = [
                initial_amplitude_guess(counts, bin_edges, mu, sig) for mu, sig in zip(initial_peaks, initial_sigma)
            ]
        initial_amps = np.asarray(initial_amps, dtype=float)
        fitres = _fit_histogram_full(
            counts, bin_edges, centres=initial_peaks, sigmas=initial_sigma, amplitudes=initial_amps, **kwargs
        )
    else:
        raise ValueError("Unsupported method; choose 'hybrid' or 'full'")

    # Add convenient bin centers/widths attributes
    fitres.bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fitres.bin_widths = float(np.mean(np.diff(bin_edges)))

    x_highres = np.linspace(bin_edges[0], bin_edges[-1], num=10 * counts.size)
    amps = fitres.peaks["A"].to_numpy()
    mus = fitres.peaks["mu"].to_numpy()
    sigs = fitres.peaks["sigma"].to_numpy()
    if amps.size == 0:
        y_highres = np.full_like(x_highres, fill_value=fitres.background)
    else:
        y_highres = pointwise_gaussian(
            x_highres, mus, sigs, amps, background=fitres.background, binwidth=fitres.bin_widths
        )
    fitres.model_x = x_highres
    fitres.model_y = y_highres

    return fitres
