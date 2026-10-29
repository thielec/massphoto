"""
Utility helpers — input standardization, unit detection, bounds and amplitude guessing.

Functions
- standardize_histogram_input(...) → (counts, bin_edges) from various input formats
- default_bounds(n_peaks, bin_edges, units=None, sigma_hint=None, centre_padding=0.05)
— initial_amplitude_guess(counts, bin_edges, mu, sigma, n_sigma=1.5)
"""

import numpy as np
import pandas as pd

DF_KEYS = ("masses_kDa", "contrasts")


def standardize_histogram_input(
    data: pd.DataFrame | np.ndarray | tuple[np.ndarray, np.ndarray],
    bins: int | np.ndarray | None = None,
    binwidth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardize various histogram input formats to (counts, bin_edges).

    Accepts multiple input patterns:
    - data=(counts, edges): tuple of pre-computed histogram
    - data=DataFrame with columns 'masses_kDa' or 'contrasts': extracts data from DataFrame
    - data=samples: unbinned data with automatic binning (100 bins)
    - data=samples, bins=N: unbinned data with N bins
    - data=samples, bins=edges: unbinned data with explicit bin edges
    - data=samples, binwidth=w: unbinned data with specified bin width

    Parameters
    ----------
    data : unbinned sample data OR tuple (counts, bin_edges)
    bins : number of bins (int) or bin edges (array)
    binwidth : width of bins (used to compute number of bins from data range)

    Returns
    -------
    counts : histogram bin counts as float array
    bin_edges : histogram bin edges as float array

    Examples
    --------
    >>> counts, edges = standardize_histogram_input(samples)
    >>> counts, edges = standardize_histogram_input(samples, bins=50)
    >>> counts, edges = standardize_histogram_input(samples, binwidth=0.1)
    >>> counts, edges = standardize_histogram_input((counts, edges))
    """
    # Case 1: data is a tuple (counts, bin_edges)
    if isinstance(data, tuple) and len(data) == 2:
        counts_arr = np.asarray(data[0], dtype=float)
        edges_arr = np.asarray(data[1], dtype=float)
        if edges_arr.ndim != 1 or edges_arr.size != counts_arr.size + 1:
            raise ValueError("bin_edges must have length len(counts) + 1")
        return counts_arr, edges_arr

    # Case 2: unbinned data
    # Extract data from DataFrame if needed
    if isinstance(data, pd.DataFrame):
        for key in DF_KEYS:
            if key in data.columns:
                data = data[key].to_numpy(dtype=float)
                break
        else:
            raise ValueError(f"DataFrame must contain one of the columns: {DF_KEYS}")
    data_arr = np.asarray(data, dtype=float)

    # Determine binning strategy
    if bins is not None:
        # bins can be int (number of bins) or array (bin edges)
        bins_arg = bins
    elif binwidth is not None:
        # Compute number of bins from binwidth
        data_range = data_arr.max() - data_arr.min()
        n_bins = max(1, int(np.ceil(data_range / binwidth)))
        bins_arg = n_bins
    else:
        # Default: 100 bins
        bins_arg = 100

    counts_arr, edges_arr = np.histogram(data_arr, bins=bins_arg)
    return counts_arr.astype(float), edges_arr.astype(float)


# (Unit detection is performed inside `default_bounds`.)


def default_bounds(
    n_peaks: int,
    bin_edges: np.ndarray,
    units: str | None = None,
    sigma_hint: float | None = None,
    centre_padding: float = 0.05,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Return default bounds for centres and widths.

    Parameters
    ----------
    n_peaks : number of peaks
    bin_edges : histogram edges
    units : 'mass' or 'contrast' (auto-detected if None)
    sigma_hint : typical sigma to bias bounds
    centre_padding : fraction of data range to pad centre bounds

    Returns
    -------
    bounds_mu, bounds_sigma : lists of (min, max) for each peak
    """
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must be a 1D array with at least two elements")
    data_min, data_max = float(edges[0]), float(edges[-1])
    data_range = data_max - data_min
    if data_range <= 0:
        raise ValueError("bin_edges must have increasing values")

    if units is None:
        # Inline the tiny heuristic used to detect units so we don't keep a
        # single-use helper in the public API. This duplicates the previous
        # logic: if the median absolute value > 1 we interpret the units as
        # 'mass' else 'contrast'. Using edges is OK because the scale of the
        # histogram edges matches the data values.
        scale = float(np.median(np.abs(edges)))
        units = "mass" if scale > 1 else "contrast"

    # Unit-specific defaults
    if sigma_hint is not None:
        sigma_default = sigma_hint
    elif units == "mass":
        sigma_default = 10.0
    else:
        sigma_default = 5.0e-5

    sigma_min = sigma_default / 10
    sigma_max = sigma_default * 10

    pad = centre_padding * data_range
    mu_min = data_min - pad
    mu_max = data_max + pad

    bounds_mu = [(mu_min, mu_max) for _ in range(n_peaks)]
    bounds_sigma = [(float(sigma_min), float(sigma_max)) for _ in range(n_peaks)]

    return bounds_mu, bounds_sigma


def initial_amplitude_guess(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    mu: float,
    sigma: float,
    n_sigma: float = 1.5,
) -> float:
    """
    Estimate initial integrated counts A for a peak at mu with width sigma.

    Sums counts in bins within ± n_sigma * sigma of mu.
    """
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    half_width = max(1e-12, float(n_sigma * sigma))
    mask = np.abs(centres - mu) <= half_width
    if not np.any(mask):
        idx = int(np.argmin(np.abs(centres - mu)))
        return float(counts[idx])
    return float(counts[mask].sum())
