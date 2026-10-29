"""
Histogram peak detection and bandwidth estimation using SciPy.

Provides:
- estimate_bandwidth_from_fwhm(...) — estimate sigma (same units as bin_edges) from FWHM of the highest peak.
- find_peak_guesses(...) — automatic or user-guided initial peak guesses (mu, sigma, A, idx).

Notes:
- Functions return values in the same units as the supplied bin_edges.
- Functions support various input formats via :func:`distfit.utils.standardize_histogram_input`.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .utils import standardize_histogram_input


def estimate_bandwidth_from_fwhm(
    data: np.ndarray | tuple[np.ndarray, np.ndarray],
    bins: int | np.ndarray | None = None,
    binwidth: float | None = None,
    peak_index: int | None = None,
    pre_smooth_sigma_bins: float = 1.0,
    fallback_to_moment: bool = True,
) -> float:
    """
    Estimate bandwidth (sigma) from FWHM of the tallest peak.

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
    binwidth : bin width for automatic binning
    peak_index : index of peak to measure (default: tallest peak)
    pre_smooth_sigma_bins : smoothing sigma in bin units before FWHM measurement
    fallback_to_moment : if FWHM fails, estimate from second moment

    Returns
    -------
    sigma : estimated bandwidth in same units as bin_edges
    """
    counts, bin_edges = standardize_histogram_input(data, bins, binwidth)
    if counts.size + 1 != bin_edges.size:
        raise ValueError("bin_edges length must be counts.size + 1")

    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    avg_bin_width = np.mean(bin_widths) if not np.allclose(bin_widths, bin_widths[0]) else bin_widths[0]

    smoothed = gaussian_filter1d(counts, sigma=pre_smooth_sigma_bins, mode="nearest")
    if peak_index is None:
        peak_index = int(np.argmax(smoothed))
    peak_height = smoothed[peak_index]
    if peak_height <= 0:
        raise ValueError("Peak height non-positive – cannot estimate FWHM")

    # Find half-maximum crossings
    half = 0.5 * peak_height
    left_idx = right_idx = None
    for i in range(peak_index, 0, -1):
        if smoothed[i - 1] <= half <= smoothed[i] or (smoothed[i - 1] < half and smoothed[i] < half):
            left_idx = i - 1
            break
    for i in range(peak_index, smoothed.size - 1):
        if smoothed[i] >= half >= smoothed[i + 1] or (smoothed[i] < half and smoothed[i + 1] < half):
            right_idx = i + 1
            break

    if (left_idx is not None) and (right_idx is not None) and (left_idx < right_idx):
        x_left = float(np.interp(half, smoothed[left_idx : left_idx + 2], bin_centres[left_idx : left_idx + 2]))
        x_right = float(
            np.interp(
                half,
                [smoothed[right_idx], smoothed[right_idx - 1]],
                [bin_centres[right_idx], bin_centres[right_idx - 1]],
            )
        )
        fwhm = float(x_right - x_left)
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return float(sigma)

    # Fallback to moment-based estimate
    if fallback_to_moment:
        window_bins = max(1, int(round(3 * max(1.0, pre_smooth_sigma_bins))))
        i0 = max(0, peak_index - window_bins)
        i1 = min(counts.size, peak_index + window_bins + 1)
        sub_centres = bin_centres[i0:i1]
        sub_counts = np.maximum(counts[i0:i1], 0.0)
        total = sub_counts.sum()
        if total <= 0:
            return float(max(avg_bin_width, 1e-12))
        mean = (sub_centres * sub_counts).sum() / total
        var = (sub_counts * (sub_centres - mean) ** 2).sum() / total
        sigma = float(np.sqrt(var))
        sigma = float(max(sigma, avg_bin_width * 0.5))
        return sigma

    raise RuntimeError("Could not estimate FWHM – half-maximum crossings not found")


def find_peak_guesses(
    data: np.ndarray | tuple[np.ndarray, np.ndarray],
    bins: int | np.ndarray | None = None,
    binwidth: float | None = None,
    sigma: float | None = None,
    min_prominence: float | None = None,
    expected_positions: list[float] | None = None,
    min_distance_sigma: float = 1.0,
    max_peaks: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Find initial peak guesses for fitting.

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
    binwidth : bin width for automatic binning
    sigma : expected peak width (auto-estimated if None, otherwise returned unchanged)
    min_prominence : minimum peak prominence (auto-estimated if None)
    expected_positions : if provided, return guesses at these positions instead of auto-detection
    min_distance_sigma : minimum distance between peaks in units of sigma
    max_peaks : maximum number of peaks to return

    Returns
    -------
    peak_positions : array of peak centres (mu values)
    sigma : peak width (same as input if provided, otherwise estimated)
    """
    counts, bin_edges = standardize_histogram_input(data, bins, binwidth)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    avg_bin_width = np.mean(bin_widths)

    # Auto-estimate sigma if not provided
    if sigma is None:
        sigma = estimate_bandwidth_from_fwhm((counts, bin_edges))

    # User-provided positions: return those positions
    if expected_positions is not None:
        return np.asarray(expected_positions), float(sigma)

    sigma_bins = float(max(0.5, sigma / max(avg_bin_width, 1e-15)))
    smooth_sigma_bins = max(1.0, 1.5 * sigma_bins)
    smoothed = gaussian_filter1d(counts, sigma=smooth_sigma_bins, mode="nearest")

    # Auto-detect peaks
    if min_prominence is None:
        vals = counts
        low = vals[vals <= np.percentile(vals, 50)]
        noise = np.std(vals) if low.size < 5 else 1.4826 * np.median(np.abs(low - np.median(low)))
        min_prominence = float(max(1.0, 3.0 * noise))

    min_distance_bins = max(1, int(round(min_distance_sigma * sigma_bins)))
    peak_idxs, props = find_peaks(smoothed, prominence=min_prominence, distance=min_distance_bins)

    if peak_idxs.size == 0:
        peak_idxs = np.array([int(np.argmax(smoothed))], dtype=int)

    if max_peaks is not None and peak_idxs.size > max_peaks:
        prominences = (
            props.get("prominences", smoothed[peak_idxs]) if peak_idxs.size > 1 else np.array([smoothed[peak_idxs[0]]])
        )
        order = np.argsort(prominences)[::-1][:max_peaks]
        peak_idxs = peak_idxs[order]

    positions = bin_centres[peak_idxs]
    return positions, float(sigma)
