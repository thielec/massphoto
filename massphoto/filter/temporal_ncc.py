"""
Normalised cross-correlation (NCC) with a step template along the temporal (first) axis.
"""

import bottleneck as bn
import numpy as np

from ..filter.median_filter import median_filter_sep_nd
from ..utils import defaults


@defaults.sync_apply_defaults()
def step_correlation(
    img: np.ndarray, navg: int = 20, ngap: int = 1, median_filter_size: int = 0, axis: int = 0
) -> np.ndarray:
    """
    Normalised step-correlation (Pearson-style NCC) along time axis (axis 0).

    For each time index t, compute trailing-window means/variances (length n = navg)
    on the two sides of a potential step, separated by ngap masked frames:

        mean_after(t),  var_after(t)   - window after the step
        mean_before(t), var_before(t)  - window before the step

    Correlation with a ±1 step template [-1,..., -1, +1,..., +1] (length 2n, mean 0) reduces to:

        r(t) = (mean_after - mean_before) /
               sqrt( (mean_after - mean_before)**2 + 2 * ( var_after + var_before ) )

    With this sign convention, an up-step (low->high) gives +1 for an ideal noiseless step,
    and a down-step gives -1. Edges where windows are invalid are set to NaN.

    Parameters
    ----------
    img : np.ndarray
        Movie array with time on axis 0 (shape [T, ...]).
    navg : int, default=20
        Number of frames averaged on each side of the step.
    ngap : int, default=1
        Number of masked frames between the two averaging windows.
    median_filter_size : int, default=0
        Optional spatial median filter kernel size applied to each frame before
        temporal correlation. Set to 0 to disable.
    axis : int, default=0
        Axis along which to compute the step-correlation. Default is 0 (time axis).

    Returns
    -------
    np.ndarray
        Step-correlation array with the same shape as img.
    """
    # 1) Validate axis / length and return early when no valid windows fit.
    axis = axis % img.ndim  # handle negative axis
    if img.shape[axis] < navg + ngap:
        return np.full(img.shape, np.nan, dtype=float)

    # 2) Compute trailing-window mean/variance (window length = navg).
    # bottleneck.move_var uses ddof=0 by default.
    window_mean = bn.move_mean(img, navg, axis=axis)
    window_variance = bn.move_var(img, navg, axis=axis)

    # 3) Build aligned slices for valid output region and the two comparison windows
    #    around each potential step location.
    start = navg + (ngap - 1) // 2
    stop = img.shape[axis] - navg - (ngap // 2)
    after_shift = navg + (ngap // 2)
    before_shift = -((ngap + 1) // 2)

    valid_slice = [slice(None)] * img.ndim
    valid_slice[axis] = slice(start, stop)
    valid_slice = tuple(valid_slice)

    after_slice = [slice(None)] * img.ndim
    before_slice = [slice(None)] * img.ndim
    after_slice[axis] = slice(start + after_shift, stop + after_shift)
    before_slice[axis] = slice(start + before_shift, stop + before_shift)
    after_slice = tuple(after_slice)
    before_slice = tuple(before_slice)

    mean_after = window_mean[after_slice]
    mean_before = window_mean[before_slice]
    var_after = window_variance[after_slice]
    var_before = window_variance[before_slice]

    # 4) Apply the closed-form normalized step-correlation expression.
    mean_diff = mean_after - mean_before
    denominator = np.sqrt(mean_diff * mean_diff + 2.0 * (var_after + var_before))

    correlation = np.full(img.shape, np.nan, dtype=float)
    np.divide(mean_diff, denominator, out=correlation[valid_slice], where=denominator > 0)

    # 5) Optional spatial median filter post-processing.
    if median_filter_size > 0:
        correlation = median_filter_sep_nd(
            correlation,
            kernel_size=median_filter_size,
            axes=(-2, -1),
        )

    return correlation
