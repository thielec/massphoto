"""
Median filter for n-dimensional arrays.
Provides separable median filtering and background subtraction using bottleneck and scipy.
"""

import bottleneck as bn
import numpy as np
import scipy as sp


def median_filter_sep_nd(
    img: np.ndarray,
    kernel_size: int,
    smoothing: float = 0.0,
    axes: tuple[int, ...] | int = (-2, -1),
    pad_mode: str = "constant",
    subtract_bg: bool = True,
) -> np.ndarray:
    """
    Apply a separable median filter to an n-dimensional array.

    The separable median filter is implemented by applying 1D median filters
    sequentially along the specified axes of the array.

    Parameters
    ----------
    img : np.ndarray
        Input n-dimensional array to be filtered.
    kernel_size : int
        Size of the median filter kernel.
    smoothing : float, default=0.0
        Amount of Gaussian smoothing to apply after filtering. Default is 0 (no smoothing).
    axes : int or tuple of int, default=(-2, -1)
        Axes along which to apply the median filter. Default is the last two axes.
    pad_mode : str, default='constant'
        Padding mode to use for the array. Options are 'edge', 'constant', 'reflect', 'symmetric'.
    subtract_bg : bool, default=True
        If True, subtract the background from the filtered image. If False, return the filtered image

    Returns
    -------
    np.ndarray
        Filtered array.
    """
    if isinstance(axes, int):
        axes = (axes,)
    axes = tuple(a % img.ndim for a in axes)
    img_arr = np.asarray(img)
    work_dtype = img_arr.dtype if img_arr.dtype in (np.float32, np.float64) else np.float64
    img_filt = np.asarray(img_arr, dtype=work_dtype)
    if img_filt.size == 0:
        return img_filt

    radius = kernel_size // 2
    pad_hi = ((kernel_size + 1) // 2) - 1
    pad_arg = [(0, 0)] * img_filt.ndim
    unpad_slice = [slice(None)] * img_filt.ndim
    for axis in axes:
        pad_arg[axis] = (radius, pad_hi)
        unpad_slice[axis] = slice(kernel_size - 1, None)

    img_filt = np.pad(img_filt, pad_arg, mode=pad_mode)
    for axis in axes:
        img_filt = bn.move_median(img_filt, window=kernel_size, min_count=1, axis=axis)
    img_filt = img_filt[tuple(unpad_slice)]
    if smoothing > 0:
        img_filt = sp.ndimage.gaussian_filter(img_filt, sigma=smoothing, mode="nearest", axes=axes)
    if subtract_bg:
        img_filt = img - img_filt
    return img_filt
