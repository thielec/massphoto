"""
Utility functions for image filter.
"""

from collections.abc import Callable

import numpy as np


def stack_apply_filter(filter_func: Callable, movie: np.ndarray, **kwargs) -> np.ndarray:
    """
    Apply a filter function to a stack of images.

    The function iterates over all dimensions except the last two (y, x).

    Parameters
    ----------
    filter_func : Callable
        Filter function to apply.
    movie : np.ndarray
        Stack of input images.
    **kwargs
        Additional keyword arguments to pass to the filter function.

    Returns
    -------
    np.ndarray
        Stack of filtered images.
    """
    if not callable(filter_func):
        raise TypeError(f"filter_func must be callable, got {type(filter_func).__name__}")
    if movie.size == 0:
        # return an empty array with the same shape as the input
        try:
            output = filter_func(np.zeros(movie.shape[-2:]), **kwargs)
            dtype = output.dtype
        except Exception:
            dtype = movie.dtype
        return np.empty(movie.shape, dtype=dtype)
    return np.asanyarray([filter_func(movie[idx], **kwargs) for idx in np.ndindex(movie.shape[:-2])])
