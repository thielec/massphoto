"""
This module contains functions for filtering images or movies.

- `correlation_map`: Computes the correlation map of a movie with a template.
- `step_correlation`: Correlates the time axis with a step function to detect step changes.
- `gaussian_laplace`: Apply a Gaussian Laplace filter to each image in a stack of images.
- `symmetry_filter`: Calculate the local radial symmetry of the images (last two dimensions).
- `median_filter_sep_nd`: Apply a separable median filter to an n-dimensional array.
"""

from .correlation_map import correlation_map
from .gaussian_laplace import gaussian_laplace
from .median_filter import median_filter_sep_nd
from .symmetry_filter import symmetry_filter
from .temporal_ncc import step_correlation
from .util import stack_apply_filter

__all__ = [
    "correlation_map",
    "gaussian_laplace",
    "median_filter_sep_nd",
    "symmetry_filter",
    "step_correlation",
    "stack_apply_filter",
]
