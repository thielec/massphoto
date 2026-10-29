"""
Test the correlation map function in the massphoto module:
Only basic tests. The mathematical correctness of the correlation map is not tested here.
"""

import numpy as np
import pytest

from massphoto.filter.correlation_map import correlation_map


def test_correlation_map():
    """Test the correlation map function."""

    # correlation_map applied to a 3D array should preserve the shape
    test_movie = np.random.random((4, 50, 100))
    result_ncor, result_cor = correlation_map(test_movie)
    assert result_ncor.shape == test_movie.shape, "Output shape should match input shape."
    assert result_cor.shape == test_movie.shape, "Output shape should match input shape."

    # The value range of the normalized correlation map should be between -1 and 1
    assert np.all(result_ncor >= -1) and np.all(
        result_ncor <= 1
    ), "Normalized correlation map values should be between -1 and 1."

    # frames that contain only NaNs should propagate through as NaN in outputs
    test_movie[1] = np.nan
    result_ncor, result_cor = correlation_map(test_movie)
    assert (
        np.isnan(result_ncor[1]).all() and np.isnan(result_cor[1]).all()
    ), "Entirely invalid frames must remain NaN in both output maps"


def test_correlation_map_invalid():
    """Test that an error is raised for invalid input dimensions."""

    # 1D array should raise a ValueError
    with pytest.raises(ValueError):
        correlation_map(np.arange(10))

    # 2D array should raise a ValueError
    with pytest.raises(ValueError):
        correlation_map(np.ones((4, 4)))

    with pytest.raises(ValueError):
        correlation_map(np.ones((4, 4, 4)), kern="invalid_psf")

    assert correlation_map(np.ones((0, 0, 0)))[0].ndim == 3, "Output should be 3D even if input is empty."
