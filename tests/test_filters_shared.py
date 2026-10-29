import numpy as np
import pytest

from massphoto.filter.correlation_map import correlation_map
from massphoto.filter.gaussian_laplace import gaussian_laplace
from massphoto.filter.median_filter import median_filter_sep_nd
from massphoto.filter.symmetry_filter import symmetry_filter
from massphoto.filter.temporal_ncc import step_correlation
from massphoto.filter.util import stack_apply_filter

FILTER_SPECS = [
    ("gaussian_laplace", gaussian_laplace, {"sigma": 1.0}),
    ("median_filter_sep_nd", median_filter_sep_nd, {"kernel_size": 3, "smoothing": 0.5}),
    ("median_filter_sep_nd", median_filter_sep_nd, {"kernel_size": 3, "axes": 0, "subtract_bg": False}),
    ("symmetry_filter", symmetry_filter, {"radius": 2, "alpha": 3, "beta": 0.5}),
    ("symmetry_filter", symmetry_filter, {"radius": 2, "alpha": 3, "beta": 0.0}),
    ("step_correlation", step_correlation, {"navg": 2, "ngap": 1}),
    ("step_correlation", step_correlation, {"navg": 2, "ngap": 1, "median_filter_size": 3}),
    ("correlation_map", correlation_map, {"kern": np.ones((3, 3), dtype=np.float32)}),
]


def _normalize_output(result):
    if isinstance(result, tuple):
        return list(result)
    return [result]


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
@pytest.mark.parametrize("_name, func, kwargs", FILTER_SPECS)
def test_filters_accept_3d_and_return_float(dtype, _name, func, kwargs):
    rng = np.random.RandomState(0)
    if np.issubdtype(dtype, np.integer):
        movie = rng.randint(0, 10, size=(3, 5, 7)).astype(dtype)
    else:
        movie = rng.random((3, 5, 7)).astype(dtype)

    result = func(movie, **kwargs)
    for output in _normalize_output(result):
        assert output.size == movie.size, "Output size should match input size."
        assert np.issubdtype(output.dtype, np.floating) or np.issubdtype(
            output.dtype, movie.dtype
        ), "Output dtype should be floating point or match input dtype."


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("_name, func, kwargs", FILTER_SPECS)
def test_filters_handle_nans(_name, func, kwargs):
    movie = np.random.random((3, 5, 7)).astype(np.float32)
    movie[0, 0, 0] = np.nan

    result = func(movie, **kwargs)
    for output in _normalize_output(result):
        assert output.size == movie.size
        assert np.issubdtype(output.dtype, np.floating)


@pytest.mark.parametrize("_name, func, kwargs", FILTER_SPECS)
def test_filters_handle_empty(_name, func, kwargs):
    movie = np.empty((0, 4, 4), dtype=np.float32)

    result = func(movie, **kwargs)
    for output in _normalize_output(result):
        assert output.size == movie.size
        assert np.issubdtype(output.dtype, np.floating)


def test_stack_apply_filter_requires_callable():
    movie = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(TypeError, match="filter_func must be callable"):
        stack_apply_filter(42, movie)
