"""
Tests for movie processing functionality.
"""

import itertools as it

import numpy as np
import pytest

from massphoto.movie_process import (
    calc_differential,
    calc_mobileframes,
    calc_ratiometric,
    normalize_frames,
)


@pytest.mark.parametrize("window_size, ngap", list(it.product([1, 2, 3, 5, 10, 20], [0, 1, 2, 3])))
def test_calc_ratiometric_combinations(window_size, ngap):
    """Test calc_ratiometric with all combinations of window_size and ngap."""
    test_frames = np.ones((100, 16, 16)) * (1 + np.linspace(1, 2, 100)[:, None, None])
    ratio = calc_ratiometric(
        test_frames, window_size, ngap=ngap, medfilt_kernel_size=0
    )  # Disable median filter as the test frames are smooth

    # Check dimensions
    assert ratio.shape == test_frames.shape, "Output shape should match input shape."

    # Check NaN regions
    valid_start = window_size + (ngap - 1) // 2
    valid_end = -window_size - ngap // 2
    assert np.all(np.isnan(ratio[:valid_start])), f"First {valid_start} frames should be NaN."
    assert np.all(np.isnan(ratio[valid_end:])), f"Last {valid_end} frames should be NaN."
    assert np.all(np.isfinite(ratio[valid_start:valid_end])), "Middle frames should be finite."

    # Check statistical properties
    middle_frames = ratio[valid_start:valid_end]
    assert (
        np.mean(np.std(middle_frames, axis=(1, 2))) < 1e-5
    ), "Standard deviation of middle frames should be close to zero."
    assert np.mean(middle_frames) > 0, "Mean of middle frames should be greater than zero."


def test_calc_ratiometric_invalid_window_size():
    """Test calc_ratiometric with invalid window size."""
    sample_frames = np.ones((10, 1, 1))
    ratio = calc_ratiometric(sample_frames, navg=20, ngap=1)
    assert np.isnan(ratio).all(), "All frames should be NaN when window size is larger than the number of frames."
    assert ratio.shape == sample_frames.shape, "Output shape should match input shape."


def test_normalize_frames_uses_target_mean():
    movie = np.ones((3, 2, 2), dtype=np.float32)
    movie *= np.array([1, 2, 4], dtype=np.float32)[:, None, None]

    normalized = normalize_frames(movie, target_mean=8)

    np.testing.assert_allclose(normalized.mean(axis=(1, 2)), 8)


def test_normalize_frames_leaves_zero_mean_frames_unscaled():
    movie = np.ones((2, 2, 2), dtype=np.float32)
    movie[0] = 0

    normalized = normalize_frames(movie, target_mean=5)

    np.testing.assert_array_equal(normalized[0], movie[0])
    np.testing.assert_allclose(normalized[1].mean(), 5)


@pytest.mark.parametrize("window_size, ngap", list(it.product([1, 2, 3, 5, 10, 20], [0, 1, 2, 3])))
def test_calc_differential_combinations(window_size, ngap):
    """Test calc_differential with all combinations of window_size and ngap."""
    test_frames = np.ones((100, 16, 16)) * np.linspace(1, 2, 100)[:, None, None]
    diff = calc_differential(test_frames, window_size, ngap=ngap)

    # Check dimensions
    assert diff.shape == test_frames.shape, "Output shape should match input shape."

    # Check NaN regions
    valid_start = window_size + (ngap - 1) // 2
    valid_end = -window_size - ngap // 2
    assert np.all(np.isnan(diff[:valid_start])), f"First {valid_start} frames should be NaN."
    assert np.all(np.isnan(diff[valid_end:])), f"Last {valid_end} frames should be NaN."
    assert np.all(np.isfinite(diff[valid_start:valid_end])), "Middle frames should be finite."

    # Check statistical properties
    middle_frames = diff[valid_start:valid_end]
    assert (
        np.mean(np.std(middle_frames, axis=(1, 2))) < 1e-5
    ), "Standard deviation of middle frames should be close to zero."
    assert np.mean(middle_frames) > 0, "Mean of middle frames should be greater than zero."
    assert np.allclose(np.diff(middle_frames, axis=0), 0), "All frames should be equal."


def test_calc_differential_invalid_window_size():
    """Test calc_differential with invalid window size."""
    sample_frames = np.ones((10, 1, 1))
    ratio = calc_differential(sample_frames, navg=20, ngap=1)
    assert np.isnan(ratio).all(), "All frames should be NaN when window size is larger than the number of frames."
    assert ratio.shape == sample_frames.shape, "Output shape should match input shape."


@pytest.mark.parametrize(
    "local_bg_halfsize, medfilt_kernel_size",
    list(
        it.product(
            [1, 2, 3, 5, 10, 20, 100],  # local_bg_halfsize values (note: 100 frames is larger than the test_frames)
            [0, 3, 5],  # medfilt_kernel_size values
        )
    ),
)
def test_calculate_mobileframes_combinations(local_bg_halfsize, medfilt_kernel_size):
    """Test calc_mobileframes with various combinations of parameters."""
    test_frames = np.ones((100, 16, 16)) * (1 + np.linspace(1, 2, 100)[:, None, None])
    mobile_frames = calc_mobileframes(test_frames, local_bg_halfsize, medfilt_kernel_size=medfilt_kernel_size)

    # Check dimensions
    assert mobile_frames.shape == test_frames.shape, "Output shape should match input shape."

    # The valid region will be reduced by the local_bg_halfsize on both ends
    valid_start = local_bg_halfsize
    valid_end = -local_bg_halfsize if local_bg_halfsize > 0 else None

    if test_frames.shape[0] >= (2 * local_bg_halfsize) + 1:
        # Only check finite values if the window is smaller than the number of frames
        assert np.all(np.isfinite(mobile_frames[valid_start:valid_end])), "Middle frames should be finite."

        # After normalization and background division, mobile frames should have values around 0
        middle_frames = mobile_frames[valid_start:valid_end]
        assert np.abs(np.mean(middle_frames)) < 0.1, "Mean of mobile frames should be close to zero."

        # There should be no NaN values in the mobile frames
        assert not np.any(np.isnan(middle_frames)), "Mobile frames should not contain NaN values."
    else:
        # If window size is larger than frame count, check that we get zeros
        assert np.all(mobile_frames == 0.0), "All frames should be zero when window size is larger than frame count."

        assert mobile_frames.shape == test_frames.shape, "Output shape should match input shape."
