"""
Tests for the drift correction.
"""

import numpy as np
import pytest
import scipy as sp

from massphoto.movie_drift_correction import correct_drift, get_shift_rfft


def test_correct_drift_exp(sample_frames):
    """Test drift correction using real experimental data."""
    correct_frames, drift_yx = correct_drift(sample_frames, return_drift=True)
    assert correct_frames.shape == sample_frames.shape, "Output shape should match input shape."
    assert np.all(np.isfinite(correct_frames)), "Corrected frames should be finite."
    # the temporal change should be lower
    temp_change = np.sum(np.abs(np.diff(sample_frames, axis=0)))
    temp_change_corrected = np.sum(np.abs(np.diff(correct_frames, axis=0)))
    assert temp_change_corrected < temp_change, "Corrected frames should have lower temporal change."

    # the maximum drift should be lower than 2 pixels
    assert np.max(np.abs(drift_yx)) < 2, "Maximum drift should be lower than 2 pixels."
    # the drift should be a 2D array
    assert drift_yx.shape[1] == 2, "Drift should be a 2D array."
    # the drift should be a 2D array
    assert drift_yx.shape[0] == sample_frames.shape[0], "Drift should have the same number of frames as the input."

    # adding a padding should not change the temporal change by much
    pad = 5
    correct_frames_pad = correct_drift(sample_frames, pad=pad)
    assert correct_frames_pad.shape == sample_frames.shape, "Output shape should match input shape."
    assert np.all(np.isfinite(correct_frames_pad)), "Corrected frames should be finite."
    # the temporal change should be lower
    temp_change_corrected_pad = np.sum(np.abs(np.diff(correct_frames_pad, axis=0)))
    assert temp_change_corrected_pad < temp_change, "Corrected frames should have lower temporal change."
    assert (
        np.abs(temp_change_corrected_pad - temp_change_corrected) / temp_change_corrected < 0.1
    ), "Padding should not change the temporal change by more than 10%."


def test_correct_drift_subpixel(sample_frames):
    """Test drift correction using one frame with known subpixel drift."""
    base_frame = sample_frames[0]
    frames = 5

    # Use arbitrary sub-pixel shifts
    rng = np.random.RandomState(42)
    shift_y = rng.uniform(0.05, 0.5, size=(frames - 1,))
    shift_x = rng.uniform(0.05, 0.5, size=(frames - 1,))
    shift_yx = np.concatenate((np.zeros((1, 2)), np.column_stack((shift_y, shift_x))))

    # Create a movie with controlled drift using sample_frames[0] as base
    movie = np.zeros((frames,) + base_frame.shape)

    for i in range(frames):
        # First frame is unshifted
        if i == 0:
            movie[i] = base_frame
        else:
            movie[i] = sp.ndimage.shift(base_frame, (shift_yx[i, 0], shift_yx[i, 1]), order=3, mode="nearest")

    # Apply drift correction and return the drift values
    corrected_movie, detected_drift = correct_drift(movie, return_drift=True, pad=1)

    # Check drift correction
    assert np.allclose(detected_drift, shift_yx, atol=0.05), "Detected drift should match the known drift."
    assert np.all(
        np.mean(
            np.abs(corrected_movie[:, 2:-2, 2:-2] - base_frame[None, 2:-2, 2:-2]),
            axis=(1, 2),
        )
        / np.mean(base_frame)
        < 1e-2
    ), "Corrected frames should be close to the base frame."

    detected_drift = get_shift_rfft(movie=movie, template=base_frame, thumbnail_size=7)
    assert np.allclose(detected_drift, shift_yx, atol=0.05), "Detected drift should match the known drift."


def test_get_shift_rfft_requires_movie_or_rfft():
    with pytest.raises(ValueError, match="Either movie or movie_rfft"):
        get_shift_rfft()


def test_correct_drift_rejects_mismatched_drift_length():
    movie = np.zeros((4, 8, 8), dtype=float)
    drift = np.zeros((3, 2), dtype=float)
    with pytest.raises(ValueError, match="same length"):
        correct_drift(movie, drift_yx=drift)
