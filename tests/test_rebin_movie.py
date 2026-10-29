"""
Tests for the rebin_movie function in movie_process module.
"""

import numpy as np
import pytest

from massphoto.movie_process import rebin_movie


def test_rebin_movie_2d():
    """Test rebinning of a 2D array."""
    # Create a simple 2D array with known pattern
    test_array = np.arange(16).reshape(4, 4)

    # Test with no binning (should return the same array)
    result = rebin_movie(test_array, pixel_binning=1)
    np.testing.assert_array_equal(result, test_array)

    # Test with pixel_binning=2
    expected = np.array([[2.5, 4.5], [10.5, 12.5]])  # Mean of each 2x2 block
    result = rebin_movie(test_array, pixel_binning=2)
    np.testing.assert_array_equal(result, expected)


def test_rebin_movie_3d():
    """Test rebinning of a 3D array (movie)."""
    # Create a simple 3D array with known pattern
    test_movie = np.arange(24).reshape(2, 4, 3)

    # Test with no binning (should return the same array)
    result = rebin_movie(test_movie, pixel_binning=1, frame_binning=1)
    np.testing.assert_array_equal(result, test_movie)

    # Test with pixel_binning=2 (spatial only)
    # also tests handling of odd dimensions
    expected = np.array([[[2.0], [8.0]], [[14.0], [20.0]]])
    result = rebin_movie(test_movie, pixel_binning=2, frame_binning=1)
    np.testing.assert_array_equal(result, expected)

    # Test with pixel_binning=3
    expected = np.array([[[4.0]], [[16.0]]])
    result = rebin_movie(test_movie, pixel_binning=3, frame_binning=1)
    np.testing.assert_array_equal(result, expected)

    # Test with frame_binning=2 (temporal only)
    test_movie = np.arange(48).reshape(4, 3, 4)
    expected = np.array(
        [
            [[6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0]],
            [
                [30.0, 31.0, 32.0, 33.0],
                [34.0, 35.0, 36.0, 37.0],
                [38.0, 39.0, 40.0, 41.0],
            ],
        ]
    )
    result = rebin_movie(test_movie, pixel_binning=1, frame_binning=2)
    np.testing.assert_array_equal(result, expected)

    # Test with pixel_binning=2 and frame_binning=2
    expected = np.array([[[8.5, 10.5]], [[32.5, 34.5]]])
    result = rebin_movie(test_movie, pixel_binning=2, frame_binning=2)
    np.testing.assert_array_equal(result, expected)


def test_rebin_movie_error_case():
    """Test that an error is raised for invalid input dimensions."""
    # 1D array should raise a ValueError
    with pytest.raises(ValueError):
        rebin_movie(np.arange(10))

    # Check that frame_binning for 2D arrays raises a clear validation error
    with pytest.raises(ValueError, match="frame_binning must be 1"):
        rebin_movie(np.ones((4, 4)), frame_binning=2)


def test_rebin_movie_large():
    """Test rebinning with larger array."""
    # Create a movie similar to ones typically processed with massphoto
    test_movie = np.random.random((20, 64, 64))

    # Test various binning configurations
    result1 = rebin_movie(test_movie, pixel_binning=2)
    assert result1.shape == (20, 32, 32)

    result2 = rebin_movie(test_movie, frame_binning=2)
    assert result2.shape == (10, 64, 64)

    result3 = rebin_movie(test_movie, pixel_binning=4, frame_binning=5)
    assert result3.shape == (4, 16, 16)


def test_rebin_movie_real_data(sample_frames):
    """Test rebinning using real data from the sample_frames fixture."""
    # Get the original shape
    original_shape = sample_frames.shape

    # Rebin with pixel_binning=2
    result = rebin_movie(sample_frames, pixel_binning=2)
    expected_shape = (original_shape[0], original_shape[1] // 2, original_shape[2] // 2)

    # Check the shape is as expected
    assert result.shape == expected_shape

    # Check the mean of the rebinned data is close to the mean of the original data
    np.testing.assert_almost_equal(
        np.mean(sample_frames[:, : 2 * result.shape[1], : 2 * result.shape[2]]),
        np.mean(result),
    )


def test_rebin_movie_metadata_3d_updates():
    """Test metadata updates for 3D input."""
    test_movie = np.arange(5 * 5 * 4).reshape(5, 5, 4)
    metadata = {
        "framerate": 100.0,
        "framebinning": 2,
        "pixelsize": 0.5,
        "pixelsize_unit": "um",
        "pixelbinning": 3,
        "image_shape": (5, 4),
        "framerate_effective": 50.0,
        "pixelsize_effective": 1.5,
        "exposuretime": 0.01,
        "exposuretime_effective": 0.02,
        "field_of_view": (7.5, 6.0),
        "field_of_view_unit": "um",
    }

    rebinned, updated_metadata = rebin_movie(test_movie, pixel_binning=2, frame_binning=2, metadata=metadata)

    assert rebinned.shape == (2, 2, 2)
    assert updated_metadata is not metadata
    assert metadata["framebinning"] == 2
    assert metadata["pixelbinning"] == 3

    assert updated_metadata["framebinning"] == 4
    assert updated_metadata["pixelbinning"] == 6
    assert updated_metadata["framerate_effective"] == 25.0
    assert updated_metadata["pixelsize_effective"] == 3.0
    assert updated_metadata["image_shape"] == (2, 2)
    assert updated_metadata["exposuretime_effective"] == 0.04
    assert updated_metadata["field_of_view"] == (6.0, 6.0)
    assert updated_metadata["field_of_view_unit"] == "um"


def test_rebin_movie_metadata_2d_updates():
    """Test metadata updates for 2D input."""
    test_array = np.arange(20).reshape(4, 5)
    metadata = {
        "framerate": 90.0,
        "framebinning": 3,
        "pixelsize": 0.4,
        "pixelsize_unit": "um",
        "pixelbinning": 2,
        "image_shape": (4, 5),
        "framerate_effective": 30.0,
        "pixelsize_effective": 0.8,
    }

    rebinned, updated_metadata = rebin_movie(test_array, pixel_binning=2, metadata=metadata)

    assert rebinned.shape == (2, 2)
    assert updated_metadata["framebinning"] == 3
    assert updated_metadata["pixelbinning"] == 4
    assert updated_metadata["framerate_effective"] == 30.0
    assert updated_metadata["pixelsize_effective"] == 1.6
    assert updated_metadata["image_shape"] == (2, 2)


def test_rebin_movie_metadata_missing_required_key_raises():
    """Test that invalid metadata raises a clear error."""
    metadata = {
        "framerate": 100.0,
        "framebinning": 1,
        "pixelsize": 0.5,
        "pixelbinning": 1,
    }

    with pytest.raises(ValueError, match="convert_metadata"):
        rebin_movie(np.ones((4, 4)), pixel_binning=2, metadata=metadata)
