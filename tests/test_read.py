"""
Tests for reading functionality.
"""

import pytest

import massphoto as mp
from massphoto.read import read_movie


def test_read_movie_invalid_file():
    with pytest.raises(FileNotFoundError):
        read_movie("invalid/path/to/movie.mp")


def test_read_movie(test_data_path):
    """Test reading MP file format."""
    frames, meta_data = read_movie(test_data_path, convert_metadata=True)

    # Check dimensions of the frames
    assert frames.shape[0] == 14949
    assert frames.shape[1] == 59
    assert frames.shape[2] == 150

    # Check metadata fields
    assert isinstance(meta_data, dict)
    meta_data_template = mp.convert_metadata({})
    assert len(meta_data_template.keys()) > 8
    # Check metadata fields
    for key in meta_data_template.keys():
        assert key in meta_data.keys(), f"Missing metadata field: {key}"

    frames, meta_data = read_movie(test_data_path, convert_metadata=False)
    meta_data = mp.convert_metadata(meta_data, include_native=True)
    # Check "native" is non empty dict
    assert isinstance(meta_data["native"], dict), "Native metadata should be a dictionary."
    assert len(meta_data["native"].keys()) > 0, "Native metadata should not be empty."


def test_tiff_write_read(test_data_path):
    """Test writing and reading a TIFF file if tifffile is installed."""
    import os
    import tempfile

    import numpy as np

    pytest.importorskip("tifffile")

    from massphoto.read import read_movie
    from massphoto.write import save_tiff

    # Read the movie from the test MP file
    movie, metadata = read_movie(test_data_path)

    # Prepare some scalar metadata for testing
    test_metadata = {"test_key": 42, "another_key": "test_value", "float_key": 3.14}

    # Write to a temporary TIFF file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_filepath = os.path.join(tmp_dir, "test.tiff")

        save_tiff(tmp_filepath, movie, metadata=test_metadata, dtype="uint16")

        # Read back the TIFF file
        read_movie_back, read_metadata = read_movie(tmp_filepath)

    # Assert the movie data matches
    np.testing.assert_array_equal(movie, read_movie_back)

    # Check that the metadata is preserved in the TIFF file (if saved as tags)
    for key, value in test_metadata.items():
        assert key in read_metadata, f"Metadata key '{key}' not found in read metadata."
        assert read_metadata[key] == value, f"Metadata value for key '{key}' does not match."
