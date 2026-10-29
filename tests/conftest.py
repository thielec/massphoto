from functools import lru_cache
from pathlib import Path

import pytest

from massphoto.movie_process import calc_ratiometric
from massphoto.read import read_movie


def pytest_configure(config):
    """Early pytest hook to initialize llvmlite and precompile numba functions.

    This runs before fixtures and helps avoid llvmlite/numba native JIT finalization
    races when tests are run from IDE test runners (e.g., VS Code test explorer).

    Without the precompilation, concurrent tests usually fail silently with an LLVM ERROR
    """
    try:
        # Ensure we only attempt this when numba is installed
        import importlib

        spec = importlib.util.find_spec("numba")
        if spec is None:
            return

        # Trigger compilation entry points
        try:
            from massphoto.psf_fitting import PSF2D

            PSF2D().get_lookup_and_fitfun()
            # Also precompile the experimental PSF path.
            base = PSF2D(psf_size=11, cache_divider=64, cache_interpolate=True)
            thumb = base.get_thumbnail()
            PSF2D(psf_experimental=thumb, psf_size=11, cache_divider=64, cache_interpolate=True).get_lookup_and_fitfun()
        except Exception:
            import warnings

            warnings.warn("Numba precompilation failed during pytest configure", stacklevel=1)
    except Exception:
        # Keep pytest start-up resilient
        pass


@lru_cache()
def download_test_data():
    """Use a cached calibration movie, or download it on the first run."""
    cached_movie = Path("test_data/005_mfp1_500x.mp")
    if cached_movie.is_file():
        return cached_movie

    from zenodozipdownloader import ZenodoZipDownloader

    files = ZenodoZipDownloader(
        doi="10.5281/zenodo.15800729",
        download_dir="test_data",
    ).download(inner_pattern="005_mfp1_500x.mp")
    if not files:
        raise RuntimeError(
            "Failed to download example data. Please download it manually from https://zenodo.org/record/15800729 "
            "or set path_calib to your own calibration movie."
        )
    return files[0]


@lru_cache()
def read_test_frames():
    """Load frames from test data file."""
    test_data_path = download_test_data()
    frames, _ = read_movie(test_data_path)
    return frames


@pytest.fixture(scope="session")
def test_data_path():
    """Download test data and return the path."""
    return download_test_data()


@pytest.fixture(scope="session")
def frames(test_data_path):
    """Load frames from test data file."""
    return read_test_frames()


@pytest.fixture(scope="session")
def sample_frames(frames):
    """Return the first 100 frames for testing."""
    return frames[:100]


@pytest.fixture(scope="session")
def ratiometric(sample_frames):
    """Calculate ratiometric representation for the first 100 frames."""
    return calc_ratiometric(sample_frames, 10)
