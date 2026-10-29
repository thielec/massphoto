"""
Functions to write movies to h5, mp, and tiff files.
"""

import os

import h5py
import numpy as np

# import tifffile if installed
try:
    import tifffile  # type: ignore

    has_tifffile = True
except ImportError:
    has_tifffile = False


def save_h5(filepath: os.PathLike, movie: np.ndarray, metadata: dict | None = None, **kwargs) -> None:
    """Save a movie to an HDF5 file.

    Parameters
    ----------
    filepath
        Output file path.
    movie
        Movie array with shape ``(n_frames, height, width)``.
    metadata
        Optional metadata mapping to store as datasets/attributes.
    **kwargs
        Reserved for future compatibility.
    """
    save_mp(filepath, movie, metadata=metadata, mp_version=0)


def save_mp(
    filepath: os.PathLike,
    movie: np.ndarray,
    binning_time: int | None = None,
    binning_pixel: int | None = None,
    metadata: dict = None,
    mp_version: int = 0,
) -> None:
    """Save a movie to the MassPhotometer ``.mp`` HDF5 container format.

    Parameters
    ----------
    filepath
        Output file path (usually ``.mp``).
    movie
        Movie array with shape ``(n_frames, height, width)``.
    binning_time, binning_pixel
        Optional acquisition binning values written to the expected HDF5 keys
        when ``metadata`` is not provided.
    metadata
        Optional metadata mapping written into the HDF5 file. When provided, it
        takes precedence over ``binning_time`` / ``binning_pixel``.
    mp_version
        MP file layout version. Use ``3`` for the newer ``movie/frame`` layout,
        otherwise the legacy top-level ``frame`` dataset layout is used.
    """
    # metadata overwrites binning_time and binning_pixel
    # this is the old type of MP files
    with h5py.File(filepath, "w") as f:
        if mp_version == 3:
            f["movie/frame"] = movie.astype(np.uint16)
            if metadata:
                for key, value in metadata.items():
                    f[key] = value
            else:
                if binning_time is not None:
                    f["movie/configuration/Devices/AcqCam/SoftwareFrameBinning"] = binning_time
                if binning_pixel is not None:
                    f["movie/configuration/Devices/AcqCam/SoftwarePixelBinning"] = binning_pixel
        else:
            f["frame"] = movie.astype(np.uint16)
            if metadata:
                for key, value in metadata.items():
                    f[key] = value
            else:
                if binning_time is not None:
                    f["configuration/Engines/AcqMovieEngine/FrameBinning"] = binning_time
                if binning_pixel is not None:
                    f["configuration/Engines/AcqMovieEngine/PixelBinning"] = binning_pixel


def save_tiff(
    filepath: os.PathLike,
    movie: np.ndarray,
    binning_time: int | None = None,
    binning_pixel: int | None = None,
    metadata: dict = None,
    mp_version: int = 0,
    dtype: str | np.dtype = "uint16",
) -> None:
    """Save a movie as a multi-page TIFF file.

    Parameters
    ----------
    filepath
        Output file path (usually ``.tif`` / ``.tiff``).
    movie
        Movie array with shape ``(n_frames, height, width)``.
    metadata
        Optional scalar metadata to embed into the TIFF (non-scalar entries are
        ignored).
    dtype
        Data type used for saving. Default is ``uint16``.

    Notes
    -----
    The ``binning_time``, ``binning_pixel`` and ``mp_version`` arguments are
    accepted for API compatibility with :func:`save_mp` but are not used.
    """
    if not has_tifffile:
        raise ImportError("tifffile is not installed. Please install it to write TIFF files.")
    if metadata:
        _metadata = {
            x: (y.decode("utf-8") if type(y) is bytes else y.item() if hasattr(y, "item") else y)
            for x, y in metadata.items()
            if np.isscalar(y)
        }
    else:
        _metadata = None
    tifffile.imwrite(filepath, np.array(movie, dtype=dtype), metadata=_metadata)
