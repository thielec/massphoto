"""Movie I/O for mass photometry data.

Public entry points are :func:`read_movie`, :func:`read_mp` and :func:`read_tdms`.

- MP/MPR files are HDF5-based and are read via :mod:`h5py` (and :mod:`hdf5plugin`
  for compressed datasets).
- TDMS support is optional and requires :mod:`nptdms`.
- TIFF support is optional and requires :mod:`tifffile`.
"""

import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path

import h5py
import numpy as np

from .metadata import convert_metadata as mp_convert_metadata

logger = logging.getLogger(__name__)

# import tifffile if installed
try:
    import tifffile  # type: ignore

    has_tifffile = True
except ImportError:
    has_tifffile = False

# import nptdms if installed
try:
    import nptdms  # type: ignore

    has_nptdms = True
except ImportError:
    has_nptdms = False

try:
    import hdf5plugin  # type: ignore  # noqa: F401
except ImportError:
    logger.warning("hdf5plugin is not installed. Compressed MP/MPR files may not be readable.")


Reader = Callable[..., tuple[np.ndarray, dict] | tuple[np.ndarray, dict, np.ndarray]]
_READERS: dict[str, Reader] = {}


def register_reader(suffixes: str | Iterable[str], reader: Reader, *, overwrite: bool = False) -> None:
    """Register a reader callable for one or more file extensions."""
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    for suffix in suffixes:
        key = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        if not overwrite and key in _READERS:
            raise ValueError(f"Reader already registered for '{key}'.")
        _READERS[key] = reader


def read_movie(filepath: str | os.PathLike, convert_metadata: bool = False) -> tuple[np.ndarray, dict]:
    """
    Read a movie file and return ``(frames, metadata)``.

    Dispatches based on file extension via the reader registry:

    - ``.mp`` / ``.mpr`` -> :func:`read_mp`
    - ``.tdms`` -> :func:`read_tdms`
    - ``.tif`` / ``.tiff`` -> :func:`read_tiff`

    Parameters
    ----------
    filepath : str or os.PathLike
        Path to the movie file.
    convert_metadata : bool, optional
        If True, convert metadata to a more readable form via
        :func:`massphoto.metadata.convert_metadata`.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(frames, metadata)``.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file type is not supported.

    See Also
    --------
    read_mp
    read_tdms
    read_tiff
    register_reader
    """
    # ensure filepath is a path and exists
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(filepath)
    # select the right import function
    reader = _READERS.get(filepath.suffix.lower())
    if reader is None:
        raise ValueError(f"Invalid file type: {filepath}")
    return reader(filepath, convert_metadata=convert_metadata)


def _hdf5_to_dict(h5file: h5py.File) -> dict:
    """
    Converts an HDF5 file to a flattened dictionary.

    Iterates over all Groups and entries. Flattening the dictionary avoids side-effects when copying the object.

    Parameters
    ----------
    h5file : h5py.File
        The HDF5 file to convert.

    Returns
    -------
    dict
        A dictionary containing the flattened HDF5 file data.
    """
    # Iterate over all Groups and entries in the hdf5 file and output them as a flattened dictionary
    # Flattening the dictionary avoids sideeffect when copying the object.
    d = {}

    def helper(name, obj):
        try:
            if isinstance(obj, h5py.Dataset):
                d[name] = obj[()]
            elif isinstance(obj, h5py.Group):
                for key, val in obj.items():
                    helper(name + "/" + key, val)
        except Exception as e:
            logger.warning("Error accessing %s: %s", name, e)

    for key, val in h5file.items():
        helper(key, val)
    return d


def read_mp(filepath: str | os.PathLike, convert_metadata: bool = False) -> tuple[np.ndarray, dict]:
    """
    Read a DiscoverMP ``.mp``/``.mpr`` (HDF5) file.

    Parameters
    ----------
    filepath : str or os.PathLike
        Path to the MP/MPR file.
    convert_metadata : bool, optional
        If True, convert metadata to a more readable form via
        :func:`massphoto.metadata.convert_metadata`.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(frames, metadata)``.

    Raises
    ------
    ValueError
        If no entry for frames is found in file.

    Notes
    -----
    Some MP files store frames in a compressed representation. This function attempts
    to detect that case and will transparently decompress via
    :func:`_decode_compressed_movie`.
    """
    with h5py.File(filepath, "r") as file:
        metadata = _hdf5_to_dict(file)

    frames = metadata.pop("movie/frame", None)  # for metadata["mp_version_number"] == 3
    if frames is None:
        frames = metadata.pop("frame", None)  # for metadata["mp_version_number"] == 2
    if frames is None:
        raise ValueError(f"No entry for frames found: {filepath}")

    try:
        # checkif the data is compressed
        if np.divide(*np.std(frames[:2], axis=(1, 2))) < 0.5:
            # Check the ratio of the std of the first two frames.
            # The first frame is uncompressed and has the normal std.
            # If the second frame is compressed, it has a much higher std due to unit overflow.
            frames = _decode_compressed_movie(frames, metadata)
    except Exception as exc:
        logger.warning("Compression check failed: %s", exc)
    if convert_metadata:
        metadata = mp_convert_metadata(metadata)

    return frames, metadata


def _decode_compressed_movie(frames: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Decode a delta-encoded MP movie.

    In a compressed movie, the first frame is the original data, and all following
    frames are the difference to the previous frame.

    Parameters
    ----------
    frames : np.ndarray
        The compressed movie data.
    metadata : dict
        The metadata of the compressed movie.

    Returns
    -------
    np.ndarray
        The decompressed movie data.

    Raises
    ------
    ValueError
        If decompression fails when comparing to keyframes.

    Notes
    -----
    This function mutates ``metadata["movie/keyframe"]`` (clears it) when the
    decompression check passes.
    """
    # get the compression parameters
    maxint = np.iinfo(frames.dtype).max

    unwrapped = frames.astype(float) - (frames > maxint // 2) * (float(maxint) + 1)
    unwrapped[0] = frames[0]
    unwrapped = np.cumsum(unwrapped, axis=0)
    # check if correct
    # not reliable, for shorter movies; the keyframe frequency seems to be around 100 frames but I am not sure whether that is guaranteed.
    # keyframe_frequency = np.ceil(frames.shape[0] / metadata["movie/keyframe"].shape[0]).astype(int)
    # if np.allclose(unwrapped[::keyframe_frequency], metadata["movie/keyframe"]):
    # just compare the last keyframe (usually the last frame, but not always, so also check the last 100 frames to be safe)
    if (
        np.allclose(unwrapped[-1], metadata["movie/keyframe"][-1])
        or np.sum(np.all(np.isclose(unwrapped[-100:], metadata["movie/keyframe"][-1]), axis=(-1, -2))) == 1
    ):
        metadata["movie/keyframe"] = np.empty((0, *frames.shape[1:]), dtype=frames.dtype)
        return unwrapped
    else:
        raise ValueError("Decompression failed.")


def read_tdms(
    filepath: str | os.PathLike,
    channel: str = "",
    read_timestamp: bool = False,
    image_size: list | None = None,
    convert_metadata: bool = False,
) -> tuple[np.ndarray, dict] | tuple[np.ndarray, dict, np.ndarray]:
    """
    Read image data from a TDMS file.

    TDMS support is optional and requires :mod:`nptdms`.

    Parameters
    ----------
    filepath : str or os.PathLike
        Path to the TDMS file.
    channel : str, optional
        Channel name to read (post-written TDMS). If empty, the first image
        channel is used.
    read_timestamp : bool, optional
        Whether to read the timestamp data. Default is False.
    image_size : list, optional
        Optional image size override as ``[y, x]`` (post-written TDMS only).
    convert_metadata : bool, optional
        Whether to convert metadata to a more readable format. Default is False.

    Returns
    -------
    tuple
        ``(frames, metadata)`` or ``(frames, metadata, timestamps)`` when
        ``read_timestamp=True``.

    Raises
    ------
    ImportError
        If the nptdms library is not installed.

    See Also
    --------
    read_movie
    """
    if has_nptdms:
        with nptdms.TdmsFile.open(filepath) as f:
            if "Pixels per frame including pad" in f.properties:
                # streamed file, ignores most of the options
                output = _read_tdms_streamed(f, read_timestamp=read_timestamp)
            else:
                # post written file
                output = _read_tdms_post(f, channel=channel, read_timestamp=read_timestamp, image_size=image_size)
    else:
        raise ImportError("nptdms is not installed. Please install it to read TDMS files.")

    if convert_metadata:
        # convert metadata to a more readable format, preserve the timestamp if present
        output = (output[0], mp_convert_metadata(output[1]), *output[2:])
    return output


def read_tiff(filepath: str | os.PathLike, convert_metadata: bool = False) -> tuple[np.ndarray, dict]:
    """
    Read a multi-page TIFF file.

    Parameters
    ----------
    filepath : str or os.PathLike
        Path to the TIFF file.
    convert_metadata : bool, optional
        Whether to convert metadata to a more readable format. Default is False.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(frames, metadata)``.

    Raises
    ------
    ImportError
        If the tifffile library is not installed.

    See Also
    --------
    read_movie
    """
    if not has_tifffile:
        raise ImportError("tifffile is not installed. Please install it to read TIFF files.")

    with tifffile.TiffFile(filepath) as tiff:
        frames = tiff.asarray()
        metadata = {}
        if tiff.pages:
            for tag in tiff.pages[0].tags.values():
                if tag.name == "ImageDescription" and isinstance(tag.value, str):
                    try:
                        import json

                        metadata.update(json.loads(tag.value))
                    except json.JSONDecodeError:
                        metadata[tag.name] = tag.value
                else:
                    metadata[tag.name] = tag.value
        if tiff.imagej_metadata:
            metadata.update(tiff.imagej_metadata)

    if convert_metadata:
        metadata = mp_convert_metadata(metadata)

    return frames, metadata


if has_nptdms:

    def _read_tdms_post(
        file: nptdms.tdms.TdmsFile, channel: str = "", read_timestamp: bool = False, image_size: list | None = None
    ) -> tuple[np.ndarray, dict] | tuple[np.ndarray, dict, np.ndarray]:
        """
        Read a post-written TDMS file (image data stored as a single channel).
        """
        # find the name of group containing the image data
        if "img" in list(file):
            movie_field = "img"
        elif "frames" in list(file):
            movie_field = "frames"
        else:
            raise KeyError("TDMS file does not contain 'img' or 'frames' group.")
        if not len(list(file[movie_field])):
            raise ValueError("TDMS does not contain any image channels")
        if channel in (None, ""):
            # if no channel is specified take the first one
            channel = file[movie_field].channels()[0].name

        data_cam = file[movie_field][channel][:]
        metadata = file[movie_field][channel].properties
        if image_size:
            size_y, size_x = image_size
        else:
            if "Image size" in metadata:
                # assume square image if only one size is given
                size_y = int(metadata["Image size"])
                size_x = int(metadata["Image size 2"]) if "Image size 2" in metadata else size_y
            elif "Pixel size" in metadata:
                size_y = int(metadata["Pixel size"])
                size_x = size_y
            else:
                size_x, size_y = 1, 1
        n = data_cam.size // (size_y * size_x)
        data_cam = data_cam[: n * size_y * size_x].reshape(n, size_y, size_x)
        if read_timestamp:
            time_cam = file["Timestamp"][channel][:]

        if read_timestamp:
            return data_cam, metadata, time_cam
        else:
            return data_cam, metadata

    def _read_tdms_streamed(
        file: nptdms.tdms.TdmsFile, read_timestamp: bool = False
    ) -> tuple[np.ndarray, dict] | tuple[np.ndarray, dict, np.ndarray]:
        """
        Read a streamed TDMS file (frames stored with padding).
        """
        # Prereading
        xdim = int(file.properties["Image size"])
        ydim = int(file.properties["Image size 2"])
        # pre_bin = int(file.properties["Bin frame size"])
        # framerate = file.properties["Frame rate ( before binning )"]
        framesizeincpad = int(file.properties["Pixels per frame including pad"])
        padsize = framesizeincpad - (xdim * ydim)

        # Reading Data
        data1D = file.groups()[0].channels()[0][:]
        metadata = file.properties
        if read_timestamp:
            time_cam = file.groups()[1].channels()[0][:]

        # Arranging into 2D structure
        npixels = len(data1D)
        nframes = int(npixels / framesizeincpad)
        padsize = int(framesizeincpad - (xdim * ydim))
        framemask = np.concatenate((np.repeat(True, int(xdim * ydim), axis=0), np.repeat(False, padsize, axis=0)))
        data1D_sorted_into_frames = np.reshape(data1D, (nframes, framesizeincpad))
        del data1D
        framemask2D = np.tile(framemask, (nframes, 1))
        data1D_sorted_into_frames_pad_removed = data1D_sorted_into_frames[framemask2D]
        del data1D_sorted_into_frames
        data_cam = np.reshape(data1D_sorted_into_frames_pad_removed, (nframes, xdim, ydim))
        del data1D_sorted_into_frames_pad_removed

        if read_timestamp:
            return data_cam, metadata, time_cam
        else:
            return data_cam, metadata


register_reader([".mp", ".mpr"], read_mp)
register_reader(".tdms", read_tdms)
register_reader([".tif", ".tiff"], read_tiff)
