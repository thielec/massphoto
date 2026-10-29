from typing import overload

import bottleneck as bn
import numpy as np
import scipy as sp

from .filter.median_filter import median_filter_sep_nd
from .metadata import update_derived_metadata
from .utils import NativeMovie, RatiometricMovie, defaults


def normalize_frames(movie: np.ndarray, target_mean: float | None = None) -> np.ndarray:
    """Normalize frame means to remove slow illumination drift.

    Each frame is scaled so that its mean equals ``target_mean``. When
    ``target_mean`` is not provided, the target is the global mean across all
    frame means.

    Parameters
    ----------
    movie : np.ndarray
        3D movie array with shape ``(frames, height, width)``.
    target_mean : float | None, default=None
        Mean intensity that each frame should be scaled to.

    Returns
    -------
    np.ndarray
        Normalized movie with the same shape as ``movie``.
    """
    frame_means = movie.mean(axis=(1, 2), keepdims=True)
    if target_mean is None:
        target_mean = float(np.nanmean(frame_means))

    scales = np.ones_like(frame_means, dtype=np.result_type(movie.dtype, float))
    np.divide(target_mean, frame_means, out=scales, where=frame_means != 0)
    return movie * scales


def _get_windowed_means(img: np.ndarray, navg: int, ngap: int) -> tuple[np.ndarray, slice, np.ndarray, np.ndarray]:
    """Return trailing moving means and aligned before/after windows."""
    img_mean = bn.move_mean(img, navg, axis=0)
    valid = slice(navg + (ngap - 1) // 2, img.shape[0] - navg - (ngap // 2))
    after_shift = navg + (ngap // 2)
    before_shift = -((ngap + 1) // 2)
    after = img_mean[slice(valid.start + after_shift, valid.stop + after_shift)]
    before = img_mean[slice(valid.start + before_shift, valid.stop + before_shift)]
    return img_mean, valid, after, before


@defaults.sync_apply_defaults(promote=("navg",))
def calc_ratiometric(
    img: NativeMovie,
    navg: int = 10,
    ngap: int = 1,
    medfilt_kernel_size: int = 15,
    illumination_correction: bool | float = True,
) -> RatiometricMovie:
    """
    Calculate ratiometric movie to reveal contrast changes over time.

    Parameters
    ----------
    img : np.ndarray
        Input image sequence with shape (frames, height, width).

    navg : int
        Number of frames to average for noise reduction.

    ngap : int, default=1
        Temporal offset between signal and reference windows.

    medfilt_kernel_size : int, default=15
        Size of the median filter kernel. 0 disables filtering.

    illumination_correction : bool or float, default=True
        Apply correction for uneven illumination:
        - True: Uses default Gaussian smoothing (sigma=9)
        - float: Uses specified sigma value for Gaussian smoothing

    Returns
    -------
    np.ndarray
        Ratiometric image with the same shape as input.
        First and last navg+ngap/2 frames contain NaN values.
    """
    if img.ndim == 0 or img.shape[0] < navg + ngap:
        return np.full(img.shape, np.nan)
    img_mean, valid, after, before = _get_windowed_means(img, navg, ngap)
    img_rat = np.full(img_mean.shape, np.nan, dtype=img_mean.dtype)
    np.divide(after, before, out=img_rat[valid])
    img_rat[valid] -= 1

    if illumination_correction is not False:  # True or a float
        if illumination_correction is True:
            illumination_correction = 9

        illumination = sp.ndimage.gaussian_filter(np.median(img, axis=0), illumination_correction, axes=(-2, -1))
        img_rat[valid] *= np.sqrt(after / illumination)

    if medfilt_kernel_size > 1:
        img_rat = median_filter_sep_nd(
            img_rat,
            kernel_size=medfilt_kernel_size,
            subtract_bg=True,
        )

    return img_rat


@defaults.sync_apply_defaults(promote=("navg",))
def calc_differential(img: np.ndarray, navg: int, ngap: int = 1) -> np.ndarray:
    """
    Calculate the differential image based on the given parameters.

    Parameters
    ----------
    img : np.ndarray
        The input image as a numpy array.
    navg : int
        The number of frames to average.
    ngap : int, optional
        The offset between windows. Defaults to 1.

    Returns
    -------
    np.ndarray
        The differential image as a numpy array.
    """
    if img.shape[0] < navg + ngap:
        return np.full(img.shape, np.nan)
    img_mean, valid, after, before = _get_windowed_means(img, navg, ngap)
    img_rat = np.full(img_mean.shape, np.nan, dtype=img_mean.dtype)
    img_rat[valid] = after - before
    return img_rat


@defaults.sync_apply_defaults()
def calc_mobileframes(
    frames: np.ndarray,
    local_bg_halfsize: int,
    medfilt_kernel_size: int = 0,
) -> np.ndarray:
    """
    Calculate "mobile frames" by local median background subtraction.

    Parameters
    ----------
    frames : np.ndarray
        Input movie with shape ``(frames, y, x)``.
    local_bg_halfsize : int
        Half-size (in frames) of the moving-median background window.
    medfilt_kernel_size : int, optional
        2D median-filter kernel size (pixels); ``0`` disables.

    Returns
    -------
    np.ndarray
        Mobile frames with the same shape as ``frames``.
    """
    window_size = (2 * local_bg_halfsize) + 1
    if frames.shape[0] < window_size:
        return np.zeros(frames.shape)

    bg_frames = np.roll(bn.move_median(frames, window=(local_bg_halfsize * 2) + 1, axis=0), -local_bg_halfsize, axis=0)
    bg_frames /= np.sum(bg_frames, axis=(1, 2), keepdims=True) + np.finfo(np.float64).eps

    mobile_frames = frames / (np.sum(frames, axis=(1, 2), keepdims=True) + np.finfo(np.float64).eps)

    mobile_frames = mobile_frames / bg_frames - 1.0
    mobile_frames[np.any(np.isnan(mobile_frames), axis=(1, 2))] = 0.0

    if medfilt_kernel_size > 1:
        mobile_frames = median_filter_sep_nd(
            mobile_frames,
            kernel_size=medfilt_kernel_size,
            subtract_bg=True,
        )

    return mobile_frames


@overload
def rebin_movie(
    frames: np.ndarray, pixel_binning: int = 1, frame_binning: int = 1, metadata: None = None
) -> np.ndarray: ...


@overload
def rebin_movie(
    frames: np.ndarray, pixel_binning: int = 1, frame_binning: int = 1, *, metadata: dict
) -> tuple[np.ndarray, dict]: ...


def rebin_movie(
    frames: np.ndarray, pixel_binning: int = 1, frame_binning: int = 1, metadata: dict | None = None
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Rebin a 2D image or 3D movie by averaging blocks.

    Parameters
    ----------
    frames : np.ndarray
        2D array ``(y, x)`` or 3D array ``(frames, y, x)``.
    pixel_binning : int, optional
        Spatial binning factor (applied to y and x).
    frame_binning : int, optional
        Temporal binning factor (3D input only).
    metadata : dict, optional
        Standardized metadata dictionary (as returned by
        :func:`massphoto.metadata.convert_metadata`). If provided, a copied and
        updated metadata dictionary is returned together with the rebinned
        frames.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, dict]
        Rebinned array. Any remainder pixels/frames are trimmed before binning.
        If ``metadata`` is provided, returns ``(rebinned, updated_metadata)``.
    """
    if np.ndim(frames) == 2:
        if frame_binning >= 2:
            raise ValueError(f"frame_binning must be 1 for 2D input, got {frame_binning}")
        Ly, Lx = frames.shape

        end_yidx = -(Ly % pixel_binning)
        end_xidx = -(Lx % pixel_binning)
        rebinned = (
            frames[: end_yidx if end_yidx else None, : end_xidx if end_xidx else None]
            .reshape(int(Ly / pixel_binning), pixel_binning, int(Lx / pixel_binning), pixel_binning)
            .mean(axis=(1, 3))
        )

    elif np.ndim(frames) == 3:
        N, Ly, Lx = frames.shape

        end_fidx = -(N % frame_binning)
        end_yidx = -(Ly % pixel_binning)
        end_xidx = -(Lx % pixel_binning)
        rebinned = (
            frames[: end_fidx if end_fidx else None, : end_yidx if end_yidx else None, : end_xidx if end_xidx else None]
            .reshape(
                int(N / frame_binning),
                frame_binning,
                int(Ly / pixel_binning),
                pixel_binning,
                int(Lx / pixel_binning),
                pixel_binning,
            )
            .mean(axis=(1, 3, 5))
        )
    else:
        raise ValueError("Invalid input shape")

    if metadata is None:
        return rebinned

    updated_metadata = metadata.copy()
    update_derived_metadata(updated_metadata)  # Checks for required keys
    updated_metadata["pixelbinning"] = updated_metadata["pixelbinning"] * pixel_binning
    if rebinned.ndim == 3:
        updated_metadata["framebinning"] = updated_metadata["framebinning"] * frame_binning
    updated_metadata["image_shape"] = tuple(rebinned.shape[-2:])
    update_derived_metadata(updated_metadata)  # Refresh derived fields

    return rebinned, updated_metadata


def calc_frame_sharpness(movie: np.ndarray, medfilt_size: int = 15) -> np.ndarray:
    """Calculate relative sharpness for each frame.

    Sharpness is measured as the standard deviation of the median-filter
    background-subtracted frame divided by the frame mean.

    Parameters
    ----------
    movie : np.ndarray
        Movie array with shape ``(frames, y, x)`` or 2D image ``(y, x)``.
    medfilt_size : int, default=15
        Kernel size for the separable median filter.

    Returns
    -------
    np.ndarray
        1D array of sharpness values per frame.
    """
    arr = np.asarray(movie)
    if arr.ndim == 2:
        arr = arr[None, ...]

    bg_sub = median_filter_sep_nd(arr, kernel_size=medfilt_size, subtract_bg=True)
    stds = np.std(bg_sub, axis=(-2, -1))
    means = np.mean(arr, axis=(-2, -1))
    sharpness = np.zeros_like(stds, dtype=np.float32)
    np.divide(stds, means, out=sharpness, where=means != 0)
    return sharpness
