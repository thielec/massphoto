import os
import warnings
from collections.abc import Callable, Sequence

import dask.array as da
import numpy as np
import pandas as pd
import scipy as sp

from .filter.correlation_map import correlation_map, correlation_map_kern_norm
from .filter.temporal_ncc import step_correlation
from .movie_process import calc_ratiometric
from .pipeline import _apply_analysis_record, process
from .utils import NativeMovie, RatiometricMovie, defaults, numba_njit

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


@numba_njit(cache=True)
def _thumbnail_max_numba(
    maps: np.ndarray,
    positions: np.ndarray,
    temp_radius: int,
    spat_radius: int,
) -> np.ndarray:
    """Return local neighborhood maxima for detected positions.

    This explicit scan is equivalent to the simpler but slower pattern::

        np.max(map_[frame0:frame1, y0:y1, x0:x1])

    For these small neighborhoods, the nested loops benchmark faster under
    numba than repeatedly slicing thumbnails and calling ``np.max``.
    """
    n_maps = maps.shape[0]
    n_pos = positions.shape[0]
    out = np.empty((n_maps, n_pos), dtype=maps.dtype)
    for i in range(n_maps):
        map_ = maps[i]
        n_frames, n_y, n_x = map_.shape
        for j in range(n_pos):
            frame, y, x = positions[j]
            frame0 = max(0, frame - temp_radius)
            frame1 = min(n_frames, frame + temp_radius + 1)
            y0 = max(0, y - spat_radius)
            y1 = min(n_y, y + spat_radius + 1)
            x0 = max(0, x - spat_radius)
            x1 = min(n_x, x + spat_radius + 1)
            max_val = map_[frame0, y0, x0]
            for frame_idx in range(frame0, frame1):
                for y_idx in range(y0, y1):
                    for x_idx in range(x0, x1):
                        max_val = max(max_val, map_[frame_idx, y_idx, x_idx])
            out[i, j] = max_val
    return out


@defaults.sync_apply_defaults()
def detect_on_map(
    maps: list | np.ndarray,
    thresholds: Sequence | np.ndarray = (0.0,),
    threshold_absolute: bool | list[bool] = False,
    position_map: int = 0,
    contrast_map: int | None = None,
    threshold_distance_temp: int = 1,
    threshold_distance_spat: int = 3,
    min_distance_temp: int = 10,
    min_distance_spat: int = 5,
    output_map_values: bool = False,
) -> pd.DataFrame:  # disable=R0913
    """
    Detects peaks in a multi-dimensional array of maps.

    Parameters
    ----------
    maps : list or np.ndarray
        The multi-dimensional array/list of maps [map1, map2, ...].
    thresholds : list or np.ndarray, optional
        The thresholds for peak detection. It can be a single value or a list of
        the same length as the number of maps. Defaults to 0.0.
    threshold_absolute : bool, optional
        If True, the absolute value of the maps will be used for thresholding.
        Defaults to False.
    position_map : int, optional
        The index of the map to use for peak position determination. Defaults to 0
        (first map).
    contrast_map : int or None, optional
        The index of the map to use for peak contrast determination. If None,
        no contrast estimate will be returned. Defaults to None.
    threshold_distance_temp : int, optional
        Temporal extent of the area in which the map has to peak above
        the threshold. Defaults to 1.
    threshold_distance_spat : int, optional
        Spatial extent of the area in which the map has to peak above
        the threshold. Defaults to 3.
    min_distance_temp : int, optional
        The minimum distance between peaks in the temporal dimension. Defaults to 10.
    min_distance_spat : int, optional
        The minimum distance between peaks in the spatial dimensions. Defaults to 5.
    output_map_values : bool, optional
        If True, include ``map_i`` columns in the output DataFrame. Each value is
        the local maximum of map ``i`` within the threshold-distance neighborhood
        around the detected event, i.e. the largest value in the support region
        that would still have detected that event. Defaults to False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the peak positions and optionally the map values.
    """
    if isinstance(maps, np.ndarray):
        if maps.shape[0] > len(thresholds):
            # single map? -> add leading singleton dimension
            maps = maps[np.newaxis]
    else:
        maps = np.asanyarray(maps)
    if len(thresholds) not in (1, maps.shape[0]):
        raise ValueError(
            f"Thresholds must be a single value or match number of maps ({maps.shape[0]}), got {len(thresholds)}"
        )

    if not maps.size:
        dtypes = [("frame", np.int64), ("y", np.int64), ("x", np.int64)]
        if contrast_map is not None:
            dtypes.append(("c", np.float64))
        if output_map_values:
            for i, map_ in enumerate(maps):
                dtypes.append((f"map_{i}", map_.dtype))
        return pd.DataFrame(np.empty(0, dtype=dtypes))

    if contrast_map is not None:
        # save the contrast map before calculating the absolute value
        contrast_map = maps[contrast_map]

    maps, detection_map = _get_detection_map(
        maps,
        thresholds,
        threshold_absolute,
        position_map,
        threshold_distance_temp,
        threshold_distance_spat,
        min_distance_temp,
        min_distance_spat,
    )

    positions = np.transpose(np.nonzero(detection_map))
    positions_df = pd.DataFrame(positions, columns=["frame", "y", "x"])
    if contrast_map is not None:
        positions_df["c"] = contrast_map[positions[:, 0], positions[:, 1], positions[:, 2]][:, None]
    if output_map_values:
        temp_radius = max(0, threshold_distance_temp // 2)
        spat_radius = max(0, threshold_distance_spat // 2)
        values_arr = _thumbnail_max_numba(maps, positions, temp_radius, spat_radius)
        positions_df = pd.concat(
            [
                positions_df,
                pd.DataFrame({f"map_{i}": values_arr[i] for i in range(maps.shape[0])}),
            ],
            axis=1,
        )
    return positions_df


def _get_detection_map(
    maps,
    thresholds,
    threshold_absolute,
    position_map,
    threshold_distance_temp,
    threshold_distance_spat,
    min_distance_temp,
    min_distance_spat,
):
    """
    Generate a detection map based on the provided maps and thresholds.

    Threshold support is expanded by first thresholding each map and then taking
    a boolean neighborhood maximum (binary dilation). This is equivalent to the
    simpler but slower float-map variant::

        maps = sp.ndimage.maximum_filter(maps, size=(1, t, s, s))
        candidate_mask = np.all(maps > thresholds[:, None, None, None], axis=0)

    The implementation here uses the equivalent boolean form::

        threshold_masks = sp.ndimage.maximum_filter(
            maps > thresholds[:, None, None, None],
            size=(1, t, s, s),
        )
        candidate_mask = np.all(threshold_masks, axis=0)

    Peak selection still uses a local maximum on ``position_map``.

    See :func:`detect_on_map` for parameter explanations.
    """
    thresholds = np.atleast_1d(thresholds)
    if np.isscalar(threshold_absolute):
        maps = np.abs(maps) if threshold_absolute else maps
    else:
        threshold_absolute = np.asarray(threshold_absolute, dtype=bool)
        maps = np.where(threshold_absolute[:, None, None, None], np.abs(maps), maps)

    # keep the original position map for the position determination
    _position_map = maps[position_map]
    if threshold_distance_temp > 1 or threshold_distance_spat > 1:
        threshold_masks = sp.ndimage.maximum_filter(
            maps > thresholds[:, None, None, None],
            size=(
                1,
                threshold_distance_temp,
                threshold_distance_spat,
                threshold_distance_spat,
            ),
        )
    else:
        threshold_masks = maps > thresholds[:, None, None, None]

    min_distance = (min_distance_temp, min_distance_spat, min_distance_spat)

    detection_map = np.bitwise_and(
        np.all(threshold_masks, axis=0),
        _position_map == sp.ndimage.maximum_filter(_position_map, size=min_distance),
    )

    return maps, detection_map


# parallel processing


@defaults.sync_apply_defaults()
def event_correlation(
    movie: RatiometricMovie,
    correlation_args: dict | None = None,
    thresholds: Sequence | np.ndarray = (0.4, 0.001),
    threshold_absolute: bool | list[bool] = True,
    min_distance_temp: int = 10,
    contrast_map: int | None = 1,
    **detection_args: dict,
):
    """
    Detect events in a ratiometric movie using spatial correlation maps.

    This is a convenience wrapper around :func:`~massphoto.filter.correlation_map.correlation_map`
    followed by :func:`detect_on_map`.

    See Also
    --------
    detect_on_map
    detect
    """
    return detect_on_map(
        correlation_map(movie, **(correlation_args or {})),
        thresholds=thresholds,
        threshold_absolute=threshold_absolute,
        min_distance_temp=min_distance_temp,
        contrast_map=contrast_map,
        **detection_args,
    )


@defaults.sync_apply_defaults()
def event_correlation_step(
    movie: NativeMovie,
    movie_r: RatiometricMovie,
    correlation_args: dict | None = None,
    correlation_native_args: dict | bool = False,
    correlation_step_args: dict | bool = False,
    thresholds: Sequence | np.ndarray = (
        0.6,  # step correlation threshold
        0.5,  # normalised spatial correlation threshold
        0.001,  # contrast threshold
    ),
    threshold_absolute: bool | list[bool] = True,
    threshold_distance_spat: int = 3,
    min_distance_temp: int = 10,
    position_map: int = 2,
    contrast_map: int | None = 2,
    **detection_args: dict,
):
    """
    Detect events using a normalised correlation with a step function in the native movie
    and a normalised correlation with a PSF template in the ratiometric movie.

    This function computes three maps and then calls :func:`detect_on_map`:

    0. Step correlation on the (optionally pre-filtered) native movie. (default threshold: 0.6)
    1. Normalised spatial correlation on the ratiometric movie. (default threshold: 0.5)
    2. Raw spatial correlation (contrast) on the ratiometric movie. (default threshold: 0.001)

    Defaults are chosen for typical mass photometry data and can be overridden
    explicitly or via :data:`massphoto.defaults` (this function is decorated with
    :meth:`~massphoto.defaults_store.DefaultsStore.sync_apply_defaults`), e.g.:
    `massphoto.defaults.set('event_correlation_step.thresholds', [0.25, 0.35, 0.0005])`

    Parameters
    ----------
    movie : NativeMovie
        Native movie, used for the step-correlation map.
    movie_r : RatiometricMovie
        Ratiometric movie, used for the spatial correlation maps.
    correlation_args : dict | None, default=None
        Keyword arguments forwarded to :func:`~massphoto.filter.correlation_map.correlation_map`.
    correlation_native_args : dict | bool, default=False
        If truthy, pre-filter ``movie`` with :func:`~massphoto.filter.correlation_map.correlation_map_kern_norm`
        before the step correlation. If ``True``, a shallow copy of ``correlation_args`` is used.
        If a dict, it is used as the kwargs for the native correlation.
    correlation_step_args : dict | bool, default=False
        Keyword arguments forwarded to :func:`~massphoto.filter.temporal_ncc.step_correlation`.
        If ``False``, defaults of that function are used.
    thresholds : list | np.ndarray, default=[0.6, 0.5, 0.001]
        Per-map detection thresholds ``[step_corr, norm_corr, contrast]``.
    threshold_absolute : bool | list[bool], default=True
        Whether to threshold on absolute map values.
    threshold_distance_spat : int, default=3
        Spatial window (pixels) that must exceed the threshold for a detection.
    min_distance_temp : int, default=10
        Minimum temporal separation (frames) between detections.
    position_map : int, default=2
        Map index used to determine peak positions (default uses the raw correlation map).
    contrast_map : int | None, default=2
        Map index used to report contrast estimates in the output (``c`` column).
    **detection_args : dict
        Forwarded to :func:`detect_on_map` (e.g. ``min_distance_spat``,
        ``threshold_distance_temp``, ``output_map_values``).

    Returns
    -------
    pd.DataFrame
        Detected event positions with columns ``frame``, ``y``, ``x`` and
        optionally ``c`` (when ``contrast_map`` is not ``None``).

    See Also
    --------
    detect_on_map
    detect
    """
    if correlation_native_args:
        if correlation_native_args is True:
            correlation_native_args = correlation_args.copy() if correlation_args else {}
        movie = correlation_map_kern_norm(movie, **correlation_native_args)

    tcorr_map = step_correlation(movie, **correlation_step_args if correlation_step_args else {})
    ncorr_map, corr_map = correlation_map(movie_r, **correlation_args if correlation_args else {})
    return detect_on_map(
        (tcorr_map, ncorr_map, corr_map),
        thresholds=thresholds,
        threshold_absolute=threshold_absolute,
        threshold_distance_spat=threshold_distance_spat,
        min_distance_temp=min_distance_temp,
        position_map=position_map,
        contrast_map=contrast_map,
        **detection_args,
    )


@defaults.apply_defaults()
def detect(
    movie: da.Array | np.ndarray | os.PathLike,
    movie_template: np.ndarray | None = None,
    movie_normalize_frame_means: bool = False,
    movie_correct_drift: bool = False,
    return_ratiometric: bool = False,
    ratiometric_args: dict | None = None,
    ratiometric_func: Callable | None = calc_ratiometric,
    detection_func: Callable = event_correlation_step,
    detection_args: dict | None = None,
    chunk_overlap: int = -1,
    chunk_size: int = 1000,
    postprocess: Sequence[Callable[[pd.DataFrame], pd.DataFrame]] = (),
    scheduler: str | None = None,
    progress_bar: bool = True,
    attach_record: bool = True,
):
    """
    Detect events in a movie with chunked (Dask) processing.

    This is a thin wrapper over :func:`~massphoto.pipeline.process` that can
    optionally apply drift correction and ratiometric conversion before calling
    ``detection_func`` on each chunk.

    Parameters
    ----------
    movie : da.Array | np.ndarray | os.PathLike
        Movie array of shape ``(frames, y, x)``, or a path readable by
        :func:`~massphoto.read_movie`.
    movie_template : np.ndarray | None, default=None
        Global native reference image used for frame-mean normalization and
        drift correction. If needed and not supplied, it is computed as the mean
        image of the input movie.
    movie_normalize_frame_means : bool
        If True, scale each native frame so its mean matches the global frame
        mean before drift correction and ratiometric conversion.
    movie_correct_drift : bool
        If True, apply :func:`~massphoto.movie_drift_correction.correct_drift` first.
    return_ratiometric : bool
        If True, also return the ratiometric movie.
    ratiometric_args : dict | None, default=None
        Passed to ``ratiometric_func``.
    ratiometric_func : Callable | None
        Ratiometric conversion (default: :func:`~massphoto.movie_process.calc_ratiometric`).
    detection_func : Callable
        Detection function applied to each chunk (default:
        :func:`event_correlation_step`).
    detection_args : dict | None
        Passed to ``detection_func``.
    chunk_overlap : int, default=-1
        Chunk overlap in frames; ``-1`` chooses a value based on processing args.
    chunk_size : int, default=1000
        Chunk size in frames; default is 1000.
    postprocess : Sequence[Callable[[pd.DataFrame], pd.DataFrame]]
        Post-processing callbacks applied to the final DataFrame.
    scheduler : str | None, default=None
        Dask scheduler name (e.g. ``'threads'``/``'processes'``).
    progress_bar : bool, default=True
        If True, display a tqdm progress bar during Dask computation.
    attach_record : bool, default=True
        If True, attach an ``analysis_record`` dictionary to
        ``DataFrame.attrs`` containing the effective defaults state,
        origin function, and origin kwargs for reproducible replay with
        :func:`massphoto.resolve_record`.

    Returns
    -------
    pd.DataFrame | tuple[pd.DataFrame, np.ndarray]
        Detected event table, and optionally the ratiometric movie.

    See Also
    --------
    massphoto.pipeline.process
    """
    result = process(
        movie,
        steps=[(detection_func, detection_args or {})],
        movie_template=movie_template,
        movie_normalize_frame_means=movie_normalize_frame_means,
        movie_correct_drift=movie_correct_drift,
        return_ratiometric=return_ratiometric,
        ratiometric_args=ratiometric_args,
        ratiometric_func=ratiometric_func,
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size,
        postprocess=postprocess,
        scheduler=scheduler,
        progress_bar=progress_bar,
        attach_record=attach_record,
    )
    return _apply_analysis_record(
        result,
        attach_record=attach_record,
        func=detect,
        local_vars=locals(),
        exclude={"movie"},
    )
