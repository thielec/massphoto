import os
from collections.abc import Callable, Sequence

import dask.array as da
import numpy as np
import pandas as pd
import scipy as sp

from .detection import event_correlation_step
from .mass_calibration import apply
from .movie_process import calc_ratiometric
from .pipeline import _apply_analysis_record, process
from .psf_fitting import PSF2D
from .utils import RatiometricMovie, defaults, numba_njit


def get_psf_thumbnails(
    processed_movie: np.ndarray,
    events: np.ndarray | pd.DataFrame,
    thumbnail_size: int = 11,
    thumbnail_frames: int = 0,
    exclude_nan: bool = False,
    return_df: bool = False,
) -> tuple[np.ndarray, pd.DataFrame] | np.ndarray:
    """
    Extracts valid point spread function (PSF) thumbnails from a processed movie based on the provided coordinates.

    Parameters:
    -----------
    processed_movie : np.ndarray
        A 3D numpy array representing the processed movie with dimensions (frames, height, width).
    events : pd.DataFrame | np.ndarray
        DataFrame containing candidate event positions with columns 'frame', 'x', and 'y'.
    thumbnail_size : int, optional
        Size of the square thumbnail. Default is 11.
    thumbnail_frames : int, optional
        Number of frames to extract for each thumbnail. Default is 0.
        For 0, a 2D thumbnail is returned. For values > 0, a 3D thumbnail is returned.
    exclude_nan : bool, optional
        Exclude thumbnails containing NaN values. NaNs only occurs when thumbnail_frames > 2*navg+ngap. Default is False.
    return_df : bool, optional
        Whether to return a DataFrame of valid thumbnail positions. Default is False.

    Returns:
    --------
    np.ndarray
        A 4D numpy array of extracted thumbnails with dimensions (number of valid thumbnails, thumbnail_size, thumbnail_size).
        This is generally only 3D however as we only provide a single frame for index 0, but passing a 4D slice selection would work.
    """

    # prepare the tyx array
    if isinstance(events, pd.DataFrame):
        # dataframe or similar
        tyx = events[["frame", "y", "x"]].to_numpy()
    else:
        # numpy array
        tyx = events[:, 0:3]
    if tyx.size == 0:
        return np.array([]), events if return_df else np.array([])

    if np.issubdtype(tyx.dtype, np.floating):
        tyx = np.rint(tyx).astype(int)

    # call the numba function
    thumbnails, ind_valid = _get_psf_thumbnails(
        processed_movie,
        tyx,
        thumbnail_size=thumbnail_size,
        thumbnail_frames=thumbnail_frames,
        exclude_nan=exclude_nan,
    )
    if thumbnail_frames == 0:
        # remove the frame dimension
        thumbnails = thumbnails[:, 0, :, :]

    if return_df:
        return thumbnails, events[ind_valid]
    return thumbnails


@numba_njit(nogil=True)
def _get_psf_thumbnails(
    movie,
    tyx,
    thumbnail_size: int = 11,
    thumbnail_frames: int = 1,
    exclude_nan: bool = False,
):
    """
    Internal function to extract valid thumbnails from a movie based on the provided coordinates.
    Always returns a 4D array of thumbnails [n, t, y, x] and a boolean array of valid indices.
    See get_psf_thumbnails_numba for the public function.
    """
    half_size = thumbnail_size // 2

    thumbnail_frames = max(1, thumbnail_frames)
    t_range = (thumbnail_frames // 2, -(-thumbnail_frames // 2))  # floor, ceil division

    ind_valid = (
        (tyx[:, 0] >= t_range[0])
        & (tyx[:, 0] <= movie.shape[0] - t_range[1])
        & (tyx[:, 1] - half_size >= 0)
        & (tyx[:, 1] + half_size < movie.shape[1])
        & (tyx[:, 2] - half_size >= 0)
        & (tyx[:, 2] + half_size < movie.shape[2])
    )

    thumbnails = np.zeros(
        (np.sum(ind_valid), thumbnail_frames, thumbnail_size, thumbnail_size),
        dtype=movie.dtype,
    )
    if exclude_nan:
        ind_not_nan = np.full(thumbnails.shape[0], True)

    for i, k in enumerate(np.nonzero(ind_valid)[0]):
        t, y, x = tyx[k]
        thumbnails[i] = movie[
            t - t_range[0] : t + t_range[1],
            y - half_size : y + half_size + 1,
            x - half_size : x + half_size + 1,
        ]
        if exclude_nan:
            ind_not_nan[i] = ~np.any(np.isnan(thumbnails[i]))

    if exclude_nan:
        ind_valid[ind_valid] = ind_not_nan
        thumbnails = thumbnails[ind_not_nan]

    return thumbnails, ind_valid


# needs a wrapper to handle dataframe input/output
def fit_psf_thumbnails_numba(movie, tyx, fitfun, thumbnail_size=11):
    """
    Low-level thumbnail fitter helper.

    Extracts thumbnails at ``tyx`` and runs ``fitfun`` on each thumbnail.
    Most users should call :func:`fit_event` or :func:`fit` instead.
    """
    half_size = thumbnail_size // 2

    ind_valid = np.bitwise_and(
        np.bitwise_and(
            np.bitwise_and(tyx[:, 0] >= 0, tyx[:, 0] < movie.shape[0]),
            np.bitwise_and(
                np.bitwise_and(tyx[:, 1] - half_size >= 0, tyx[:, 1] + half_size < movie.shape[1]),
                np.bitwise_and(tyx[:, 2] - half_size >= 0, tyx[:, 2] + half_size < movie.shape[2]),
            ),
        )
    )

    thumbnails = np.zeros((np.sum(ind_valid), thumbnail_size, thumbnail_size), dtype=movie.dtype)
    _contrast = np.zeros(np.sum(ind_valid), dtype=np.float64)
    _x = np.zeros(np.sum(ind_valid), dtype=np.float64)
    _y = np.zeros(np.sum(ind_valid), dtype=np.float64)
    _res = np.zeros(np.sum(ind_valid), dtype=np.float64)
    _r2 = np.zeros(np.sum(ind_valid), dtype=np.float64)
    _contrast_se = np.zeros(np.sum(ind_valid), dtype=np.float64)
    for i, k in enumerate(np.nonzero(ind_valid)[0]):
        t, y, x = tyx[k]
        thumbnails[i] = movie[t, y - half_size : y + half_size + 1, x - half_size : x + half_size + 1]
        _contrast[i], _y[i], _x[i], _res[i], _r2[i], _contrast_se[i] = fitfun(thumbnails[i], max_shift=3)
    return thumbnails, ind_valid, _contrast, _x, _y, _res, _r2, _contrast_se


@defaults.sync_apply_defaults(promote=("thumbnail_size",), exclude=("psf_parameters", "psf_experimental"))
def fit_event(
    movie: RatiometricMovie,
    events: pd.DataFrame,
    thumbnail_size: int = 11,
    psf_parameters: dict | None = None,
    psf_experimental: np.ndarray | None = None,
    method: str = "gradient-descent",
    max_shift: int = 3,
    keep_invalid: bool = False,
) -> pd.DataFrame:
    """
    Fit PSFs to detected events for precise position and contrast determination.

    Parameters
    ----------
    movie : np.ndarray
        Input movie array with shape (frames, y, x).

    events : pd.DataFrame
        DataFrame with columns 'frame', 'y', 'x' and optionally 'c'.

    thumbnail_size : int, default=11
        Size of square region around each event for fitting.

    psf_parameters : dict | None, default=None
        Parameters for the theoretical PSF model.
        If None, uses defaults from DefaultsStore.

    psf_experimental : np.ndarray | None, default=None
        Experimentally determined PSF. Takes precedence over theoretical models.

    method : str, default="gradient-descent"
        Fitting method. Options:

        - "gradient-descent": Optimized and numba-complied code for PSF fitting
        - Other scipy.optimize.minimize methods (e.g., "Nelder-Mead")

    max_shift : int, default=3
        Maximum allowed shift from initial position in pixels. Passed to fitting function but not enforced.

    keep_invalid : bool, default=False
        If True, the output DataFrame has the same length and index as ``events``.
        Rows whose thumbnails fall outside the movie bounds are retained with NaN in all fit columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with fitted parameters as columns:

        - frame: Original frame number
        - x, y: Refined event positions
        - contrasts: Fitted contrast values
        - x_det, y_det: Original detection positions
        - x_fit, y_fit: Position refinement offsets
        - contrasts_det: Original detection contrasts (if provided)
        - contrasts_se: Standard error of contrast (gradient-descent only)
        - r2_fit: R-squared of fit (gradient-descent only)
        - res_fit: Residual of fit (gradient-descent only)
    """

    psf = PSF2D(psf_params=psf_parameters, psf_experimental=psf_experimental, psf_size=thumbnail_size)

    # Reset to a positional index to ensure iloc-based assignment is safe,
    # regardless of whether events carries duplicate or non-default labels.
    original_index = events.index
    events = events.reset_index(drop=True)

    # Drop any previously computed fit columns to allow refitting without column conflicts
    _fit_cols = {"x_det", "y_det", "contrasts_det", "contrasts", "x_fit", "y_fit", "contrasts_se", "r2_fit", "res_fit"}
    events = events.drop(columns=[c for c in _fit_cols if c in events.columns])

    thumbnails, df_valid = get_psf_thumbnails(movie, events, thumbnail_size=thumbnail_size, return_df=True)

    contrasts = []
    x = []
    y = []
    if method == "gradient-descent":
        ffun = psf.fitfun
        # The compiled gradient-descent fitter has a float64 image/scalar
        # signature, while input movies commonly arrive as float32.
        thumbnails = thumbnails.astype(np.float64, copy=False)
        max_shift = float(max_shift)
        n = len(thumbnails)
        contrasts = np.zeros(n)
        x = np.zeros(n)
        y = np.zeros(n)
        contrasts_se = np.zeros(n)
        r2 = np.zeros(n)
        res = np.zeros(n)
        for i, thumbnail in enumerate(thumbnails):
            _contrast, _x, _y, _res, _r2, _contrast_se = ffun(thumbnail, max_shift)
            contrasts[i] = _contrast
            x[i] = _x
            y[i] = _y
            contrasts_se[i] = _contrast_se
            r2[i] = _r2
            res[i] = _res

        fit_res = {
            "contrasts": contrasts,
            "x_fit": x,
            "y_fit": y,
            "contrasts_se": contrasts_se,
            "r2_fit": r2,
            "res_fit": res,
        }
    else:
        lfun = psf.lookupfun
        c_guess = df_valid["c"].to_list() if "c" in df_valid.columns else np.ones(len(thumbnails))
        for thumbnail, _c in zip(thumbnails, c_guess):
            res = sp.optimize.minimize(
                lambda cxy: np.sum((thumbnail - cxy[0] * lfun(cxy[1:])) ** 2),  # noqa: B023
                np.array([_c, 0, 0]),
                method=method,
                options={"maxiter": 1000, "disp": False},
            )
            contrasts.append(res.x[0])
            x.append(res.x[1])
            y.append(res.x[2])

        fit_res = {"contrasts": np.array(contrasts), "x_fit": np.array(x), "y_fit": np.array(y)}

    df_valid = df_valid.rename(columns={"x": "x_det", "y": "y_det", "c": "contrasts_det"})
    for key, value in fit_res.items():
        df_valid[key] = value
    df_valid["x"] = df_valid["x_det"] + df_valid["x_fit"]
    df_valid["y"] = df_valid["y_det"] + df_valid["y_fit"]

    valid_positions = df_valid.index.to_numpy()

    if not keep_invalid:
        return df_valid.reset_index(drop=True)

    df_out = events.rename(columns={"x": "x_det", "y": "y_det", "c": "contrasts_det"})
    for col in df_valid.columns:
        if col not in df_out.columns:
            df_out[col] = np.nan
    df_out.iloc[valid_positions] = df_valid[df_out.columns].values
    df_out.index = original_index
    return df_out


@defaults.apply_defaults()
def fit(
    movie: da.Array | np.ndarray | os.PathLike,
    events: pd.DataFrame | np.ndarray | list | None = None,
    movie_template: np.ndarray | None = None,
    movie_normalize_frame_means: bool = False,
    movie_correct_drift: bool = False,
    return_ratiometric: bool = False,
    ratiometric_args: dict | None = None,
    ratiometric_func: Callable | None = calc_ratiometric,
    detection_func: Callable = event_correlation_step,
    detection_args: dict | None = None,
    fit_func: Callable = fit_event,
    fit_args: dict | None = None,
    chunk_overlap: int = -1,
    chunk_size: int = 1000,
    postprocess: Sequence[Callable[[pd.DataFrame], pd.DataFrame]] = (apply,),
    scheduler: str | None = None,
    progress_bar: bool = True,
    attach_record: bool = True,
    compute_movie_metrics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """
    Detect events and fit PSFs with chunked (Dask) processing.

    This is the main high-level entry point for most analyses. By default, it runs
    a detection step followed by a fitting step using
    :func:`~massphoto.pipeline.process`. Defaults can be overridden via
    :data:`massphoto.defaults` (see :doc:`/api/defaults`).

    Parameters
    ----------
    movie : da.Array | np.ndarray | os.PathLike
        Movie array of shape ``(frames, y, x)``, or a path readable by
        :func:`~massphoto.read_movie`.
    events : pd.DataFrame | np.ndarray | list | None, default=None
        Precomputed detections/candidates with ``frame``, ``y``, and ``x`` fields.
        When provided, the detection step is skipped and only fitting is run.
    movie_template : np.ndarray | None, default=None
        Global native reference image used for frame-mean normalization and
        drift correction. If needed and not supplied, it is computed as the mean
        image of the input movie.
    movie_normalize_frame_means : bool, default=False
        If True, scale each native frame so its mean matches the global frame
        mean before drift correction, ratiometric conversion, detection, and
        fitting.
    movie_correct_drift : bool, default=False
        If True, apply :func:`~massphoto.movie_drift_correction.correct_drift` before
        ratiometric conversion / detection.
    return_ratiometric : bool, default=False
        If True, also return the ratiometric movie used for detection/fitting.
    ratiometric_args : dict | None, default=None
        Passed to ``ratiometric_func``. Defaults are taken from
        :data:`massphoto.defaults`.
    ratiometric_func : Callable | None, default=:func:`~massphoto.movie_process.calc_ratiometric`
        Function that converts the native movie to a ratiometric representation.
    detection_func : Callable, default=event_correlation_step
        Event detection function. Common choices:
        :func:`~massphoto.detection.event_correlation_step` (default) or
        :func:`~massphoto.detection.event_correlation` (faster).
    detection_args : dict | None, default=None
        Extra kwargs passed to ``detection_func``.
    fit_func : Callable, default=fit_event
        PSF fitting function applied to detections (default: :func:`fit_event`).
    fit_args : dict | None, default=None
        Extra kwargs passed to ``fit_func``.
    chunk_overlap : int, default=-1
        Chunk overlap in frames; ``-1`` lets :func:`~massphoto.pipeline.process`
        infer a suitable value from processing args.
    chunk_size : int, default=1000
        Chunk size in frames; default is 1000.
    postprocess : Sequence[Callable[[pd.DataFrame], pd.DataFrame]]
        Post-processing callbacks applied to the final DataFrame. By default,
        applies :func:`~massphoto.mass_calibration.apply` when a calibration is available.
    scheduler : str | None, default=None
        Dask scheduler name (e.g. ``"threads"`` or ``"processes"``).
    progress_bar : bool, default=True
        If True, display a tqdm progress bar during Dask computation.
    attach_record : bool, default=True
        If True, attach an ``analysis_record`` dictionary to
        ``DataFrame.attrs`` containing the effective defaults state,
        origin function, and origin kwargs for reproducible replay with
        :func:`massphoto.resolve_record`.
    compute_movie_metrics : bool, default=False
        If True, compute per-frame movie metrics (``intensity``, ``sharpness``,
        ``signal``, and drift metrics when ``movie_correct_drift=True``) and attach
        them to ``DataFrame.attrs["frame_metrics"]``.

    Returns
    -------
    pd.DataFrame | tuple[pd.DataFrame, np.ndarray]
        Event table with detection + fit results. If ``return_ratiometric=True``,
        also returns the ratiometric movie.

    Examples
    --------
    Run end-to-end detection + fitting from a file::

        import massphoto

        events = massphoto.fit(
            "path/to/data.mp",
            movie_correct_drift=True,
            ratiometric_args={"navg": 10, "ngap": 1},
            detection_func=massphoto.detection.event_correlation,
            fit_args={"thumbnail_size": 11},
            scheduler="threads",
        )

    See Also
    --------
    massphoto.detection.detect
    fit_event
    massphoto.pipeline.process
    """
    steps = (
        [(fit_func, fit_args or {})]
        if events is not None
        else [(detection_func, detection_args or {}), (fit_func, fit_args or {})]
    )

    result = process(
        movie,
        steps=steps,
        events=events,
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
        compute_movie_metrics=compute_movie_metrics,
    )
    return _apply_analysis_record(
        result,
        attach_record=attach_record,
        func=fit,
        local_vars=locals(),
        exclude={"movie", "events"},
    )
