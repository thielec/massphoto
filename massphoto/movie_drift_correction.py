"""
Drift correction utilities.

The main entry point is :func:`correct_drift`, which estimates (or applies) a
per-frame drift trace and returns a drift-corrected movie.
"""

# %% imports
import numpy as np
import scipy as sp


# %% functions
def get_shift_rfft(
    movie: np.ndarray | None = None,
    template: np.ndarray | None = None,
    movie_rfft: np.ndarray | None = None,
    template_rfft: np.ndarray | None = None,
    thumbnail_size: int = 7,
) -> np.ndarray:
    """
    Estimate a per-frame (y, x) drift trace by FFT cross-correlation.

    Provide either ``movie`` or ``movie_rfft``. If ``template`` is not given,
    the first frame of the movie is used as the template.

    Parameters
    ----------
    movie : np.ndarray | None
        Movie array of shape ``(frames, y, x)``.
    template : np.ndarray | None
        Template image of shape ``(y, x)``.
    movie_rfft : np.ndarray | None
        Pre-computed ``rfft2`` of ``movie``.
    template_rfft : np.ndarray | None
        Pre-computed ``rfft2`` of ``template``.
    thumbnail_size : int
        Thumbnail size (pixels) around the correlation peak used for the phasor
        sub-pixel estimate.

    Returns
    -------
    np.ndarray
        Drift trace of shape ``(frames, 2)`` as ``(y, x)`` shifts in pixels.
    """

    if movie is None and movie_rfft is None:
        raise ValueError("Either movie or movie_rfft must be provided")
    if movie_rfft is None:
        movie_rfft = sp.fft.rfft2(movie)

    if template_rfft is None:
        if template is None:
            template_rfft = movie_rfft[0]
        else:
            template_rfft = sp.fft.rfft2(template)

    movie_cc = sp.fft.fftshift(np.real(sp.fft.irfft2(movie_rfft * np.conj(template_rfft))), axes=(-2, -1))

    # get drift by finding the maximum of the CC with phasor method
    # based on https://pubs.aip.org/aip/jcp/article/148/12/123311/1023886/Phasor-based-single-molecule-localization

    thumbnail = movie_cc[
        :,
        int(movie_cc.shape[1] / 2 - thumbnail_size / 2 + 0.5) : int(movie_cc.shape[1] / 2 + thumbnail_size / 2 + 0.5),
        int(movie_cc.shape[2] / 2 - thumbnail_size / 2 + 0.5) : int(movie_cc.shape[2] / 2 + thumbnail_size / 2 + 0.5),
    ]
    del movie_cc

    thumb_rfft = np.fft.rfft2(thumbnail)
    thumb_size = np.array(thumbnail.shape[-2:])
    # get the complex angle for the first fourier component in y and x
    angles = np.angle(thumb_rfft[..., (1, 0), (0, 1)])
    # shift to -2pi to 0
    angles = (angles % (2 * np.pi)) - 2 * np.pi
    # convert to position in the image
    pos = np.abs(angles) / (2 * np.pi / thumb_size) - (thumb_size - 1) / 2

    return pos


def correct_drift(
    movie: np.ndarray,
    template: np.ndarray | None = None,
    drift_yx: np.ndarray | None = None,
    thumbnail_size: int = 7,
    pad: int = 0,
    return_drift: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Correct drift in a movie by applying a per-frame translation.

    If ``drift_yx`` is not provided, it is estimated with :func:`get_shift_rfft`.

    Parameters
    ----------
    movie : np.ndarray
        Movie array of shape ``(frames, y, x)``.
    template : np.ndarray | None
        Template image used to estimate drift (ignored when ``drift_yx`` is given).
    drift_yx : np.ndarray | None
        Pre-computed drift trace of shape ``(frames, 2)`` as ``(y, x)`` shifts.
    thumbnail_size : int
        Thumbnail size used by :func:`get_shift_rfft`.
    pad : int
        Symmetric padding (pixels) applied before correction to reduce edge artefacts.
    return_drift : bool
        If True, also return the drift trace used.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray]
        Drift-corrected movie, and optionally the drift trace.

    See Also
    --------
    massphoto.detection.detect
    massphoto.fitting.fit
    """

    if pad:
        movie = np.pad(movie, ((0, 0), (pad, pad), (pad, pad)), mode="symmetric")
        if template is not None:
            template = np.pad(template, ((pad, pad), (pad, pad)), mode="symmetric")
    movie_rfft = sp.fft.rfft2(movie)

    if drift_yx is None:
        drift_yx = get_shift_rfft(movie_rfft=movie_rfft, template=template, thumbnail_size=thumbnail_size)
    if drift_yx.shape[0] != movie.shape[0]:
        raise ValueError(
            f"Drift trace must have the same length as the movie, got {drift_yx.shape[0]} and {movie.shape[0]}"
        )

    # apply drift correction
    # adapted from https://github.com/vortex-exoplanet/VIP (frame_shift)
    shift = -drift_yx
    ramp_x = sp.fft.rfftfreq(movie.shape[-1])[None, :]
    ramp_y = sp.fft.fftfreq(movie.shape[-2])[:, None]
    tilt = (-2 * np.pi) * (shift[:, 1, None, None] * ramp_x + shift[:, 0, None, None] * ramp_y)
    fact = np.cos(tilt) + 1j * np.sin(tilt)

    # Important: specify output shape so odd-sized images preserve dimensions.
    # Without `s=...`, irfft2 assumes an even last axis length (2*(n-1)).
    movie_corrected = np.real(sp.fft.irfft2(movie_rfft * fact, s=movie.shape[-2:]))
    if pad:
        movie_corrected = movie_corrected[:, pad:-pad, pad:-pad]
    if return_drift:
        return movie_corrected, drift_yx
    else:
        return movie_corrected


def get_drift_metrics(drift_yx: np.ndarray, fps: float = 1.0) -> dict[str, np.ndarray]:
    """Calculate drift displacement and instantaneous speed from a drift trace.

    Parameters
    ----------
    drift_yx : np.ndarray
        Drift trace of shape ``(frames, 2)`` as ``(y, x)`` shifts in pixels.
    fps : float, default=1.0
        Frame rate multiplier. When 1.0 (default), drift_speed is in pixels/frame.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:
        - ``"drift_y"``: Per-frame y displacement in pixels.
        - ``"drift_x"``: Per-frame x displacement in pixels.
        - ``"drift_speed"``: Instantaneous displacement speed in pixels/frame (or pixels/s if fps given).
    """
    drift_arr = np.asarray(drift_yx, dtype=float)
    if drift_arr.ndim != 2 or drift_arr.shape[1] != 2:
        raise ValueError(f"drift_yx must have shape (frames, 2), got {drift_arr.shape}")

    drift_y = drift_arr[:, 0]
    drift_x = drift_arr[:, 1]

    if len(drift_arr) == 0:
        return {
            "drift_y": np.empty(0, dtype=np.float32),
            "drift_x": np.empty(0, dtype=np.float32),
            "drift_speed": np.empty(0, dtype=np.float32),
        }

    # Step differences with 0 at frame 0
    dy = np.diff(drift_y, prepend=drift_y[0])
    dx = np.diff(drift_x, prepend=drift_x[0])
    step_dist = np.hypot(dy, dx)
    drift_speed = (step_dist * fps).astype(np.float32)

    return {
        "drift_y": drift_y.astype(np.float32),
        "drift_x": drift_x.astype(np.float32),
        "drift_speed": drift_speed,
    }
