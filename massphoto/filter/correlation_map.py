from collections.abc import Callable

import numpy as np
import scipy as sp

from ..psf_fitting import PSF2D

# Adapted from skimage.template matching (BSD-3-Clause), with shared work for
# normalized + raw maps and a fast path for depth-1 templates.


def _window_sum_2d(image: np.ndarray, window_shape: tuple[int, ...]) -> np.ndarray:
    window_sum = np.cumsum(image, axis=0)
    window_sum = window_sum[window_shape[0] : -1] - window_sum[: -window_shape[0] - 1]

    window_sum = np.cumsum(window_sum, axis=1)
    window_sum = window_sum[:, window_shape[1] : -1] - window_sum[:, : -window_shape[1] - 1]
    return window_sum


def _window_sum_3d(image: np.ndarray, window_shape: tuple[int, ...]) -> np.ndarray:
    window_sum = _window_sum_2d(image, window_shape)
    window_sum = np.cumsum(window_sum, axis=2)
    window_sum = window_sum[:, :, window_shape[2] : -1] - window_sum[:, :, : -window_shape[2] - 1]
    return window_sum


def _window_sum_spatial_only(image: np.ndarray, window_shape: tuple[int, int]) -> np.ndarray:
    window_sum = np.cumsum(image, axis=1)
    window_sum = window_sum[:, window_shape[0] : -1, :] - window_sum[:, : -window_shape[0] - 1, :]
    window_sum = np.cumsum(window_sum, axis=2)
    window_sum = window_sum[:, :, window_shape[1] : -1] - window_sum[:, :, : -window_shape[1] - 1]
    return window_sum


def _compute_shared_correlation_maps(image: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized and raw correlation maps from shared FFT work.

    Notes
    -----
    This implementation is adapted from
    :func:`skimage.feature.template.match_template` and related helper logic in
    ``skimage/feature/template.py`` (BSD-3-Clause), with modifications to:

    - expose the unnormalized correlation alongside the normalized response,
    - reuse the shared cross-correlation work for both outputs, and
    - specialize the common ``template.shape[0] == 1`` case so spatial
      correlation on movies does not do unnecessary work along the frame axis.
    """
    # 1) Validate dimensions and shape compatibility.
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got {image.ndim}D")
    if image.ndim < template.ndim:
        raise ValueError("Dimensionality of template must be less than or equal to the dimensionality of image.")
    if np.any(np.less(image.shape, template.shape)):
        raise ValueError("Image must be larger than template.")

    # 2) Normalize dtype for numerically stable FFT/statistics operations.
    image_shape = image.shape
    float_dtype = image.dtype if image.dtype in (np.float32, np.float64) else np.float32
    if np.issubdtype(image.dtype, np.floating) and image.dtype not in (np.float32, np.float64):
        float_dtype = np.float64
    image = image.astype(float_dtype, copy=False)

    # 3) Compute raw cross-correlation and matching valid output slices.
    if image.ndim == 3 and template.shape[0] == 1:
        template_eff = template[0]
        image = np.pad(
            image,
            pad_width=(
                (0, 0),
                (template_eff.shape[0], template_eff.shape[0]),
                (template_eff.shape[1], template_eff.shape[1]),
            ),
            mode="constant",
            constant_values=0,
        )
        window_sum_func = _window_sum_spatial_only
        xcorr = sp.signal.fftconvolve(image, template_eff[None, ::-1, ::-1], mode="valid", axes=(-2, -1))[:, 1:-1, 1:-1]
        y0 = (template_eff.shape[0] - 1) // 2
        y1 = y0 + image_shape[1]
        x0 = (template_eff.shape[1] - 1) // 2
        x1 = x0 + image_shape[2]
        slices = (slice(None), slice(y0, y1), slice(x0, x1))
        template_shape = template_eff.shape

    else:
        template_eff = template
        pad_width = tuple((width, width) for width in template_eff.shape)
        image = np.pad(image, pad_width=pad_width, mode="constant", constant_values=0)

        if image.ndim == 2:
            window_sum_func = _window_sum_2d
            xcorr = sp.signal.fftconvolve(image, template_eff[::-1, ::-1], mode="valid")[1:-1, 1:-1]
        else:
            window_sum_func = _window_sum_3d
            xcorr = sp.signal.fftconvolve(image, template_eff[::-1, ::-1, ::-1], mode="valid")[1:-1, 1:-1, 1:-1]

        slices = tuple(
            slice((template_eff.shape[i] - 1) // 2, ((template_eff.shape[i] - 1) // 2) + image_shape[i])
            for i in range(template_eff.ndim)
        )
        template_shape = template_eff.shape

    # 4) Compute per-window sums needed for normalized correlation.
    image_window_sum = window_sum_func(image, template_shape)
    image_window_sum2 = window_sum_func(np.square(image), template_shape)

    template_mean = template_eff.mean()
    template_volume = template_eff.size
    template_ssd = np.sum((template_eff - template_mean) ** 2)

    # 5) Build normalized response using the standard NCC numerator/denominator.
    numerator = xcorr - image_window_sum * template_mean
    denominator = image_window_sum2
    np.multiply(image_window_sum, image_window_sum, out=image_window_sum)
    np.divide(image_window_sum, template_volume, out=image_window_sum)
    denominator -= image_window_sum
    denominator *= template_ssd
    np.maximum(denominator, 0, out=denominator)
    np.sqrt(denominator, out=denominator)

    response = np.zeros_like(xcorr, dtype=float_dtype)
    mask = denominator > np.finfo(float_dtype).eps
    response[mask] = numerator[mask] / denominator[mask]

    # 6) Also return raw correlation normalized only by kernel energy.
    corr_scale = np.sum(template_eff**2)
    return response[slices], xcorr[slices] / corr_scale


def correlation_map(
    movie: np.ndarray, kern: np.ndarray | Callable | None = None, kern_params: dict | None = None, kern_hsize: int = 6
):
    """Calculate spatial correlation maps between a movie and a 2D PSF template.

    Parameters
    ----------
    movie : np.ndarray
        Input 2D image or 3D movie array ``(frames, y, x)``.
    kern : np.ndarray | Callable | None, default=None
        2D PSF kernel array or a callable generating the kernel. If None, the default
        empirical PSF model is used.
    kern_params : dict | None, default=None
        Parameters passed to ``kern`` if it is callable.
    kern_hsize : int, default=6
        Half-size (in pixels) of the kernel box when generated from a callable.
        The full kernel size is ``2 * kern_hsize + 1``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of ``(norm_corr_map, corr_map)``:
        - ``norm_corr_map``: Normalized cross-correlation map (Pearson correlation in each local window).
        - ``corr_map``: Unnormalized cross-correlation map scaled by the kernel energy.

        Both arrays preserve NaN locations from the input ``movie``. Frames containing
        only invalid values remain NaN.

    Raises
    ------
    ValueError
        If ``kern`` is neither a NumPy array, a callable, nor None.
    """
    # TODO: extend to 3D templates (triangle PSF)
    psf = _get_kern(kern, kern_params, kern_hsize)

    if not movie.size:
        return np.zeros(movie.shape), np.zeros(movie.shape)

    valid_frame = np.isfinite(movie).any(axis=(1, 2))
    ncorr_map = np.full(movie.shape, np.nan, dtype=float)
    corr_map = np.full(movie.shape, np.nan, dtype=float)
    psf = psf if psf.ndim == 3 else psf[None, ...]
    ncorr_map[valid_frame], corr_map[valid_frame] = _compute_shared_correlation_maps(movie[valid_frame], psf)
    return ncorr_map, corr_map


def _get_kern(kern: np.ndarray | Callable | None, kern_params: dict | None, kern_hsize: int) -> np.ndarray:
    """Get the kernel from either a callable or a numpy array."""
    if isinstance(kern, np.ndarray):
        return kern.astype(float)
    if kern is None or callable(kern):
        return PSF2D(kern, kern_params, psf_size=kern_hsize * 2 + 1).get_thumbnail()
    raise ValueError("kern must be either a callable or a numpy array")


def correlation_map_kern_norm(
    movie: np.ndarray, kern: np.ndarray | Callable | None = None, kern_params: dict | None = None, kern_hsize: int = 6
):
    """
    Calculate the correlation map normalized by the sum of the kernel squared.
    See `correlation_map` for more information.

    Returns:
        np.ndarray: The correlation map normalized by the sum of the kernel squared.
    """
    psf = _get_kern(kern, kern_params, kern_hsize)

    if not movie.size:
        return np.zeros(movie.shape)

    valid_frame = np.isfinite(movie).any(axis=(1, 2))

    corr_map = np.full(movie.shape, np.nan, dtype=float)
    psf = psf if psf.ndim == 3 else psf[None, ...]
    corr_map[valid_frame] = _compute_shared_correlation_maps(movie[valid_frame], psf)[1]
    return corr_map


def correlation_map_norm(
    movie: np.ndarray, kern: np.ndarray | Callable | None = None, kern_params: dict | None = None, kern_hsize: int = 6
):
    """
    Calculate the normalized correlation map.
    See `correlation_map` for more information.

    Returns:
        np.ndarray: The normalized correlation map.
    """
    psf = _get_kern(kern, kern_params, kern_hsize)

    if not movie.size:
        return np.zeros(movie.shape)

    valid_frame = np.isfinite(movie).any(axis=(1, 2))

    ncorr_map = np.full(movie.shape, np.nan, dtype=float)
    psf = psf if psf.ndim == 3 else psf[None, ...]
    ncorr_map[valid_frame] = _compute_shared_correlation_maps(movie[valid_frame], psf)[0]
    return ncorr_map
