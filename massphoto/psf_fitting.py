"""
This module provides functions fitting a 2D point spread function (PSF) to a thumbnail.
By default, only the position and amplitude (contrast) are fitted. For efficient fitting, the PSF is evaluated on a grid and the fitting is done using a lookup table.

The module is designed, to enable compilation with numba, caching and parallelisation.
"""

import functools
from collections.abc import Callable

import numpy as np
import scipy as sp

# %% define the PSF
# PSF parameters are defined in the defaults file
from .psf import DEFAULT_PSF_PARAMS, psf_2d
from .utils import HAS_NUMBA, defaults, numba_njit

DEFAULT_PSF_SIZE: int = 11
DEFAULT_PSF_CACHE_DIVIDER: int = 128
DEFAULT_PSF_CACHE_INTERPOLATE: bool = True

defaults.update(
    {
        "psf_fun": None,
        "psf_size": DEFAULT_PSF_SIZE,
        "psf_cache_divider": DEFAULT_PSF_CACHE_DIVIDER,
        "psf_cache_interpolate": DEFAULT_PSF_CACHE_INTERPOLATE,
    }
)


# %% Helper functions that reads the defaults if no parameters are given
class PSF2D:
    """
    A class to represent a 2D point spread function (PSF) with custom or default parameters.
    Provides methods to get the PSF thumbnail, fit function, and lookup function.
    Currently just a wrapper around the psf_2d function and get_lookup_fun which has to
    be a static method for compilation, caching, and parallelisation.

    """

    def __init__(
        self,
        psf_fun: Callable | np.ndarray | None = None,
        psf_params: dict | None = None,
        psf_experimental: np.ndarray | None = None,
        psf_size: int | None = None,
        cache_divider: int | None = None,
        cache_interpolate: bool | None = None,
    ):
        """
        Initialize the PSF2D object with a PSF function and parameters.

        Args:
            psf_fun (Callable): The PSF function to use.
            psf_params (dict, optional): Parameters for the PSF function. Defaults to None.
            psf_experimental (np.ndarray, optional): Experimental PSF data. Defaults to None.
            psf_size (int, optional): Size of the PSF. Defaults to None.
            cache_divider (int, optional): Divider for caching the PSF function. None uses the default value from defaults.psf_cache_divider.
        """
        if isinstance(psf_fun, self.__class__):
            # Already a PSF2D object -> copy the parameters
            self.psf_fun = psf_fun.psf_fun
            self.psf_params = psf_fun.psf_params.copy()
            self.psf_experimental = psf_fun.psf_experimental
            self.psf_size = psf_fun.psf_size
            self.psf_cache_divider = cache_divider if cache_divider is not None else psf_fun.psf_cache_divider
            self.psf_cache_interpolate = (
                cache_interpolate if cache_interpolate is not None else psf_fun.psf_cache_interpolate
            )
            return

        self.psf_fun = psf_fun if callable(psf_fun) else defaults.psf_fun
        if self.psf_fun is None:
            self.psf_fun = psf_2d
        if psf_params is not None:
            self.psf_params = psf_params
        elif isinstance(defaults.psf_params, dict):
            self.psf_params = defaults.psf_params.copy()
        else:
            self.psf_params = DEFAULT_PSF_PARAMS.copy()
        if psf_experimental is None:
            self.psf_experimental = psf_fun if isinstance(psf_fun, np.ndarray) else defaults.psf_experimental
        else:
            self.psf_experimental = psf_experimental
        self.psf_size = (
            psf_size
            if psf_size is not None
            else (defaults.psf_size if defaults.psf_size is not None else DEFAULT_PSF_SIZE)
        )
        self.psf_cache_divider = (
            cache_divider
            if cache_divider is not None
            else (defaults.psf_cache_divider if defaults.psf_cache_divider is not None else DEFAULT_PSF_CACHE_DIVIDER)
        )
        self.psf_cache_interpolate = (
            cache_interpolate
            if cache_interpolate is not None
            else (
                defaults.psf_cache_interpolate
                if defaults.psf_cache_interpolate is not None
                else DEFAULT_PSF_CACHE_INTERPOLATE
            )
        )

    def get_thumbnail(self, thumbnail_size: int | None = None) -> np.ndarray:
        # Note: for even sizes, the centre pixel is at thumbnail_size // 2 -1
        thumbnail_size = thumbnail_size or self.psf_size
        if self.psf_experimental is not None:
            if thumbnail_size == self.psf_experimental.shape[0]:
                return self.psf_experimental
            elif thumbnail_size < self.psf_experimental.shape[0]:
                # crop the PSF to the thumbnail size
                start_idx = (self.psf_experimental.shape[0] - thumbnail_size) // 2
                return self.psf_experimental[
                    start_idx : start_idx + thumbnail_size, start_idx : start_idx + thumbnail_size
                ]
            else:
                # pad with zeros
                pad_width = (thumbnail_size - self.psf_experimental.shape[0]) // 2
                pad_extra = (thumbnail_size - self.psf_experimental.shape[0]) % 2
                return np.pad(
                    self.psf_experimental,
                    ((pad_width, pad_width + pad_extra), (pad_width, pad_width + pad_extra)),
                    mode="constant",
                    constant_values=0,
                )

            # else: generate the PSF thumbnail
        thumbnail_half_size = thumbnail_size // 2
        [yy, xx] = np.mgrid[
            thumbnail_half_size - thumbnail_size + 1 : thumbnail_half_size + 1,
            thumbnail_half_size - thumbnail_size + 1 : thumbnail_half_size + 1,
        ]
        if not callable(self.psf_fun):
            raise TypeError(f"psf_fun must be callable, got {type(self.psf_fun).__name__}")
        return self.psf_fun(xx.flatten(), yy.flatten(), **self.psf_params).reshape(xx.shape)

    def get_lookup_and_fitfun(self) -> tuple[Callable, Callable]:
        if self.psf_experimental is not None:
            # Use the experimental PSF
            # turn into a tuple to make it hashable
            psf_experimental = tuple(map(tuple, self.psf_experimental))
            lfun, ffun = get_lookup_fun(
                psf_experimental,
                size=self.psf_size,
                divider=self.psf_cache_divider,
                pad=4,
                interpolate=self.psf_cache_interpolate,
            )
            return lfun, ffun
        else:
            # Use the PSF function
            lfun, ffun = get_lookup_fun(
                self.psf_fun,
                size=self.psf_size,
                divider=self.psf_cache_divider,
                pad=4,
                D4_symmetry=True,
                interpolate=self.psf_cache_interpolate,
                **self.psf_params,
            )
            return lfun, ffun

    @property
    def fitfun(self) -> Callable:
        return self.get_lookup_and_fitfun()[1]

    @property
    def lookupfun(self) -> Callable:
        return self.get_lookup_and_fitfun()[0]

    def get_fitfun(self) -> Callable:
        return self.fitfun

    def get_lookupfun(self) -> Callable:
        return self.lookupfun


# %% functions to estimate the PSF from data
def estimate_psf(
    data: np.ndarray,
    max_freq: float | bool = True,
    image_mask: bool = True,
    image_mask_blur: float = 1.0,
    subtract_image_mean: bool = True,
    subtract_otf_background: bool | float = True,
    psf_size: int | None = None,
    return_otf: bool = False,
    centre_psf: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Estimate the Point Spread Function (PSF) of an image by calculating the Optical Transfer Function (OTF) and applying an inverse Fourier Transform.

    Args:
        data (np.ndarray): The input image data.
        max_freq (float | bool, optional): The radius of the circular mask used to filter the OTF. If True, the maximum frequency is estimated automatically. Defaults to True.
        image_mask (bool, optional): Whether to apply an image mask to the data. Defaults to True.
        image_mask_blur (float, optional): The standard deviation of the Gaussian blur applied to the image mask. Defaults to 1.0.
        subtract_image_mean (bool, optional): Whether to subtract the mean of the data. Defaults to True.
        subtract_otf_background (bool | float, optional): Whether to subtract the background from the OTF. If True, the background is estimated using the mask. If a float, it is used as the background value. Defaults to True.
        psf_size (int | None, optional): The size of the returned PSF. If specified, the PSF will be cropped to this size. Defaults to None.
        return_otf (bool, optional): Whether to return the Optical Transfer Function (OTF) instead of the PSF. Defaults to False.
        centre_psf (bool, optional): Whether to centre the PSF by taking the absolute value of the OTF. Note: this removes phase information and may introduce unwanted symmetry. Defaults to False.

    Returns:
        np.ndarray: The estimated PSF or OTF.
        np.ndarray: (optional) If return_otf is True, the estimated OTF is returned as well.
    """

    if data.ndim == 2:
        data = data[None, :, :]
    # ensure that data has an odd number of pixels in each dimension
    # data = data[:, 1 - data.shape[1] % 2 :, 1 - data.shape[2] % 2 :]

    if subtract_image_mean:
        data = data - np.mean(data, axis=(1, 2), keepdims=True)
    if image_mask:
        mask = np.linspace(-1, 1, data.shape[2])[None, :] ** 2 + np.linspace(-1, 1, data.shape[1])[:, None] ** 2 < 1
    else:
        mask = np.ones(data.shape[1:])
    if image_mask_blur:
        mask = sp.ndimage.binary_erosion(mask, iterations=int(image_mask_blur * 2))
        mask = sp.ndimage.gaussian_filter(mask.astype(float), image_mask_blur)

    data_masked = mask * data[np.all(np.isfinite(data), axis=(1, 2))]
    if centre_psf:
        otf_guess = np.sum(np.abs(np.fft.fft2(data_masked, axes=(1, 2))), 0)
        # TODO: find a better way to centre the PSF. This removes all phase components, instead just removing the first (for centre of mass)
        # could be better. See drift correction code for an example.
    else:
        otf_guess = np.sum(np.fft.fft2(data_masked, axes=(1, 2)), 0)
    if max_freq is None:
        max_freq = 1
    elif isinstance(max_freq, bool) and max_freq is True:
        # estimate the max frequency from the OTF, third argument is the frequency at threshold
        max_freq = 1.1 * frc_profile(np.fft.fftshift(otf_guess), threshold=0.5, min_frequency=0.1)[2]
    mask = np.fft.fftfreq(otf_guess.shape[0])[:, None] ** 2 + np.fft.fftfreq(otf_guess.shape[1]) ** 2 < max_freq**2
    if isinstance(subtract_otf_background, bool) and subtract_otf_background is True:
        if not np.any(mask):
            subtract_otf_background = mask * np.median(otf_guess[1 - mask])
        else:
            subtract_otf_background = mask * np.min(otf_guess)
    otf_masked = mask * otf_guess - subtract_otf_background
    psf_guess = np.real(np.fft.fftshift(np.fft.ifft2(otf_masked)))
    psf_guess /= np.max(psf_guess)
    if psf_size is not None:
        psf_size = int(psf_size)
        start_idx = np.maximum(np.array(psf_guess.shape) // 2 - psf_size / 2 + 1, 0).astype(int)
        psf_guess = psf_guess[start_idx[0] : start_idx[0] + psf_size, start_idx[1] : start_idx[1] + psf_size]
    if return_otf:
        return psf_guess, otf_masked
    return psf_guess


def frc_profile(fft_img: np.ndarray, threshold=0.5, min_frequency=0.1) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate the radial profile of a given FFT image.

    Args:
        fft_img (np.ndarray): The input FFT image. Expects the center of the image to be the center of the FFT.
        threshold (float, optional): The threshold value for determining the resolution. Defaults to 0.5.
        min_frequency (float, optional): The minimum frequency value. Defaults to 0.1.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: A tuple containing the radius bin centers, ring brightness values, and the estimated max_frequency.
    """

    # center = [int(fft_img.shape[0] / 2), int(fft_img.shape[1] / 2)]
    # y, x = np.indices((fft_img.shape))  # first determine radii of all pixels
    # r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = np.fft.fftshift(np.sqrt(np.fft.fftfreq(fft_img.shape[0])[:, None] ** 2 + np.fft.fftfreq(fft_img.shape[1]) ** 2))
    # radius of the image.
    _bins = int(np.min(fft_img.shape) / 2)
    ring_brightness, radius = np.histogram(r[r > 0], weights=np.abs(fft_img[r > 0]), bins=_bins)
    ring_counts, radius = np.histogram(r[r > 0], bins=_bins)
    ring_brightness /= ring_counts + np.finfo(float).eps
    radius_bin_center = (radius[1:] + radius[:-1]) / 2

    if threshold is not None:
        min_frequency = max(min_frequency, radius_bin_center[np.argmax(ring_brightness)])
        radius_fine = np.linspace(min_frequency, radius[-1], 500)
        ring_brightness_fine = sp.interpolate.PchipInterpolator(radius_bin_center, ring_brightness)(radius_fine)
        guess_fine = np.nonzero(
            ring_brightness_fine < ring_brightness[-2] + (ring_brightness[1] - ring_brightness[-2]) * threshold
        )[0]
        max_frequency = radius_fine[guess_fine[0]]
    else:
        max_frequency = radius[-1]

    return radius_bin_center, ring_brightness, max_frequency


def fit_psf_params(
    psf_guess: np.ndarray, p0: list | None = None, fit_params=("a12", "w", "s", "r_mask"), update_defaults: bool = False
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Fit the parameters of the PSF model to a given PSF guess.

    Args:
        psf_guess (np.ndarray): The PSF guess.
        p0 (list[float]): The initial parameter values.

    Returns:
        tuple[np.ndarray, list[float]]: The fitted PSF and the fitted parameters.
    """
    y, x = np.indices(psf_guess.shape)
    y = y - np.mean(y)
    x = x - np.mean(x)

    def err_fun(p):
        return np.sum((psf_guess - psf_2d(x, y, **dict(zip(fit_params, p)))) ** 2)

    if p0 is None:
        psf_defaults = (
            {**DEFAULT_PSF_PARAMS, **defaults.psf_params}
            if isinstance(defaults.psf_params, dict)
            else DEFAULT_PSF_PARAMS
        )
        p0 = [val for key, val in psf_defaults.items() if key in fit_params]
    if len(p0) != len(fit_params):
        raise ValueError(f"p0 and fit_params must have the same length, got {len(p0)} and {len(fit_params)}")

    res = sp.optimize.minimize(err_fun, p0, method="Nelder-Mead")

    psf_params_fit = dict(zip(fit_params, res.x.tolist()))

    if update_defaults:
        if isinstance(defaults.psf_params, dict):
            defaults.psf_params.update(psf_params_fit)
        else:
            defaults.set("psf_params", {**DEFAULT_PSF_PARAMS, **psf_params_fit})

    psf_fit = psf_2d(x, y, **psf_params_fit)
    return psf_fit, psf_params_fit


# %% Helper functions
@numba_njit("(int64,int64,int64,int64)")
def get_indices(x_idx, size_padded, pad, divider):
    if np.abs(x_idx) > size_padded // 2:
        # all indices out of bound
        return 0, 0, (size_padded - 2 * pad) // divider + 1
    else:
        # there are at least some valid indces
        x_start = pad + x_idx
        x_end = (
            size_padded + x_idx - pad + 1
        )  # we don't use negative indexing because we could not access the last element with it
        # check whether the ourput has the right size or needs padding
        x_pad = 0
        if x_start < 0:
            x_pad = x_start // divider
            x_start = x_start % divider  # keeping the subpixel position correct
        elif x_end > size_padded + 1:
            x_pad = -((size_padded - x_end + 1) // divider)
            x_end = size_padded + 1
        return x_start, x_end, x_pad


@numba_njit("float64[:,:](float64[:,:],int64,int64,int64)")
def pad_xy(psf, x_pad, y_pad, size):
    # pad psf to size. The sign of x_pad and y_pad control the side for padding.
    # We use concatenate with zero arrays since np.pad is not available in numba
    if psf.size == 0:
        psf = np.zeros((size, size))
    else:
        if y_pad < 0:
            psf = np.concatenate((np.zeros((abs(y_pad), psf.shape[1])), psf), axis=0)
        elif y_pad > 0:
            psf = np.concatenate((psf, np.zeros((abs(y_pad), psf.shape[1]))), axis=0)
        if x_pad < 0:
            psf = np.concatenate((np.zeros((size, abs(x_pad))), psf), axis=1)
        elif x_pad > 0:
            psf = np.concatenate((psf, np.zeros((size, abs(x_pad)))), axis=1)
    return psf


# %% Support for experimental PSF implementation
def fft_upscale(array, scale):
    """
    Upsample the last two dimensions of an array by an integer factor using FFT.
    Preserves the amplitude of the input array.

    Parameters
    ----------
    array : 2D numpy ndarray
        Input array to be upsampled.
    scale : int
        Integer scale factor for upsampling.

    Returns
    -------
    array_up : numpy ndarray
        Upsampled array.

    """
    if scale == 1:
        return array

    # Calculate dimensions
    dim_y, dim_x = array.shape[-2:]
    new_dim_y = dim_y * scale
    new_dim_x = dim_x * scale

    # Perform FFT and pad with zeros
    array_f = np.fft.ifftshift(np.fft.fft2(array))
    pad_y_before = (new_dim_y - dim_y) // 2
    pad_y_after = new_dim_y - dim_y - pad_y_before
    pad_x_before = (new_dim_x - dim_x) // 2
    pad_x_after = new_dim_x - dim_x - pad_x_before
    pad_values = (*((0, 0),) * (array.ndim - 2), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after))
    array_f_padded = np.pad(
        array_f,
        pad_values,
        mode="constant",
        constant_values=0,
    )

    # Inverse FFT and correct the amplitude
    array_up = np.fft.ifft2(np.fft.fftshift(array_f_padded))
    array_up = array_up.real * (scale**2)
    return array_up


# %% Grid generation
def calculate_grid(fun, size=1, divider=256, pad=1, D4_symmetry=False, **kwargs):
    pad_fine = np.floor(pad * divider).astype(int)  # convert pad to grid scaling
    lh = size // 2

    if not callable(fun):
        # experimental PSF: we got a thumbnail and need to upsample it
        fun = np.pad(fun, ((pad + 1, pad + 1), (pad + 1, pad + 1)), mode="constant", constant_values=0)
        if fun.ndim != 2 or fun.shape[0] < size or fun.shape[1] < size:
            raise ValueError(
                f"The PSF array must be 2D and at least as large as requested size {size}, got shape {fun.shape}"
            )
        # psf_upsample = fft_upscale(fun,divider)
        # psf_upsample = np.pad(psf_upsample, ((0, 1), (0, 1)), mode="edge")
        psf_upsample = sp.ndimage.zoom(
            fun, divider, order=5, mode="grid-constant", cval=0.0, prefilter=False, grid_mode=True
        )
        # Cut out the center
        # psf_fine = psf_upsample[
        #     psf_upsample.shape[0]//2-lh*divider-pad:psf_upsample.shape[0]//2+lh*divider+1+pad,
        #     psf_upsample.shape[1]//2-lh*divider-pad:psf_upsample.shape[1]//2+lh*divider+1+pad
        #     ]
        # trim the edges: (half a pixel (divider/2) on all sides)
        psf_fine = psf_upsample[
            psf_upsample.shape[0] // 2 - (lh + pad) * divider : psf_upsample.shape[0] // 2 + (lh + pad) * divider + 1,
            psf_upsample.shape[1] // 2 - (lh + pad) * divider : psf_upsample.shape[1] // 2 + (lh + pad) * divider + 1,
        ]
    else:
        # callable PSF function: we need to evaluate it on a grid
        if D4_symmetry:
            xd = np.arange(-lh - pad_fine / divider, 1 / divider, 1 / divider)
            Xd, Yd = np.meshgrid(xd, xd)
            psf_seg = fun(Xd, Yd, **kwargs)
            psf_fine = np.block(
                [[psf_seg, psf_seg[:, -2::-1]], [psf_seg[-2::-1], psf_seg[-2::-1, -2::-1]]]
            )  # this flips the segment and removes the last line (x=0 or y=0)
        else:
            xd = np.arange(-lh - pad_fine / divider, lh + (pad_fine + 1) / divider, 1 / divider)
            Xd, Yd = np.meshgrid(xd, xd)
            psf_fine = fun(Xd, Yd, **kwargs)

    return psf_fine


@functools.lru_cache(maxsize=4)
def get_lookup_fun(fun, img_only=False, size=1, divider=256, pad=1, D4_symmetry=False, interpolate=False, **kwargs):
    psf_fine = calculate_grid(fun, size=size, divider=divider, pad=pad, D4_symmetry=D4_symmetry, **kwargs)
    pad_fine = np.floor(pad * divider).astype(int)  # convert pad to grid scaling
    psf_fine_gy = np.gradient(psf_fine, 1 / divider, axis=0)
    psf_fine_gx = np.gradient(psf_fine, 1 / divider, axis=1)

    lookup_fun_signature = "float64[:,:](float64[:])"
    if not interpolate:

        def lookup_fun_(pos):
            if pos.size == 3:
                c, x0, y0 = pos
            elif pos.size == 2:
                x0, y0 = pos
                c = 1.0
            else:
                raise ValueError
            x_idx = int(np.round(-x0 * divider))
            y_idx = int(np.round(-y0 * divider))

            x_start, x_end, x_pad = get_indices(x_idx, psf_fine.shape[1], pad_fine, divider)
            y_start, y_end, y_pad = get_indices(y_idx, psf_fine.shape[0], pad_fine, divider)

            psf = c * psf_fine[y_start:y_end:divider, x_start:x_end:divider]
            if x_pad != 0 or y_pad != 0:
                # we need to pad
                psf = pad_xy(psf, x_pad, y_pad, size)
            return psf

    else:

        def lookup_fun_(pos):
            if pos.size == 3:
                c, x0, y0 = pos
            elif pos.size == 2:
                x0, y0 = pos
                c = 1.0
            else:
                raise ValueError

            # Compute fractional grid indices
            x_f = -x0 * divider
            y_f = -y0 * divider
            x_idx = int(np.floor(x_f))
            y_idx = int(np.floor(y_f))
            x_alpha = x_f - x_idx
            y_alpha = y_f - y_idx

            # Get indices for the four corners
            x_start, x_end, x_pad = get_indices(x_idx, psf_fine.shape[1], pad_fine, divider)
            y_start, y_end, y_pad = get_indices(y_idx, psf_fine.shape[0], pad_fine, divider)
            x_start1, x_end1, x_pad1 = get_indices(x_idx + 1, psf_fine.shape[1], pad_fine, divider)
            y_start1, y_end1, y_pad1 = get_indices(y_idx + 1, psf_fine.shape[0], pad_fine, divider)

            # Get four PSFs at the corners
            psf00 = psf_fine[y_start:y_end:divider, x_start:x_end:divider]
            psf01 = psf_fine[y_start:y_end:divider, x_start1:x_end1:divider]
            psf10 = psf_fine[y_start1:y_end1:divider, x_start:x_end:divider]
            psf11 = psf_fine[y_start1:y_end1:divider, x_start1:x_end1:divider]

            # Pad if needed
            if x_pad != 0 or y_pad != 0:
                psf00 = pad_xy(psf00.copy(), x_pad, y_pad, size)  # Copy to make editable in numba
            if x_pad1 != 0 or y_pad != 0:
                psf01 = pad_xy(psf01.copy(), x_pad1, y_pad, size)
            if x_pad != 0 or y_pad1 != 0:
                psf10 = pad_xy(psf10.copy(), x_pad, y_pad1, size)
            if x_pad1 != 0 or y_pad1 != 0:
                psf11 = pad_xy(psf11.copy(), x_pad1, y_pad1, size)

            # Bilinear interpolation
            psf = (
                (1 - x_alpha) * (1 - y_alpha) * psf00
                + x_alpha * (1 - y_alpha) * psf01
                + (1 - x_alpha) * y_alpha * psf10
                + x_alpha * y_alpha * psf11
            )
            return c * psf

    lookup_fun = numba_njit(lookup_fun_signature, cache=False)(lookup_fun_) if HAS_NUMBA else lookup_fun_
    if img_only:
        return lookup_fun

    errfun_pos_signature = "float64(float64[:], float64[:,:])"

    def errfun_pos_(pos, img):

        psf = lookup_fun(pos)
        eps = np.finfo(np.float64).eps
        alpha = np.sum(psf * img) / (np.sum(psf**2) + eps)

        L2 = np.sum((img - alpha * psf) ** 2)
        return L2

    err_fun = numba_njit(errfun_pos_signature, cache=False)(errfun_pos_) if HAS_NUMBA else errfun_pos_

    lookup_fun_grad_signature = "UniTuple(float64[:,:],3)(float64[:])"

    def lookup_fun_grad_(pos):
        if pos.size == 2:
            x0, y0 = pos
        else:
            raise ValueError
        x0, y0 = pos
        x_idx = int(np.round(-x0 * divider))
        y_idx = int(np.round(-y0 * divider))

        x_start, x_end, x_pad = get_indices(x_idx, psf_fine.shape[1], pad_fine, divider)
        y_start, y_end, y_pad = get_indices(y_idx, psf_fine.shape[0], pad_fine, divider)

        psf = psf_fine[
            y_start:y_end:divider,
            x_start:x_end:divider,
        ].copy()
        psf_gy = psf_fine_gy[
            y_start:y_end:divider,
            x_start:x_end:divider,
        ].copy()
        psf_gx = psf_fine_gx[
            y_start:y_end:divider,
            x_start:x_end:divider,
        ].copy()
        if x_pad != 0 or y_pad != 0:
            # we need to pad
            psf = pad_xy(psf, x_pad, y_pad, size)
            psf_gy = pad_xy(psf_gy, x_pad, y_pad, size)
            psf_gx = pad_xy(psf_gx, x_pad, y_pad, size)
        return psf, psf_gy, psf_gx

    lookup_fun_grad = (
        numba_njit(lookup_fun_grad_signature, cache=False)(lookup_fun_grad_) if HAS_NUMBA else lookup_fun_grad_
    )

    errgrad_signature = "Tuple((float64,float64[:]))(float64[:], float64[:,:])"

    def errgrad_(pos, img):  # pos = (x0,y0)
        psf, gyf, gxf = lookup_fun_grad(pos)

        eps = np.finfo(np.float64).eps
        alpha = np.sum(psf * img) / (np.sum(psf**2) + eps)
        yax = img - alpha * psf  # y - alpha x
        ssx = np.sum(yax**2)
        ssx_gy = -2 * np.sum(yax * gyf) * alpha
        ssx_gx = -2 * np.sum(yax * gxf) * alpha
        return ssx, -np.array([ssx_gx, ssx_gy])

    errgrad_fun = numba_njit(errgrad_signature, cache=False, nogil=True)(errgrad_) if HAS_NUMBA else errgrad_

    get_lsq_param_signature = "UniTuple(float64,4)(float64[:], float64[:,:])"

    def get_lsq_param_(pos, img):
        psf = lookup_fun(pos)
        dof = size * size - 3  # assume we are estimating 3 parameters (c,x,y).

        eps = np.finfo(np.float64).eps
        alpha = np.sum(psf * img) / (np.sum(psf**2) + eps)
        yax = img - alpha * psf  # y - alpha x
        ssr = np.sum(yax**2)
        ssx = np.sum((psf - np.mean(psf)) ** 2)
        sst = np.sum((img - np.mean(img)) ** 2)
        r2 = 1 - ssr / (sst + eps)
        # sigma = np.sqrt(ssr/dof)
        alpha_se = np.sqrt(ssr / (dof * ssx + eps))
        # chi2 = np.sum(yax**2)/dof  # we need to way by the expected variance, which is not available here
        return alpha, alpha_se, r2, ssr

    get_lsq_param = numba_njit(get_lsq_param_signature, cache=False)(get_lsq_param_) if HAS_NUMBA else get_lsq_param_

    grid_search_signature = "float64[:](float64[:], float64[:,:],float64,int32,int32)"

    def grid_search_(pos0, img, range=3, nstep=4, max_divider=divider):
        current_pos = np.array([np.round(pos0[0] * max_divider), np.round(pos0[1] * max_divider)])
        current_range = range * max_divider
        step_size = np.ceil(current_range / nstep)

        min_pos = current_pos
        min_err = np.inf
        while True:
            steps = np.arange(-current_range // 2, current_range // 2 + 1, step_size)
            for dx in steps:
                for dy in steps:
                    _pos = (current_pos + np.array([dx, dy])) / max_divider
                    err = err_fun(_pos, img)
                    if err < min_err:
                        min_err = err
                        min_pos = _pos
            if step_size > 1.0:
                current_pos = min_pos * max_divider
                current_range = step_size
                step_size = np.ceil(current_range / nstep)
            else:
                break
        return min_pos

    grid_search = (
        numba_njit(grid_search_signature, cache=False, nogil=True)(grid_search_) if HAS_NUMBA else grid_search_
    )

    gradient_descent_signature = "float64[:](float64[:], float64[:,:])"

    def gradient_descent_(pos0, img):
        alpha = 0.5
        max_iter = 100
        detect_overshoot = True
        tolerance = 1e-100

        min_pos = pos0

        eps = np.finfo(np.float64).eps
        signal_scale = np.var(lookup_fun(np.zeros(2))) / (np.var(img) + eps)
        tolerance /= signal_scale

        alpha_shrinked = False

        min_err = np.inf
        current_pos = min_pos.copy()
        current_alpha = alpha * signal_scale
        min_grad = np.zeros(2)
        grad_sign = np.zeros(2)
        for _idx in range(max_iter):
            err, grad = errgrad_fun(current_pos, img)
            if max(abs(grad[0]), abs(grad[1])) < tolerance:
                break

            if err > min_err:
                # shrink alpha
                current_alpha *= 0.5
                alpha_shrinked = True
            else:
                min_err = err
                min_pos = current_pos.copy()
                min_grad = grad

                if detect_overshoot and np.all(grad_sign == -np.sign(grad)):
                    # overshoot: shrink alpha
                    current_alpha *= 0.5
                    alpha_shrinked = True
                grad_sign = np.sign(grad)

            shift = -min_grad * current_alpha
            if interpolate:
                min_step = 0.01
            else:
                # without interpolation stop
                min_step = 0.5
            if max(abs(shift[0]), abs(shift[1])) * divider < min_step:
                if not alpha_shrinked:
                    # try raising alpha
                    current_alpha = 10 * current_alpha
                else:
                    break
            current_pos = min_pos + shift

        return min_pos

    gradient_descent = (
        numba_njit(gradient_descent_signature, cache=False, nogil=True)(gradient_descent_)
        if HAS_NUMBA
        else gradient_descent_
    )

    fit_gd_grid_guess_signature = "UniTuple(float64,6)(float64[:,:], float64)"

    def fit_gd_grid_guess_(img, max_shift):
        pos_guess = grid_search(np.zeros(2), img, range=2 * max_shift, nstep=(2 * max_shift + 1), max_divider=1)
        min_pos = gradient_descent(pos_guess, img)
        alpha, alpha_se, r2, l2 = get_lsq_param(min_pos, img)
        return alpha, min_pos[0], min_pos[1], np.sqrt(l2), r2, alpha_se

    fit_gd_grid_guess = (
        numba_njit(fit_gd_grid_guess_signature, cache=False, nogil=True)(fit_gd_grid_guess_)
        if HAS_NUMBA
        else fit_gd_grid_guess_
    )

    # return lookup, err, errgrad, get_lsq_param
    return lookup_fun, fit_gd_grid_guess


# %%
