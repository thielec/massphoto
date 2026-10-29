"""
Point Spread Function (PSF) models for mass photometry.

Provides analytical PSF as described in:
Foley et al. Nature Methods (2021): https://doi.org/10.1038/s41592-021-01261-w
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import j1

from .utils import defaults

DEFAULT_PSF_PARAMS: dict[str, float] = {
    "a12": -6.3337259134,
    "w": 2.1370299329,
    "s": 5.0425755943,
    "r_mask": 1.5 * 2.5e-3,
    "r_aperture": 8.52e-3,
}

defaults.set("psf_params", DEFAULT_PSF_PARAMS.copy())


def psf_jinc(
    r: ArrayLike,
    a1: float,
    w1: float,
    a2: float = 0.0,
    w2: float = 1.0,
    sigma: float = float("inf"),
) -> NDArray[np.float64]:
    """
    Two-component jinc-based point spread function (PSF) with Gaussian envelope.

    The jinc function is normalised such that :math:`\\mathrm{jinc}(0) = 1`:

    .. math::

        \\mathrm{jinc}(x) = 2\\,\\frac{J_1(\\pi x)}{\\pi x}

    The PSF is then defined as:

    .. math::

        I(r) = \\left[a_1\\,\\mathrm{jinc}\\left(\\frac{r}{w_1}\\right)
        + a_2\\,\\mathrm{jinc}\\left(\\frac{r}{w_2}\\right)\\right]
        e^{-1/2*(r/\\sigma)^2}

    so that at :math:`r=0` the peak intensity equals :math:`a_1 + a_2`.

    Notes
    -----
    This PSF represents the difference between two jinc functions multiplied by a Gaussian envelope.
    It follows the form described in Equation (2) of the publication:

    Foley *et al.* **Nature Methods** (2021).
    `https://doi.org/10.1038/s41592-021-01261-w <https://doi.org/10.1038/s41592-021-01261-w>`_

    Parameters
    ----------
    r : NDArray[np.float64]
        Radial coordinate or array of radial coordinates.
    a1, a2 : float
        Amplitudes of the two jinc components.
    w1, w2 : float
        Width parameters of the two jinc components.
    sigma : float
        Gaussian envelope width.

    Returns
    -------
    NDArray[np.float64]
        Intensity at radius ``r``.
    """
    r_arr = np.asarray(r, dtype=float)
    x1 = np.pi * r_arr / w1
    x2 = np.pi * r_arr / w2

    # normalised jinc: jinc(0) = 1
    jinc1 = np.ones_like(r_arr, dtype=float)
    jinc2 = np.ones_like(r_arr, dtype=float)
    mask = r_arr != 0
    np.divide(2.0 * j1(x1), x1, out=jinc1, where=mask, dtype=float)
    np.divide(2.0 * j1(x2), x2, out=jinc2, where=mask, dtype=float)

    return (a1 * jinc1 + a2 * jinc2) * np.exp(-0.5 * (r_arr / sigma) ** 2)


def psf_2d(
    x: ArrayLike,
    y: ArrayLike,
    c: float | None = None,
    a12: float | None = None,
    w: float | None = None,
    s: float | None = None,
    r_mask: float | None = None,
    r_aperture: float | None = None,
) -> NDArray[np.float64]:
    """
    Compute the two-dimensional point spread function (PSF) based on a dual-jinc
    model multiplied by a Gaussian envelope.

    The PSF is evaluated using :func:`psf_jinc` with parameters derived from
    the supplied arguments or, if omitted, from :data:`defaults.psf_params`:

    - :math:`a_1 + a_2 = c`
    - :math:`a_1 / a_2 = a_{12}`
    - :math:`w_1 = w`
    - :math:`w_2 = w \\times r_{\\text{aperture}} / r_{\\text{mask}}`
    - :math:`\\sigma = s`

    Parameters
    ----------
    x, y : ArrayLike
        Cartesian coordinates.
    c : float, optional
        Overall amplitude (sum of ``a1 + a2``). Defaults to 1.0.
    a12, w, s, r_mask, r_aperture : float, optional
        PSF parameters. Missing values are read from :data:`defaults.psf_params`.

    Returns
    -------
    NDArray[np.float64]
        Two-dimensional PSF intensity distribution.
    """
    # Load defaults if arguments are not provided
    psf_defaults = defaults.psf_params if isinstance(defaults.psf_params, dict) else {}
    a12 = psf_defaults.get("a12", DEFAULT_PSF_PARAMS["a12"]) if a12 is None else a12
    w = psf_defaults.get("w", DEFAULT_PSF_PARAMS["w"]) if w is None else w
    s = psf_defaults.get("s", DEFAULT_PSF_PARAMS["s"]) if s is None else s
    r_mask = psf_defaults.get("r_mask", DEFAULT_PSF_PARAMS["r_mask"]) if r_mask is None else r_mask
    r_aperture = psf_defaults.get("r_aperture", DEFAULT_PSF_PARAMS["r_aperture"]) if r_aperture is None else r_aperture
    c = 1.0 if c is None else c

    # invalid configuration guard
    if abs(a12 + 1.0) < 1e-12:
        raise ValueError("Parameter 'a12' must not be -1 (would make a1/a2 undefined for non-zero c).")

    # amplitude ratio and sum constraint
    a1 = float(c) * (a12 / (a12 + 1.0))
    a2 = float(c) * (1.0 / (a12 + 1.0))

    w1 = float(w)
    w2 = float(w) * float(r_aperture) / float(r_mask)
    sigma = float(s)

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    r = np.sqrt(x_arr**2 + y_arr**2)

    return psf_jinc(r, a1=a1, w1=w1, a2=a2, w2=w2, sigma=sigma)
