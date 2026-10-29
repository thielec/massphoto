"""Mass calibration utilities.

This module provides functions to fit and apply polynomial mass calibration
between image contrast and mass (kDa), manage calibration coefficients stored
in `defaults`, and validate or invert first-order polynomials.

The calibration function maps contrast to mass, e.g. a first-order polynomial
(linear) function of the form::

    mass_kDa = coeffs[0] + coeffs[1] * contrast

This is different from the DiscoverMP convention which uses::

    contrast = slope * mass_kDa + offset

To convert between the two conventions, the parameter ``direction`` can be used
to indicate the direction of the calibration.

"""

import logging
from collections.abc import Callable, Sequence
from typing import Any, overload

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from .utils import EventList, defaults

CALIBRATION_DIRECTIONS = ("contrast_to_mass", "mass_to_contrast")
CALIBRATION_INPUT_UNITS = ("contrast", "mass", "auto")
logger = logging.getLogger(__name__)


def apply(
    event_list: EventList | pd.DataFrame | np.ndarray,
    mass_calibration: Callable[[np.ndarray], np.ndarray] | Sequence[float] | None = None,
) -> EventList | np.ndarray:
    """
    Apply mass calibration to fit results.

    When ``event_list`` is an :class:`~pandas.DataFrame` (``EventList``), the
    ``masses_kDa`` column is created or overwritten.
    If no calibration is available (argument and defaults are ``None``),
    the input is returned unchanged and a log message is emitted.

    Parameters
    ----------
    event_list : EventList | np.ndarray
        EventList containing a ``contrasts`` column, or a 1D array of contrast values.
    mass_calibration : Callable[[np.ndarray], np.ndarray] | Sequence[float] | None
        Callable that maps contrast values to mass (kDa), or polynomial coefficients in
        :class:`~numpy.polynomial.Polynomial` order (increasing), or ``None`` to read
        from :data:`massphoto.defaults`.

    Returns
    -------
    EventList | np.ndarray
        Updated EventList, or an array of masses.

    See Also
    --------
    get_calibration
    fit
    """
    mass_calibration = get_calibration(
        mass_calibration,
        direction="contrast_to_mass",
        allow_none=True,
    )
    if isinstance(event_list, np.ndarray):
        if mass_calibration is None:
            raise ValueError("mass_calibration must be provided or set in defaults")
        return mass_calibration(event_list)
    if mass_calibration is None:
        logger.info("Mass calibration unavailable; returning input unchanged.")
        return event_list
    if "contrasts" not in event_list.columns:
        raise ValueError("fit_results must contain 'contrasts' field")
    event_list["masses_kDa"] = mass_calibration(event_list["contrasts"])
    # uncertainty propagation if possible
    if "contrasts_se" in event_list.columns and getattr(mass_calibration, "deriv", False):
        deriv = mass_calibration.deriv()
        event_list["masses_kDa_se"] = deriv(event_list["contrasts"]) * event_list["contrasts_se"]
    return event_list


def get_calibration(
    mass_calibration: Callable[[np.ndarray], np.ndarray] | Sequence[float] | Polynomial | None = None,
    direction: str = "contrast_to_mass",
    allow_none: bool = False,
) -> Callable[[np.ndarray], np.ndarray] | None:
    """
    Return a callable mass calibration (contrast -> mass) or its inverse.

    Sequences of coefficients are converted to a `np.polynomial.Polynomial` (callable).
    If a callable is provided and `direction` is 'contrast_to_mass', it is returned as-is.
    If `direction` is 'mass_to_contrast' and a non-polynomial callable is given, a
    ValueError is raised because inversion is ambiguous for arbitrary callables.

    Parameters
    ----------
    mass_calibration : Callable[[np.ndarray], np.ndarray] | Sequence[float] | np.polynomial.Polynomial | None
        Callable mapping contrast to mass, or sequence of polynomial coefficients
        in increasing order (index i is coefficient for x^i), or a Polynomial.
        If None, the calibration is read from `defaults.mass_calibration`.
    direction : str
        'contrast_to_mass' or 'mass_to_contrast'. When requesting 'mass_to_contrast'
        the returned callable maps mass -> contrast.
    allow_none : bool, default=False
        If True, return ``None`` when no calibration is available (both argument
        and defaults are ``None``), instead of raising.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray] | None
        Callable implementing the requested mapping, or ``None`` if
        ``allow_none=True`` and no calibration is available.
    """
    if direction not in CALIBRATION_DIRECTIONS:
        raise ValueError(f"direction must be one of {CALIBRATION_DIRECTIONS}, got {direction!r}")
    if mass_calibration is None:
        mass_calibration = defaults.mass_calibration
    if mass_calibration is None:
        if allow_none:
            return None
        raise ValueError("mass_calibration must be provided or set in defaults")
    if not callable(mass_calibration):
        mass_calibration = get_coefficients(mass_calibration, direction=direction)
        mass_calibration = Polynomial(mass_calibration)
    elif direction == "mass_to_contrast":
        if isinstance(mass_calibration, Polynomial):
            mass_calibration = _invert_polynomial_coefficients(mass_calibration)
        else:
            raise ValueError(
                "Cannot invert a non-polynomial callable mass_calibration. Provide polynomial coefficients or a `Polynomial`."
            )
    return mass_calibration


def get_coefficients(
    mass_calibration: Sequence[float] | Polynomial | None = None,
    direction: str = "contrast_to_mass",
) -> Sequence[float]:
    """
    Return the raw coefficients of the mass calibration.

    Always returns a sequence of coefficients (in increasing order). If a
    `Polynomial` is provided it is converted to a coefficient sequence.
    If `direction == 'mass_to_contrast'`, the returned coefficients are inverted
    so they represent the requested mapping.

    Parameters
    ----------
    mass_calibration : Sequence[float] | np.polynomial.Polynomial | None
        Sequence of polynomial coefficients in increasing order or a Polynomial.
        If None, `defaults.mass_calibration` is used.
    direction : str
        'contrast_to_mass' or 'mass_to_contrast'.

    Returns
    -------
    Sequence[float]
        Coefficients (in increasing order) representing the requested mapping.
    """
    if mass_calibration is None:
        mass_calibration = defaults.mass_calibration
    if direction not in CALIBRATION_DIRECTIONS:
        raise ValueError(f"direction must be one of {CALIBRATION_DIRECTIONS}, got {direction!r}")
    if mass_calibration is None:
        raise ValueError("mass_calibration must be provided or set in defaults")

    if isinstance(mass_calibration, Polynomial):
        coeffs = mass_calibration.convert().coef
    elif isinstance(mass_calibration, Sequence) and not isinstance(
        mass_calibration, (str, bytes)
    ):  # numerical sequence and not string
        coeffs = mass_calibration
    else:
        raise TypeError(
            "mass_calibration must be a sequence of coefficients or a Polynomial when requesting coefficients"
        )

    if direction == "mass_to_contrast":
        coeffs = _invert_polynomial_coefficients(coeffs)
    return coeffs


def set_coefficients(
    coeffs: Sequence[float] | Polynomial,
    direction: str = "contrast_to_mass",
    check: bool = True,
) -> None:
    """
    Set the coefficients of the mass calibration function in defaults.

    Parameters
    ----------
    coeffs : Sequence[float] | np.polynomial.Polynomial
        Polynomial coefficients in :class:`~numpy.polynomial.Polynomial` order
        (increasing), or a ``Polynomial`` instance.

    direction : str
        Direction of the calibration. One of 'contrast_to_mass' or 'mass_to_contrast'.
    """
    if isinstance(coeffs, Polynomial):
        coeffs = coeffs.convert().coef
    if direction == "mass_to_contrast":
        coeffs = _invert_polynomial_coefficients(coeffs)
    defaults.set("mass_calibration", tuple(float(c) for c in coeffs))
    logger.info(
        "Updated defaults.mass_calibration with %d coefficient(s): %s",
        len(coeffs),
        tuple(coeffs),
    )
    if check:
        check_coefficients(coeffs)


def check_coefficients(
    coeffs: Sequence[float] | Polynomial,
    direction: str = "contrast_to_mass",
) -> None:
    """
    Check that the coefficients of the mass calibration function are valid.

    Parameters
    ----------
    coeffs : Sequence[float] | np.polynomial.Polynomial
        Polynomial coefficients in :class:`~numpy.polynomial.Polynomial` order
        (increasing), or a ``Polynomial`` instance.
    """
    if isinstance(coeffs, Polynomial):
        coeffs = coeffs.convert().coef
    if direction == "mass_to_contrast":
        coeffs = _invert_polynomial_coefficients(coeffs)
    # Apply to a value of 0.1 and check that the result is above 10. That's a very conservative check but catches
    # inverted coefficients.
    val = Polynomial(coeffs)(0.1)
    if val < 10:
        raise ValueError("Mass calibration coefficients are invalid. Resulting mass for contrast 0.1 is below 10 kDa.")


def fit(
    contrasts: np.ndarray | pd.DataFrame | Sequence[float] | Any,
    masses_kDa: np.ndarray,
    *,
    degree: int | list[int] = 1,
    update_defaults: bool = False,
    direction: str = "contrast_to_mass",
    input_unit: str = "contrast",
    reference_calibration: Callable[[np.ndarray], np.ndarray] | Sequence[float] | Polynomial | None = None,
) -> Polynomial:
    """
    Fit a mass calibration polynomial to given contrasts and masses.

    Parameters
    ----------
    contrasts : np.ndarray | Sequence[float] | pd.DataFrame | FitResult-like
        Peak-position values used as calibration input.
        Supported inputs are:

        - 1D array-like values (contrast by default)
        - a DataFrame with a ``mu`` column (e.g. ``fitres.peaks``)
        - an object with a ``peaks`` DataFrame containing ``mu`` (e.g. ``FitResult``)

        When a DataFrame or FitResult-like object is provided, peak positions are
        extracted from ``mu`` and sorted ascending before fitting.
    masses_kDa : np.ndarray
        Array of mass values in kDa.
    degree : int | list[int]
        Maximum degree of the polynomial to fit. Provide a list of integers to specify
        the degrees of the terms to include, e.g. [1,] for linear without offset.
        Default is 1 (=[0,1] linear fit with offset).
    update_defaults : bool
        If True, store the fitted calibration in :data:`massphoto.defaults`.
    direction : str
        Fit either ``'contrast_to_mass'`` (default) or ``'mass_to_contrast'``.
        The returned polynomial always represents the fitted direction.
    input_unit : str
        Unit of ``contrasts``. One of ``'contrast'``, ``'mass'``, or ``'auto'``.
        If ``'mass'``, values are converted to contrast via inversion of an existing
        calibration before fitting. If ``'auto'``, values are interpreted as mass when
        the median absolute value is greater than 1, otherwise as contrast.
    reference_calibration : Callable[[np.ndarray], np.ndarray] | Sequence[float] | np.polynomial.Polynomial | None
        Calibration to invert when ``input_unit`` resolves to ``'mass'``.
        If ``None``, :data:`massphoto.defaults.mass_calibration` is used.
        Inversion requires an invertible first-order polynomial calibration.

    Returns
    -------
    np.polynomial.Polynomial
        Fitted calibration polynomial.
    """
    if input_unit not in CALIBRATION_INPUT_UNITS:
        raise ValueError(f"input_unit must be one of {CALIBRATION_INPUT_UNITS}, got {input_unit!r}")

    contrasts = _normalize_calibration_inputs(contrasts)
    masses_kDa = np.asarray(masses_kDa, dtype=float)

    if input_unit == "auto":
        scale = float(np.median(np.abs(contrasts))) if contrasts.size else 0.0
        input_unit = "mass" if scale > 1 else "contrast"

    contrasts_for_fit = contrasts
    if input_unit == "mass":
        logger.info("fit received mass-domain peak positions; converting to contrast using existing calibration")
        try:
            mass_to_contrast = get_calibration(
                reference_calibration,
                direction="mass_to_contrast",
                allow_none=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Cannot convert mass-domain peaks to contrast values. "
                "Provide an invertible first-order polynomial calibration via "
                "reference_calibration or defaults.mass_calibration."
            ) from exc
        contrasts_for_fit = np.asarray(mass_to_contrast(contrasts), dtype=float)

    if direction == "contrast_to_mass":
        p = Polynomial.fit(contrasts_for_fit, masses_kDa, deg=degree, domain=[])
    else:
        p = Polynomial.fit(masses_kDa, contrasts_for_fit, deg=degree, domain=[])
    if update_defaults:
        set_coefficients(p, direction=direction)
    return p


def _normalize_calibration_inputs(contrasts: np.ndarray | pd.DataFrame | Sequence[float] | Any) -> np.ndarray:
    """Normalize supported calibration-input formats to a 1D float array."""
    peaks = getattr(contrasts, "peaks", None)
    if isinstance(peaks, pd.DataFrame):
        contrasts = peaks

    if isinstance(contrasts, pd.DataFrame):
        if "mu" not in contrasts.columns:
            raise ValueError("DataFrame calibration input must contain a 'mu' column")
        values = np.sort(contrasts["mu"].to_numpy(dtype=float))
    else:
        values = np.asarray(contrasts, dtype=float)

    if values.ndim != 1:
        raise ValueError(f"Calibration input values must be 1D, got shape {values.shape}")
    return values


@overload
def _invert_polynomial_coefficients(coeffs: Polynomial) -> Polynomial: ...


@overload
def _invert_polynomial_coefficients(coeffs: Sequence[float]) -> Sequence[float]: ...


def _invert_polynomial_coefficients(
    coeffs: Sequence[float] | Polynomial,
) -> Sequence[float] | Polynomial:
    """
    Invert a polynomial of first order. Ambiguous for higher orders. Input type is preserved.

    Parameters
    ----------
    coeffs : Sequence[float] | np.polynomial.Polynomial
        Coefficients of the polynomial to invert, where the i-th element corresponds to the coefficient for x^i.
        Or a np.polynomial.Polynomial instance.

    Returns
    -------
    Sequence[float]
        Coefficients of the inverted polynomial.
    Or
    np.polynomial.Polynomial
        Inverted polynomial if input was a np.polynomial.Polynomial.
    """
    input_was_polynomial = isinstance(coeffs, Polynomial)
    if input_was_polynomial:
        coeffs = coeffs.convert().coef
    if len(coeffs) != 2:
        raise ValueError("Only first order polynomials can be inverted unambiguously")
    # y = a * x + b  =>  x = (y - b) / a  =>  x = (1/a) * y + (-b/a)
    b, a = coeffs
    coeffs = [-b / a, 1 / a]
    return Polynomial(coeffs) if input_was_polynomial else coeffs
