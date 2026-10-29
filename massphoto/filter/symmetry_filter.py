"""
ADAPTATION FROM MATLAB TO PYTHON BY MAX HANTKE (max.hantke@ox.chem.ac.uk):

FASTRADIAL - Loy and Zelinski's fast radial feature detector

An implementation of Loy and Zelinski's fast radial feature detector

Usage: S = fastradial(im, radii, alpha, beta)

Arguments:
           im    - Image to be analysed
           radii - Array of integer radius values to be processed
                   suggested radii might be [1 3 5]
           alpha - Radial strictness parameter.
                   1 - slack, accepts features with bilateral symmetry.
                   2 - a reasonable compromise.
                   3 - strict, only accepts radial symmetry.
                       ... and you can go higher
           beta  - Gradient threshold.  Gradients below this threshold do
                   not contribute to symmetry measure, defaults to 0.

Returns    S     - Symmetry map.  Bright points with high symmetry are
                   marked with large positive values. Dark points of
                   high symmetry marked with large -ve values.

To localize points use NONMAXSUPPTS on S, -S or abs(S) depending on
what you are seeking to find.

Reference:
Loy, G.  Zelinsky, A.  Fast radial symmetry transform for detecting points of
interest.  IEEE PAMI, Vol. 25, No. 8, August 2003. pp 959-973.

Copyright (c) 2004-2010 Peter Kovesi
Centre for Exploration Targeting
The University of Western Australia
http://www.csse.uwa.edu.au/~pk/research/matlabfns/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
"""

import warnings

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

from .util import stack_apply_filter


def _frsym(im: np.ndarray, radii: np.ndarray | int, alpha: float, beta: float = 0) -> tuple[np.ndarray, np.ndarray]:
    """Compute magnitude-weighted (``S``) and orientation-only (``S_orient``) fast radial symmetry maps."""
    radii = np.atleast_1d(radii)
    round_err = abs(radii.astype(float).round() - radii)
    if (round_err > np.finfo("float64").precision).any():
        raise ValueError("radii must be integers")

    Ny, Nx = im.shape
    NxNy = im.size

    # Compute derivatives in x and y via Farid and Simoncelli's 5 tap derivative filters
    imgx, imgy = _imgrad(im)
    mag = np.sqrt(imgx**2 + imgy**2) + np.finfo("float64").eps  # (+eps to avoid division by 0)

    # Normalise gradient values so that [imgx imgy] form unit direction vectors.
    imgx = imgx / mag
    imgy = imgy / mag

    S = np.zeros(shape=(len(radii), Ny, Nx), dtype="float64")  # Symmetry matrix
    S_orient = np.zeros(shape=(len(radii), Ny, Nx), dtype="float64")  # Orientation only symmetry matrix

    x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))

    for i_r, r in enumerate(radii):
        # Coordinates of 'positively' and 'negatively' affected pixels
        posx = np.asarray(x + np.round(r * imgx), dtype="i")
        posy = np.asarray(y + np.round(r * imgy), dtype="i")

        negx = np.asarray(x - np.round(r * imgx), dtype="i")
        negy = np.asarray(y - np.round(r * imgy), dtype="i")

        # Clamp coordinate values to range of image dimensions
        posx[posx < 0] = 0
        posx[posx > Nx - 1] = Nx - 1
        posy[posy < 0] = 0
        posy[posy > Ny - 1] = Ny - 1

        negx[negx < 0] = 0
        negx[negx > Nx - 1] = Nx - 1
        negy[negy < 0] = 0
        negy[negy > Ny - 1] = Ny - 1

        # Indices
        posI = posy.ravel() * Nx + posx.ravel()
        negI = negy.ravel() * Nx + negx.ravel()

        # Orientation projection image:
        # Integrate positively affected pixels by increments of +1 and negatively affected pixels by increments of -1
        if beta > 0:
            mask = mag.ravel() > beta
            if mask.any():
                posI = posI[mask]
                negI = negI[mask]
                mag = mag.ravel()[mask]
        O = np.bincount(posI, minlength=NxNy) - np.bincount(negI, minlength=NxNy)

        # Magnitude projection image:
        # Integrate positively affected pixels by increments of +grad and negatively affected pixels by increments of -grad
        M = np.bincount(posI, weights=mag.ravel(), minlength=NxNy) - np.bincount(
            negI, weights=mag.ravel(), minlength=NxNy
        )

        # Clamp Orientation projection matrix values to a maximum of
        # +/-kappa,  but first set the normalization parameter kappa to the
        # values suggested by Loy and Zelinski
        kappa = 8 if r == 0 else 9.9

        O[O > kappa] = kappa
        O[O < -kappa] = -kappa

        # Unsmoothed symmetry measure at this radius value
        F = M / kappa * (abs(O) / kappa) ** alpha
        F_orient = np.sign(O) * (abs(O) / kappa) ** alpha  # Orientation only based measure

        # Smooth and spread the symmetry measure with a Gaussian proportional to
        # r.  Also scale the smoothed result by r so that large scales do not
        # lose their relative weighting.
        S[i_r, :, :] = gaussian_filter(F.reshape(im.shape), 0.25 * r) * r
        S_orient[i_r, :, :] = gaussian_filter(F_orient.reshape(im.shape), 0.25 * r) * r

    S = S / len(radii)  # Average
    S_orient = S_orient / len(radii)

    return S, S_orient


def _imgrad(im: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray([0.030320, 0.249724, 0.439911, 0.249724, 0.030320])
    d = np.asarray([0.104550, 0.292315, 0.000000, -0.292315, -0.104550])
    # d2 = np.asarray([0.232905,  0.002668, -0.471147,  0.002668,  0.232905])
    cdk = np.convolve(d, k)

    ckk = np.convolve(k, k)

    dx = -convolve1d(convolve1d(im, cdk, axis=1, mode="constant"), ckk, axis=0, mode="constant")
    dy = convolve1d(convolve1d(im, cdk, axis=0, mode="constant"), ckk, axis=1, mode="constant")

    return (dx, dy)


def symmetry_filter(img: np.ndarray, radius: int = 2, alpha: int = 3, beta: float = 0.0) -> np.ndarray:
    """Fast radial symmetry transform for detecting rotationally symmetric features.

    Parameters
    ----------
    img : np.ndarray
        Input 2D image or n-D image stack ``(..., y, x)``.
    radius : int, default=2
        Feature radius in pixels to detect.
    alpha : int, default=3
        Radial strictness parameter (higher values select strictly radial symmetry,
        whereas lower values allow bilateral features).
    beta : float, default=0.0
        Gradient threshold; gradients below this value do not contribute.

    Returns
    -------
    np.ndarray
        Radial symmetry map with the same shape as ``img``.
    """

    def filter_func(im: np.ndarray, radius: int, alpha: int, beta: float) -> np.ndarray:
        S, S_orient = _frsym(im, radii=radius, alpha=alpha, beta=beta)
        return np.abs(S[0])

    return stack_apply_filter(filter_func, img, radius=radius, alpha=alpha, beta=beta)
