import numpy as np
import scipy as sp

from .util import stack_apply_filter


def gaussian_laplace(movie: np.ndarray, sigma: float = 1.5, **kwargs) -> np.ndarray:
    """Apply a Gaussian Laplace filter to each image in a movie stack.

    Parameters
    ----------
    movie : np.ndarray
        Stack of images represented as a NumPy array. The last two dimensions
        correspond to height and width ``(..., y, x)``.
    sigma : float, default=1.5
        Standard deviation of the Gaussian kernel.
    **kwargs
        Additional keyword arguments passed to :func:`scipy.ndimage.gaussian_laplace`.

    Returns
    -------
    np.ndarray
        Filtered images with the same shape as the input array.
    """
    return stack_apply_filter(sp.ndimage.gaussian_laplace, movie, sigma=sigma, **kwargs)
