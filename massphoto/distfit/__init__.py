"""
Distfit - fitting of histogram-like mass or contrast distributions.

Typical usage
--------------
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from massphoto import distfit
>>> # Flexible input: pass raw data or pre-computed histogram
>>> result = distfit.fit(samples)  # auto-detect peaks from raw data
>>> # Or with custom binning
>>> result = distfit.fit(samples, bins=50)
>>> # Or with pre-computed histogram
>>> counts, edges = np.histogram(samples, bins=50)
>>> result = distfit.fit((counts, edges))
>>> plt.plot(result.model_counts)

Modules
--------
detect : bandwidth estimation and peak detection
model  : analytical bin-integrated Gaussian model
fitting : fitting implementation (select hybrid or full through ``fit(method=...)``)
utils  : input standardization, unit detection, bounds, amplitude guessing
"""

from . import detect, model, utils
from .fitting import FitResult, fit
from .model import bin_integrated_gaussian, jacobian_bin_integrated
from .utils import standardize_histogram_input

# re-export key symbols
__all__ = [
    "detect",
    "fit",
    "FitResult",
    "model",
    "utils",
    "bin_integrated_gaussian",
    "jacobian_bin_integrated",
    "standardize_histogram_input",
]
