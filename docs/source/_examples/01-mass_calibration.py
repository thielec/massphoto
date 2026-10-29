"""
Mass calibration
================

This tutorial demonstrates how to estimate a contrast-to-mass calibration from a
calibration movie:

1. Load a calibration movie.
2. Run detection + PSF fitting to obtain a table of event contrasts.
3. Fit peaks in the contrast histogram.
4. Fit a (linear) contrast-to-mass calibration and plot the result.
"""

# sphinx_gallery_thumbnail_number = 2
# %%
# Imports
# -------
# We first import necessary packages and modules.

import matplotlib.pyplot as plt
import numpy as np

# %%
# Download example data
# ---------------------
#
# This tutorial expects a calibration movie. For now, the example downloads a
# small calibration file from Zenodo. You can replace this block with a direct
# file path once you have a local calibration movie.
from zenodozipdownloader import ZenodoZipDownloader

import massphoto as mp

files = ZenodoZipDownloader(
    doi="10.5281/zenodo.15800729",
    download_dir="example_data",
).download(inner_pattern="005_mfp1_500x.mp")
if not files:
    raise RuntimeError(
        "Failed to download example data. Please download it manually from https://zenodo.org/record/15800729 "
        "or set path_calib to your own calibration movie."
    )
path_calib = files[0]

print(f"Using calibration movie: {path_calib}")

# %%
# Inputs
# ------
#
# `expected_masses_kDa` should match the ladder present in the calibration movie.
# `contrast_slope_guess` is only used to choose sensible histogram bins and
# initial peak locations (kDa/contrast).

# Expected masses for the first N peaks (kDa). Adjust to match your ladder.
expected_masses_kDa = np.arange(1, 8) * 86.0

# A rough contrast-to-mass slope estimate used to choose sensible binning and
# initial peak locations (kDa/contrast). If unsure, keep the default.
contrast_slope_guess = 3e4

# %%
# Run detection + fitting
# -----------------------
#
# This is the main pipeline step. The output is a DataFrame with fitted event
# positions and a `contrasts` column.


movie, _metadata = mp.read_movie(path_calib)
fitres_calib = mp.fit(movie, movie_correct_drift=True)

print(f"Detected+fit events used for calibration: n={len(fitres_calib)}")
fitres_calib[["frame", "x", "y", "contrasts"]].head()


# %%
# Fit contrast histogram peaks
# ----------------------------
#
# We model the histogram as a sum of bin-integrated Gaussians. The fitted peak
# centres (`mu`) are used as contrast values for the calibration fit.
bins = np.arange(-20, 660, 4) / contrast_slope_guess
initial_peaks = expected_masses_kDa / contrast_slope_guess

mpfit_calib = mp.distfit.fit(
    fitres_calib["contrasts"],
    bins=bins,
    initial_peaks=initial_peaks,
    include_background=False,
)

print("\nFitted peak centres (contrast):")
mpfit_calib.peaks[["mu", "sigma", "A"]]

# %%
# Plot contrast histogram
# -----------------------
#
# This is a useful sanity check: you should see distinct peaks and a reasonable
# fit overlay.
plt.figure(figsize=(6, 3))
plt.hist(fitres_calib["contrasts"], bins=bins, edgecolor="none")
plt.plot(mpfit_calib.model_x, mpfit_calib.model_y, color="black", lw=1, label="fit")
plt.xlabel("Contrast c")
plt.ylabel("Counts")
plt.title("Contrast histogram")
plt.legend()
plt.tight_layout()

# %%
# Fit contrast-to-mass calibration
# --------------------------------
#
# We fit a polynomial (default: linear) mapping contrast to mass (kDa). Often the
# lower-mass peaks are the cleanest, so we only use the first few for the fit.

n_values_for_fit = min(5, len(expected_masses_kDa), len(mpfit_calib.peaks))
calibration = mp.mass_calibration.fit(
    mpfit_calib.peaks["mu"][:n_values_for_fit],
    expected_masses_kDa[:n_values_for_fit],
)
print("\nFitted calibration (contrast -> mass, kDa):", calibration)

# %%
# Update from existing calibration (mass-domain peaks)
# ----------------------------------------------------
#
# If you already have a previous calibration, measured peak positions may be in
# mass units already. In that case, pass those values to ``fit``
# with ``input_unit='auto'`` (or ``'mass'``) and provide the previous calibration
# for inversion to contrast before fitting.

previous_calibration = (-5.2, 3.1e4)
measured_peak_masses_kDa = mp.mass_calibration.apply(
    mpfit_calib.peaks["mu"][:n_values_for_fit].to_numpy(),
    previous_calibration,
)
calibration_updated = mp.mass_calibration.fit(
    measured_peak_masses_kDa,
    expected_masses_kDa[:n_values_for_fit],
    input_unit="auto",
    reference_calibration=previous_calibration,
)
print("Updated calibration from mass-domain peak positions:", calibration_updated)

# %%
# Optionally, update the package wide defaults (`massphoto.defaults.mass_calibration`) with the fitted calibration.
calibration = mp.mass_calibration.fit(
    mpfit_calib.peaks["mu"][:n_values_for_fit],
    expected_masses_kDa[:n_values_for_fit],
    update_defaults=True,
)
# %%
# Plot mass calibration
# ---------------------
#
# This plot shows the fitted peak positions (with width estimates) and the
# resulting calibration curve.


plt.figure(figsize=(6, 3))
plt.errorbar(
    x=mpfit_calib.peaks["mu"][:n_values_for_fit],
    y=expected_masses_kDa[:n_values_for_fit],
    xerr=mpfit_calib.peaks["sigma"][:n_values_for_fit],
    fmt="|",
    color="black",
    capsize=0,
    linewidth=2,
    label="peaks used for fit",
)
x_fit = np.array([bins[0], bins[-1]], dtype=float)
plt.plot(x_fit, calibration(x_fit), lw=1, label="linear fit")
plt.xlabel("Contrast c")
plt.ylabel("Expected mass m (kDa)")
plt.title("Mass calibration")
plt.legend()
plt.tight_layout()

# %%
# Apply calibration
# -----------------
#
# Apply the mass calibration to add a ``masses_kDa`` column to the fit results.

fitres_calib = mp.mass_calibration.apply(fitres_calib, calibration)

fitres_calib[["frame", "x", "y", "contrasts", "masses_kDa"]].head()

# %%
# Mass distribution after calibration
# -----------------------------------
#
# Finally, inspect the calibrated mass distribution. The expected ladder masses
# are shown as dashed vertical lines.
plt.figure(figsize=(6, 3))
plt.hist(
    fitres_calib["masses_kDa"],
    bins=np.arange(-100, 700, 2.5),
    edgecolor="none",
    label="Single-protein events",
)
for mass in expected_masses_kDa:
    plt.axvline(
        mass,
        color="red",
        ls="--",
        lw=1,
        label="Expected masses" if mass == expected_masses_kDa[0] else None,
    )
plt.xlabel("Mass (kDa)")
plt.ylabel("Counts")
plt.title("Mass distribution after calibration")
plt.legend()
plt.tight_layout()

# %%
