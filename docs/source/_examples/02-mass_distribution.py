"""
Mass distribution
=================

This tutorial shows how to compute and fit mass distributions from a movie once
a contrast-to-mass calibration is available:

1. Download an example movie.
2. Set a contrast-to-mass calibration.
3. Run detection + PSF fitting to obtain a table of event masses.
4. Fit peaks in the mass histogram and visualize the result.
"""

# %%
# Imports
# -------
# We first import necessary packages and modules.

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

# %%
# Download example data
# ---------------------
#
# This tutorial uses a short example movie from Zenodo. You can replace this
# block with a direct file path once you have a local movie.
from zenodozipdownloader import ZenodoZipDownloader

import massphoto as mp

files = ZenodoZipDownloader(
    doi="10.5281/zenodo.15800729",
    download_dir="example_data",
).download("Figure 2.zip/006_dynamin_6.mp")
if not files:
    raise RuntimeError(
        "Failed to download example data. Please download it manually from https://zenodo.org/record/15800729 "
        "or set path_movie to your own movie."
    )
path_movie = files[0]

print(f"Using movie: {path_movie}")

# %%
# Analyse with default parameters
# -------------------------------
#
# Set mass calibration parameters. See the mass calibration example for how
# to obtain these from a calibration movie:
# :ref:`sphx_glr_auto_examples_01-mass_calibration.py`.

# Disable progress bars for cleaner output in this tutorial
mp.defaults.set("progress_bar", False)

# Set contrast-to-mass calibration parameters (intercept, slope). Adjust to match your calibration.
mp.defaults.set("mass_calibration", (-5.2, 3.1e4))

# Run detection and fitting. Keep the ratiometric frames for inspecting events.
# Sequential execution limits peak memory while retaining those frames.
fitres, movie_r = mp.fit(path_movie, return_ratiometric=True, scheduler="synchronous")
fitres[["frame", "x", "y", "contrasts", "masses_kDa"]].head().round({"x": 2, "y": 2, "contrasts": 5, "masses_kDa": 1})

# %%
# Fit mass distribution
histfit = mp.distfit.fit(fitres["masses_kDa"], bins=np.arange(-100, 700, 2.5), include_background=True)
histfit.peaks[["mu", "sigma", "A"]].round(2)

# %%
# Plot results
plt.figure(figsize=(6, 3))
plt.bar(histfit.bin_centers, histfit.counts, width=histfit.bin_widths, edgecolor="none", label="Data")
plt.plot(histfit.model_x, histfit.model_y, color="black", lw=1, label="Fit")
plt.xlabel("Mass (kDa)")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()

# %%
# Analyse with custom parameters
# ------------------------------
#

# Read the movie
movie, metadata = mp.read_movie(path_movie, convert_metadata=True)

# sphinx_gallery_start_ignore
# Homepage panels are generated only in the documentation source directory.
# Save and close them explicitly so they are not captured as tutorial figures.
from pathlib import Path  # noqa: E402
import json  # noqa: E402


def _save_homepage_panels(movie, movie_r, events, histogram, output_dir):
    """Generate the four homepage workflow panels and provenance file.

    Creates raw.png, events.png, profile.svg, histogram.svg, and provenance.json
    under `auto_examples/images/homepage` for display in the documentation landing page.
    Saves and closes each dedicated figure explicitly so they are not captured
    as tutorial gallery outputs.
    """
    from matplotlib.patches import Rectangle

    output_dir.mkdir(parents=True, exist_ok=True)
    navg = int(mp.defaults.get("calc_ratiometric.navg", mp.defaults.get("navg")))
    half_width = 20
    candidates = events.loc[
        events["masses_kDa"].between(160, 200)
        & (events["contrasts"] > 0)
        & (events["frame"] > 2 * navg)
        & (events["frame"] < movie.shape[0] - 2 * navg)
        & events["x"].between(half_width + 1, movie.shape[2] - half_width - 2)
        & events["y"].between(half_width + 1, movie.shape[1] - half_width - 2)
    ].sort_values(["r2_fit", "frame", "y", "x"], ascending=[False, True, True, True])
    event = None
    for _, candidate in candidates.iterrows():
        nearby = (np.abs(events["frame"] - candidate.frame) <= 2 * navg) & (
            np.hypot(events["x"] - candidate.x, events["y"] - candidate.y) < 16
        )
        if nearby.sum() == 1:
            event = candidate
            break
    if event is None:
        raise RuntimeError("No isolated event near 180 kDa is available for the homepage panels.")

    frame, row, col = int(event.frame), int(round(event.y)), int(round(event.x))
    frames = np.array([frame - navg, frame - navg // 2, frame])
    ys, xs = slice(row - half_width, row + half_width + 1), slice(col - half_width, col + half_width + 1)
    raw = np.asarray(movie[frames, ys, xs])
    ratio = np.asarray(movie_r[frames, ys, xs])
    if not np.isfinite(ratio).all():
        raise RuntimeError("Homepage event crop contains invalid ratiometric pixels.")

    dark_blue = "#002147"
    accent_red = "#e22319"

    def save(fig, name):
        fig.savefig(output_dir / name, dpi=180, metadata={"Date": None} if name.endswith(".svg") else None)
        plt.close(fig)

    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "massphoto-homepage",
        }
    ):
        for data, cmap, name in ((raw, "gray", "raw.png"), (ratio, "viridis", "events.png")):
            fig, ax = plt.subplots(figsize=(2.8, 2.2))
            fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
            low, high = np.quantile(data, [0.001, 0.999])
            size = data.shape[1]
            for i, pixels in enumerate(data):
                offset = 5 * i
                alpha = 1.0 if i == len(data) - 1 else 0.6
                ax.imshow(
                    pixels,
                    cmap=cmap,
                    vmin=low,
                    vmax=high,
                    interpolation="nearest",
                    extent=(offset, offset + size, offset + size, offset),
                    zorder=2 * i,
                    alpha=alpha,
                )
                ax.add_patch(
                    Rectangle(
                        (offset, offset), size, size, fill=False, edgecolor=dark_blue, linewidth=2, zorder=2 * i + 1
                    )
                )
            if name == "events.png":
                # Coordinate convention matches imshow's pixel centres and origin.
                cx = offset + half_width + 0.5 + event.x - col
                cy = offset + half_width + 0.5 + event.y - row
                ax.add_patch(
                    Rectangle((cx - 7, cy - 7), 14, 14, fill=False, edgecolor="white", linewidth=0.8, zorder=10)
                )
            ax.set(xlim=(-2, offset + size + 2), ylim=(offset + size + 2, -2))
            ax.set_axis_off()
            save(fig, name)

        # Sample the fitted PSF along the same integer row as the measured data.
        # Both coordinates include the fitted subpixel offsets.
        pixel_x = np.arange(col - 13, col + 14)
        measured = np.asarray(movie_r[frame, row, pixel_x])
        model_x = np.linspace(pixel_x[0], pixel_x[-1], 400)
        psf = mp.PSF2D()
        model = event.contrasts * psf.psf_fun(model_x - event.x, np.full_like(model_x, row - event.y), **psf.psf_params)
        fig, ax = plt.subplots(figsize=(2.8, 2.2), layout="constrained")
        ax.axhline(0, color="0.8", lw=0.6)
        ax.step(pixel_x - event.x, measured * 1000, where="mid", color="0.45", lw=1, label="Data")
        ax.plot(model_x - event.x, model * 1000, color=accent_red, lw=1.5, label="Fit")
        ax.set(xlabel="Position (px)", ylabel="Contrast (×10⁻³)", xlim=(-13, 13), xticks=[-10, 0, 10])
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.legend(frameon=False, fontsize=11, loc="upper right", handlelength=1)
        save(fig, "profile.svg")

        fig, ax = plt.subplots(figsize=(2.8, 2.2), layout="constrained")
        ax.bar(
            histogram.bin_centers,
            histogram.counts,
            width=histogram.bin_widths,
            color=dark_blue,
            edgecolor="none",
            alpha=0.6,
        )
        ax.plot(histogram.model_x, histogram.model_y, color=accent_red, lw=1.2)
        ax.set(xlabel="Mass (kDa)", ylabel="Counts", xlim=(-100, 700), xticks=[0, 200, 400, 600])
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        save(fig, "histogram.svg")

    # A build artifact records the measured event and plotting inputs for review.
    provenance = {
        "dataset": "https://doi.org/10.5281/zenodo.15800729",
        "movie": "Figure 2.zip/006_dynamin_6.mp",
        "event": {key: float(event[key]) for key in ("frame", "x", "y", "contrasts", "masses_kDa", "r2_fit")},
        "frames": frames.tolist(),
        "crop_y": [ys.start, ys.stop],
        "crop_x": [xs.start, xs.stop],
        "navg": navg,
        "mass_calibration": [-5.2, 3.1e4],
        "event_count": len(events),
        "histogram_count": int(histogram.counts.sum()),
        "profile_x": (pixel_x - event.x).tolist(),
        "profile_data": measured.tolist(),
        "preview": events[["frame", "x", "y", "contrasts", "masses_kDa"]]
        .head()
        .round({"x": 2, "y": 2, "contrasts": 5, "masses_kDa": 1})
        .to_dict(orient="records"),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if Path.cwd().name == "_examples" and (Path.cwd().parent / "conf.py").is_file():
    _save_homepage_panels(movie, movie_r, fitres, histfit, Path.cwd().parent / "auto_examples/images/homepage")
del movie_r
# sphinx_gallery_end_ignore

# Inspect the metadata
print("Metadata:")
pprint(metadata)

# Check default thresholds for detection
print("Default detection thresholds:", mp.defaults.get("event_correlation_step.thresholds"))

# We could adjust these if we want to be more or less sensitive, for all following analyses. For example:
# mp.defaults.set("event_correlation_step.thresholds", [0.5, 0.4, 0.001])

# Check default averaging window
print("Default navg:", mp.defaults.get("navg"))

# We could adjust the default like this:
# mp.defaults.set("navg", 5)


##
# We can also specify detection arguments here, which will only apply to this fit call.
fitres = mp.fit(
    movie,
    movie_correct_drift=True,
    ratiometric_args={
        "navg": 5,
    },
    detection_args={
        "thresholds": [
            0.5,  # temporal threshold
            0.4,  # spatial threshold
            0.001,  # contrast threshold
        ],
        # navg for the temporal filter (longer than navg for the ratiometric filter is often beneficial)
        "correlation_step_args": {"navg": 10},
    },
)

# We decreased the integration time and lowered the thresholds, which leads to more events being detected, but also more noise.
# To remove some of the noise, we can filter based on fit quality or mass uncertainty.
filter_str = "masses_kDa_se < 10 & r2_fit > 0.4"
histfit = mp.distfit.fit(
    fitres.query(filter_str),
    bins=np.arange(-100, 700, 2.5),
    initial_peaks=(90, 180, 360, 540),
    include_background=False,
)
histfit.peaks[["mu", "sigma", "A"]].round(2)

# %%
# Compare the unfiltered distribution, selected events, and fitted peaks.

plt.figure(figsize=(6, 3))
plt.hist(fitres["masses_kDa"], bins=histfit.bin_edges, edgecolor="none", facecolor="orange", label="Data (unfiltered)")
plt.bar(histfit.bin_centers, histfit.counts, width=histfit.bin_widths, edgecolor="none", label="Data")
plt.plot(histfit.model_x, histfit.model_y, color="black", lw=1, label="Fit")
plt.xlabel("Mass (kDa)")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()

# %%
