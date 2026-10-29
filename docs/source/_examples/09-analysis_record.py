"""
Analysis record and replay
==========================

Every call to :func:`massphoto.fit`, :func:`massphoto.detect`, and
:func:`massphoto.process` automatically attaches an analysis record to the
returned DataFrame. This tutorial shows how to:

1. Run an analysis with custom parameters on one movie.
2. Inspect the attached ``analysis_record``.
3. Resolve the record into a replay callable and apply it to a second movie.

.. seealso::

    :ref:`sphx_glr_auto_examples_02-mass_distribution.py` example for details on
    custom detection and fitting parameters.

    :ref:`Analysis Record and Reproducibility <usage-analysis-record>` section in
    the usage guide.
"""

# sphinx_gallery_thumbnail_number = 1
# %%
# Record once, replay on another movie
# ----------------------------------------
#
# An analysis record captures the settings used for Movie 1. Replay applies those
# settings to Movie 2, producing a separate results table for the new data.
# The orange path shows how the settings pass between the two analyses.

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.set_axis_off()

nodes = [
    (0.15, 0.80, "Movie 1", "#e8f0f8", "#456785"),
    (0.50, 0.80, "Analysis", "#e8f0f8", "#456785"),
    (0.85, 0.80, "Results 1", "#e8f0f8", "#456785"),
    (0.50, 0.49, "Analysis\nrecord", "#fff0d6", "#b56916"),
    (0.15, 0.18, "Movie 2", "#e8f0f8", "#456785"),
    (0.50, 0.18, "Replay", "#fff0d6", "#b56916"),
    (0.85, 0.18, "Results 2", "#e8f0f8", "#456785"),
]
for x, y, label, fill, edge in nodes:
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=15,
        color="#23364a",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": fill, "edgecolor": edge, "linewidth": 1.5},
    )

for y in (0.80, 0.18):
    for start, end in ((0.28, 0.37), (0.63, 0.72)):
        ax.annotate(
            "",
            xy=(end, y),
            xytext=(start, y),
            arrowprops={"arrowstyle": "-|>", "color": "#456785", "lw": 2, "mutation_scale": 16},
        )

for start, end in ((0.71, 0.61), (0.37, 0.27)):
    ax.annotate(
        "",
        xy=(0.50, end),
        xytext=(0.50, start),
        arrowprops={"arrowstyle": "-|>", "color": "#b56916", "lw": 2.5, "mutation_scale": 18},
    )

plt.show()
# sphinx_gallery_end_ignore

# %%
# Imports
# -------
# We first import necessary packages and modules.

from pprint import pprint

from zenodozipdownloader import ZenodoZipDownloader

import massphoto as mp

# %%
# Download example data
# ---------------------
#
# This tutorial uses two short movies from Zenodo: a calibration movie
# (``005_mfp1_500x.mp``) and a dynamin movie (``006_dynamin_6.mp``). We analyse
# the calibration movie with custom parameters and replay the same analysis on
# the dynamin movie.
downloader = ZenodoZipDownloader(doi="10.5281/zenodo.15800729", download_dir="example_data")

files_calib = downloader.download(inner_pattern="005_mfp1_500x.mp")
files_dynamin = downloader.download("Figure 2.zip/006_dynamin_6.mp")

if not files_calib or not files_dynamin:
    raise RuntimeError(
        "Failed to download example data. Please download it manually from https://zenodo.org/record/15800729."
    )

path_movie_1 = files_calib[0]
path_movie_2 = files_dynamin[0]

print(f"Movie 1 (calibration): {path_movie_1}")
print(f"Movie 2 (dynamin)    : {path_movie_2}")

# %%
# Analyse Movie 1 with custom parameters
# ----------------------------------------
#
# We run :func:`massphoto.fit` with a shorter averaging window and lower
# detection thresholds. The ``analysis_record`` is attached to the result
# DataFrame by default (``attach_record=True``).

# Disable progress bars for cleaner output in this tutorial
mp.defaults.set("progress_bar", False)

movie_1, _ = mp.read_movie(path_movie_1)

fitres_1 = mp.fit(
    movie_1,
    ratiometric_args={"navg": 4},
    detection_args={
        "thresholds": [0.5, 0.4, 0.001],
        "correlation_step_args": {"navg": 10},
    },
    movie_correct_drift=True,
)

fitres_1[["frame", "x", "y", "contrasts"]].head()

# %%
# Inspect the analysis record
# ---------------------------
#
# The record is stored in ``DataFrame.attrs`` under the key
# ``"analysis_record"``. Schema version 2 records the origin (``fit``, ``detect``,
# or ``process``), captured call kwargs, a snapshot of ``massphoto.defaults``,
# and basic movie metadata. Replay requires the movie and any precomputed events
# again.

record = fitres_1.attrs["analysis_record"]

print("schema     :", record["schema_version"])
print("origin     :", record["origin"])
print("movie_info :", record["movie_info"])

# %%
# The ``origin_kwargs`` entry captures effective arguments to :func:`massphoto.fit`,
# including resolved defaults, while excluding the movie and precomputed events.
print("\norigin_kwargs:")
pprint(record["origin_kwargs"])

# %%
# ``defaults_state`` is a full snapshot of ``massphoto.defaults`` at the time the
# analysis ran — including any package-wide settings such as PSF parameters or
# mass calibration. Here we just confirm the key is present:
print(f"\ndefaults_state: {len(record['defaults_state'])} keys")

# %%
# Replay the analysis on Movie 2
# --------------------------------
#
# :func:`massphoto.resolve_record` converts the stored record into:
#
# - ``defaults_overrides``: the merged defaults state (defaults snapshot + origin
#   kwargs) that will reproduce the run.
# - ``replay``: a defaults-wrapped :func:`massphoto.fit` callable preloaded
#   with the stored state. Only the new movie needs to be supplied.

overrides, replay = mp.resolve_record(fitres_1)

movie_2, _ = mp.read_movie(path_movie_2)
fitres_2 = replay(movie_2)

fitres_2[["frame", "x", "y", "contrasts"]].head()

# %%
# Confirm that the same custom parameters were used.

record_2 = fitres_2.attrs["analysis_record"]
print("Replayed origin_kwargs match original:", record_2["origin_kwargs"] == record["origin_kwargs"])

# %%
# Persistence note
# ----------------
#
# ``DataFrame.attrs`` is an in-memory dictionary. Tabular exports such as CSV,
# XLSX, and ``DataFrame.to_hdf`` **do not** preserve it. To save the DataFrame
# together with its ``attrs`` (including ``analysis_record`` and
# ``frame_metrics``), use :meth:`pandas.DataFrame.to_pickle` and
# :func:`pandas.read_pickle`:
#
# .. code-block:: python
#
#     import pandas as pd
#
#     fitres_1.to_pickle("fitres.pkl")
#     loaded = pd.read_pickle("fitres.pkl")
#     _, replay = mp.resolve_record(loaded)
#     fitres_replayed = replay(movie_1)
#
# If you export the event table to another format and want to preserve the record,
# save the record dictionary alongside the data explicitly:
#
# .. code-block:: python
#
#     import pickle
#
#     # Save the record separately
#     with open("fitres_record.pkl", "wb") as f:
#         pickle.dump(fitres_1.attrs["analysis_record"], f)
#
#     # Later, load and replay
#     with open("fitres_record.pkl", "rb") as f:
#         saved_record = pickle.load(f)
#
#     _, replay = mp.resolve_record(saved_record)
#     fitres_replayed = replay(movie_1)
#
# To opt out of attaching the record entirely, pass ``attach_record=False``:
#
# .. code-block:: python
#
#     fitres = mp.fit(movie_1, attach_record=False)

# %%
