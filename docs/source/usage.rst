Usage Guide
===========

**massphoto** turns camera movies into tables of single-molecule landing events.
The top-level API provides a complete analysis pipeline while the underlying
functions remain accessible in the domain namespaces. This guide explains the
analysis workflow, the available entry points, and how to adjust settings for your data.

Use this guide to understand the data, choose an entry point, and decide what to
adjust. The :doc:`examples <auto_examples/index>` contain the runnable workflows,
sample data, and plots.

Choose a starting point
--------------------------

Depending on your goal, start with one of the two primary workflows:

* **Calibrate an instrument**: If you have standard proteins (e.g. BSA, MFP1)
  and need to determine contrast-to-mass conversion factors, follow the
  calibration workflow.
* **Measure an unknown sample**: If you already have calibration parameters, follow the mass distribution
  workflow.

.. minigallery:: _examples/01-mass_calibration.py _examples/02-mass_distribution.py

A calibration is specific to your instrument and measurement conditions. Example coefficients
and filtering thresholds illustrate a workflow; they are not universal settings.

What happens to a movie
---------------------------

Raw mass photometry images contain a strong, spatially patterned background caused
by the scattered light from the coverglass-water interface. The usual analysis
looks for the small intensity changes associated with particles binding to or
detaching from the surface:

1. **Reveal changes.** Ratiometric processing compares temporal averages before
   and after each frame to suppress the static background. Optional drift
   correction aligns the raw images before this comparison.
2. **Find candidates.** The default detector combines temporal step correlation,
   spatial PSF correlation, and a contrast threshold. Detection selects
   candidate events; it does not yet provide the final fitted contrast.
3. **Measure each event.** Point spread function (PSF) fitting estimates event
   positions and contrasts from the ratiometric images. Contrast describes an
   intensity change, rather than a mass.
4. **Calibrate and summarise.** A contrast-to-mass calibration converts fitted
   contrasts to masses. Inspect the event table, select suitable events, and
   plot or fit the resulting distribution.

:func:`massphoto.fit` manages the movie processing, detection, and event
fitting. Its default postprocessing also adds masses when a calibration is set.
Without a calibration, fitted contrasts are still available.
Next, :func:`massphoto.distfit.fit` can fit peaks in a
**distribution of events**, rather than fitting individual event images.

For intermediate images and a walkthrough of each stage, see the
:ref:`step-by-step pipeline <sphx_glr_auto_examples_07-step_by_step_pipeline.py>`.

Choose how much control you need
------------------------------------

After ``import massphoto as mp``, most analyses can stay at the top level.
Domain namespaces group alternative detection strategies, calibration, and
histogram fitting.

.. list-table:: Entry points by task
   :header-rows: 1
   :widths: 48 52

   * - Task
     - Entry point
   * - Detect events and fit their PSF images
     - :func:`massphoto.fit`
   * - Detect candidates without PSF fitting
     - :func:`massphoto.detect`
   * - Assemble custom steps in the chunked pipeline
     - :func:`massphoto.process`
   * - Inspect a movie and its metadata
     - :func:`massphoto.read_movie`
   * - Work directly with individual processing stages
     - :func:`massphoto.calc_ratiometric`, :func:`massphoto.correct_drift`, :func:`massphoto.fit_event`
   * - Choose a detection strategy
     - :mod:`massphoto.detection`
   * - Fit or apply a contrast-to-mass calibration
     - :mod:`massphoto.mass_calibration`
   * - Fit contrast or mass distributions
     - :mod:`massphoto.distfit`

:func:`massphoto.fit`, :func:`massphoto.detect`, and :func:`massphoto.process` accept a file path or a movie array with axes ``(frames, y, x)`` and
manage chunked processing. You can replace ``detection_func`` or ``fit_func`` in :func:`massphoto.fit`
without assembling the whole pipeline yourself. The
:ref:`custom detection example <sphx_glr_auto_examples_08-custom_detection.py>`
shows this extension point in use.

Settings that affect the result
-------------------------------

Most users will use the :func:`massphoto.fit` entry point. The defaults are tuned for typical mass photometry movies, but you
may need to adjust them for your data. The following settings are most commonly needed:

**Averaging and detection.** ``ratiometric_args`` configures background
subtraction; ``detection_args`` configures candidate selection; ``fit_args``
configures event fitting.
The averaging window ``navg`` trades shot-noise suppression (:math:`\text{SNR} \propto \sqrt{n_\text{avg}}`)
against temporal resolution: shorter windows allow for a higher density of events,
while longer windows can improve signal-to-noise.
The default detector (:func:`~massphoto.detection.event_correlation_step`) applies
three thresholds ``[step_corr, norm_corr, contrast]`` (default ``[0.6, 0.5, 0.001]``):
temporal step correlation on the native movie, normalised spatial correlation with
the PSF template, and raw ratiometric contrast.
Use the :ref:`mass distribution example <sphx_glr_auto_examples_02-mass_distribution.py>`
for parameter adjustments, and the
:ref:`shot noise example <sphx_glr_auto_examples_05-shot_noise_curve.py>`
for the relationship between averaging and noise.
For experimental guidelines and recommended settings, see
`Kratochvíl et al. (2025) <https://doi.org/10.1038/s41596-025-01255-4>`_.

**Image corrections and diagnostics.** ``movie_correct_drift=True`` compensates for lateral
motion before ratiometric processing; the
:ref:`drift correction example <sphx_glr_auto_examples_04-drift_correction.py>`
shows how to inspect that motion. Frame-level quality diagnostics (intensity,
sharpness, signal, and drift speed) can be tracked over time with ``compute_movie_metrics=True``,
as demonstrated in the
:ref:`movie properties example <sphx_glr_auto_examples_06-movie_metrics.py>`.
Illumination correction, enabled by default
in :func:`massphoto.calc_ratiometric`, accounts for spatial variation in the
illumination and background when estimating contrast. These corrections address
different sources of artefacts often encountered in mass photometry.

**PSF and mass calibration.** The PSF model describes the event image and is used to
estimate the contrast. Mass calibration maps the fitted contrast to mass. Changing
a mass calibration does not change the PSF model. See the
:ref:`calibration example <sphx_glr_auto_examples_01-mass_calibration.py>`.

**One analysis or many.** Explicit arguments apply to one call and take
precedence over stored defaults. ``mp.defaults`` holds shared settings such as
calibration; ``mp.defaults.context(...)`` applies temporary settings and restores
them afterwards. Defaults may be global (``navg``) or function-specific
(``event_correlation_step.thresholds``); function-specific keys take precedence
over global keys. See :doc:`api/defaults` for configuration examples and lookup
rules.

Understand the event table
------------------------------

``mp.fit`` returns a pandas DataFrame with one row per fitted event. Select
columns, filter rows, and combine results using normal pandas operations.

.. list-table:: Useful result columns
   :header-rows: 1
   :widths: 32 68

   * - Columns
     - Meaning
   * - ``frame``, ``x``, ``y``
     - Event frame and position in the movie.
   * - ``contrasts``
     - Contrast estimated by fitting the event image.
   * - ``contrasts_se``, ``r2_fit``
     - Contrast standard error and fit quality, when provided by the fitting method.
   * - ``masses_kDa``
     - Calibrated mass, when a mass calibration is available.
   * - ``masses_kDa_se``
     - Propagated contrast uncertainty in mass units, when contrast error and a calibration derivative are available.

Inspect both selected and unfiltered distributions when choosing quality
filters: removing events changes the population being analysed and might introduce
bias. The mass uncertainty column propagates contrast error; it does not include
uncertainty in the calibration coefficients.

To filter out poor fits and high-uncertainty events before distribution fitting:

.. code-block:: python

    # Quality filtering (e.g. fit quality R² and mass uncertainty)
    events_filtered = events.query("r2_fit > 0.6 and masses_kDa_se < 5.0")

.. _usage-analysis-record:

Keep analyses reproducible
------------------------------

``mp.fit``, ``mp.detect``, and ``mp.process`` attach an ``analysis_record`` to
``DataFrame.attrs`` by default. :func:`massphoto.resolve_record` creates a replay
callable using the recorded pipeline settings that lets you reproduce the analysis
on the same or a different movie. Follow the
:ref:`analysis record and replay example <sphx_glr_auto_examples_09-analysis_record.py>`
for inspecting, replaying, and saving a record.
