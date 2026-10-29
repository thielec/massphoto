Examples
========

These runnable examples demonstrate the core workflows of massphoto, organised from fundamental single-molecule analysis to diagnostics and pipeline customisation:

* **Core analysis (1-3)**: Contrast-to-mass calibration, mass distribution fitting, and event landing rate over time.
* **Diagnostics & quality control (4-6)**: Stage drift compensation, noise floor evaluation with averaging windows, and movie properties over time.
* **Pipeline & extensibility (7-9)**: Step-by-step walk-through, custom candidate detection, and reproducible analysis records with replay.

.. admonition:: Benchmark Datasets & Best Practice Protocol

   The sample data used across these examples originates from the benchmark mass photometry
   recordings archived on `Zenodo (10.5281/zenodo.15800729) <https://doi.org/10.5281/zenodo.15800729>`_,
   accompanying Kratochvíl et al., *Nat. Protoc.* (2025), `doi:10.1038/s41596-025-01255-4 <https://doi.org/10.1038/s41596-025-01255-4>`_.

   When running these examples for the first time, required data files (~1 GB total) are downloaded automatically
   from Zenodo and cached locally. An active internet connection is required on the first run;
   if a download times out due to network issues, re-running the command will retry files that were not fully downloaded.
