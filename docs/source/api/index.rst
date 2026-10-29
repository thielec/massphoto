API Reference
=============

Detailed reference documentation for the **massphoto** package, organised by workflow stage.

.. tip::

   Looking for a specific function or class? Consult the :ref:`genindex` (alphabetical list) or the :ref:`modindex` (Python module hierarchy).

Primary Workflows & Quantification
----------------------------------

High-level entry points, calibration, distribution fitting, and package-wide configuration.

.. toctree::
   :maxdepth: 1

   massphoto
   mass_calibration
   distfit
   events
   defaults

Core Pipeline & Imaging
-----------------------

Event candidate detection, point spread function (PSF) fitting, drift compensation, and image filtering.

.. toctree::
   :maxdepth: 1

   detection
   fitting
   psf_definition
   drift_correction
   movie_process
   filter

File I/O & Advanced Infrastructure
----------------------------------

Reading and writing supported camera formats, metadata handling, and custom execution pipelines.

.. toctree::
   :maxdepth: 1

   read
   write
   metadata
   pipeline

Indices
-------

* :ref:`genindex` — Alphabetical index of all symbols and definitions
* :ref:`modindex` — Python module hierarchy
* :ref:`search` — Search documentation
