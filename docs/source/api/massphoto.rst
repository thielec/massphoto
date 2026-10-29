Top-level API
=============

The primary entry points of massphoto are available directly at the package root
after ``import massphoto as mp``.

This page summarises all top-level functions and classes and links to their full reference documentation.

.. currentmodule:: massphoto

Primary Functions
-----------------

.. autosummary::
   :nosignatures:

   fit
   read_movie
   calc_ratiometric

Event Detection & PSF Fitting
------------------------------

.. autosummary::
   :nosignatures:

   detect
   detect_on_map
   fit_event
   PSF2D
   fit_psf_params
   get_psf_thumbnails

Movie Transformations & Drift
-----------------------------

.. autosummary::
   :nosignatures:

   correct_drift
   calc_differential
   calc_mobileframes
   calc_frame_sharpness
   normalize_frames
   rebin_movie

Event Metrics & Metadata
------------------------

.. autosummary::
   :nosignatures:

   add_local_density
   add_nearest_neighbour
   annotate_frame_metrics
   convert_metadata

Pipeline Orchestration & Replay
--------------------------------

.. autosummary::
   :nosignatures:

   process
   resolve_record

File Export
-----------

.. autosummary::
   :nosignatures:

   save_mp
   save_h5
   save_tiff
