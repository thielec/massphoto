Histogram Fitting
=================

.. py:module:: massphoto.distfit

Use :func:`massphoto.distfit.fit` for raw samples or precomputed histograms.
It returns :class:`massphoto.distfit.FitResult`. Select the fitting engine with
``method="hybrid"`` (default) or ``method="full"``.

.. currentmodule:: massphoto.distfit

.. autofunction:: fit

.. autoclass:: FitResult
   :members:
