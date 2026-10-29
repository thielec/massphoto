Custom Pipelines
================

The pipeline module provides a general chunked pipeline entry point
(:func:`massphoto.process`) and workflow helpers.

By default, parallel outputs carry an ``analysis_record`` in
``DataFrame.attrs`` (``attach_record=True`` in high-level APIs). Use
:func:`massphoto.resolve_record` to turn that record back into a defaults-aware
replay callable.

Pipeline Type Markers
---------------------

When assembling custom pipelines with :func:`massphoto.process`, function parameters
use type annotations to declare what data representation they consume (see
:ref:`Define a custom detection function <define_custom_detection_function>` in the tutorial gallery):

.. py:data:: massphoto.NativeMovie
   :type: NewType("NativeMovie", np.ndarray)

   Annotates parameters that expect raw camera movie frames ``(frames, y, x)``.
   The chunked pipeline runner passes slices of the original input movie to these parameters.

.. py:data:: massphoto.RatiometricMovie
   :type: NewType("RatiometricMovie", np.ndarray)

   Annotates parameters that expect background-subtracted ratiometric images.
   When requested, the pipeline ensures ratiometric frames are generated before invoking the step.

.. py:data:: massphoto.EventList
   :type: NewType("EventList", pd.DataFrame)

   Annotates functions returning or processing detected events in a :class:`pandas.DataFrame`
   containing required spatial coordinates (``x``, ``y``) and frame indices (``frame``).

Module Reference
----------------

.. automodule:: massphoto.pipeline
   :members:
   :undoc-members:
   :show-inheritance:
