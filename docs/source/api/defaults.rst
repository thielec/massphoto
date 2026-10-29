Managing Defaults
=================

This page documents :data:`massphoto.defaults`, backed by
:class:`massphoto.defaults_store.DefaultsStore`, which provides centralized,
reproducible configuration for processing pipelines, detection filters, and calibration.

User Guide: Analysis Configuration
----------------------------------

``massphoto.defaults`` allows setting parameters once and having them automatically
injected across the analysis pipeline.

Common Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Primary Analysis Settings
   :header-rows: 1
   :widths: 28 15 15 42

   * - Key
     - Type
     - Default
     - Description / Physical Context
   * - ``navg``
     - ``int``
     - ``10``
     - Temporal half-window (frames) for ratiometric background suppression.
   * - ``psf_size``
     - ``int``
     - ``11``
     - Square thumbnail size (pixels) used for PSF cross-correlation and fitting.
   * - ``psf_params``
     - ``dict``
     - *(see below)*
     - Interferometric PSF parameters (``a12``, ``w``, ``s``, ``r_mask``, ``r_aperture``).
   * - ``mass_calibration``
     - ``tuple | None``
     - ``None``
     - ``(intercept, slope)`` mapping contrast to mass (kDa): :math:`m = c_0 + c_1 \cdot C`.
   * - ``event_correlation_step.thresholds``
     - ``list``
     - ``[0.6, 0.5, 0.001]``
     - Detection thresholds: ``[step_correlation, spatial_PSF_correlation, contrast]``.
   * - ``fit.chunk_size``
     - ``int``
     - ``1000``
     - Number of frames per chunk for parallel Dask execution.

Default PSF parameters on import::

    {
        "psf_params": {
            "a12": -6.3337259134,
            "w": 2.1370299329,
            "s": 5.0425755943,
            "r_mask": 1.5 * 2.5e-3,
            "r_aperture": 8.52e-3,
        },
        "psf_fun": None,
        "psf_size": 11,
        "psf_cache_divider": 128,
        "psf_cache_interpolate": True,
        "mass_calibration": None,
    }

Setting and Querying Defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`~massphoto.defaults_store.DefaultsStore.set` and :meth:`~massphoto.defaults_store.DefaultsStore.get`
to inspect and modify parameters. Keys fall into two categories:

**Global keys** apply across every pipeline stage that accepts that parameter:

.. code-block:: python

    import massphoto as mp

    # Set calibration and averaging window globally
    mp.defaults.set("mass_calibration", (-5.2, 31000.0))
    mp.defaults.set("navg", 15)

    # Query current value
    current_navg = mp.defaults.get("navg")

**Function-specific keys** use the ``<function_name>.<parameter>`` format to target a specific stage
without affecting others. Function-specific keys always take precedence over global keys.
For example, to configure detection thresholds for :func:`~massphoto.detection.event_correlation_step`:

.. code-block:: python

    # Set detection thresholds: [step_correlation, spatial_PSF_correlation, contrast]
    mp.defaults.set("event_correlation_step.thresholds", [0.55, 0.45, 0.0008])

    # Query current thresholds
    thresholds = mp.defaults.get("event_correlation_step.thresholds")

Similarly, execution chunks for parallel processing can be configured via ``mp.defaults.set("fit.chunk_size", 500)``.

Scoped Temporary Overrides
~~~~~~~~~~~~~~~~~~~~~~~~~~

To test parameters on a single movie without altering global defaults, use the
context manager:

.. code-block:: python

    with mp.defaults.context({
        "fit.chunk_size": 500,
        "event_correlation_step.thresholds": [0.55, 0.45, 0.0008],
    }):
        events = mp.fit("movie.mp")

When the context block exits, previous settings are restored automatically.

Saving and Loading Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Persist settings to a YAML file for reproducible sharing across lab workstations:

.. code-block:: python

    # Save to file
    mp.defaults.save("lab_instrument_defaults.yml")

    # Restore on another machine
    mp.defaults.load("lab_instrument_defaults.yml")

Advanced Developer Reference
----------------------------

This section documents internal mechanics, function decorators, and worker state serialization
for developers and advanced pipeline customizers who want to equip their own functions with
automatic defaults injection.

API Methods
~~~~~~~~~~~

Common methods on :data:`massphoto.defaults`:

- :meth:`massphoto.defaults_store.DefaultsStore.get_state`
- :meth:`massphoto.defaults_store.DefaultsStore.set_state`
- :meth:`massphoto.defaults_store.DefaultsStore.get`
- :meth:`massphoto.defaults_store.DefaultsStore.set`
- :meth:`massphoto.defaults_store.DefaultsStore.update`
- :meth:`massphoto.defaults_store.DefaultsStore.pop`
- :meth:`massphoto.defaults_store.DefaultsStore.save`
- :meth:`massphoto.defaults_store.DefaultsStore.load`

Decorators for Custom Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions can be decorated to automatically register their parameters and inject defaults:

.. code-block:: python

    from massphoto import defaults

    @defaults.sync_defaults(prefix='test')
    def fn(a=1, psf_size=11):
        """Registers defaults from signature but does not inject at call-time."""
        return a, psf_size

    @defaults.sync_apply_defaults(prefix='test')
    def fn2(a=1, psf_size=11):
        """Registers defaults and injects values from the store at call-time."""
        return a, psf_size

    # After updating the store, fn2 will pick up the value automatically
    defaults.set('test.psf_size', 15)
    assert fn2() == (1, 15)

Removing keys and un-shadowing defaults:

.. code-block:: python

    # Raises KeyError if 'some_key' is absent
    prev = massphoto.defaults.pop('some_key')

    # Returns the supplied default instead of raising
    prev = massphoto.defaults.pop('some_key', None)

Worker-friendly callable wrapping:

.. code-block:: python

    # Prepare a callable for a worker and include a snapshot of the state
    wrapped = defaults.wrap_callable(some_function, state=defaults.get_state(copy=True))

.. rubric:: Notes and tips


- ``save`` and ``load`` default to YAML's safe dumper/loader; it can be necessary to use ``unsafe=True``
  when python objects are stored in the defaults (e.g. a callable PSF function). Only use for files of trusted origin.

- The ``sync_apply_defaults`` decorator uses function-specific (``func.param``),
  prefix (``prefix.param``) and global (``param``) keys with the following
  precedence (highest-first): explicit call args > ``func.param`` > ``prefix.param`` >
  global ``param`` > signature default.

- With an explicit prefix, registration uses ``prefix.param`` instead of
  ``func.param`` except for parameters promoted to global keys. Existing
  function-name keys still take precedence during lookup.

.. rubric:: DefaultsStore API

.. automodule:: massphoto.defaults_store
    :members:
    :undoc-members:
    :show-inheritance:
