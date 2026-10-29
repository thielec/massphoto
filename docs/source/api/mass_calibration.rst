Mass Calibration
================

Mass calibration maps optical interferometric contrast to molecular mass (kDa).
The calibration is typically obtained by measuring a reference standard containing proteins
of known molecular weight (e.g., MassFerenceP1, BSA, or Urease) under identical optical and buffer conditions.

Calibration Conventions
-----------------------

.. note::
   **massphoto vs. DiscoverMP Calibration Conventions**

   In commercial software (Refeyn DiscoverMP), linear calibrations are commonly expressed as:

   .. math::

      \text{contrast} = \text{slope} \cdot \text{mass (kDa)} + \text{offset}

   In **massphoto**, the polynomial maps contrast directly to mass:

   .. math::

      \text{mass (kDa)} = c_0 + c_1 \cdot \text{contrast}

   where :math:`c_0` is the intercept (kDa) and :math:`c_1` is the slope (kDa / contrast unit).
   To convert parameters from DiscoverMP:

   .. math::

      c_1 = \frac{1}{\text{slope}}, \quad c_0 = -\frac{\text{offset}}{\text{slope}}

   When using :func:`~massphoto.mass_calibration.fit`, you can specify ``direction="contrast_to_mass"``
   (default) or ``direction="mass_to_contrast"`` to match the desired convention.

For experimental protocols on preparing calibration standards and selecting protein reference peaks, see
`Kratochvíl et al., Nature Protocols (2025) <https://doi.org/10.1038/s41596-025-01255-4>`_.

Module Reference
----------------

.. automodule:: massphoto.mass_calibration
   :members:
   :undoc-members:
   :show-inheritance:
