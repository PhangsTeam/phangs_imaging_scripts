##########
Target key
##########

This file defines the the targets in a project.

In the ``ms_key`` each measurement set is mapped to a "target", which
has an associated position, velocity, and velocity width. These are
defined in this file. In cases where linear mosaicking is desired,
this file should also define targets for the combined linear
mosaics.

This file should be space or tab delimited. Each entry should have **5** columns,
as described below.

- **Column 1:** Target name. This should match the target names used in the ``ms_key``.
- **Column 2:** Right ascension for the phase centre in sexagesimal format.
- **Column 3:** Declination for the phase centre in sexagesimal format.
- **Column 4:** Systemic velocity in km/s. This assumes LSRK velocity convention.
- **Column 5:** Velocity width in km/s. This is used to define the velocity range, and
  window for continuum subtraction.

.. note::

   Note that the phase center isn't necessarily the same as the object center.
   This is an important distinction in the case of mosaics that cover
   only part of the object, where the phase centre should be the mosaic center.

=======================
Example target key file
=======================

.. include:: ../../../phangs-alma_keys/target_definitions.txt
    :literal:
