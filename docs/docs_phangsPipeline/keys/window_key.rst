##########
Window key
##########

The window key defines files holding user-supplied velocity field FITS files.

When the user requests to construct the flat masks used to produce the
flat moment-0 maps the velocity window is needed. This file provides that
velocity window.

This file should be comma delimited. Each entry should have **3** columns,
as described below.

- **Column 1:** Target name.
- **Column 2:** Velocity window in km/s.
- **Column 3:** File name. The filename is searched for under the
  ``vfield_root`` parent directory defined in the ``master_key``.

=======================
Example window key file
=======================

.. include:: ../../../phangs-alma_keys/window_key.txt
    :literal:
