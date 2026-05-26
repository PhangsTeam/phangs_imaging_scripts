#############
Cleanmask key
#############

The cleanmask key defines files holding user-supplied clean mask FITS files.

The key defines the location of clean masks for each target, product
combination. These files are read in and aligned (by velocity)
during imaging and used as a prior on deconvolution. If not clean
mask is supplied then a primary beam mask is used.

This file should be space or tab delimited. Each entry should have **3** columns,
as described below.

- **Column 1:** Target name.
- **Column 2:** Spectral product for this single dish image cube.
- **Column 3:** File name. The filename is searched for under the
  ``cleanmask`` parent directory defined in the ``master_key``.

==========================
Example cleanmask key file
==========================

.. include:: ../../../phangs-alma_keys/cleanmask_key.txt
    :literal:
