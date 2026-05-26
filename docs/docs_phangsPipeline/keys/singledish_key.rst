##############
Singledish key
##############

The singledish key defines files holding user-supplied single dish FITS files.

If the files defined in the singledish file are not provided, then they can be
generated using the single dish pipeline (this only works for ALMA data).
Otherwise, these must be provided as FITS files.

The pipeline requires that any single dish data used for feathering be provided
by the user. This file defines the locations of those data and maps them to
a target and product.

This file should be space or tab delimited. Each entry should have **3** columns,
as described below.

- **Column 1:** Target name.
- **Column 2:** Spectral product for this single dish image cube.
- **Column 3:** File name. The filename is searched for under the
  ``singledish_root`` parent directory defined in the ``master_key``.

===========================
Example singledish key file
===========================

.. include:: ../../../phangs-alma_keys/singledish_key.txt
    :literal:
