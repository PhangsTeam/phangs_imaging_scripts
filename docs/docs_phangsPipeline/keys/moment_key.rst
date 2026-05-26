##########
Moment key
##########

This file defines "moments" calculated by the pipeline. These are
derived two dimensional map products created by applying a mask to
the cube and then collapsing it using some algorithm.

This file should be space or tab delimited. Each entry should have **3** columns,
as described below.

- **Column 1:** Moment name as used in the ``derived_key``. This can be
  anything, it's used for cross-linking. Avoid spaces and other
  problem characters.
- **Column 2:** Parameter being defined. There are a number of options:

  - ``algorithm``: a tag fed to the moment routines to return the function
    used to calculate the moment. Needs to be one of the known
    routines. See those programs or the PHANGS-ALMA examples for viable
    routines.
  - ``mask``: ``strictmask`` or ``broadmask`` or ``none`` or some other viable extension.
    Notes which mask to apply. Generally use ``strictmask`` or ``broadmask``.
  - ``ext``: the extension added to the end of the moment map file
    name. Can be anything, though pick something that can sensibly work
    in a file name - avoid _ or spaces.
  - ``ext_error``: the extension added to the end of the uncertainty map
    file name. e.g., ``emom0`` for the error on the "mom0" map.
  - ``round``: the order of the calculation. Calculations start with round
    1 and count up to the maximum round. This is needed when subsequent
    calculations depend on a first round of moments (e.g., velocity field
    priors or intensity-based pruning). Generally, this will be 1 unless
    there's a reason for it to be a higher value.
  - ``kwargs``: a dictionary of kwargs passed to the moment calculation
    call. The value here is a dictionary read literally. Enter it with
    no spaces or other problem characters.

- **Column 3:** Value. Should be appropriate for the parameter above.

=======================
Example moment key file
=======================

.. include:: ../../../phangs-alma_keys/moment_key.txt
    :literal:
