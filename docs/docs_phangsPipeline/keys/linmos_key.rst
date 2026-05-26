##########
Linmos key
##########

.. note::
   This file is not needed if you are not doing linear mosaicking.

This key files defines the mapping of imaging targets to linear mosaics.
Imaging targets will be separately cleaned and deconvolved and
then combined into a single data product via linear mosaic in the
postprocessing.

This file should be space or tab delimited. Each entry should have **2** columns,
as described below.

- **Column 1:** Name of the target for the full linear mosaic.
- **Column 2:** Name of the imaging target that belongs to that mosaic.

=======================
Example linmos key file
=======================

.. include:: ../../../phangs-alma_keys/linearmosaic_definitions.txt
    :literal:
