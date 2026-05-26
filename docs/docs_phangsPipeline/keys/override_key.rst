############
Override key
############

.. warning::
   This is not currently thoroughly implemented throughout the pipeline.

This key files gives the user the ability to override
certain algorithmic choices made by the pipeline for individual
targets or files.

The two major uses in the previous pipeline incarnation were to
tweak ``tclean`` to avoid divergence in challenging cases and to
manually specify the extent of linear mosaics.

This file should be space or tab delimited. Each entry should have **3** columns,
as described below.

- **Column 1:** Keyword - image name or other file name
- **Column 2:** Parameter to override
- **Column 3:** Value to override with

=========================
Example override key file
=========================

.. include:: ../../../phangs-alma_keys/overrides.txt
    :literal:
