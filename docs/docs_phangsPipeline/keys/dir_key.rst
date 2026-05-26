#######
Dir key
#######

.. note::
   You probably don't need this file if not doing linear mosaicking.

This key files gives the user the ability to map a target name to a working
directory.

By default, the working directory for any target is a directory with
the same name as the target. The programs will create the
appropriate directory in the imaging, postprocessing,
etc. directories. This allows overwriting of that behavior.

The most common use is to combine several targets into a single
directory. This comes up for PHANGS-ALMA in the context of linear
mosaics. Several targets are observed and then later combined into
a single data cube. Thus we prefer to image and process these parts
together in a single directory.

This file should be space or tab delimited. Each entry should have **2** columns,
as described below.

- **Column 1:** Target name.
- **Column 2:** Working directory name.

====================
Example dir key file
====================

.. include:: ../../../phangs-alma_keys/dir_key.txt
    :literal:
