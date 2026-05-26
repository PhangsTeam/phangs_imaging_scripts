##########
Master key
##########

The master key defines the directory structure for the project and
links to the other keys that define configurations, targets, uv
data, algorithmic choices, etc. for the project. The master key is
the key input to the pipeline that links all of the keys to the
handler objects.

.. warning::
    The pipeline (currently) operates on absolute path definitions.
    Even if you are using keys an otherwise already set up project,
    please set these absolute file paths to be appropriate to
    for your system before running the pipeline!

===============
Key definitions
===============

-----------
Directories
-----------

``key_dir``: This is the location all the key files used by an individual pipeline product.
The master key file is almost certainly in your ``key_dir``.

``ms_root``: This is any directory in which the calibrated measurement sets
(listed in the ``ms_key``) are located. These directories are pre-pended to
the file paths in ms_key to identify the locations of the uv data.
Often this will be the root of the tree where you unpack and calibrate your data.

.. note::
    You can set more than one ``ms_root`` in order to have multiple search
    paths for the measurement sets defined in the ``ms_key``.

``imaging_root``: This is the root directory for staging and process the
visibility data and imaging. Each individual target will have a subdirectory
of this imaging directory where processing for that target occurs.

``cleanmask_root``: This is the root directory for the clean mask FITS cubes.

.. note::
    ``cleanmask_root`` is optional. You only need this if you are using clean masks.

``postprocess_root``: This is the root directory for processing the imaged cubes,
including feathering and linear mosaicking. Each individual target will have a
subdirectory of this imaging directory where processing for that target occurs.

``singledish_root``: This is the root directory storing all the single dish imaged
FITS cubes. These are inputs for feathering and postprocessing.

.. note::
    You can set more than one ``singledish_root`` below to have multiple
    search paths for the measurement sets defined in the ``singledish_key``.

.. note::
    ``singledish_root`` is optional. You only need this if you are feathering with
    short spacing data, or imaging singledish data.

``derived_root``: This is the root directory for creating derived data products, like
moment maps and masks from the postprocessed image cubes. Each individual target
will have a subdirectory of this parent derived directory where processing for that
target occurs.

``release_root``: This is the root directory for where "released" products are stored.
Each individual target will have a subdirectory of this parent derived directory where
files are placed.

``vfield_root``: This is the root directory for velocity fields used for the shuffled
products. Each individual target will have a subdirectory of this parent derived
directory where processing for that target occurs. The velocity field can either be
provided or created from the moment-1 in the derived products stage.

------------------------------------
Files giving locations of input data
------------------------------------

These are files giving the location of input data to be processed by
the pipeline. Right now, we expect list of input uv data (``ms_key``),
single dish data (``singledish_key``), and clean masks (``cleanmask_key``).

These key files should be exist (or be defined relative to the
``key_dir`` path set above). The easiest solution is to put all the
keys in the same directory.

``ms_key``: This is the data file(s) that list the individual u-v data used
by the project. Each entry maps from a calibrated u-v data set to a target
name, array tag, and project tag.

.. warning::
    These measurement sets need to be calibrated in advance, e.g., with
    the ALMA calibration pipeline or VLA pipeline.

``singledish_key``: This is the data file(s) that list individual single
dish FITS files for use in feathering. Each entry maps from a target and
product to a single dish cube.

.. note::
    If the singledish data don't already exist, then this will be used to
    run the Total Power pipeline.

``cleanmask_key``: This is the data file(s) that list individual clean masks
as FITS files. Each entry maps from a target and product to a clean mask.

``distance_key``: This is the data file(s) that list distances to each target.
Each entry maps from a target to a distance in Mpc.

``window_key``: This is the data file(s) that list the velocity windows for each target and product.
These are used for shuffled cube and flat map creation

-------------------------------------
Files defining targets and algorithms
-------------------------------------

``config_key``: This file defines the configurations of lines, interferometric
arrays, and combinations of interferometric and total power data looped over
by the pipeline. It defines "products" (which are spectral data products,
like CO 2-1 or continuum) and "configurations" (which are combinations of
interferometer and single dish data).

``target_key``: This file contains the phase centers, velocities, and widths for
each target. The velocity information is used for gridding, continuum extraction,
identifying spectral windows, etc.

``moment_key``: This file contains the definitions of the moment maps to be created in the
derived products stage.

``derived_key``: This file contains various recipes for derived product creation.

``imaging_key``: This file maps between product and configuration combinations to clean
parameter files of the type saved by save_input in CASA. These parameters get read in as
the baseline values for cleaning using that config/product combination, they are then
modified by the imaging scripts and cleaning procedures.

``linmos_key``: This file maps targets to linear mosaics. Linear mosaics combine
individual targets into a single image as part of post-processing. This has some
redundancy with the directory key, but in theory targets might be processed separately
and still mosaicked, so the keys both exist but look very similar. In principle, the
same file could be assigned to both.

.. note::
    ``linmos_key`` is optional.

``dir_key``: This file contains mappings between target and directory within the working
directory. Use this, e.g., to image and process target 'ngc3627_1' in the directory 'ngc3627'.

.. note::
    ``dir_key`` is optional.

``override_key``: This file contains by hand overrides for the post-processing and imaging
steps.

.. note::
    ``override_key`` is optional.

=======================
Example master key file
=======================

.. include:: ../../../phangs-alma_keys/master_key.txt
    :literal:
