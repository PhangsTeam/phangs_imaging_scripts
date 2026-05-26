##########
Config key
##########

This file defines the spectral "products" and array
"configurations" that to be processed and combined by the
pipeline. Here, the user provides detailed parameters for each
product and config.

This file should be space or tab delimited. Each entry should have **3** columns,
as described below.

==================
Column Definitions
==================

-----------------------
Column 1: Type of field
-----------------------

This must be one of the following:

- ``line_product``: a spectral line imaging product (e.g., ``co21_2p5kms``)
- ``cont_product``: a continuum imaging product (e.g., ``cont``)
- ``interf_config``: a configuration containing only interferometric data (e.g., ``12m``)
- ``feather_config``: a configuration combining single dish and interferometric data (e.g., ``7m+tp``)


--------------
Column 2: Name
--------------

This can be anything, but should not have any spaces or _ characters
to avoid confusion.

------------------
Column 3: Pameters
------------------

This is evaluated as a literal dictionary (i.e., the line is fed to
python). Please avoid spaces and follow the fields required below.

.. note::

   Configuration definitions can repeat multiple lines, in which
   case the dictionary parameters are accumulated across all
   lines. Don't repeat parameters, but if you do then in theory the
   later lines will overwrite the earlier ones.

=================
Parameter options
=================

----------------
``line_product``
----------------

This defines a spectral line product, which will be produced by
clean using specmode='cube'. The name can be anything, it does not
have to be the line name, e.g., ``lowresco`` is fine.

**Parameters:**

- ``line_tag``: this is the line tag we use in our pipeline and should
  agree with the line names in our ``utilsLines.py`` module. If this is
  missing, then you should add the line to your utilsLines module (and
  file an issue).
- ``channel_kms``: this is the target channel width of our output image
  cubes in units of km/s. Depending on the chosen regridding algorithm
  this may be the exact channel width or an approximate target.
- ``exclude_freq_ranges_ghz``: given as a list of list, this gives a set
  [low, high] frequency range in GHz to avoid when using uvcontsub.
  e.g. ``'exclude_freq_ranges_ghz':[[1.40,1.42]]``.
  This is useful when multiple targets emit in the same transition along
  a line-of-sight (e.g., Galactic and extragalactic 21-cm HI).
- ``flag_edge_fraction`` (optional): float between 0.0 and 0.5
  This is an input to avoid using the edges of a SPW for
  fitting the continuum in uvcontsub.
- ``lines_to_flag`` : these are the spectral line families (for groups
  of lines) or spectral line tags (for individual lines) to exclude
  (at the velocity and bandwidth of the source) from continuum
  imaging. They need to be known by our ``utilsLines.py``.

----------------
``cont_product``
----------------

This defines a continuum product, which will be produced by clean
using specmode='mfs'. The name can be anything, e.g., ``850umcont``.

**Parameters:**

- ``freq_ranges_ghz``: These are given as a list of lists, and define
  a ``[low, high]`` frequency range in GHz for the continuum imaging.
- ``lines_to_flag`` : these are the spectral line families (for groups
  of lines) or spectral line tags (for individual lines) to exclude
  (at the velocity and bandwidth of the source) from continuum
  imaging. They need to be known by our ``utilsLines.py``.

-----------------
``interf_config``
-----------------

This defines an interferometric configuration, which is a collection
of data from interferometric arrays. These will eventually be
combined into a single visibility file and imaged together. The name
can be anything, e.g., ``BCD`` or ``12m+7m``. Note that these cover
both individual interferometric configurations (e.g. ``12m``), as
well as more complicated joint array combinations (e.g. ``12m+7m``).

**Parameters:**

- ``array_tags``: defines a list of array tags to be concatenated
  into this config. These array tags are assigned to each measurement
  set in the ``ms_key``, which lists the visibility data. Thus,
  they only mean anything inasmuch as they agree with those defined in
  that file. The natural mapping is, e.g., VLA config, ALMA array, and
  maybe a code roughly indicating the ALMA subarray (e.g.,
  ``12mext``). This is fundamentally up to the user to decide. Despite
  the names used by PHANGS-ALMA (12m, 7m) the pipeline does NOT
  attempt to detect and assign data to arrays on its own.
- ``clean_scales_arcsec``: These are the scales used for multiscale
  cleaning of data from this array. The units are arcseconds.
- ``requires``: By default, we require ALL
  arrays that make up the configuration, but this can be changed
  to an OR if you only need one of a certain combination, e.g.
  ``{'requires':['12m_1|12m_2']}`` requires only one of ``12m_1`` or
  ``12m_2`` to be present. If an array is not in the ``requires`` list,
  then it is assumed to be required.

------------------
``feather_config``
------------------

This defines a "feather configuration," which is a combination of an
interferometric configuration and total power data. The name can be
anything, e.g., ``BCD+tp`` or ``12m+7m+tp``.

**Parameters:**

- ``interf_config`` : each feather config needs to map back to an
  interferometric config. This mapping is used to carry out the
  feathering process. This field defines the ``interf_config``
  corresponding to this feather config. When data from that
  ``interf_config`` is feathered it produces data labeled with this
  ``feather_config``.

=======================
Example config key file
=======================

.. include:: ../../../phangs-alma_keys/config_definitions.txt
    :literal:
