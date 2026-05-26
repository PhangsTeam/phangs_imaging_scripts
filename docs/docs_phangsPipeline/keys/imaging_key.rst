###########
Imaging key
###########

The imaging pipeline builds ``clean_call`` objects that repeatedly
call ``tclean`` to build a deconvolved interferometric image. This
imaging recipes key links combinations of configuration ("line"),
spectral "product", and imaging step (see below) to "imaging
recipes."

The ``imaging_key`` defines sets of ``tclean`` parameters similar to
those written out by ``save_input`` in CASA. They are read in to the
pipeline and then passed along to ``tclean``. So these offer a way to
tune the imaging behavior.

Note that some pipeline imaging calls automatically overwrite some
parameters, like ``niter``, ``scales``, or (when using dynamic sizing)
``cellsize`` and ``imsize``. The parameters provided here serve as a base
but will get clobbered.

.. note::

   Order is very important here!

   The recipes are read in as lists, which are applied in the order
   that they are read in. So a parameter in later recipe will
   override the same parameter in the earlier recipe.

This file should be space or tab delimited. Each entry should have **4** columns,
as described below.

- **Column 1:** Array configuration (e.g. ``12m``, ``7m+tp``, etc). ``all`` is OK, and will
  apply to all configurations.
- **Column 2:** Spectral configuration (e.g. ``co21``, ``hi21cm``, etc). ``all_line`` is OK,
  and will apply to all spectral configurations. ``all_cont`` is OK, and will apply to all
  continuum configurations.
- **Column 3:** Imaging stage to apply this recipe to. Valid stages are ``all``, ``dirty``,
  ``multiscale``, and ``singlescale``.
- **Column 4:** The name of the clean parameter file to be read in. Should be in ``key_dir``.

========================
Example imaging key file
========================

.. include:: ../../../phangs-alma_keys/imaging_recipes.txt
    :literal:
