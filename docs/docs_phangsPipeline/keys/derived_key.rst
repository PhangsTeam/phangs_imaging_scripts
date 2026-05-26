###########
Derived key
###########

This files defines the derived science-ready data products produced by
the pipeline.

This file should be space or tab delimited. Each entry should have **4** columns,
as described below.

- **Column 1:** Array configuration (e.g. ``12m``, ``7m+tp``, etc). ``all`` is OK, and will
  apply to all configurations.
- **Column 2:** Spectral configuration (e.g. ``co21``, ``hi21cm``, etc). ``all`` is OK, and
  will apply to all spectral configurations.
- **Column 3:** Field for keyword arguments. There are a number of options:

  - ``phys_res``: Target physical resolutions (in pc). Format is a dictionary of ``{'tag': value}``
    pairs. Targets will need defined distances for physical resolutions to be used, and ``tag`` will
    be inserted into the filename.
  - ``ang_res``: Target angular resolutions (in arcsec). Format is a dictionary of ``{'tag': value}``
    pairs, and ``tag`` will be inserted into the filename.
  - ``convolve_kw``: Keywords for convolution.
  - ``shuffle_kw``: Keywords for velocity shuffling.
  - ``strictmask_kw``: Keywords for generation of strict masks.
  - ``broadmask_kw``: Keywords for generation of broad masks.
  - ``flatstrictmask_kw``: Keywords for generation of flat stict masks.
  - ``flatbroadmask_kw``: Keywords for generation of flat broad masks.
  - ``mask_configs``: List of names of other configurations to link when
    creating broad masks. All masks for all linked configurations will
    be combined to create the broad masks.
  - ``moments``: List of moments (defined in the moment key) to
    calculate for that that config and spectral product pair.

- **Column 4:** Values input for that field. This is read as a literal
  dictionary, avoid spaces. Multiple lines for a single field + config
  + spectral product combination are fine. In general, later lines
  overwrite previous ones.

========================
Example derived key file
========================

.. include:: ../../../phangs-alma_keys/derived_key.txt
    :literal:
