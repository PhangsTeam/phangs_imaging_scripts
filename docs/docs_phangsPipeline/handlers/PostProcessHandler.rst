##################
PostProcessHandler
##################

The PHANGS pipeline to handle post-processing of cubes. Works through
a single big class (the PostProcessHandler). This needs to be attached
to a keyHandler to access the target, product, and configuration
keys.

There are two options for how postprocessing is handled. The default
uses CASA functions, which is the default but can be slow (especially
for large cubes). The other option is to use the faster spectral-cube
routines, which are faster and give comparable results, but are less
well-tested. To change between these, in the ``loop_postprocess`` call,
set ``postprocessing_method='casa'`` or ``postprocessing_method='spectralcube'``.

.. autofunction:: phangsPipeline.PostProcessHandler.loop_postprocess
    :noindex:
