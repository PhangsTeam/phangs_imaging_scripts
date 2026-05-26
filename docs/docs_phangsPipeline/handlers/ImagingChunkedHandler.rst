#####################
ImagingChunkedHandler
#####################

This module makes images and cubes out of the uv data products created by VisHandler, imaging in
chunks of channels when cube sizes are large.

Note that unlike the ImagingHandler, ImagingChunkedHandler requires passing an individual target,
config, and product each time.

Requires CASA to run.

.. autoclass:: phangsPipeline.ImagingChunkedHandler
    :noindex:

.. autofunction:: phangsPipeline.ImagingChunkedHandler.run_imaging
    :noindex:
