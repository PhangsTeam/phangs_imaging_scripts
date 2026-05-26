##########
VisHandler
##########

The PHANGS pipeline to handle staging and pre-processing of uv data
before imaging. Works through a single big class (the
VisHandler). This needs to be attached to a keyHandler to access the
target, product, and configuration keys and locations of the uv data.

Requires CASA to run.

.. autofunction:: phangsPipeline.VisHandler.loop_stage_uvdata
    :noindex:
