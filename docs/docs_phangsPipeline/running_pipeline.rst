####################
Running the pipeline
####################

==========
Setting up
==========

Once you have :doc:`installed the pipeline <installation>`, and set up
your :doc:`key files <keys>`, you are ready to run the pipeline!

We first need to make sure that the CASA analysisUtils package can be imported:

.. code-block:: python

    import sys
    sys.path.append("/path/to/analysis_scripts/")

After which we are ready to import the pipeline itself:

.. code-block:: python

    import phangsPipeline as ppl

By default, running the pipeline will run on all targets and configurations,
so it can be handy to specify a subset of targets and configurations to run
on:

.. code-block:: python

    targets = ["m33"]
    line_products = ["co21"]
    interf_configs = ["12m+7m"]
    feather_configs = ["12m+7m+tp"]
    no_cont = True

Set up the logger:

.. code-block:: python

    ppl.phangsLogger.setup_logger(level="INFO", logfile=None)

Note that you can also save this to file by specifying ``logfile``.

Next, we initialise the keyHandler, which will be fed into each further step:

.. code-block:: python

    key_handler = ppl.KeyHandler(master_key=master_key_file)

    # Create any missing directories
    key_handler.make_missing_directories(
        imaging=True,
        postprocess=True,
        derived=True,
    )

Note that you may want to switch on and off some stages as appropriate.

=======
Running
=======

The pipeline is controlled by a number of :doc:`handlers <handlers>`, which perform
various tasks such as staging the data, imaging, postprocessing, and creating derived products.
See the specific pages for details on what each handler does, what follows here is a minimal
code example for running the pipeline end-to-end.

.. code-block:: python

    # Set up the singledish handler
    sd_handler = ppl.SingleDishHandler(key_handler=key_handler)
    sd_handler.set_targets(only=targets)
    sd_handler.set_line_products(only=line_products)
    sd_handler.set_no_cont_products(no_cont)

    # Run the singledish handler
    sd_handler.loop_singledish(do_all=True)

    # Set up the uv handler
    uv_handler = ppl.VisHandler(key_handler=key_handler)
    uv_handler.set_targets(only=targets)
    uv_handler.set_interf_configs(only=interf_configs)
    uv_handler.set_line_products(only=line_products)
    uv_handler.set_no_cont_products(no_cont)

    # Run the uv handler
    uv_handler.loop_stage_uvdata(do_all=True)

    # Set up the imaging handler
    im_handler = ppl.ImagingHandler(key_handler=key_handler)
    im_handler.set_targets(only=targets)
    im_handler.set_interf_configs(only=interf_configs)
    im_handler.set_line_products(only=line_products)
    im_handler.set_no_cont_products(no_cont)

    # Run imaging
    im_handler.loop_imaging(do_all=True)

    # Set up the postprocessing handler
    pp_handler = ppl.PostProcessHandler(key_handler=key_handler)
    pp_handler.set_targets(only=targets)
    pp_handler.set_interf_configs(only=interf_configs)
    pp_handler.set_line_products(only=line_products)
    pp_handler.set_feather_configs(only=feather_configs)
    pp_handler.set_no_cont_products(no_cont)

    # Run postprocessing
    pp_handler.loop_postprocess(do_all=True)

    # Set up the derived handler
    derived_handler = ppl.DerivedHandler(key_handler=key_handler)
    derived_handler.set_targets(only=targets)
    derived_handler.set_interf_configs(only=interf_configs)
    derived_handler.set_feather_configs(only=feather_configs)
    derived_handler.set_line_products(only=line_products)
    derived_handler.set_no_cont_products(no_cont)

    # Run derived products
    derived_handler.loop_derive_products(do_all=True)

This will create singledish products, stage and image interferometric data, postprocess data,
and create derived products (moment maps etc) for the specified targets, line products, and configurations.

.. note::

    This example uses the standing ImagingHandler. For large cubes and running jobs on HPC,
    you may want to use the :doc:`ImagingChunkedHandler <handlers/ImagingChunkedHandler>` instead.
