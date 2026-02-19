'''
Run this script INSIDE CASA or with CASA available.

This is an example using ImagingChunkedHandler for a single imaging produce.
The purpose of ImagingChunkedHandler is to make it easier to produce very large
cubes using CASA without relying on the internal chunking. This makes it simpler
to use the PHANGS imaging in a cluster environment.

This script loads the project data, constructs the PHANGS pipeline
handlers, and then executes each step: staging, imaging,
postprocessing. The user has control over which targets, spectral
products, and steps run.

This is a documented version that we provide with the main pipeline
repository as an example tou users. You should be able to modify
this script to get a good start on your own wrapper to the pipeline.
'''


##############################################################################
# Load routines, initialize handlers
##############################################################################

import os
import sys

# Location of the master key. Set this to the master key that points
# to all of the keys for your project.

key_file = '/Users/ekoch/storage/LGLBS/chunked_imaging_tests/lglbs_keys/master_key.txt'

# Import the logger and initialize the logging. You can change the
# level of message that you want to see by changing "level" here or
# save to a logfile with the keyword.

from phangsPipeline import phangsLogger as pl
pl.setup_logger(level='DEBUG', logfile=None)

# Imports

# sys.path.insert(1, )
from phangsPipeline import handlerKeys as kh
from phangsPipeline import handlerVis as uvh
from phangsPipeline import handlerImaging as imh
from phangsPipeline import handlerPostprocess as pph

from phangsPipeline.handlerImagingChunked import ImagingChunkedHandler


# Initialize the various handler objects. First initialize the
# KeyHandler, which reads the master key and the files linked in the
# master key. Then feed this keyHandler, which has all the project
# data, into the other handlers (VisHandler, ImagingHandler,
# PostProcessHandler), which run the actual pipeline using the project
# definitions from the KeyHandler.

this_kh = kh.KeyHandler(master_key=key_file)
this_uvh = uvh.VisHandler(key_handler=this_kh)
this_pph = pph.PostProcessHandler(key_handler=this_kh)

# Make any missing directories

this_kh.make_missing_directories(imaging=True, derived=True, postprocess=True, release=True)

##############################################################################
# Set up what we do this run
##############################################################################


# Set the configs (arrays), spectral products (lines), and targets to
# consider.

# Set the targets. Called with only () it will use all targets. The
# only= , just= , start= , stop= criteria allow one to build a smaller
# list.

# Set the configs. Set both interf_configs and feather_configs just to
# determine which cubes will be processed. The only effect in this
# derive product calculation is to determine which cubes get fed into
# the calculation.

# Set the line products. Similarly, this just determines which cubes
# are fed in. Right now there's no derived product pipeline focused on
# continuum maps.

# ASIDE: In PHANGS-ALMA we ran a cheap parallelization by running
# several scripts with different start and stop values in parallel. If
# you are running a big batch of jobs you might consider scripting
# something similar.

# Note here that we need to set the targets, configs, and lines for
# *all three* relevant handlers - the VisHandler (uvh), ImagingHandler
# (imh), and PostprocessHandler (pph). The settings below will stage
# combined 12m+7m data sets (including staging C18O and continuum),
# image the CO 2-1 line from these, and then postprocess the CO 2-1
# cubes.

this_uvh.set_targets()
this_uvh.set_interf_configs(only=['C'])
this_uvh.set_line_products(only=['hilores'])
this_uvh.set_no_cont_products(False)

# e.g., could be to be more selective:
# this_uvh.set_targets(only=['ngc3489','ngc3599','ngc4476'])
# this_uvh.set_interf_configs(only=['C'])
# this_uvh.set_line_products(only=['hilores'])

this_pph.set_targets()
this_pph.set_interf_configs(only=['C'])
this_pph.set_feather_configs(only=['C+tp'])

# Use boolean flags to set the steps to be performed when the pipeline
# is called. See descriptions below (but only edit here).

do_staging = True
do_imaging = True
do_postprocess = True

##############################################################################
# Run staging
##############################################################################

# "Stage" the visibility data. This involves copying the original
# calibrated measurement set, continuum subtracting (if requested),
# extraction of requested lines and continuum data, regridding and
# concatenation into a single measurement set. The overwrite=True flag
# is needed to ensure that previous runs can be overwritten.

if do_staging:
    this_uvh.loop_stage_uvdata(do_copy=True, do_contsub=True,
                               do_extract_line=False, do_extract_cont=False,
                               do_remove_staging=False, overwrite=True,
                               strict_config=False)

    this_uvh.loop_stage_uvdata(do_copy=False, do_contsub=False,
                               do_extract_line=True, do_extract_cont=False,
                               do_remove_staging=False, overwrite=True,
                               strict_config=False)

    this_uvh.loop_stage_uvdata(do_copy=False, do_contsub=False,
                               do_extract_line=False, do_extract_cont=True,
                               do_remove_staging=False, overwrite=True,
                               strict_config=False)

    this_uvh.loop_stage_uvdata(do_copy=False, do_contsub=False,
                               do_extract_line=False, do_extract_cont=False,
                               do_remove_staging=True, overwrite=True,
                               strict_config=False)

##############################################################################
# Step through imaging
##############################################################################

# Image the concatenated, regridded visibility data. The full loop
# involves applying any user-supplied clean mask, multiscale imaging,
# mask generation for the single scale clean, and single scale
# clean. The individual parts can be turned on or off with flags to
# the imaging loop call but this call does everything.

if do_imaging:

    # Initialize the ImagingChunkedHandler
    # Unlike ImageHandler, you must specify the target, configuration and line_name
    # ImagingChunkedHandler is designed to image individual products that require
    # chunking to process efficiently.
    this_imh = ImagingChunkedHandler('ic10ctr', 'C', 'hilores', this_kh,
                                     chunksize=10,
                                     )

    # this_imh.task_make_dirty_image()

    # Loop over each chunk to completion
    for chunk_num in range(this_imh.nchunks):
        print(f"Chunk {chunk_num} of {this_imh.nchunks}")

        this_imh.run_imaging(do_all=True, chunk_num=chunk_num)

    # When running per chunk, combining into final cubes is a separate call
    this_imh.task_complete_gather_into_cubes(root_name='all')

    # (Optional) generate FITS cubes
    this_imh.task_export_to_fits(tag=None)
    # this_imh.task_export_to_fits(tag='dirty')
    # this_imh.task_export_to_fits(tag='multiscale')
    # this_imh.task_export_to_fits(tag='singlescale')

    # Cleanup per chunk imaging products
    this_imh.task_cleanup(tag=None)
    # this_imh.task_cleanup(tag='dirty')
    # this_imh.task_cleanup(tag='multiscale')
    # this_imh.task_cleanup(tag='singlescale')

##############################################################################
# Step through postprocessing
##############################################################################

# Postprocess the data in CASA after imaging. This involves primary
# beam correction, linear mosaicking, feathering, conversion to Kelvin
# units, and some downsampling to save space.

if do_postprocess:
    this_pph.loop_postprocess(do_prep=True, do_feather=True,
                              do_mosaic=True, do_cleanup=True)
