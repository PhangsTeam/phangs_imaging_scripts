'''
To be run in a slurm/pbs jobarray.

Requires the output of `create_imaging_job_config_files.py`
'''


import os
import sys
from copy import copy

from pathlib import Path


config_filename_full = sys.argv[-1]

config_filename = config_filename_full.split("/")[-1]

line_name = config_filename.split("_")[0]

line_list = [line_name]

with open(config_filename_full, 'r') as f:
    field_name = f.readlines()

field_name = field_name[0].strip()

target_list = [field_name]


# Location of the master key. Set this to the master key that points
# to all of the keys for your project.

key_file = '/home/erickoch/full_aca_band6/keys_hydra/master_key.txt'

# Import the logger and initialize the logging. You can change the
# level of message that you want to see by changing "level" here or
# save to a logfile with the keyword.

from phangsPipeline import phangsLogger as pl
pl.setup_logger(level='DEBUG', logfile=None)

from phangsPipeline import handlerKeys as kh
from phangsPipeline import handlerVis as uvh
# from phangsPipeline import handlerImaging as imh
from phangsPipeline

# Initialize the various handler objects. First initialize the
# KeyHandler, which reads the master key and the files linked in the
# master key. Then feed this keyHandler, which has all the project
# data, into the other handlers (VisHandler, ImagingHandler,
# PostProcessHandler), which run the actual pipeline using the project
# definitions from the KeyHandler.

this_kh = kh.KeyHandler(master_key=key_file)
this_uvh = uvh.VisHandler(key_handler=this_kh)
this_imh = imh.ImagingHandler(key_handler=this_kh)

# Make any missing directories

this_kh.make_missing_directories(imaging=True,
                                 derived=True,
                                 postprocess=True,
                                 release=True)

##############################################################################
# Set up what we do this run
##############################################################################

this_uvh.set_targets(only=target_list)
this_uvh.set_interf_configs(only=['7m'])
this_uvh.set_line_products(only=line_list)
this_uvh.set_no_cont_products(True)

this_imh.set_targets(only=target_list)
this_imh.set_interf_configs(only=['7m'])
this_imh.set_no_cont_products(True)
this_imh.set_line_products(only=line_list)

# Use boolean flags to set the steps to be performed when the pipeline
# is called. See descriptions below (but only edit here).

do_staging = True
do_imaging = True

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
                               do_remove_staging=False, overwrite=True)

    this_uvh.loop_stage_uvdata(do_copy=False, do_contsub=False,
                               do_extract_line=True, do_extract_cont=False,
                               do_remove_staging=False, overwrite=True)

    this_uvh.loop_stage_uvdata(do_copy=False, do_contsub=False,
                               do_extract_line=False, do_extract_cont=True,
                               do_remove_staging=False, overwrite=True)

    this_uvh.loop_stage_uvdata(do_copy=False, do_contsub=False,
                               do_extract_line=False, do_extract_cont=False,
                               do_remove_staging=True, overwrite=True)

##############################################################################
# Step through imaging
##############################################################################

# Image the concatenated, regridded visibility data. The full loop
# involves applying any user-supplied clean mask, multiscale imaging,
# mask generation for the single scale clean, and single scale
# clean. The individual parts can be turned on or off with flags to
# the imaging loop call but this call does everything.

if do_imaging:
    this_imh.loop_imaging(do_all=True)
