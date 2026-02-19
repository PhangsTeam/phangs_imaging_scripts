'''
Run this script INSIDE CASA or with CASA available.

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
from phangsPipeline import handlerPostprocess as pph

from phangsPipeline.handlerImagingChunked import ImagingChunkedHandler


# Initialize the various handler objects. First initialize the
# KeyHandler, which reads the master key and the files linked in the
# master key. Then feed this keyHandler, which has all the project
# data, into the other handlers (VisHandler, ImagingHandler,
# PostProcessHandler), which run the actual pipeline using the project
# definitions from the KeyHandler.

this_kh = kh.KeyHandler(master_key=key_file)

# Make any missing directories

this_kh.make_missing_directories(imaging=True, derived=True, postprocess=True, release=True)


# Initialize the ImagingChunkedHandler
# Unlike ImageHandler, you must specify the target, configuration and line_name
# ImagingChunkedHandler is designed to image individual products that require
# chunking to process efficiently.
this_imh = ImagingChunkedHandler('ic10ctr', 'C', 'hilores', this_kh,
                                 chunksize=10,
                                 )

# When running per chunk, combining into final cubes is a separate call
this_imh.task_complete_gather_into_cubes(root_name='all')

##############################################################################
# Step through postprocessing
##############################################################################

# Postprocess the data in CASA after imaging. This involves primary
# beam correction, linear mosaicking, feathering, conversion to Kelvin
# units, and some downsampling to save space.

this_pph = pph.PostProcessHandler(this_kh)
this_pph.loop_postprocess(do_prep=True, do_feather=True,
                            do_mosaic=True, do_cleanup=True)
