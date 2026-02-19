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

# Pass the chunk number from the cmd line

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

# Make any missing directories

this_kh.make_missing_directories(imaging=True, derived=True, postprocess=True, release=True)


# Initialize the ImagingChunkedHandler
# Unlike ImageHandler, you must specify the target, configuration and line_name
# ImagingChunkedHandler is designed to image individual products that require
# chunking to process efficiently.
this_imh = ImagingChunkedHandler('ic10ctr', 'C', 'hilores', this_kh,
                                 chunksize=10,
                                 )

print(f"Total number of chunks: {this_imh.nchunks}")
