"""
Grab bag of tests implemented for the various CASA routines. This
isn't a systematic unit test, but if you write something useful put it
here. This collection for be for tests that can be run with only the
pipeline itself in place. There are other test files in the scripts/
directory.
"""

import logging

import analysisUtils as au
import numpy as np

from . import casaMaskingRoutines as cma

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def test_estimate_noise(
    ):
    """
    Test the noise estimation routine.
    """

    tol = 1e-2

    vec = np.random.randn(1e5)
    mad_est = cma.estimate_noise(vec, method='mad')
    std_est = cma.estimate_noise(vec, method='std')
    chauv_est = cma.estimate_noise(vec, method='chauv')
    chauvmad_est = cma.estimate_noise(vec, method='chauvmad')

    logger.info("mad estimate accuracy: "+str(np.abs(mad_est-1.0)))
    if np.abs(mad_est - 1.0) > tol:
        logger.error("mad estimate exceeds tolerance.")

    logger.info("std estimate accuracy: "+str(np.abs(std_est-1.0)))
    if np.abs(std_est - 1.0) > tol:
        logger.error("std estimate exceeds tolerance.")

    logger.info("chauv estimate accuracy: "+str(np.abs(chauv_est-1.0)))
    if np.abs(chauv_est - 1.0) > tol:
        logger.error("chauv estimate exceeds tolerance.")

    logger.info("chauvmad estimate accuracy: "+str(np.abs(chauvmad_est-1.0)))
    if np.abs(chauvmad_est - 1.0) > tol:
        logger.error("chauvmad estimate exceeds tolerance.")

    return(None)
