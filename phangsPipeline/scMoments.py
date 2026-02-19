import inspect
import logging
import warnings

import astropy.units as u
import numpy as np
from spectral_cube import SpectralCube

from . import scDerivativeRoutines as scdr

warnings.filterwarnings("ignore")

def _nicestr(quantity):
    if quantity.value == int(quantity.value):
        return(str(int(quantity.value))+' '+str(quantity.unit))
    else:
        return(str(quantity))

def _func_and_kwargs_for_moment(moment_tag=None):
    """
    Return function name and defalt kwargs for a moment tag.
    """

    func = None
    kwargs = None
    if moment_tag is None:
        return(func,kwargs)

    if moment_tag == 'mom0':
        func = scdr.write_moment0
        kwargs ={'unit': u.K * u.km / u.s}
    elif moment_tag == 'mom1':
        func = scdr.write_moment1
        kwargs = {'unit': u.km / u.s}
    elif moment_tag == 'mom2':
        func = scdr.write_moment2
        kwargs = {'unit': u.km / u.s}
    elif moment_tag == 'ew':
        func = scdr.write_ew
        kwargs = {'unit': u.km / u.s}
    elif moment_tag == 'vquad':
        func = scdr.write_vquad
        kwargs = {'unit': u.km / u.s}
    elif moment_tag == 'vpeak':
        func = scdr.write_vmax
        kwargs = {'unit': u.km / u.s}
    elif moment_tag == 'tpeak':
        func = scdr.write_tmax
        kwargs = {'unit': u.K}
    elif moment_tag == 'mom1wprior':
        func = scdr.write_moment1_hybrid
        kwargs = {'unit': u.km / u.s}
    elif moment_tag == 'mom0flat':
        func = scdr.write_moment0_flat
        kwargs ={'unit': u.K * u.km / u.s}

    return(func, kwargs)

def moment_tag_known(moment_tag=None):
    """
    Test whether the programs know about a moment tag.
    """
    func, kwargs = _func_and_kwargs_for_moment(moment_tag)
    if func is None:
        return(False)
    return(True)

def moment_generator(
        cubein, mask=None, noise=None,
        moment=None, momkwargs=None,
        outfile=None, errorfile=None,
        channel_correlation=None,
        context=None, assignkunits=False):

    """
    Generate one moment map from input cube, noise, and masks.
    """

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Set up the call
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # Get the relevant function and keyword arguments for this moment
    func, kwargs = _func_and_kwargs_for_moment(moment)
    if func is None:
        logging.error("Moment tag not recognized: "+str(moment))
        raise NotImplementedError
        return(None)

    # Add any user-supplied kwargs to the dictionary
    if momkwargs is not None:
        if type(momkwargs) != type({}):
            logging.error("Type of momkwargs should be dictionary.")
            raise NotImplementedError
        for this_kwarg in momkwargs:
            kwargs[this_kwarg] = momkwargs[this_kwarg]

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Read in the data
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # Read in the cube (if needed)
    if type(cubein) is str:
        cube = SpectralCube.read(cubein)
    elif type(cubein) is SpectralCube:
        cube = cubein
    else:
        logging.error('Unrecognized input type for cubein')
        raise NotImplementedError

    cube.allow_huge_operations = True
        
    # Force Kelvin. We will be unit agnostic later.
    cube = cube.to(u.K)
    
    # Attach a mask if needed
    if mask is not None:
        if type(mask) is str:
            mask = SpectralCube.read(mask)
        elif type(mask) is SpectralCube:
            mask = mask
        else:
            logging.error('Unrecognized input type for mask')
            raise NotImplementedError

        # Ensure the mask is booleans and attach it to the cube. This
        # just assumes a match in astrometry. Could add reprojection
        # here or (better) build a masking routine to apply masks with
        # arbitrary astrometry.

        mask = np.array(mask.filled_data[:].value, dtype=bool)
        cube = cube.with_mask(mask, inherit_mask=False)

    # Read in the noise (if present)
    if noise is not None:        
        if type(noise) is str:
            noisecube = SpectralCube.read(noise)
        elif type(noise) is SpectralCube:
            noisecube = noise
        else:
            logging.error('Unrecognized input type for noise.')
            raise NotImplementedError

        noisecube.allow_huge_operations = True

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Call the moment generation
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Probably not needed anymore
    theseargs = (inspect.getfullargspec(func)).args

    if 'context' in theseargs:
        moment_map, error_map = func(
            cube, rms=noisecube,
            outfile=outfile, errorfile=errorfile,
            channel_correlation=channel_correlation,
            #context=context,
            **kwargs)
    else:
        moment_map, error_map = func(
            cube, rms=noisecube,
            outfile=outfile, errorfile=errorfile,
            channel_correlation=channel_correlation,
            **kwargs)
        
    return(moment_map, error_map)
    

