import logging
import warnings

import astropy.units as u
import numpy as np
from spectral_cube import SpectralCube

from . import scDerivativeRoutines as scdr

warnings.filterwarnings("ignore")


def _func_and_kwargs_for_moment(
    moment_tag: str | None = None,
):
    """
    Return function name and defalt kwargs for a moment tag.

    Args:
        moment_tag (str): Moment tag to look up. Defaults to None, which will return None.
    """

    func_dict = {
        None: None,
        "failure_case": scdr.write_moment0,
        "mom0": scdr.write_moment0,
        "mom1": scdr.write_moment1,
        "mom2": scdr.write_moment2,
        "ew": scdr.write_ew,
        "vquad": scdr.write_vquad,
        "vpeak": scdr.write_vmax,
        "tpeak": scdr.write_tmax,
        "mom1wprior": scdr.write_moment1_hybrid,
    }
    kwargs_dict = {
        None: None,
        "failure_case": "not_kwargs",
        "mom0": {"unit": u.K * u.km / u.s},
        "mom1": {"unit": u.km / u.s},
        "mom2": {"unit": u.km / u.s},
        "ew": {"unit": u.km / u.s},
        "vquad": {"unit": u.km / u.s},
        "vpeak": {"unit": u.km / u.s},
        "tpeak": {"unit": u.K},
        "mom1wprior": {"unit": u.km / u.s},
    }

    func = func_dict.get(moment_tag, None)
    kwargs = kwargs_dict.get(moment_tag, None)

    return func, kwargs


def moment_generator(
    cubein: str | SpectralCube,
    mask: str | SpectralCube | None = None,
    noise: str | SpectralCube | None = None,
    moment: str | None = None,
    momkwargs: dict | None = None,
    outfile: str | None = None,
    errorfile: str | None = None,
    channel_correlation: np.ndarray | None = None,
):
    """
    Generate one moment map from input cube, noise, and masks.

    Args:
        cubein (str or SpectralCube): Input cube to generate moment from.
        mask (str or SpectralCube, optional): Mask to apply to cube.
            Defaults to None, which will use no mask.
        noise (str or SpectralCube, optional): Noise cube to use for error estimation.
            Defaults to None, which will use no noise cube.
        moment (str, optional): Moment tag to generate.
            Defaults to None, which will fail.
        momkwargs (dict, optional): Additional keyword arguments for moment generation.
            Defaults to None, which will just adopt the default kwargs.
        outfile (str, optional): Output file name for moment map.
            Defaults to None, which will not produce an output file.
        errorfile (str, optional): Name for output error map.
            Defaults to None, which will not produce an output file.
        channel_correlation (np.ndarray, optional): Channel correlation coefficients.
            Defaults to None, which will assume uncorrelated channels.
    """

    # Error checking
    if moment is None:
        raise ValueError("Moment tag must be specified.")

    if not isinstance(moment, str):
        raise TypeError("Moment tag should be a string.")

    # Get the relevant function and keyword arguments for this moment
    func, kwargs = _func_and_kwargs_for_moment(moment)

    # If we don't recognise the moment tag, then return an error
    if func is None:
        raise ValueError(f"Moment tag {moment} not recognized.")

    # If we don't have a kwargs dictionary, then return an error
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs should be a dictionary.")

    # Add any user-supplied kwargs to the dictionary
    if momkwargs is not None:
        if isinstance(momkwargs, dict):
            for this_kwarg in momkwargs:
                kwargs[this_kwarg] = momkwargs[this_kwarg]
        else:
            raise TypeError("momkwargs should be a dictionary.")

    # Read in the cube (if needed)
    if isinstance(cubein, str):
        cube = SpectralCube.read(cubein)
    elif isinstance(cubein, SpectralCube):
        cube = cubein
    else:
        raise TypeError("cubein should be a string or SpectralCube object.")

    cube.allow_huge_operations = True

    # Attach a mask if needed
    if mask is not None:
        if isinstance(mask, str):
            mask = SpectralCube.read(mask)
        elif isinstance(mask, SpectralCube):
            mask = mask
        else:
            raise TypeError("If specified, mask should be a string or SpectralCube object.")

        mask.allow_huge_operations = True

        # Ensure the mask is booleans and attach it to the cube. This
        # just assumes a match in astrometry.
        mask = np.array(mask.unitless_filled_data[:], dtype=bool)
        cube = cube.with_mask(mask, inherit_mask=False)

    # Read in the noise (if present)
    if noise is not None:
        if isinstance(noise, str):
            noisecube = SpectralCube.read(noise)
        elif isinstance(noise, SpectralCube):
            noisecube = noise
        else:
            raise TypeError("If specified, noise should be a string or SpectralCube object.")

        noisecube.allow_huge_operations = True

    else:
        noisecube = None

    # Call moment generation
    moment_map, error_map = func(
        cube,
        rms=noisecube,
        outfile=outfile,
        errorfile=errorfile,
        channel_correlation=channel_correlation,
        **kwargs,
    )

    return moment_map, error_map
