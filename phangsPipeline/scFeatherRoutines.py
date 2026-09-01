import logging
import os

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.utils.console import ProgressBar
from spectral_cube import SpectralCube
from uvcombine import feather_simple_cube

from .scCubeRoutines import check_wcs_match, create_large_fits, reproject_to_other_cube

# Logging
logger = logging.getLogger(__name__)


def prep_sd_for_feather(
    sdfile_in: str | None = None,
    sdfile_out: str | None = None,
    interf_file: str | None = None,
    do_align: bool = True,
    do_checkunits: bool = True,
    overwrite: bool = False,
):
    """
    Prepare single dish data for feathering.

    Args:
        sdfile_in (str): the input single dish file. Can be a FITS file (with
            do_import) or a CASA image.
        sdfile_out (str): the output of the program. If all flags are called,
            this will be a .fits image on the same astrometric grid as the
            interferometric file with units of Jy/beam and no degenerate axes.
        interf_file (str): the interferometric file being used in the feather
            call. Used as a template here to astrometrically align the data.
        do_align (bool): If True then align the single dish data to the
            astrometric grid of the interferometer file.
        do_checkunits (bool): If True then check the units to make
            sure that they are in Jy/beam, which is required by feather.
        overwrite (bool): Delete existing files. Defaults to False.

    Returns:
        bool: True if successful, False if not.
    """

    # Check inputs
    if interf_file is None:
        logger.error("Interferometric file not defined")
        return False

    if sdfile_in is None:
        logger.error("Single dish file not defined")
        return False

    if not os.path.exists(interf_file):
        logger.error(f"Interferometric file not found: {interf_file}")
        return False

    if not os.path.exists(sdfile_in):
        logger.error(f"Single dish file not found: {sdfile_in}")
        return False

    if sdfile_out is None:
        logger.error("Output single dish file name not supplied via sdfile_out")
        return False

    if not overwrite and os.path.exists(sdfile_out):
        logger.info(f"Output file {sdfile_out} already exists and overwrite is False. Skipping")
        return True

    # Load in cubes. The singledish may be a CASA image, so check for that
    cube_format = "fits"
    if not sdfile_in.lower().endswith(".fits"):
        cube_format = "casa"
    sd_cube = SpectralCube.read(
        sdfile_in,
        format=cube_format,
    )
    sd_cube.allow_huge_operations = True

    int_cube = SpectralCube.read(interf_file)

    # Align the single dish data to the astrometric grid of the interferometric data
    if do_align:
        logger.info(f"Aligning {sdfile_in} to {interf_file}")
        sd_cube = reproject_to_other_cube(
            input_cube=sd_cube,
            target_cube=int_cube,
        )

    # Check the units on the singledish file and convert from K to Jy/beam if needed.
    if do_checkunits:
        if sd_cube.unit == u.K:
            logger.info("Unit is Kelvin. Converting to Jy/beam.")
            sd_cube = sd_cube.to(u.Jy / u.beam)

    # Make sure we're in float32 space
    sd_cube_dtype = sd_cube.unmasked_data[0, 0, 0].dtype

    # If we're not, cast it to float32. Do channel-by-channel to reduce RAM
    if sd_cube_dtype != np.float32:
        
        create_large_fits(
            sdfile_out,
            sd_cube.header,
        )

        with fits.open(sdfile_out, mode="update") as hdu:
            n_chan = hdu[0].data.shape[0]

            logger.info("Converting cube to float32 channel-by-channel")
            with ProgressBar(n_chan) as bar:
                for chan in range(n_chan):
                    hdu[0].data[chan] = sd_cube.unitless_filled_data[chan].astype(np.float32)
                    hdu.flush()
                    bar.update()

    else:
        sd_cube.write(sdfile_out, overwrite=True)

    return True


def feather_two_cubes(
    interf_file: str | None = None,
    sd_file: str | None = None,
    out_file: str | None = None,
    apod_file: str | None = None,
    do_apodize: bool = False,
    apod_cutoff: float | int = -1.0,
    sdgain: float = 1.0,
    overwrite: bool = False,
):
    """
    Feather together interferometric and total power data using uvcombine.
    Optionally, first apply some steps to homogenize
    the two data sets.

    TODO: Feathering with apodization hasn't been tested

    Args:
        interf_file (str): the interferometric cube to feather.
        sd_file (str): the single dish cube to feather.
        out_file (str): the output file name.
        apod_file (str): the apodization file to use.
        do_apodize (bool): if True, then apodize BOTH data sets using the provided apodization map.
            Defaults to False.
        apod_cutoff (float | int): the cutoff in the apodization file below which data are blanked.
            Defaults to -1.0.
        sdgain (float | int): the gain factor to apply to the single dish data. Defaults to 1.0.
        overwrite (bool): if True, overwrite existing files.
            Defaults to False

    Returns:
        bool: True if successful, False if not.
    """

    # Check inputs
    if interf_file is None:
        logger.error("Interferometric file not defined")
        return False

    if sd_file is None:
        logger.error("Single-dish cube not defined")
        return False

    if out_file is None:
        logger.error("Output file not defined")
        return False

    if do_apodize and apod_file is None:
        logger.error("Apodization requested, but apodization file not defined")
        return False

    if not os.path.exists(sd_file):
        logger.error(f"Single dish file not found: {sd_file}")
        return False

    if not os.path.exists(interf_file):
        logger.error(f"Interferometric file not found: {interf_file}")
        return False

    if do_apodize and apod_file is not None and not os.path.exists(apod_file):
        logger.error(f"Apodization requested, but file not found: {apod_file}")
        return False

    if os.path.exists(out_file) and not overwrite:
        logger.info(f"outfile {out_file} already exists and overwrite is False. Skipping")
        return True

    # Load in cubes
    int_cube = SpectralCube.read(interf_file)
    sd_cube = SpectralCube.read(sd_file)

    apod_data = 0

    # If we're apodizing, multiply by the apod file
    if do_apodize:
        apod_cube = SpectralCube.read(apod_file)

        # Use unmasked data to avoid unit complaints
        apod_data = apod_cube.unmasked_data[:]

        int_cube *= apod_data
        sd_cube *= apod_data

    # Check if WCS matches up, and if so disable the resampling
    wcs_match = check_wcs_match(int_cube, sd_cube)

    resample = True
    if wcs_match:
        resample = False

    else:
        logger.warning("WCS mismatch. Will not combine masks, and may cause issues in feathering.")

    # Do the feather!
    feathered_cube = feather_simple_cube(
        int_cube,
        sd_cube,
        allow_spectral_resample=resample,
        allow_lo_reproj=resample,
        lowresscalefactor=sdgain,
    )
    feathered_cube.allow_huge_operations = True

    # Apply the masks from the two cubes for safety
    feathered_cube = feathered_cube.with_mask(int_cube.mask)
    feathered_cube = feathered_cube.with_mask(sd_cube.mask)

    # If we apodized, now divide out the common kernel.
    if do_apodize:
        # Make sure we actually have the apodization file
        if apod_data == 0:
            raise ValueError("apod_data should be defined")

        feathered_cube /= apod_data

        # Apply the apodization cutoff as a mask. The sign is greater than
        # since True is for included data
        feathered_cube = feathered_cube.with_mask(apod_data > apod_cutoff)

    # Make sure we're in float32 space
    feathered_cube_dtype = feathered_cube.unmasked_data[0, 0, 0].dtype

    # If we're not, cast it to float32. Do channel-by-channel to reduce RAM
    if feathered_cube_dtype != np.float32:
        create_large_fits(
            out_file,
            feathered_cube.header,
        )

        with fits.open(out_file, mode="update") as hdu:
            n_chan = hdu[0].data.shape[0]

            logger.info("Converting cube to float32 channel-by-channel")
            with ProgressBar(n_chan) as bar:
                for chan in range(n_chan):
                    hdu[0].data[chan] = feathered_cube.unitless_filled_data[chan].astype(np.float32)
                    hdu.flush()
                    bar.update()

    else:
        # Write out the cube
        feathered_cube.write(out_file, overwrite=True)

    return True
