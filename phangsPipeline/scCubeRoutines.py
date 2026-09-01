import logging
import os

import astropy.units as u
import numpy as np
import scipy.ndimage as nd
from astropy.convolution import convolve, convolve_fft
from astropy.io import fits
from astropy.utils.console import ProgressBar
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from radio_beam import Beam
from spectral_cube import SpectralCube, VaryingResolutionSpectralCube
from spectral_cube.utils import NoBeamError

from . import __version__

# Logging
logger = logging.getLogger(__name__)


def create_large_fits(
    f: str,
    header: fits.header.Header,
):
    """Create a large FITS file without loading large array into memory

    Args:
        f (str): Output file name
        header (fits.header.Header): FITS header to use for the output file
    """

    header.tofile(f, overwrite=True)

    shape = tuple(header[f"NAXIS{ii}"] for ii in range(1, header["NAXIS"] + 1))
    with open(f, "rb+") as fobj:
        file_length = len(header.tostring()) + (np.prod(shape) * np.abs(header["BITPIX"] // 8))
        file_length = ((file_length + 2880 - 1) // 2880) * 2880 - 1
        fobj.seek(file_length)
        fobj.write(b"\0")

    return True


def reproject_to_other_cube(
    input_cube: SpectralCube,
    target_cube: SpectralCube,
):
    """Spectrally interpolate and then reproject a cube slice-by-slice to the WCS of another

    Args:
        input_cube (SpectralCube): Cube to reproject
        target_cube (SpectralCube): Cube to reproject to

    Returns:
        SpectralCube: The reprojected cube
    """

    # Start with spectral interpolation, only if we're not on the same spectral grid
    in_spec_axis = input_cube.spectral_axis
    targ_spec_axis = target_cube.spectral_axis

    do_spectral_interp = True
    if len(in_spec_axis) == len(targ_spec_axis):
        if np.isclose(in_spec_axis, target_cube.spectral_axis).all():
            do_spectral_interp = False

    if not do_spectral_interp:
        reproj_cube = input_cube.with_mask(input_cube.mask)
    else:
        logger.info("Performing spectral interpolation to match target cube")

        reproj_cube = input_cube.spectral_interpolate(target_cube.spectral_axis)

    # Next up is spatial reprojection, where we check the celestial WCS to see if we need
    # to do this
    spatial_wcs_match = reproj_cube.wcs.celestial.wcs.compare(target_cube.wcs.celestial.wcs)
    shape_match = reproj_cube.shape == target_cube.shape

    if not spatial_wcs_match or not shape_match:
        logger.info("Performing spatial regridding to match target cube")

        # Build a 2D header for the reprojection
        hdr_2d = target_cube.wcs.celestial.to_header()
        hdr_2d["NAXIS"] = 2
        hdr_2d["NAXIS1"] = target_cube.wcs.pixel_shape[0]
        hdr_2d["NAXIS2"] = target_cube.wcs.pixel_shape[1]

        # Create an empty array to put values into. Use float32 to save space
        # and for consistency elsewhere
        full_cube_array = np.ones(target_cube.shape, dtype=np.float32) * np.nan

        # Do each slice of the reprojection. Use a progress bar since this
        # is quite slow
        with ProgressBar(reproj_cube.shape[0]) as bar:
            for i in range(reproj_cube.shape[0]):
                slice_reproj = reproj_cube[i, :, :].reproject(hdr_2d)
                full_cube_array[i, :, :] = slice_reproj.filled_data[:]

                bar.update()

        # Build a new cube to put these reprojected slices into
        meta = reproj_cube.meta
        try:
            beam = reproj_cube.beam
        except NoBeamError:
            beam = None

        mask = np.isfinite(full_cube_array)

        reproj_cube = SpectralCube(
            data=full_cube_array,
            wcs=target_cube.wcs,
            meta=meta,
            header=target_cube.header,
            beam=beam,
        )

        # Add in the mask
        reproj_cube = reproj_cube.with_mask(mask)

    return reproj_cube


def reproject_to_other_wcs(
    cube: SpectralCube,
    wcs: WCS,
):
    """Spectrally interpolate and then reproject a cube slice-by-slice to a target WCS

    Args:
        cube (SpectralCube): Cube to reproject
        wcs (WCS): Target WCS

    Returns:
        SpectralCube: The reprojected cube
    """

    # Reproject to WCS. Start by getting out the spectral axis for interpolation
    w_spec = wcs.spectral
    w_spec_len = wcs.array_shape[0]
    w_spec_vals = w_spec.pixel_to_world(range(w_spec_len))

    # Start with spectral interpolation, only if we're not on the same spectral grid
    in_spec_axis = cube.spectral_axis

    do_spectral_interp = True
    if len(in_spec_axis) == len(w_spec_vals):
        if np.isclose(in_spec_axis, w_spec_vals).all():
            do_spectral_interp = False

    if not do_spectral_interp:
        reproj_cube = cube.with_mask(cube.mask)
    else:
        logger.info("Performing spectral interpolation to match target WCS")
        reproj_cube = cube.spectral_interpolate(w_spec_vals)

    # Next up is spatial reprojection, where we check the celestial WCS to see if we need
    # to do this
    spatial_wcs_match = reproj_cube.wcs.celestial.wcs.compare(wcs.celestial.wcs)
    shape_match = reproj_cube.shape == wcs.array_shape
    if not spatial_wcs_match or not shape_match:
        logger.info("Performing spatial regridding to match target WCS")

        # Build a final header
        hdr = wcs.to_header()

        # We need to add in NAXIS as well
        hdr["NAXIS"] = len(wcs.pixel_shape)
        hdr["NAXIS1"] = wcs.pixel_shape[0]
        hdr["NAXIS2"] = wcs.pixel_shape[1]
        hdr["NAXIS3"] = wcs.pixel_shape[2]

        # Build a 2D header for the reprojection
        hdr_2d = wcs.celestial.to_header()
        hdr_2d["NAXIS"] = 2
        hdr_2d["NAXIS1"] = wcs.pixel_shape[0]
        hdr_2d["NAXIS2"] = wcs.pixel_shape[1]

        # Create an empty array to put values into. Use float32 to save space
        # and for consistency elsewhere
        full_cube_array = np.ones(wcs.array_shape, dtype=np.float32) * np.nan

        # Do each slice of the reprojection. Use a progress bar since this
        # is quite slow
        with ProgressBar(reproj_cube.shape[0]) as bar:
            for i in range(reproj_cube.shape[0]):
                slice_reproj = reproj_cube[i, :, :].reproject(hdr_2d)
                full_cube_array[i, :, :] = slice_reproj.filled_data[:]

                bar.update()

        # Build a new cube to put these reprojected slices into
        meta = reproj_cube.meta
        try:
            beam = reproj_cube.beam
        except NoBeamError:
            beam = None

        mask = np.isfinite(full_cube_array)

        reproj_cube = SpectralCube(
            data=full_cube_array,
            wcs=wcs,
            meta=meta,
            header=hdr,
            beam=beam,
        )

        # Add in the mask
        reproj_cube = reproj_cube.with_mask(mask)

    return reproj_cube


def check_wcs_match(
    cube1: SpectralCube,
    cube2: SpectralCube,
):
    """Check if two cubes have the same wcs.

    Args:
        cube1 (SpectralCube): First cube
        cube2 (SpectralCube): Second cube

    Returns:
        bool: True if cubes have the same WCS, False otherwise
    """

    # Check shape
    if cube1.shape != cube2.shape:
        logger.error("Shape mismatch!")
        return False
    
    # Check spectral
    spectral_wcs_match = np.isclose(cube1.spectral_axis, cube2.spectral_axis).all()
    if not spectral_wcs_match:
        logger.error("Cube spectral WCS mismatch!")
        return False

    # Check spatial
    spatial_wcs_match = cube1.wcs.celestial.wcs.compare(cube2.wcs.celestial.wcs)
    if not spatial_wcs_match:
        logger.error("Cube spatial WCS mismatch!")
        return False

    return True


def export_and_cleanup(
    infile: str | None = None,
    outfile: str | None = None,
    overwrite: bool = False,
    remove_cards: list | None = None,
    add_cards: dict | None = None,
    add_history: list | None = None,
    zap_history: bool = True,
    round_beam: bool = True,
    roundbeam_tol: float | int = 0.01,
):
    """
    Do final cleanup for a cube, tidying up header keywords that are
    usually confusing or useless. Optionally add new keywords to the header
    and check whether the beam is close enough to being round that it makes
    sense to overwrite it.

    Args:
        infile (str): Input file name
        outfile (str): Output file name
        overwrite (bool): Overwrite existing file. Default is False
        remove_cards (list): List of keywords to remove from header
        add_cards (dict): Dictionary of keywords to add to header
        add_history (list): List of keywords to add to header
        zap_history (bool): If True, zap history keywords
        round_beam (bool): If True, then force round beam in header
        roundbeam_tol (float | int): Fractional tolerance for round beam in header

    Returns:
        bool: True if successful, False otherwise
    """

    if infile is None:
        logger.error("Missing required input.")
        return False

    if outfile is None:
        logger.error("Missing required output.")
        return False

    if not os.path.exists(infile):
        logger.error(f"Input file does not exist - {infile}")
        return False

    if os.path.isfile(outfile) and not overwrite:
        logger.info(f"Output exists {outfile} and overwrite set to False. Will skip")
        return True

    if add_history is None:
        add_history = []

    if add_cards is None:
        add_cards = {}

    if remove_cards is None:
        remove_cards = []

    cube = SpectralCube.read(infile)

    # Convert to km/s and write to outfile
    cube = cube.with_spectral_unit(
        u.km / u.s,
        velocity_convention="radio",
    )

    cube.write(outfile, overwrite=True)

    # Read back in with astropy so we can manipulate the header
    with fits.open(outfile) as hdu:
        hdr = hdu[0].header
        data = hdu[0].data

        # Cards to remove by default
        default_cards_to_remove = [
            "BLANK",
            "DATE-OBS",
            "OBSERVER",
            "O_BLANK",
            "O_BSCALE",
            "O_BZERO",
            "OBSRA",
            "OBSDEC",
            "OBSGEO-X",
            "OBSGEO-Y",
            "OBSGEO-Z",
            "DISTANCE",
        ]

        for card in default_cards_to_remove:
            if card in hdr.keys():
                hdr.remove(card)

        # User cards to remove
        for card in remove_cards:
            if card in hdr.keys():
                hdr.remove(card)

        # Delete history
        if zap_history:
            while "HISTORY" in hdr.keys():
                hdr.remove("HISTORY")

        # Add history
        if len(add_history) > 0:
            # Add a blank history if we don't have one
            if "HISTORY" not in hdr.keys():
                hdr["HISTORY"] = ""

            # Append history lines as necessary
            for history_line in add_history:
                hdr["HISTORY"].append(history_line)

        for card in add_cards:
            hdr[card] = add_cards[card]

        # Get the data min and max right
        datamin = np.nanmin(data)
        datamax = np.nanmax(data)

        hdr["DATAMIN"] = datamin
        hdr["DATAMAX"] = datamax

        # Round the beam recorded in the header if it lies within the
        # specified tolerance.

        if round_beam:
            if roundbeam_tol > 0.0:
                bmaj = hdr["BMAJ"]
                bmin = hdr["BMIN"]
                if bmaj != bmin:
                    frac_dev = np.abs(bmaj - bmin) / bmaj
                    if frac_dev <= roundbeam_tol:
                        logger.info("Rounding beam.")
                        hdr["BMAJ"] = bmaj
                        hdr["BMIN"] = bmaj
                        hdr["BPA"] = 0.0
                    else:
                        logger.info("Beam too asymmetric to round.")
                        logger.info(f"... fractional deviation: {frac_dev}")

        # Never forget where you came from
        hdr["COMMENT"] = "Produced with PHANGS-ALMA pipeline version " + __version__

        hdu.writeto(outfile, overwrite=True)

    return True


def convolve_to_round_beam(
    infile: str | None = None,
    outfile: str | None = None,
    convolve_fn: str = "convolve_fft",
    nan_treatment: str = "interpolate",
    force_beam: float | int | None = None,
    overwrite: bool = False,
):
    """
    Convolve supplied image to have a round beam. Optionally, force
    that beam to some size, else it figures out the beam.

    Args:
        infile (str): Input file name
        outfile (str): Output file name
        convolve_fn (str): Convolution function name. Either "convolve"
            or "convolve_fft". Default is "convolve_fft".
        nan_treatment (str): Treatment to use for nan values in convolution
        force_beam (float): Force beam to size in arcsec.
            Default is None, which does not force the beam to any particular
            size.
        overwrite (bool): Overwrite existing file. Default is False

    Returns:
        bool: True if successful, False otherwise
    """

    if infile is None or outfile is None:
        logger.error("Missing required input.")
        return False

    if not os.path.exists(infile):
        logger.error("Input file missing - " + infile)
        return False

    if os.path.exists(outfile) and not overwrite:
        logger.info(f"outfile {outfile} already exists and overwrite is False. Skipping")
        return True

    cube = SpectralCube.read(infile)
    cube.allow_huge_operations = True

    # Get beam from cube (convert to arcsec), and then target major axis.
    # This is a little different depending on whether the cube is a VaryingResolutionSpectralCube
    # or not
    if isinstance(cube, VaryingResolutionSpectralCube):
        beam_idx = np.argmax([b.major.to(u.arcsec).value for b in cube.beams])
        beam = cube.beams[beam_idx]
    else:
        beam = cube.beam

    bmaj = beam.major
    bmaj = bmaj.to(u.arcsec)

    # Get pixel scale in arcsec. Assume square pixels
    pixel_scales = proj_plane_pixel_scales(cube.wcs.celestial) * u.deg
    pixel_as = [p.to(u.arcsec) for p in pixel_scales][0]

    # Make the beam a little larger to avoid convolution artifacts
    if force_beam is None:
        target_bmaj = np.sqrt(bmaj**2 + (2.0 * pixel_as) ** 2)
    else:
        min_bmaj = np.sqrt(bmaj**2 + (2.0 * pixel_as) ** 2)
        if force_beam < min_bmaj:
            logger.warning("Requested beam is too small for convolution.")
            return False
        target_bmaj = force_beam

    # Build beam, do the convolution, write out
    target_beam = Beam(major=target_bmaj, minor=target_bmaj, pa=0 * u.deg)

    logger.info(f"Convolving to round beam - {str(target_beam)}")

    if convolve_fn == "convolve":
        conv_fn = convolve
    elif convolve_fn == "convolve_fft":
        conv_fn = convolve_fft
    else:
        raise ValueError(f"convolve_fn {convolve_fn} not recognized")

    kwargs = {
        "nan_treatment": nan_treatment,
        "preserve_nan": True,
    }

    # Keep track of the original dtype since it can change during convolution
    orig_dtype = cube.unmasked_data[0, 0, 0].dtype

    cube = cube.convolve_to(
        target_beam,
        convolve=conv_fn,
        **kwargs,
    )

    # Because this operation can change the dtype, recreate the cube
    # Do this channel-by-channel, to keep RAM usage low
    if cube.unmasked_data[0, 0, 0].dtype != orig_dtype:
        create_large_fits(
            outfile,
            cube.header,
        )

        with fits.open(outfile, mode="update") as hdu:
            n_chan = hdu[0].data.shape[0]

            logger.info("Writing cube out channel-by-channel")
            with ProgressBar(n_chan) as bar:
                for chan in range(n_chan):
                    hdu[0].data[chan] = cube.unitless_filled_data[chan].astype(orig_dtype)
                    hdu.flush()
                    bar.update()

    else:
        cube.write(
            outfile,
            overwrite=True,
        )

    return True


def align_to_target(
    infile: str | None = None,
    outfile: str | None = None,
    template: str | None = None,
    overwrite: bool = False,
):
    """
    Align one cube to another, creating a copy.

    Args:
        infile (str): Input file name
        outfile (str): Output file name
        template (str): Template file name
        overwrite (bool): Overwrite existing file. Default is False

    Returns:
        bool: True if successful, False otherwise
    """

    if infile is None:
        logger.error("Missing required input.")
        return False

    if template is None:
        logger.error("Missing required template.")
        return False

    if outfile is None:
        logger.error("Missing required output.")
        return False

    if not os.path.exists(infile):
        logger.error(f"Input file missing - {infile}")
        return False

    if not os.path.exists(template):
        logger.error(f"Template file missing - {template}")
        return False

    if os.path.exists(outfile) and not overwrite:
        logger.info(f"outfile {outfile} already exists and overwrite is False. Skipping")
        return True

    in_cube = SpectralCube.read(infile)
    template_cube = SpectralCube.read(template)

    in_cube = reproject_to_other_cube(
        input_cube=in_cube,
        target_cube=template_cube,
    )

    in_cube.write(outfile, overwrite=True)

    return True


def primary_beam_correct(
    infile: str | None = None,
    pbfile: str | None = None,
    outfile: str | None = None,
    cutoff: float | int = 0.25,
    overwrite: bool = False,
):
    """
    Construct a primary-beam corrected image.

    Args:
        infile (str): Input file name
        pbfile (str): Primary beam file name
        outfile (str): Output file name
        cutoff (float | int): Cutoff value for primary beam correction
        overwrite (bool): Overwrite existing file. Default is False

    Returns:
        bool: True if successful, False otherwise
    """

    if infile is None or pbfile is None or outfile is None:
        logger.error("Missing required input.")
        return False

    if not os.path.exists(infile):
        logger.error(f"Input file missing - {infile}")
        return False

    if not os.path.exists(pbfile):
        logger.error(f"Primary beam file missing - {pbfile}")
        return False

    if os.path.exists(outfile) and not overwrite:
        logger.info(f"outfile {outfile} already exists and overwrite is False. Skipping")
        return True

    # Load in image and pb cubes
    cube = SpectralCube.read(infile)
    cube.allow_huge_operations = True
    pb = SpectralCube.read(pbfile)
    pb.allow_huge_operations = True

    # Add the cutoff as a mask. The sign is greater than since True is for included
    # data
    mask = pb > cutoff

    # Quick checks to make sure data shape and WCS are the same
    wcs_match = check_wcs_match(cube, pb)
    if not wcs_match:
        return False

    # Apply the PB correction. Use unmasked data to avoid unit complaints
    cube /= pb.unmasked_data[:]

    # Apply the PB mask and write out
    cube = cube.with_mask(mask)
    cube.write(outfile, overwrite=True)

    return True


def trim_cube(
    infile: str | None = None,
    outfile: str | None = None,
    overwrite: bool = False,
    min_pixperbeam: int = 3,
    pad: int = 1,
    rebin: bool = True,
):
    """
    Trim empty space from around the edge of a cube. Also rebin the
    cube to smaller size, while ensuring a minimum number of pixels
    across the beam. Used to reduce the volume of cubes.

    Args:
        infile (str): Input file name
        outfile (str): Output file name
        overwrite (bool): Overwrite existing file. Default is False
        min_pixperbeam (int): Minimum number of pixels per beam, if rebinning.
            Default is 3.
        pad (int): Padding around the edge of the cube. Default is 1.
        rebin (bool): Whether to rebin the cube to some minimum number of pixels
            per beam. Default is True.

    Returns:
        bool: True if successful, False otherwise
    """

    if infile is None or outfile is None:
        logger.error("Missing required input.")
        return False

    if not os.path.exists(infile):
        logger.error(f"Input file not found: {infile}")
        return False

    if os.path.exists(outfile) and not overwrite:
        logger.info(f"outfile {outfile} already exists and overwrite is False. Skipping")
        return True

    # Load the cube in
    cube_format = "fits"
    if not infile.lower().endswith(".fits"):
        cube_format = "casa"
    cube = SpectralCube.read(infile, format=cube_format)
    cube.allow_huge_operations = True

    orig_dtype = cube.unmasked_data[0, 0, 0].dtype

    if rebin:
        # Get out pixels per beam. spectral-cube returns as an area,
        # so sqrt
        pixperbeam = np.sqrt(cube.pixels_per_beam)

        # Calculate the rebinning factor, and rebin if >1
        rebin_factor = int(np.floor(pixperbeam / min_pixperbeam))

        if rebin_factor > 1:
            logger.info(f"Will rebin spatial axes by a factor {rebin_factor}")

            # Rebin along the spatial axes, which are 1 and 2
            cube = cube.downsample_axis(
                factor=rebin_factor,
                axis=1,
            )
            cube = cube.downsample_axis(
                factor=rebin_factor,
                axis=2,
            )

    # Pull out the mask, use this to get data extent
    mask = cube.get_mask_array()

    # Loop over each axis to check the maximal extent in the mask
    axis_slices = []
    for ax in range(len(cube.shape)):
        mask_spec_i = np.any(
            mask, axis=tuple([i for i, x in enumerate(list(mask.shape)) if i != ax])
        )
        i_min = np.max([0, np.min(np.where(mask_spec_i)) - pad])
        i_max = np.min([np.max(np.where(mask_spec_i)) + pad, mask.shape[ax] - 1])
        axis_slices.append(slice(i_min, i_max + 1))

    # Slice down the cube
    cube = cube[*axis_slices]

    # Do a final check against dtype
    cube_final_dtype = cube.unmasked_data[0, 0, 0].dtype

    # If we don't match, write out channel-by-channel
    if cube_final_dtype != orig_dtype:

        create_large_fits(
            outfile,
            cube.header,
        )

        with fits.open(outfile, mode="update") as hdu:
            n_chan = hdu[0].data.shape[0]

            logger.info("Writing out cube channel-by-channel")
            with ProgressBar(n_chan) as bar:
                for chan in range(n_chan):
                    hdu[0].data[chan] = cube.unitless_filled_data[chan].astype(np.float32)
                    hdu.flush()
                    bar.update()

    # Otherwise, just save
    else:
        cube.write(outfile, overwrite=True)

    return True


def trim_rind(
    infile: str | None = None,
    outfile: str | None = None,
    overwrite: bool = False,
    pixels: int = 1,
):
    """Binary erode a mask to trim the rind off a cube

    TODO: This hasn't been explicitly tested yet, but should work.

    Args:
        infile (str): Input file name
        outfile (str): Output file name
        overwrite (bool): Overwrite existing file.
            Default is False
        pixels (int): Number of pixels to erode. Default is 1.

    Returns:
        bool: True if successful, False otherwise
    """

    if infile is None or outfile is None:
        logger.error("Missing required input.")
        return False

    if not os.path.exists(infile):
        logger.error(f"Input file not found: {infile}")
        return False

    if os.path.exists(outfile) and not overwrite:
        logger.info(f"outfile {outfile} already exists and overwrite is False. Skipping")
        return True

    # Figure out the extent of the image inside the cube
    cube = SpectralCube.read(infile)
    mask = cube.get_mask_array()

    elt = nd.generate_binary_structure(2, 1)
    if pixels > 1:
        elt = nd.iterate_structure(elt, pixels - 1)
    mask = nd.binary_erosion(mask, elt[np.newaxis, :, :])
    cube = cube.with_mask(mask, inherit_mask=False)

    cube.write(outfile, overwrite=True)

    return True


def convert_units(
    infile: str | None = None,
    outfile: str | None = None,
    units: str | u.Unit = "K",
    overwrite: bool = False,
):
    """
    Convert a cube to a specific unit.

    Args:
        infile (str): Input file
        outfile (str): Output file
        units (str | u.Unit): Unit to convert to.
            Defaults to "K"
        overwrite (bool): Overwrite existing file.
            Defaults to False

    Returns:
        bool: True if successful, False otherwise
    """

    if infile is None:
        logger.error("Missing required input.")
        return False

    if outfile is None:
        logger.error("Missing required output.")
        return False

    if not os.path.exists(infile):
        logger.error(f"Input file not found: {infile}")
        return False

    if os.path.exists(outfile) and not overwrite:
        logger.error(f"Output file {outfile} already exists and overwrite is False. Will skip")
        return True

    # Read in cube and convert
    cube = SpectralCube.read(infile)
    cube.allow_huge_operations = True

    # Do this channel by channel to avoid needing a lot of RAM
    create_large_fits(
        outfile,
        cube.header,
    )

    with fits.open(outfile, mode="update") as hdu:
        n_chan = hdu[0].data.shape[0]

        logger.info(f"Converting cube to {units} channel-by-channel")
        with ProgressBar(n_chan) as bar:
            for chan in range(n_chan):
                hdu[0].data[chan] = cube[chan].to(units)
                hdu.flush()
                bar.update()

        # Finally, update the unit
        hdu[0].header.update({"BUNIT": units})
        hdu.flush()

    return True
