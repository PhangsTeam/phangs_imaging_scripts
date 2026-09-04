import copy
import logging
import os

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.utils.console import ProgressBar
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from radio_beam import Beam, Beams
from reproject.mosaicking import find_optimal_celestial_wcs
from spectral_cube import SpectralCube

from .scCubeRoutines import convolve_cube, create_large_fits, reproject_to_other_wcs
from .scNoiseRoutines import mad_zero_centered

# Logging
logger = logging.getLogger(__name__)


def build_common_header(
    infile_list: list | None = None,
    allow_big_image: bool = False,
    too_big_pix: int | float = 1e4,
):
    """
    Build a target header to be used as a template when
    setting up linear mosaicking operations.

    Args:
        infile_list (list): List of input image files.
        allow_big_image (bool, optional): If True, allow big images to be created.
            Defaults to False.
        too_big_pix (int|float, optional): The threshold for what constitutes a "big"
            image in pixels. Defaults to 1e4.
    Returns:
        fits.header.Header: Target header.
    """

    # Check inputs
    if infile_list is None:
        logger.error("Missing required infile_list.")
        return None

    for this_infile in infile_list:
        if not os.path.exists(this_infile):
            logger.error(f"File not found {this_infile}. Returning.")
            return None

    # Loop over files, pull out cube headers
    cube2d_hdrs = []
    target_hdr = None

    for this_infile in infile_list:
        cube = SpectralCube.read(this_infile)
        cube2d_hdrs.append(cube.wcs.celestial)

        # If we don't have a 3D header, save that here
        if target_hdr is None:
            target_hdr = cube.header

    # If we still don't have a 3D header, then raise an issue
    if not isinstance(target_hdr, fits.header.Header):
        raise ValueError("Did not find a cube header")

    # Determine optimal WCS (in 2D), remembering we need to add in shape manually
    wcs2d, shape2d = find_optimal_celestial_wcs(cube2d_hdrs)
    optimal_2d_hdr = wcs2d.to_header()
    optimal_2d_hdr["NAXIS1"] = shape2d[1]
    optimal_2d_hdr["NAXIS2"] = shape2d[0]

    if allow_big_image:
        if any(shape2d) > too_big_pix:
            logger.warning(f"This is a very big image you plan to create: {shape2d}")
            logger.warning("To make an image this big set allow_big_image to True. Returning.")
            return None

    # Put this 2D WCS into the 3D header
    target_hdr.update(optimal_2d_hdr)

    # Ensure WCSAXES=3, if it's in the cube header
    if target_hdr.get("WCSAXES", None) == 2:
        target_hdr["WCSAXES"] = 3

    return target_hdr


def common_res_for_mosaic(
    infile_list: list | None = None,
    outfile_list: list | dict | None = None,
    target_res: int | float | None = None,
    do_convolve: bool = True,
    convolve_fn: str = "convolve_uv",
    nan_treatment: str = "interpolate",
    overwrite: bool = False,
):
    """
    Convolve multi-part cubes to a common res for mosaicking.

    Args:
        infile_list (list|None): List of input image files.
        outfile_list (list|dict|None): List or dictionary of output image files.
        target_res (int|float|None): Target resolution in arcsec.
        do_convolve (bool): if True, do convolution. Else just
            calculates and returns target resolution. Defaults to True.
        convolve_fn (str): Convolution function name. "convolve_uv", "convolve"
            or "convolve_fft". Default is "convolve_uv".
        nan_treatment (str): Treatment to use for nan values in convolution.
            Defaults to "interpolate". Only used for "convolve" or "convolve_fft".
        overwrite (bool): if True, overwrite existing files. Defaults
            to False.

    Returns:
        float: Common target resolution
    """

    if infile_list is None:
        logger.error("Missing required infile_list.")
        return None

    if len(infile_list) == 0:
        logger.error("No files to process.")
        return None

    # Check input files exist
    for this_file in infile_list:
        if not os.path.exists(this_file):
            logger.error(f"File not found: {this_file}")
            return None

    # If do_convolve is True then make sure that we have output files
    # and link those to input files
    outfile_dict = {}

    if do_convolve:
        if outfile_list is None:
            logger.error("Missing outfile_list required for convolution.")
            return None

        if not isinstance(outfile_list, list) and not isinstance(outfile_list, dict):
            logger.error("outfile_list must be dictionary or list.")
            return None

        if isinstance(outfile_list, list):
            if len(infile_list) != len(outfile_list):
                logger.error("Mismatch in input and output list lengths.")
                return None

            outfile_dict = {}
            for ii in range(len(infile_list)):
                outfile_dict[infile_list[ii]] = outfile_list[ii]

        if isinstance(outfile_list, dict):
            outfile_dict = outfile_list

        missing_keys = 0
        for infile in infile_list:
            if infile not in outfile_dict.keys():
                logger.error(f"Missing output file for infile: {infile}")
                missing_keys += 1
            if missing_keys > 0:
                logger.error(f"Missing {missing_keys} output file names.")
                return None

    if target_res is None:
        logger.debug("Calculating target resolution ... ")

        # Keep track of beams and pixel sizes
        beam_list = []
        pix_list = []

        for this_infile in infile_list:
            logger.info(f"Checking {this_infile}")

            cube = SpectralCube.read(this_infile)

            pixel_scales = proj_plane_pixel_scales(cube.wcs.celestial) * u.deg
            pixel_as = [p for p in pixel_scales][0]

            beam_list.append(cube.beam)
            pix_list.append(pixel_as.to(u.arcsec).value)

        # Calculate a common beam
        beam_list = Beams(beams=beam_list)
        common_beam = beam_list.common_beam()
        bmaj = common_beam.major.to(u.arcsec).value

        max_pix = np.max(pix_list)

        # If we're not convolving directly, then we need to pad out a little
        # to avoid convolution artifacts
        pixel_padding = 0
        if convolve_fn not in ["convolve_uv"]:
            pixel_padding = 2

        target_bmaj = np.sqrt(bmaj**2 + (pixel_padding * max_pix) ** 2) * u.arcsec

        # Ensure we just have a single quantity value here
        if not isinstance(target_bmaj, u.Quantity):
            raise TypeError(f"target_bmaj must be a Quantity, not {type(target_bmaj)}")

    else:
        target_bmaj = target_res * u.arcsec

    logger.info(f"Found a common beam size of {str(target_bmaj)}")

    if not do_convolve:
        return target_bmaj.value

    # Build a target beam
    target_beam = Beam(major=target_bmaj, minor=target_bmaj, pa=0 * u.deg)

    # Convolve each cube to the target beam
    for this_infile in infile_list:
        this_outfile = outfile_dict[this_infile]

        if os.path.exists(this_outfile) and not overwrite:
            logger.info(
                f"{this_outfile} already exists and overwrite is False. Will not overwrite."
            )
            continue

        logger.info(
            f"Convolving {os.path.basename(this_infile)} to {os.path.basename(this_outfile)}"
        )

        cube = SpectralCube.read(this_infile)
        cube.allow_huge_operations = True

        convolve_cube(
            cube=cube,
            target_beam=target_beam,
            outfile=this_outfile,
            convolve_fn=convolve_fn,
            nan_treatment=nan_treatment,
        )

    return target_bmaj.value


def sample_array_edges(
    shape: np.ndarray,
    *,
    n_samples: int,
):
    """
    Get edges from an array

    Given an N-dimensional array shape, sample each edge of the array using
    the requested number of samples (which will include vertices). To do this
    we iterate through the dimensions and for each one we sample the points
    in that dimension and iterate over the combination of other vertices.
    Returns an array with dimensions (N, n_samples)

    Args:
        shape (np.ndarray): The shape of the array to sample
        n_samples (int): The number of samples to take
    """

    all_positions = []
    ndim = len(shape)
    shape = np.array(shape)
    for idim in range(ndim):
        for vertex in range(2**ndim):
            positions = -0.5 + shape * ((vertex & (2 ** np.arange(ndim))) > 0).astype(int)
            positions = np.broadcast_to(positions, (n_samples, ndim)).copy()
            positions[:, idim] = np.linspace(-0.5, shape[idim] - 0.5, n_samples)
            all_positions.append(positions)
    positions = np.unique(np.vstack(all_positions), axis=0).T
    return positions


def common_grid_for_mosaic(
    infile_list: list | None = None,
    outfile_list: list | dict | None = None,
    template_name: str | None = None,
    target_hdr: fits.Header | None = None,
    allow_big_image: bool = False,
    too_big_pix: int | float = 1e4,
    overwrite: bool = False,
):
    """
    Build a common astrometry for a mosaic and align all input image
    files to that astrometry. If the common astrometry isn't supplied
    as a header, the program calls other routines to create it based
    on the supplied parameters and stack of input images. Returns the
    common header.

    Args:
        infile_list (list|None): a list of input image files.
        outfile_list (list|dict|None): a list or dictionary of output image files.
        template_name (str): Name format for the output template cubes that
            will define the final mosaic.
        target_hdr (fits.Header): User-provided header. Defaults to None,
            which will build an optimal header
        allow_big_image (bool, optional): If True, allow big images to be created.
            Defaults to False.
        too_big_pix (int|float, optional): The threshold for what constitutes a "big"
            image in pixels. Defaults to 1e4.
        overwrite (bool): Whether to overwrite existing images. Defaults to False.

    Returns:
        fits.Header: Target common header
    """

    if infile_list is None:
        logger.error("Infile list missing.")
        return None

    if len(infile_list) == 0:
        logger.error("No files to process.")
        return None

    for this_infile in infile_list:
        if not os.path.exists(this_infile):
            logger.error(f"File {this_infile} not found. Continuing.")
            continue

    if outfile_list is None:
        logger.error("Outfile list missing.")
        return None

    if template_name is None:
        logger.error("Need to supply a template name.")
        return None

    # Make sure that the outfile list is a dictionary
    if not isinstance(outfile_list, list) and not isinstance(outfile_list, dict):
        logger.error("outfile_list must be dictionary or list.")
        return None

    outfile_dict = {}
    if isinstance(outfile_list, list):
        if len(infile_list) != len(outfile_list):
            logger.error("Mismatch in input and output list lengths.")
            return None

        outfile_dict = {}
        for ii in range(len(infile_list)):
            outfile_dict[infile_list[ii]] = outfile_list[ii]

    if isinstance(outfile_list, dict):
        outfile_dict = outfile_list

    # Get the common header if one is not supplied
    if target_hdr is None:
        logger.info("Generating target header.")

        target_hdr = build_common_header(
            infile_list=infile_list,
            allow_big_image=allow_big_image,
            too_big_pix=too_big_pix,
        )

    if target_hdr is None:
        logger.error("No target header, and was not able to build one.")
        return None

    # Create three template headers: one for flux, one for weight, one for mask.
    # They're the same thing, but with different units and dtypes
    logger.info("Creating template headers for flux, weight, and mask.")

    unitless_hdr = copy.deepcopy(target_hdr)
    unitless_hdr.pop("BUNIT", None)

    mask_hdr = copy.deepcopy(target_hdr)
    mask_hdr.update({"BITPIX": 8})

    target_hdr.totextfile(template_name.replace(".fits", "_flux.txt"), overwrite=True)
    unitless_hdr.totextfile(template_name.replace(".fits", "_weight.txt"), overwrite=True)
    mask_hdr.totextfile(template_name.replace(".fits", "_mask.txt"), overwrite=True)

    # Align the input files to the new astrometry. This will also loop
    # over and align any "weight" files.
    logger.info("Aligning image files")

    for this_infile in infile_list:
        this_outfile = outfile_dict[this_infile]

        if os.path.exists(this_outfile) and not overwrite:
            logger.info(
                f"{os.path.basename(this_outfile)} already exists and overwrite is False. Will not overwrite."
            )
            continue

        logger.info(
            f"Reprojecting {os.path.basename(this_infile)} to {os.path.basename(this_outfile)}"
        )

        # Get the spectral axis out from the target header
        w = WCS(target_hdr)

        cube = SpectralCube.read(this_infile)

        # We start by doing a reproject to the minimal size to slot
        # the array neatly into an output array. Do this in 2D
        w_2d = w[0, :, :]
        cube_w_2d = cube.wcs[0, :, :]

        edges = sample_array_edges(cube.shape[1:], n_samples=11)[::-1]
        edges_out = w_2d.world_to_pixel(cube_w_2d.pixel_to_world(*edges))[::-1]

        # Determine the cutout parameters

        # In some cases, images might not have valid coordinates in the corners,
        # such as all-sky images or full solar disk views. In this case we skip
        # this step and just use the full output WCS for reprojection.
        optimal_shape = w.array_shape[1:]
        ndim_out = len(optimal_shape)

        skip_data = False
        if np.any(np.isnan(edges_out)):
            bounds = list(zip([0] * ndim_out, optimal_shape))
        else:
            bounds = []
            for idim in range(ndim_out):
                imin = max(0, int(np.floor(edges_out[idim].min() + 0.5)))
                imax = min(optimal_shape[idim], int(np.ceil(edges_out[idim].max() + 0.5)))
                bounds.append((imin, imax))
                if imax < imin:
                    skip_data = True
                    break

        if skip_data:
            raise Exception("Could not force cube into optimal header!")

        bounds_as_slice = tuple([slice(b[0], b[1]) for b in bounds])
        slice_out = tuple([slice(imin, imax) for (imin, imax) in bounds])

        # Slice out the part of the WCS we care about, and do the reprojection
        indiv_w = w[:, *slice_out]

        cube = reproject_to_other_wcs(
            cube=cube,
            wcs=indiv_w,
        )

        # And we also need to write out the details for the slice, so we can put this back into the view
        # later
        with open(this_outfile.replace(".fits", "_slice.txt"), "w+") as f:
            f.write(str(slice_out))

        cube.write(this_outfile, overwrite=True)

    return target_hdr


def noise_for_cube(
    infile: str | None = None,
    maskfile: str | None = None,
    exclude_mask: bool = True,
):
    """
    Get a single noise estimate for an image cube.

    TODO: maskfile functionality has not been tested

    Args:
        infile (str | None): the input image file.
        maskfile (str | None): the mask image file.
        exclude_mask (default True): if True, mask is excluded.
            Defaults to True.

    Returns:
        float: Noise value
    """

    if infile is None:
        logger.error("No infile specified.")
        return None

    if not os.path.exists(infile):
        logger.error(f"infile specified but not found - {infile}")
        return None

    if maskfile is not None:
        if not os.path.exists(maskfile):
            logger.error(f"maskfile specified but not found - {maskfile}")
            return None

    # Read in cubes
    cube = SpectralCube.read(infile)
    cube.allow_huge_operations = True

    if maskfile is not None:
        mask = SpectralCube.read(maskfile)
        mask.allow_huge_operations = True
        mask.with_fill_value(0)

        mask = mask.filled_data[:].astype(bool)

        # If the mask is not an exclude, then flip the signs
        if not exclude_mask:
            mask = ~mask

    else:
        mask = None

    this_noise = mad_zero_centered(
        data=cube.unitless_filled_data[:],
        mask=mask,
    )

    return this_noise


def generate_weight_file(
    image_file: str | None = None,
    input_file: str | None = None,
    input_value: float | int | None = None,
    input_type: str = "pb",
    outfile: str | None = None,
    scale_by_noise: bool = False,
    mask_for_noise: str | None = None,
    noise_value: float | None = None,
    scale_by_factor: float | None = None,
    overwrite: bool = False,
):
    """
    Generate a weight image for use in a linear mosaic.

    The weight image will be used in the linear mosaicking as a weight,
    multiplied by the image file and then divided out. The optimal S/N
    choice is often 1/noise^2. The program gives some options to
    calculate a weight image of this type from the data or using the
    primary beam response.

    Args:
        image_file (str | None): the input image file
        input_file (str | None): the input file for weight calculation
        input_value (float | int | None): the input value for weight calculation
        input_type (str): the type of input. Should be one of
            - 'pb' for primary beam response (weight goes at pb^2)
            - 'noise' for a noise estimate (weight goes as 1/noise^2)
            - 'weight' for a weight value
        outfile (str | None): the output weight file
        scale_by_noise (bool): whether to scale by 1/noise^2
        mask_for_noise (str | None): the mask for noise calculation
        noise_value (float | None): the noise value in the image.
            If not supplied, the program will calculate it from the image.
        scale_by_factor (float | None): the factor to scale by
        overwrite (bool): whether to overwrite existing files

    Returns:
        bool: True if successful, False otherwise
    """

    # Check input
    if image_file is None and input_file is None:
        logger.error("I need either an input or an image template file.")
        return False

    if input_file is None and input_value is None:
        logger.error("I need either an input value or an input file.")
        return False

    if input_file is not None and input_value is not None:
        logger.error("I need ONE OF an input value or an input file. Got both.")
        return False

    if outfile is None:
        logger.error("Specify output file.")
        return False

    if input_file is not None:
        valid_types = ["pb", "noise", "weight"]
        if input_type not in valid_types:
            logger.error(f"Valid input types are : {valid_types}")
            return False

    if input_file is None and input_value is None:
        logger.error("Need either an input value or an input file.")
        return False

    if input_file is not None:
        if not os.path.exists(input_file):
            logger.error(f"Missing input file directory - {input_file}")
            return False

    if image_file is not None:
        if not os.path.exists(image_file):
            logger.error(f"Missing image file directory - {image_file}")
            return False

    if not overwrite and os.path.exists(outfile):
        logger.info(f"{outfile} exists and overwrite is False. Skipping")
        return True

    # If scaling by noise is requested and no estimate is provided,
    # generate an estimate

    if scale_by_noise:
        if noise_value is None and image_file is None:
            logger.error("I can only scale by the noise if I have a noise value or an image.")
            return False

        if noise_value is None:
            logger.info(f"Calculating noise for {image_file}")

            noise_value = noise_for_cube(
                infile=image_file,
                maskfile=mask_for_noise,
                exclude_mask=True,
            )
            if np.isnan(noise_value):
                raise ValueError("Could not calculate noise")

            logger.info(f"Noise: {noise_value}")

    # Define the template for the astrometry
    template = copy.deepcopy(input_file)
    if template is None:
        template = copy.deepcopy(image_file)

    logger.debug(f"Template for weight file is: {template}")

    data = SpectralCube.read(template)
    data.allow_huge_operations = True
    weight = SpectralCube.read(template)
    weight.allow_huge_operations = True

    # Case 1 : We just have an input value
    if input_file is None and input_value is not None:
        weight_value = get_weight_value(
            input_type=input_type,
            input_value=input_value,
            scale_by_factor=scale_by_factor,
            scale_by_noise=scale_by_noise,
            noise_value=noise_value,
        )

        # Make sure we end up unitless
        weight = weight / weight * weight_value

    # Case 2 : We have an input image. Manipulate data into a weight
    # array
    if input_file is not None:
        weight = get_weight_value(
            input_type=input_type,
            input_value=data,
            scale_by_factor=scale_by_factor,
            scale_by_noise=scale_by_noise,
            noise_value=noise_value,
        )

        if not isinstance(weight, SpectralCube):
            raise TypeError("weight should be a SpectralCube object")

    weight.write(outfile, overwrite=True)

    return True


def get_weight_value(
    input_type: str = "pb",
    input_value: float | int | None = None,
    scale_by_factor: float | int | None = None,
    scale_by_noise: bool = False,
    noise_value: float | int | None = None,
):
    """Get weight value, given specific input types and values

    Args:
        input_type: the type of input. This can be
            - 'pb' for primary beam response (weight goes at pb^2)
            - 'noise' for a noise estimate (weight goes as 1/noise^2)
            - 'weight' for a weight value
        input_value: an input value of some type (see below) used to form
            the basis for the weight. This is a single value. Only one of
            input_file or input_value can be set.
        scale_by_factor: a factor to scale the weight
        scale_by_noise: scale the weight by noise? Defaults to False
        noise_value: the value to use for noise

    Returns:
        float|SpectralCube: The calculated weight value
    """

    if input_value is None:
        raise ValueError("Need an input value")

    if input_type == "noise":
        weight = 1.0 / input_value**2
    elif input_type == "pb":
        weight = input_value**2
    elif input_type == "weight":
        weight = input_value
    else:
        raise ValueError(f"Unknown input type: {input_type}")

    # Apply scaling
    if scale_by_factor is not None:
        weight *= scale_by_factor
    if scale_by_noise:
        if noise_value is None:
            raise ValueError("If scaling by noise, then noise value should be defined")

        weight /= noise_value**2

    return weight


def mosaic_aligned_data(
    infile_list: list | None = None,
    weightfile_list: list | dict | None = None,
    template_name: str | None = None,
    outfile: str | None = None,
    overwrite: bool = False,
):
    """
    Combine a list of previously aligned data into a single image
    using linear mosaicking. Weight each file using a corresponding
    weight file and also create sum and integrated weight files.

    Args:
        infile_list (list). List of input files
        weightfile_list (list|dict). List of weight files. Can be a
            dictionary or a list. If it's a list then
            matching is by order, so that the first infile goes to first
            weight file, etc. If it's a dictionary, it looks for the infile
            name as a key.
        template_name (str): Name for the path to template cubes for the
            mosaic.
        outfile (str). The name of the output mosaic image.
        overwrite (bool, optional) : Delete existing files.
            Defaults to False

    Returns:
        bool: True if successful, False otherwise
    """

    # Check inputs
    if infile_list is None:
        logger.error("Input file list required.")
        return False

    if len(infile_list) == 0:
        logger.error("No files to process.")
        return False

    if weightfile_list is None:
        logger.error("Missing weightfile_list required for mosaicking.")
        return False

    if template_name is None:
        logger.error("Missing template_name required for mosaicking.")
        return False

    if outfile is None:
        logger.error("Output file is required.")
        return False

    # Define some extra outputs and then check file existence
    sum_file = outfile.replace(".fits", ".sum.fits")
    weight_file = outfile.replace(".fits", ".weight.fits")
    mask_file = outfile.replace(".fits", ".mask.fits")

    for this_file in [outfile, sum_file, weight_file, mask_file]:
        if os.path.exists(this_file) and not overwrite:
            logger.error(
                f"Output file {this_file} present and overwrite False. Will not create mosaic"
            )
            return False

    # Get the weight file list together
    if not isinstance(weightfile_list, list) and not isinstance(weightfile_list, dict):
        logger.error("Weightfile_list must be dictionary or list.")
        return False

    weightfile_dict = {}
    if isinstance(weightfile_list, list):
        if len(infile_list) != len(weightfile_list):
            logger.error("Mismatch in input and output list lengths.")
            return False

        for ii in range(len(infile_list)):
            weightfile_dict[infile_list[ii]] = weightfile_list[ii]

    if isinstance(weightfile_list, dict):
        weightfile_dict = weightfile_list

    # And get out slice indices for viewing the subcube in the full mosaic
    slice_dict = {}
    for this_infile in infile_list:
        slice_dict[this_infile] = this_infile.replace(".fits", "_slice.txt")

    # Check file existence
    for this_infile in infile_list:
        if not os.path.exists(this_infile):
            logger.error(f"Missing file - {this_infile}")
            return False

        this_weightfile = weightfile_dict[this_infile]
        if not os.path.exists(this_weightfile):
            logger.error(f"Missing file - {this_weightfile}")
            return False

        this_slice = slice_dict[this_infile]
        if not os.path.exists(this_slice):
            logger.error(f"Missing file - {this_slice}")
            return False

    # Now, we do the mosaicking! Build empty files from template headers
    template_flux_header_file = template_name.replace(".fits", "_flux.txt")
    template_flux_header = fits.Header.fromtextfile(template_flux_header_file)

    template_weight_header_file = template_name.replace(".fits", "_weight.txt")
    template_weight_header = fits.Header.fromtextfile(template_weight_header_file)

    template_mask_header_file = template_name.replace(".fits", "_mask.txt")
    template_mask_header = fits.Header.fromtextfile(template_mask_header_file)

    create_large_fits(
        f=sum_file,
        header=template_flux_header,
    )
    create_large_fits(
        f=weight_file,
        header=template_weight_header,
    )
    create_large_fits(
        f=mask_file,
        header=template_mask_header,
    )
    create_large_fits(
        f=outfile,
        header=template_flux_header,
    )

    # Load in all the cubes and slices. Use a fill value of 0
    infile_cubes = []
    weightfile_cubes = []
    slices = []

    for this_infile in infile_list:
        this_weightfile = weightfile_dict[this_infile]
        this_slice = slice_dict[this_infile]

        cube = SpectralCube.read(this_infile)
        cube.allow_huge_operations = True
        cube = cube.with_fill_value(0)

        infile_cubes.append(cube)

        weight_cube = SpectralCube.read(this_weightfile)
        weight_cube.allow_huge_operations = True
        weight_cube = weight_cube.with_fill_value(0)

        weightfile_cubes.append(weight_cube)

        with open(this_slice, "r") as f:
            slice_str = f.read()
            slice_tuple = eval(slice_str)
            slices.append(list(slice_tuple))

    # Perform the various calculations. We do this channel-by-channel to minimize RAM requirements
    with (
        fits.open(sum_file, mode="update") as sum_hdu,
        fits.open(weight_file, mode="update") as weight_hdu,
        fits.open(mask_file, mode="update") as mask_hdu,
        fits.open(outfile, mode="update") as out_hdu,
    ):
        n_chan = sum_hdu[0].data.shape[0]

        # Make sure we have the beam and units right, and if not replace them
        for hdu in [sum_hdu, weight_hdu, mask_hdu, out_hdu]:
            keys_to_update = [
                "BMAJ",
                "BMIN",
                "BPA",
                "BEAM",
                "BUNIT",
            ]

            for key in keys_to_update:
                if key in hdu[0].header:
                    if hdu[0].header[key] != infile_cubes[0].header[key]:
                        hdu[0].header[key] = infile_cubes[0].header[key]

            hdu.flush()

        logger.info("Processing mosiac channel-by-channel")

        with ProgressBar(n_chan) as bar:
            # Loop over each channel
            for chan in range(n_chan):
                # Pull out the channel from the cube
                sum_hdu_idx = sum_hdu[0].data[chan]
                weight_hdu_idx = weight_hdu[0].data[chan]

                # Loop over each input cube
                for cube_idx in range(len(infile_cubes)):
                    slice_idx = slices[cube_idx]
                    flux_idx = infile_cubes[cube_idx]
                    weight_idx = weightfile_cubes[cube_idx]

                    # For the slice this cube corresponds to, calculate weighted sum and weight
                    sum_hdu_idx[*slice_idx] += (
                        flux_idx.unitless_filled_data[chan] * weight_idx.unitless_filled_data[chan]
                    )
                    weight_hdu_idx[*slice_idx] += weight_idx.unitless_filled_data[chan]

                # Put NaNs back in
                sum_hdu_idx[sum_hdu_idx == 0] = np.nan

                # Get the mask out as anywhere we've got weight
                mask_hdu_idx = weight_hdu_idx != 0
                mask_hdu_idx = mask_hdu_idx.astype(np.uint8)

                # Calculate the overall mosaic by dividing the sum by the weight
                out_hdu_idx = sum_hdu_idx / weight_hdu_idx

                # Put channels back into cube
                sum_hdu[0].data[chan] = copy.deepcopy(sum_hdu_idx)
                weight_hdu[0].data[chan] = copy.deepcopy(weight_hdu_idx)
                mask_hdu[0].data[chan] = copy.deepcopy(mask_hdu_idx)
                out_hdu[0].data[chan] = copy.deepcopy(out_hdu_idx)

                # Save out changes to disk
                sum_hdu.flush()
                weight_hdu.flush()
                mask_hdu.flush()
                out_hdu.flush()

                bar.update()

    return True
