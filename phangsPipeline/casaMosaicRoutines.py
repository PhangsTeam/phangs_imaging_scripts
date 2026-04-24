"""
Standalone routines related to linear mosaicking of multi-part mosaics
in CASA.
"""

import copy
import logging
import os

import analysisUtils as au
import numpy as np

from . import casaMaskingRoutines as cma
from . import casaStuff
from . import casaCubeRoutines as ccr

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#region Routines to match resolution

def common_res_for_mosaic(
    infile_list = None,
    outfile_list = None,
    target_res=None,
    pixel_padding=2.0,
    do_convolve=True,
    overwrite=False
    ):
    """
    Convolve multi-part cubes to a common res for mosaicking.

    infile_list : list of input files.

    outfile_list : if do_convolve is true, a list of output files that
    will get the convolved data. Can be a dictionary or a list. If
    it's a list then matching is by order, so that the first infile
    goes to first outfile, etc. If it's a dictionary, it looks for the
    infile name as a key.

    target_res : force this target resolution.

    pixel_padding (default 2.0) : the number of pixels to add to the
    largest common beam (in quadrature) to ensure robust convolution.

    do_convolve (default True) : do the convolution. Otherwise just
    calculates and returns the target resolution.

    overwrite (default False) : Delete existing files. You probably
    want to set this to True but it's a user decision.

    Unless a target resolution is supplied, the routine first
    calculates the common resolution based on the beam size of all of
    the input files. This target resolution is returned as the output
    of the routine. The supplied pixel_padding is used to ensure that
    a convolution kernel can be built by imregrid, since CASA can't
    currently keep the major axis fixed and convolve the minor axis.

    If do_convolve is True, it also convolves all of the input files
    to output files with that resolution. For this it needs a list of
    output files matched to the input file list, either as another
    list or a dictionary.
    """

    # Check inputs.

    # First check that input files are supplied and exist.

    if infile_list is None:
        logger.error("Missing required infile_list.")
        return(None)

    if len(infile_list) == 0:
        logger.error("No files to process.")
        return None

    for this_file in infile_list:
        if os.path.isdir(this_file) == False:
            logger.error("File not found "+this_file)
            return(None)

    # If do_convolve is True then make sure that we have output files
    # and that they match the input files.

    if do_convolve:

        if outfile_list is None:
            logger.error("Missing outfile_list required for convolution.")
            return(None)

        if (type(outfile_list) != type([])) and (type(outfile_list) != type({})):
            logger.error("outfile_list must be dictionary or list.")
            return(None)

        if type(outfile_list) == type([]):
            if len(infile_list) != len(outfile_list):
                logger.error("Mismatch in input and output list lengths.")
                return(None)
            outfile_dict = {}
            for ii in range(len(infile_list)):
                outfile_dict[infile_list[ii]] = outfile_list[ii]

        if type(outfile_list) == type({}):
            outfile_dict = outfile_list

        missing_keys = 0
        for infile in infile_list:
            if infile not in outfile_dict.keys():
                logger.error("Missing output file for infile: "+infile)
                missing_keys += 1
            if missing_keys > 0:
                logger.error("Missing "+str(missing_keys)+" output file names.")
                return(None)

    # Figure out the target resolution if it is not supplied by the user

    if target_res is None:
        logger.debug("Calculating target resolution ... ")

        bmaj_list = []
        pix_list = []

        for this_infile in infile_list:
            logger.info("Checking "+this_infile)

            hdr = casaStuff.imhead(this_infile)

            if (hdr['axisunits'][0] != 'rad'):
                logger.error("ERROR: Based on CASA experience. I expected units of radians.")
                logger.error("I did not find this. Returning. Adjust code or investigate file "+this_infile)
                return(None)
            this_pixel = abs(hdr['incr'][0]/np.pi*180.0*3600.)

            if (hdr['restoringbeam']['major']['unit'] != 'arcsec'):
                logger.error("ERROR: Based on CASA experience. I expected units of arcseconds for the beam.")
                logger.error("I did not find this. Returning. Adjust code or investigate file "+this_infile)
                return(None)
            this_bmaj = hdr['restoringbeam']['major']['value']

            bmaj_list.append(this_bmaj)
            pix_list.append(this_pixel)

        max_bmaj = np.max(bmaj_list)
        max_pix = np.max(pix_list)
        target_bmaj = np.sqrt((max_bmaj)**2+(pixel_padding*max_pix)**2)
    else:
        target_bmaj = force_beam

    logger.info('I found a common beam size of '+str(target_bmaj))

    if not do_convolve:
        return(target_bmaj)

    # With a target resolution and matched lists we can proceed.

    for this_infile in infile_list:
        this_outfile = outfile_dict[this_infile]
        logger.debug("Convolving "+this_infile+' to '+this_outfile)

        if overwrite:
            os.system("rm -rf "+this_outfile)

        casaStuff.imsmooth(imagename=this_infile,
                      outfile=this_outfile,
                      targetres=True,
                      major=str(target_bmaj)+'arcsec',
                      minor=str(target_bmaj)+'arcsec',
                      pa='0.0deg',
                      overwrite=overwrite
                      )

    return(target_bmaj)

#endregion

#region Routines to match astrometry between parts of a mosaic

def calculate_mosaic_extent(
        infile_list = None,
        force_ra_ctr = None,
        force_dec_ctr = None,
        force_freq_ctr = None,
):
    """
    Given a list of input files, calculate the center and extent of
    the mosaic needed to cover them all. Return the results as a
    dictionary.

    infile_list : list of input files to loop over.

    force_ra_ctr (default None) : if set then force the RA center of
    the mosaic to be this value, and the returned extent is the
    largest separation of any image corner from this value in RA.

    force_dec_ctr (default None) : as force_ra_ctr but for
    Declination.

    If the RA and Dec. centers are supplied, then they are assumed to
    be in decimal degrees.
    """

    # Check inputs

    if infile_list is None:
        logger.error("Missing required infile_list.")
        return(None)

    for this_infile in infile_list:
        if not os.path.isdir(this_infile):
            logger.error("File not found "+this_infile+"Returning.")
            return(None)

    # Initialize the list of corner RA and Dec positions.

    ra_list = []
    dec_list = []
    # TBD - right now we assume matched frequency/velocity axis
    freq_list = []

    # Loop over input files and calculate RA and Dec coordinates of
    # the corners.

    myia = au.createCasaTool(casaStuff.iatool)

    for this_infile in infile_list:

        this_hdr = casaStuff.imhead(this_infile)

        if this_hdr['axisnames'][0] != 'Right Ascension':
            logger.error("Expected axis 0 to be Right Ascension. Returning.")
            return(None)
        if this_hdr['axisunits'][0] != 'rad':
            logger.error("Expected axis units to be radians. Returning.")
            return(None)
        if this_hdr['axisnames'][1] != 'Declination':
            logger.error("Expected axis 1 to be Declination. Returning.")
            return(None)
        if this_hdr['axisunits'][1] != 'rad':
            logger.error("Expected axis units to be radians. Returning.")
            return(None)

        this_shape = this_hdr['shape']
        xlo = 0
        xhi = this_shape[0]-1
        ylo = 0
        yhi = this_shape[1]-1

        pixbox = str(xlo)+','+str(ylo)+','+str(xlo)+','+str(ylo)
        blc = casaStuff.imval(this_infile, stokes='I', box=pixbox)

        pixbox = str(xlo)+','+str(yhi)+','+str(xlo)+','+str(yhi)
        tlc = casaStuff.imval(this_infile, stokes='I', box=pixbox)

        pixbox = str(xhi)+','+str(yhi)+','+str(xhi)+','+str(yhi)
        trc = casaStuff.imval(this_infile, stokes='I', box=pixbox)

        pixbox = str(xhi)+','+str(ylo)+','+str(xhi)+','+str(ylo)
        brc = casaStuff.imval(this_infile, stokes='I', box=pixbox)

        ra_list.append(blc['coords'][:,0])
        ra_list.append(tlc['coords'][:,0])
        ra_list.append(trc['coords'][:,0])
        ra_list.append(brc['coords'][:,0])

        dec_list.append(blc['coords'][:, 1])
        dec_list.append(tlc['coords'][:, 1])
        dec_list.append(trc['coords'][:, 1])
        dec_list.append(brc['coords'][:, 1])

        freq_list.append(blc['coords'][:, 2])
        freq_list.append(tlc['coords'][:, 2])
        freq_list.append(trc['coords'][:, 2])
        freq_list.append(brc['coords'][:, 2])

    # Get the minimum and maximum RA and Declination.

    # TBD - this breaks straddling the meridian (RA = 0) or the poles
    # (Dec = 90). Add catch cases or at least error calls for
    # this. Meridian seems more likely to come up, so just that is
    # probably fine.

    min_ra = np.min(np.concatenate(ra_list))
    max_ra = np.max(np.concatenate(ra_list))
    min_dec = np.min(np.concatenate(dec_list))
    max_dec = np.max(np.concatenate(dec_list))

    # TBD - right now we assume matched frequency/velocity axis

    min_freq = np.min(np.concatenate(freq_list))
    max_freq = np.max(np.concatenate(freq_list))

    # If we do not force the center of the mosaic, then take it to be
    # the average of the min and max, so that the image will be a
    # square.

    if force_ra_ctr == None:
        ra_ctr = (max_ra+min_ra)*0.5
    else:
        ra_ctr = force_ra_ctr*np.pi/180.

    if force_dec_ctr == None:
        dec_ctr = (max_dec+min_dec)*0.5
    else:
        dec_ctr = force_dec_ctr*np.pi/180.


    if force_freq_ctr == None:
        freq_ctr = (max_freq+min_freq)*0.5
    else:
        freq_ctr = force_freq_ctr

    # Now calculate the total extent of the mosaic given the center.

    delta_ra = 2.0*np.max([np.abs(max_ra-ra_ctr),
                           np.abs(min_ra-ra_ctr)])
    delta_ra *= np.cos(dec_ctr)
    delta_dec = 2.0*np.max([np.abs(max_dec-dec_ctr),
                            np.abs(min_dec-dec_ctr)])
    delta_freq = 2.0*np.max([np.abs(max_freq-freq_ctr),
                             np.abs(min_freq-freq_ctr)])

    # Put the output into a dictionary.

    output = {
        'ra_ctr':[ra_ctr*180./np.pi,'degrees'],
        'dec_ctr':[dec_ctr*180./np.pi,'degrees'],
        'delta_ra':[delta_ra*180./np.pi*3600.,'arcsec'],
        'delta_dec':[delta_dec*180./np.pi*3600.,'arcsec'],
        'freq_ctr':[freq_ctr,'Hz'],
        'delta_freq':[delta_freq,'Hz'],
    }
    return(output)

def build_common_header(
        infile_list = None,
        template_file = None,
        ra_ctr = None,
        dec_ctr = None,
        delta_ra = None,
        delta_dec = None,
        freq_ctr = None,
        delta_freq = None,
        allow_big_image = False,
        too_big_pix=1e4,
):
    """
    Build a target header to be used as a template by imregrid when
    setting up linear mosaicking operations.

    infile_list : the list of input files. Used to generate the
    center, extent, and pick a template file if these things aren't
    supplied by the user.

    template_file : the name of a file to use as the template. The
    coordinate axes and size are manipulated but other things like the
    pixel size and units remain the same. If this is not supplied the
    first file from the input file list is selected.

    ra_ctr : the center of the output file in right ascension. Assumed
    to be in decimal degrees. If None or not supplied, then this is
    calculated from the image stack.

    dec_ctr : as ra_ctr but for declination.

    delta_ra : the extent of the output image in arcseconds. If this
    is not supplied, it is calculated from the image stack.

    delta_dec : as delta_ra but for declination.

    allow_big_image (default False) : allow very big images? If False
    then the program throws an error if the image appears too
    big. This is often the sign of a bug.

    too_big_pix (default 1e4) : definition of pixel scale (in one
    dimension) that marks an image as too big.
    """

    # Check inputs

    if template_file is None:

        if infile_list is None:
            logger.error("Missing required infile_list and no template file.")
            return(None)

        template_file = infile_list[0]
        logger.info("Using first input file as template - "+template_file)

    if infile_list is not None:
        for this_infile in infile_list:
            if not os.path.isdir(this_infile):
                logger.error("File not found "+this_infile+" . Returning.")
                return(None)

    if template_file is not None:
        if os.path.isdir(template_file) == False:
            logger.error("The specified template file does not exist.")
            return(None)

    if infile_list is None:

        if template_file is None:
            logger.error("Without an input file stack, I need a template file.")
            return(None)

        if (delta_ra is None) or (delta_dec is None) or (ra_ctr is None) or (dec_ctr is None):
            logger.error("Without an input file stack, I need ra_ctr, dec_ctr, delta_ra, delta_dec.")
            return(None)

    # If the RA and Dec center and extent are not full specified, then
    # calculate the extent based on the image stack.

    if (delta_ra is None) or (delta_dec is None) or \
            (ra_ctr is None) or (dec_ctr is None):

        logger.info("Extent not fully specified. Calculating it from image stack.")
        extent_dict = calculate_mosaic_extent(
            infile_list = infile_list,
            force_ra_ctr = ra_ctr,
            force_dec_ctr = dec_ctr,
            force_freq_ctr = freq_ctr
            )

        if ra_ctr is None:
            ra_ctr = extent_dict['ra_ctr'][0]
        if dec_ctr is None:
            dec_ctr = extent_dict['dec_ctr'][0]
        if delta_ra is None:
            delta_ra = extent_dict['delta_ra'][0]
        if delta_dec is None:
            delta_dec = extent_dict['delta_dec'][0]

        # Just assume Doppler
        if freq_ctr is None:
            freq_ctr = extent_dict['freq_ctr'][0]
        if delta_freq is None:
            delta_freq = extent_dict['delta_freq'][0]

    # Get the header from the template file

    target_hdr = casaStuff.imregrid(template_file, template='get')

    # Get the pixel scale. This makes some assumptions. We could put a
    # lot of general logic here, but we are usually working in a
    # case where this works.

    if (target_hdr['csys']['direction0']['units'][0] != 'rad') or \
            (target_hdr['csys']['direction0']['units'][1] != 'rad'):
        logger.error("ERROR: Based on CASA experience. I expected pixel units of radians.")
        logger.error("I did not find this. Returning. Adjust code or investigate file "+infile_list[0])
        return(None)

    # Add our target center pixel values to the header after
    # converting to radians.

    ra_ctr_in_rad = ra_ctr * np.pi / 180.
    dec_ctr_in_rad = dec_ctr * np.pi / 180.

    target_hdr['csys']['direction0']['crval'][0] = ra_ctr_in_rad
    target_hdr['csys']['direction0']['crval'][1] = dec_ctr_in_rad
    target_hdr['csys']['spectral1']['wcs']['crval'] = freq_ctr

    # Calculate the size of the image in pixels and set the central
    # pixel coordinate for the RA and Dec axis.

    ra_pix_in_as = np.abs(target_hdr['csys']['direction0']['cdelt'][0]*180./np.pi*3600.)
    ra_axis_size = np.ceil(delta_ra / ra_pix_in_as) + 1
    new_ra_ctr_pix = (ra_axis_size + 1) /2.0

    dec_pix_in_as = np.abs(target_hdr['csys']['direction0']['cdelt'][1]*180./np.pi*3600.)
    dec_axis_size = np.ceil(delta_dec / dec_pix_in_as) + 1
    new_dec_ctr_pix = (dec_axis_size + 1)/2.0

    freq_pix_in_hz = np.abs(target_hdr['csys']['spectral1']['wcs']['cdelt'])
    freq_axis_size = np.ceil(delta_freq / freq_pix_in_hz) + 1
    # +1 or the 1-indexing
    new_freq_ctr_pix = (freq_axis_size + 1) / 2.0

    # Check that the axis size isn't too big. This is likely to be a
    # bug. If allowbigimage is True then bypass this, otherwise exit.

    if not allow_big_image:
        if ra_axis_size > too_big_pix or \
                dec_axis_size > too_big_pix:
            logger.error("WARNING! This is a very big image you plan to create, "+str(ra_axis_size)+ \
                             " x "+str(dec_axis_size))
            logger.error(" To make an image this big set allowbigimage=True. Returning.")
            return(None)

    # Enter the new values into the header and return.

    target_hdr['csys']['direction0']['crpix'][0] = new_ra_ctr_pix
    target_hdr['csys']['direction0']['crpix'][1] = new_dec_ctr_pix
    target_hdr['csys']['spectral1']['wcs']['crpix'] = new_freq_ctr_pix

    target_hdr['shap'][0] = int(ra_axis_size)
    target_hdr['shap'][1] = int(dec_axis_size)
    target_hdr['shap'][2] = int(freq_axis_size)
    return(target_hdr)

def common_grid_for_mosaic(
    infile_list = None,
    outfile_list = None,
    target_hdr = None,
    template_file = None,
    # could use **kwargs here if this gets much more complicated
    ra_ctr = None,
    dec_ctr = None,
    delta_ra = None,
    delta_dec = None,
    allow_big_image = False,
    too_big_pix=1e4,
    asvelocity=True,
    interpolation='cubic',
    axes=[-1],
    overwrite=False,
    ):
    """
    Build a common astrometry for a mosaic and align all input image
    files to that astrometry. If the common astrometry isn't supplied
    as a header, the program calls other routines to create it based
    on the supplied parameters and stack of input images. Returns the
    common header.

    infile_list : list of input files.

    outfile_list : a list of output files that will get the convolved
    data. Can be a dictionary or a list. If it's a list then matching
    is by order, so that firs infile goes to first outfile, etc. If
    it's a dictionary, it looks for the infile name as a key.

    target_hdr : the CASA-format header used to align the images,
    needs the same format returned by a call to imregrid with
    template='get'.

    ra_ctr, dec_ctr, delta_ra, delta_dec, allow_big_image, too_big_pix
    : keywords passed to the header creation routine. See
    documentation for "build_common_header" to explain these.

    asvelocity, interpolation, axes : keywords passed to the CASA imregrid
    call. See documentation there.

    overwrite (default False) : Delete existing files. You probably
    want to set this to True but it's a user decision.
    """

    # Error checking - mostly the subprograms do this.

    if infile_list is None:
        logger.error("Infile list missing.")
        return(None)

    if len(infile_list) == 0:
        logger.error("No files to process.")
        return None

    for this_infile in infile_list:
        if os.path.isdir(this_infile) == False:
            logger.error("File "+this_infile+" not found. Continuing.")
            continue

    if outfile_list is None:
        logger.error("Outfile list missing.")
        return(None)

    # Make sure that the outfile list is a dictionary

    if (type(outfile_list) != type([])) and (type(outfile_list) != type({})):
        logger.error("outfile_list must be dictionary or list.")
        return(None)

    if type(outfile_list) == type([]):
        if len(infile_list) != len(outfile_list):
            logger.error("Mismatch in input and output list lengths.")
            return(None)
        outfile_dict = {}
        for ii in range(len(infile_list)):
            outfile_dict[infile_list[ii]] = outfile_list[ii]

    if type(outfile_list) == type({}):
        outfile_dict = outfile_list

    # Get the common header if one is not supplied

    if target_hdr is None:

        logger.info('Generating target header.')

        target_hdr = build_common_header(
            infile_list = infile_list,
            template_file = template_file,
            ra_ctr = ra_ctr,
            dec_ctr = dec_ctr,
            delta_ra = delta_ra,
            delta_dec = delta_dec,
            allow_big_image = allow_big_image,
            too_big_pix=too_big_pix,
            )

    if target_hdr is None:
        logger.error('No target header, and was not able to build one.')
        return(None)

    # Align the input files to the new astrometry. This will also loop
    # over and align any "weight" files.

    logger.info('Aligning image files.')

    for this_infile in infile_list:

        this_outfile = outfile_dict[this_infile]

        if overwrite:
            os.system("rm -rf " + this_outfile)

        casaStuff.imregrid(imagename=this_infile,
                      template=target_hdr,
                      output=this_outfile,
                      asvelocity=asvelocity,
                      axes=axes,
                      interpolation=interpolation,
                      overwrite=overwrite)
    return(target_hdr)

#endregion

#region Routines to deal with weighting

def generate_weight_file(
    image_file = None,
    input_file = None,
    input_value = None,
    input_type = 'pb',
    outfile = None,
    scale_by_noise = False,
    mask_for_noise = None,
    noise_value = None,
    scale_by_factor = None,
    overwrite=False,
    ):
    """
    Generate a weight image for use in a linear mosaic. The weight
    image will be used in the linear mosaicking as a weight,
    multiplied by the image file and then divided out. The optimal S/N
    choice is often 1/noise^2. The program gives some options to
    calculate a weight image of this type from the data or using the
    primary beam response.

    image_file : the data cube or image associated with the
    weighting. Used for noise calculation and to supply a template
    astrometry. Not always necessary.

    input_file : an input image of some type (see below) used to form
    the basis for the weight.

    input_value : an input value of some type (see below) used to form
    the basis for the weight. This is a single value. Only one of
    input_file or input_value can be set.

    input_type : the type of input. This can be

    - 'pb' for primary beam response (weight goes at pb^2)
    - 'noise' for a noise estimate (weight goes as 1/noise^2)
    - 'weight' for a weight value

    outfile : the name of the output weight file to write

    scale_by_noise (default False) : if True, then scale the weight
    image by 1/noise^2 . Either the noise_value is supplied or it will
    be calculated from the image.

    mask_for_noise : if supplied, the "True" values in this mask image
    will be passed to the noise estimation routine and excluded from
    the noise calculation.

    noise_value : the noise in the image. If not set, this will be
    calculated from the image.

    scale_by_factor : a factor that will be applied directly to the
    weight image.

    overwrite (default False) : Delete existing files. You probably
    want to set this to True but it's a user decision.

    """

    # Check input

    if image_file is None and input_file is None:
        logger.error("I need either an input or an image template file.")
        return(None)

    if input_file is None and input_value is None:
        logger.error("I need either an input value or an input file.")
        return(None)

    if input_file is not None and input_value is not None:
        logger.error("I need ONE OF an input value or an input file. Got both.")
        return(None)

    if outfile is None:
        logger.error("Specify output file.")
        return(None)

    if input_file is not None:
        valid_types = ['pb', 'noise', 'weight']
        if input_type not in valid_types:
            logger.error("Valid input types are :"+str(valid_types))
            return(None)

    if input_file is None and input_value is None:
        logger.error("Need either an input value or an input file.")
        return(None)

    if input_file is not None:
        if not os.path.isdir(input_file):
            logger.error("Missing input file directory - "+input_file)
            return(None)

    if image_file is not None:
        if not os.path.isdir(image_file):
            logger.error("Missing image file directory - "+image_file)
            return(None)

    if overwrite:
        os.system("rm -rf " + outfile)

    # If scaling by noise is requested and no estimate is provided,
    # generate an estimate

    if scale_by_noise:

        if noise_value is None and image_file is None:
            logger.error("I can only scale by the noise if I have a noise value or an image.")
            return(None)

        if noise_value is None:

            logger.info("Calculating noise for "+image_file)

            # Could use kwargs here to simplify parameter passing. Fine right now, too.

            noise_value = cma.noise_for_cube(
                infile = image_file,
                maskfile = mask_for_noise,
                exclude_mask = True,
                method = 'chauvmad',
                niter=5,
                )

            logger.info("Noise "+str(noise_value))

    # Define the template for the astrometry

    if input_file is None:
        template = image_file
    else:
        template = input_file

    logger.debug('Template for weight file is: '+template)

    # Check the output file

    if os.path.isdir(outfile) or os.path.isfile(outfile):
        if not overwrite:
            logger.error("File exists and overwrite set to false - "+outfile)
            return(None)
        os.system('rm -rf '+outfile)

    # Copy the template and open it
    os.system("cp -r "+template+" "+outfile)

    myia = au.createCasaTool(casaStuff.iatool)
    myia.open(outfile)

    # Check for memory issues
    has_memory_issue, cube_shape = ccr.check_getchunk_putchunk_memory_issue(
        outfile, myia=myia, return_shape=True)

    # Case 1 : We just have an input value.

    if input_file is None and input_value is not None:

        if input_type == 'noise':
            weight_value = 1./input_value**2
        if input_type == 'pb':
            weight_value = input_value**2
        if input_type == 'weight':
            weight_value = input_value

        # Apply scaling
        if scale_by_factor is not None:
            weight_value *= scale_by_factor
        if scale_by_noise:
            weight_value /= (noise_value**2)

        # Write weight value (channel-by-channel if needed)
        if not has_memory_issue:
            data = myia.getchunk()
            weight_image = data*0.0 + weight_value
            myia.putchunk(weight_image)
        else:
            logger.debug('putchunk channel by channel for known memory issue')
            if len(cube_shape) == 2:
                blc = [0, 0] # [X, Y]
                trc = [-1, -1] # [X, Y]
                data_slice = myia.getchunk(blc, trc)
                weight_slice = data_slice*0.0 + weight_value
                myia.putchunk(weight_slice, blc)
            elif len(cube_shape) == 3:
                for ichan in range(cube_shape[2]):
                    blc = [0, 0, ichan]
                    trc = [-1, -1, ichan]
                    data_slice = myia.getchunk(blc, trc)
                    weight_slice = data_slice*0.0 + weight_value
                    myia.putchunk(weight_slice, blc)
            elif len(cube_shape) == 4:
                for istokes in range(cube_shape[3]):
                    for ichan in range(cube_shape[2]):
                        blc = [0, 0, ichan, istokes]
                        trc = [-1, -1, ichan, istokes]
                        data_slice = myia.getchunk(blc, trc)
                        weight_slice = data_slice*0.0 + weight_value
                        myia.putchunk(weight_slice, blc)

    # Case 2 : We have an input image. Read in the data and manipulate
    # it into a weight array.

    if input_file is not None:
        
        if not has_memory_issue:
            data = myia.getchunk()
            
            if input_type == 'noise':
                weight_image = 1./data**2
            if input_type == 'pb':
                weight_image = data**2
            if input_type == 'weight':
                weight_image = data
            
            # Apply scaling
            if scale_by_factor is not None:
                weight_image *= scale_by_factor
            if scale_by_noise:
                weight_image /= (noise_value**2)

            myia.putchunk(weight_image)
        else:
            logger.debug('putchunk channel by channel for known memory issue')
            if len(cube_shape) == 2:
                blc = [0, 0] # [X, Y]
                trc = [-1, -1] # [X, Y]
                input_slice = myia.getchunk(blc, trc)
                
                if input_type == 'noise':
                    weight_slice = 1./input_slice**2
                if input_type == 'pb':
                    weight_slice = input_slice**2
                if input_type == 'weight':
                    weight_slice = input_slice
                if scale_by_factor is not None:
                    weight_slice *= scale_by_factor
                if scale_by_noise:
                    weight_slice /= (noise_value**2)

                myia.putchunk(weight_slice, blc)
            elif len(cube_shape) == 3:
                for ichan in range(cube_shape[2]):
                    blc = [0, 0, ichan]
                    trc = [-1, -1, ichan]
                    input_slice = myia.getchunk(blc, trc)
                    
                    if input_type == 'noise':
                        weight_slice = 1./input_slice**2
                    if input_type == 'pb':
                        weight_slice = input_slice**2
                    if input_type == 'weight':
                        weight_slice = input_slice
                    
                    if scale_by_factor is not None:
                        weight_slice *= scale_by_factor
                    if scale_by_noise:
                        weight_slice /= (noise_value**2)
                    
                    myia.putchunk(weight_slice, blc)
            elif len(cube_shape) == 4:
                for istokes in range(cube_shape[3]):
                    for ichan in range(cube_shape[2]):
                        blc = [0, 0, ichan, istokes]
                        trc = [-1, -1, ichan, istokes]
                        input_slice = myia.getchunk(blc, trc)

                        if input_type == 'noise':
                            weight_slice = 1./input_slice**2
                        if input_type == 'pb':
                            weight_slice = input_slice**2
                        if input_type == 'weight':
                            weight_slice = input_slice

                        if scale_by_factor is not None:
                            weight_slice *= scale_by_factor
                        if scale_by_noise:
                            weight_slice /= (noise_value**2)

                        myia.putchunk(weight_slice, blc)

    myia.close()

    return(None)

#endregion

#region Routines to carry out the mosaicking

def mosaic_aligned_data(
    infile_list = None,
    weightfile_list = None,
    outfile = None,
    overwrite=False,
    ):
    """
    Combine a list of previously aligned data into a single image
    using linear mosaicking. Weight each file using a corresponding
    weight file and also create sum and integrated weight files.

    infile_list : list of input files. Required.

    weightfile_list : a list of weight files that correspond to the
    input files. Can be a dictionary or a list. If it's a list then
    matching is by order, so that the first infile goes to first
    weight file, etc. If it's a dictionary, it looks for the infile
    name as a key.

    outfile : the name of the output mosaic image. Will create
    associated files with ".sum" and ".weight" appended to this file
    name.

    overwrite (default False) : Delete existing files. You probably
    want to set this to True but it's a user decision.

    """

    # Check inputs

    if infile_list is None:
        logger.error("Input file list required.")
        return(None)

    if len(infile_list) == 0:
        logger.error("No files to process.")
        return None

    if outfile is None:
        logger.error("Output file is required.")
        return(None)

    # Define some extra outputs and then check file existence

    sum_file = outfile+'.sum'
    weight_file = outfile+'.weight'
    mask_file = outfile+'.mask'
    temp_file = outfile+'.temp'

    for this_file in [outfile, sum_file, weight_file, temp_file, mask_file]:
        if os.path.isdir(this_file):
            if not overwrite:
                logger.error("Output file present and overwrite off - "+this_file)
                return(None)
            os.system('rm -rf '+this_file)

    # Check the weightfile dictionary/list and get it set.

    if weightfile_list is None:
        logger.error("Missing weightfile_list required for mosaicking.")
        return(None)

    if (type(weightfile_list) != type([])) and (type(weightfile_list) != type({})):
        logger.error("Weightfile_list must be dictionary or list.")
        return(None)

    if type(weightfile_list) == type([]):
        if len(infile_list) != len(weightfile_list):
            logger.error("Mismatch in input and output list lengths.")
            return(None)
        weightfile_dict = {}
        for ii in range(len(infile_list)):
            weightfile_dict[infile_list[ii]] = weightfile_list[ii]

    if type(weightfile_list) == type({}):
        weightfile_dict = weightfile_list

    # Check file existence

    for this_infile in infile_list:
        if not os.path.isdir(this_infile):
            logger.error("Missing file - " + this_infile)
            return(None)
        this_weightfile = weightfile_dict[this_infile]
        if not os.path.isdir(this_weightfile):
            logger.error("Missing file - " + this_weightfile)
            return(None)

    # Define LEL expressions to be fed to immath. These just sum up
    # weight*image and weight. Those produce the .sum and .weight
    # output file.

    full_imlist = []

    lel_exp_sum = ''
    lel_exp_weight = ''

    first = True
    counter = 0

    # We might need to add a 4th (Stokes) axis
    # to prevent crashing
    myia = au.createCasaTool(casaStuff.iatool)

    for this_infile in infile_list:

        # Build out to a list that goes infile1, infile2, ... infilen,
        # weightfile1, weightfile2, ... weightfilen.

        # Ensure we've got all the axes needed
        myia.open(this_infile)
        in_shape = myia.shape()
        if len(in_shape) != 4:
            cur_infile = this_infile + '.add_stokes'
            myia.adddegaxes(outfile=cur_infile, stokes='I', overwrite=True)
        else:
            cur_infile = copy.deepcopy(this_infile)
        myia.close()

        # Do the same for the weight file
        this_weightfile = weightfile_dict[this_infile]
        myia.open(this_weightfile)
        in_shape = myia.shape()
        if len(in_shape) != 4:
            cur_weightfile = this_weightfile + '.add_stokes'
            myia.adddegaxes(outfile=cur_weightfile, stokes='I', overwrite=True)
        else:
            cur_weightfile = copy.deepcopy(this_weightfile)
        myia.close()

        full_imlist.append(cur_infile)
        full_imlist.append(cur_weightfile)

        # Make LEL string expressions that refer to these two images
        # and then increment the counter by 2. So IM0 is the first
        # image, IM1 the first weight, IM2 the second image, IM3 the
        # second weight, and so on.

        this_im = 'IM'+str(counter)
        this_wt = 'IM'+str(counter+1)
        counter += 2

        # LEL expressions that refer to the weighted sum and the
        # weight for this image pair.

        this_lel_sum = '('+this_im+'*'+this_wt+')'
        this_lel_weight = '('+this_wt+')'

        # Chain these together into a full string that adds all of the
        # images together.

        if first:
            lel_exp_sum += this_lel_sum
            lel_exp_weight += this_lel_weight
            first=False
        else:
            lel_exp_sum += '+'+this_lel_sum
            lel_exp_weight += '+'+this_lel_weight

    # Feed our two LEL strings into immath to make the sum and weight
    # images.

    # Check for memory issues
    has_memory_issue, cube_shape = ccr.check_getchunk_putchunk_memory_issue(
        full_imlist[0], myia=myia, return_shape=True)

    # Replace masked pixels - do this channel-by-channel for large cubes
    for thisfile in full_imlist:
        myia.open(thisfile)
        
        if not has_memory_issue:
            if not np.all(myia.getchunk(getmask=True)):
                myia.replacemaskedpixels(0.0)
                myia.set(pixelmask=1)
        else: # Large cube - process channel by channel
            logger.debug(f'Replacing masked pixels channel-by-channel for {thisfile}')
            if len(cube_shape) == 2:
                blc = [0, 0] # [X, Y]
                trc = [cube_shape[0]-1, cube_shape[1]-1] # [X, Y]
                mask_slice = myia.getchunk(blc, trc, getmask=True)
                if not np.all(mask_slice):
                    data_slice = myia.getchunk(blc, trc)
                    data_slice[~mask_slice] = 0.0
                    myia.putchunk(data_slice, blc)
                myia.set(pixelmask=1)

            elif len(cube_shape) == 3:
                for ichan in range(cube_shape[2]):
                    blc = [0, 0, ichan]
                    trc = [cube_shape[0]-1, cube_shape[1]-1, ichan]
                    mask_slice = myia.getchunk(blc, trc, getmask=True)
                    if not np.all(mask_slice):
                        data_slice = myia.getchunk(blc, trc)
                        data_slice[~mask_slice] = 0.0
                        myia.putchunk(data_slice, blc)
                myia.set(pixelmask=1)
                        
            elif len(cube_shape) == 4:
                for istokes in range(cube_shape[3]):
                    for ichan in range(cube_shape[2]):
                        blc = [0, 0, ichan, istokes]
                        trc = [cube_shape[0]-1, cube_shape[1]-1, ichan, istokes]
                        mask_slice = myia.getchunk(blc, trc, getmask=True)
                        if not np.all(mask_slice):
                            data_slice = myia.getchunk(blc, trc)
                            data_slice[~mask_slice] = 0.0
                            myia.putchunk(data_slice, blc)
                myia.set(pixelmask=1)
        myia.close()

    cwd = os.getcwd()
    ppdir = os.chdir(os.path.dirname(full_imlist[0]))
    local_imlist = [os.path.basename(ll) for ll in full_imlist]
    sum_file = os.path.basename(sum_file)
    weight_file = os.path.basename(weight_file)
    temp_file = os.path.basename(temp_file)
    local_outfile = os.path.basename(outfile)
    local_maskfile = os.path.basename(mask_file)

    if not has_memory_issue:
        # Original approach for smaller cubes
        casaStuff.immath(imagename = local_imlist, mode='evalexpr',
                         expr=lel_exp_sum, outfile=sum_file, 
                         stokes='I',
                         imagemd = local_imlist[0])

        casaStuff.immath(imagename = local_imlist, mode='evalexpr',
                         expr=lel_exp_weight, outfile=weight_file, 
                         stokes='I',
                         imagemd = local_imlist[0])
    else:
        # Process channel-by-channel for large cubes
        logger.info('Processing mosaic channel-by-channel to avoid memory issues')
        
        # Create template files
        os.system("cp -r "+local_imlist[0]+" "+sum_file)
        os.system("cp -r "+local_imlist[0]+" "+weight_file)
        
        myia_sum = au.createCasaTool(casaStuff.iatool)
        myia_weight = au.createCasaTool(casaStuff.iatool)
        myia_sum.open(sum_file)
        myia_weight.open(weight_file)
        
        ndim = len(cube_shape)
        
        try:
            if ndim == 3:
                nchan = cube_shape[2]
                logger.info(f'Processing {nchan} channels')
                
                for ichan in range(nchan):
                    if ichan % 10 == 0:
                        logger.info(f'Processing channel {ichan+1}/{nchan}')
                    
                    blc = [0, 0, ichan]
                    
                    # Extract channel slices from all input images
                    temp_chan_images = []
                    for idx, im in enumerate(local_imlist):
                        temp_chan = f'temp_chan_{ichan}_img_{idx}'
                        casaStuff.imsubimage(imagename=im, outfile=temp_chan,
                                           chans=str(ichan), dropdeg=False)
                        temp_chan_images.append(temp_chan)
                    
                    # Process sum
                    temp_sum = f'temp_sum_ch{ichan}'
                    casaStuff.immath(imagename=temp_chan_images, mode='evalexpr', 
                                    expr=lel_exp_sum, outfile=temp_sum)
                    
                    myia_temp = au.createCasaTool(casaStuff.iatool)
                    myia_temp.open(temp_sum)
                    sum_slice = myia_temp.getchunk()
                    myia_temp.close()
                    
                    myia_sum.putchunk(sum_slice, blc)
                    
                    # Process weight
                    temp_weight = f'temp_weight_ch{ichan}'
                    casaStuff.immath(imagename=temp_chan_images, mode='evalexpr', 
                                    expr=lel_exp_weight, outfile=temp_weight)
                    
                    myia_temp.open(temp_weight)
                    weight_slice = myia_temp.getchunk()
                    myia_temp.close()
                    
                    myia_weight.putchunk(weight_slice, blc)
                    
                    # Cleanup temporary files
                    for temp in temp_chan_images + [temp_sum, temp_weight]:
                        os.system('rm -rf ' + temp)
                        
            elif ndim == 4:
                nstokes = cube_shape[3]
                nchan = cube_shape[2]
                total = nstokes * nchan
                logger.info(f'Processing {nstokes} Stokes x {nchan} channels = {total} planes')
                
                counter = 0
                for istokes in range(nstokes):
                    for ichan in range(nchan):
                        counter += 1
                        if counter % 10 == 0:
                            logger.info(f'Processing plane {counter}/{total}')
                        
                        blc = [0, 0, ichan, istokes]
                        
                        # Extract channel/stokes slices
                        temp_chan_images = []
                        for idx, im in enumerate(local_imlist):
                            temp_chan = f'temp_ch{ichan}_st{istokes}_img{idx}'
                            casaStuff.imsubimage(imagename=im, outfile=temp_chan,
                                               chans=str(ichan), stokes=str(istokes), 
                                               dropdeg=False)
                            temp_chan_images.append(temp_chan)
                        
                        # Process sum
                        temp_sum = f'temp_sum_ch{ichan}_st{istokes}'
                        casaStuff.immath(imagename=temp_chan_images, mode='evalexpr',
                                        expr=lel_exp_sum, outfile=temp_sum)
                        
                        myia_temp = au.createCasaTool(casaStuff.iatool)
                        myia_temp.open(temp_sum)
                        sum_slice = myia_temp.getchunk()
                        myia_temp.close()
                        
                        myia_sum.putchunk(sum_slice, blc)
                        
                        # Process weight
                        temp_weight = f'temp_weight_ch{ichan}_st{istokes}'
                        casaStuff.immath(imagename=temp_chan_images, mode='evalexpr',
                                        expr=lel_exp_weight, outfile=temp_weight)
                        
                        myia_temp.open(temp_weight)
                        weight_slice = myia_temp.getchunk()
                        myia_temp.close()
                        
                        myia_weight.putchunk(weight_slice, blc)
                        
                        # Cleanup
                        for temp in temp_chan_images + [temp_sum, temp_weight]:
                            os.system('rm -rf ' + temp)
            
        finally:
            myia_sum.close()
            myia_weight.close()
        
        logger.info('Channel-by-channel sum/weight processing complete')

    # Just to be safe, reset the masks on the two images.

    myia = au.createCasaTool(casaStuff.iatool)
    myia.open(sum_file)
    myia.set(pixelmask=1)
    myia.close()

    myia.open(weight_file)
    myia.set(pixelmask=1)
    myia.close()

    # Now divide the sum*weight image by the weight image.
    if not has_memory_issue:

        casaStuff.immath(imagename = [sum_file, weight_file], mode='evalexpr',
                    expr='iif(IM1 > 0.0, IM0/IM1, 0.0)', outfile=temp_file,
                    imagemd = sum_file)

        # The mask for the final output is where we have any weight. This
        # may not be exactly what's desired in all cases, but it's not
        # clear to me what else to do except for some weight threshold
        # (does not have to be zero, though, I guess).

        casaStuff.immath(imagename = weight_file, mode='evalexpr',
                        expr='iif(IM0 > 0.0, 1.0, 0.0)',
                        outfile=local_maskfile)
    
    else:
        logger.info('Processing division and mask channel-by-channel')
        
        os.system("cp -r "+sum_file+" "+temp_file)
        os.system("cp -r "+weight_file+" "+local_maskfile)
        
        myia_temp = au.createCasaTool(casaStuff.iatool)
        myia_mask = au.createCasaTool(casaStuff.iatool)
        myia_temp.open(temp_file)
        myia_mask.open(local_maskfile)

        try:
            if ndim == 3:
                for ichan in range(nchan):
                    blc = [0, 0, ichan]
                    
                    myia_sum.open(sum_file)
                    sum_slice = myia_sum.getchunk(blc, [cube_shape[0]-1, cube_shape[1]-1, ichan])
                    myia_sum.close()
                    
                    myia_weight.open(weight_file)
                    weight_slice = myia_weight.getchunk(blc, [cube_shape[0]-1, cube_shape[1]-1, ichan])
                    myia_weight.close()
                    
                    # Division
                    result_slice = np.where(weight_slice > 0.0, sum_slice/weight_slice, 0.0)
                    myia_temp.putchunk(result_slice, blc)
                    
                    # Mask
                    mask_slice = np.where(weight_slice > 0.0, 1.0, 0.0)
                    myia_mask.putchunk(mask_slice, blc)
                    
            elif ndim == 4:
                for istokes in range(nstokes):
                    for ichan in range(nchan):
                        blc = [0, 0, ichan, istokes]
                        trc = [cube_shape[0]-1, cube_shape[1]-1, ichan, istokes]
                        
                        myia_sum.open(sum_file)
                        sum_slice = myia_sum.getchunk(blc, trc)
                        myia_sum.close()
                        
                        myia_weight.open(weight_file)
                        weight_slice = myia_weight.getchunk(blc, trc)
                        myia_weight.close()
                        
                        result_slice = np.where(weight_slice > 0.0, sum_slice/weight_slice, 0.0)
                        myia_temp.putchunk(result_slice, blc)
                        
                        mask_slice = np.where(weight_slice > 0.0, 1.0, 0.0)
                        myia_mask.putchunk(mask_slice, blc)
        finally:
            myia_temp.close()
            myia_mask.close()

    # Add in potential Stokes axis to the mask file
    myia = au.createCasaTool(casaStuff.iatool)
    myia.open(local_maskfile)
    in_shape = myia.shape()
    if len(in_shape) != 4:
        cur_maskfile = local_maskfile + '.add_stokes'
        myia.adddegaxes(outfile=cur_maskfile, stokes='I', overwrite=True)
    else:
        cur_maskfile = copy.deepcopy(local_maskfile)
    myia.close()

    # Strip out any degenerate axes and create the final output file.

    casaStuff.imsubimage(imagename=temp_file,
                         outfile=local_outfile,
                         mask='"'+cur_maskfile+'"',
                         overwrite=overwrite,
                         dropdeg=True,
                         )

    # Remove any temp Stokes files we've made along the way
    os.system('rm -rf *.add_stokes')

    os.chdir(cwd)
    return(None)

#endregion
