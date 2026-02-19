import logging
from functools import reduce

import astropy.units as u
import astropy.wcs as wcs
import numpy as np
import scipy.ndimage as nd
import scipy.ndimage.morphology as morph
from astropy.io import fits
from spectral_cube import Projection, SpectralCube

from .scDerivativeRoutines import convert_and_reproject
from . import __version__

np.seterr(divide='ignore', invalid='ignore')

mad_to_std_fac = 1.482602218505602

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def nchan_thresh_mask(cube, thresh=5., nchan=2):
    """
    Make mask from a cube applying a threshold across some number of
    consecutive channels.

    Parameters:

    -----------

    cube : np.array

        A cube in the same units as threshold.

    Keywords:
    ---------

    thresh : float

        Threshold for inclusion in mask if.

     nchan : int

        Number of consecutive channels at thresh required for a detection. Default: 2

    """

    # TBD Error checking on types, dimensionality, etc.

    mask = np.greater_equal(cube, thresh,
                            where=(~np.isnan(cube)),
                            out=np.full(cube.shape, False, dtype=bool))

    kernel = np.ones(nchan, dtype=bool)
    kernel = kernel[:,np.newaxis,np.newaxis]

    mask = morph.binary_opening(mask, kernel)

    return(mask)

def reject_small_regions(mask, min_volume=0, min_area=0):
    """
    Remove small regions from a mask. Small can be defined in either
    volume or area.

    Parameters:

    -----------

    mask : np.array
        A mask.

    Keywords:
    ---------

    minvolume : int
        Minimum volume in pixels. Default 0.

    minarea : int
        Minimum area in pixels. Default 0.

    """

    # TBD Error checking on types, dimensionality, etc.

    # Blob color the cube and loop over regions

    regions, regct = nd.label(mask)
    objslices = nd.find_objects(regions)

    for ii, thisslice in enumerate(objslices):
        subcube = regions[thisslice]

        if min_volume > 0:
            volume =  np.sum(subcube == (ii+1))
            if volume < min_volume:
                mask[regions == (ii+1)] = False

        if min_area > 0:
            if mask.ndim == 3:
                area = np.sum(np.any(subcube == (ii+1), axis=0))
            if mask.ndim == 2:
                area = np.sum(np.any(subcube == (ii+1)))

            if area < min_area:
                mask[regions == (ii+1)] = False

    return(mask)

def grow_mask(mask, iters_xy=0, iters_v=0, constraint=None):
    """
    Grow a boolean mask via dilation in the spectral (v) dimension,
    spatial (xy) dimension, or into a constraint.

    Logic:

    (Case I) If iters_xy and/or iters_v suppplied:

    1. Dilate original mask by iters_xy

    2. Dilate original mask by iters_v

    3. Take the union of the two new masks.

    4. Apply the constraint.

    (Case II) Only a constraint is supplied:

    1. Include all regions of the constraint that include an element
    of the original mask.

    Parameters:

    -----------

    mask : np.array
        A mask.

    Keywords:
    ---------

    iters_xy : int
        Number of iterations of expansion in spatial dimensions.

    iters_v : int
        Number of iterations of expansion in spectral dimension.

    constraint : np.array that can be broadcast to mask
        Another mask to use as a constraint.

    """

    # TBD Error checking on types, dimensionality, etc.

    if iters_xy > 0:

        # Generate the structure to dilate by
        struct = morph.iterate_structure(
            morph.generate_binary_structure(2, 1), iters_xy)

        # Fill in a third dimension if needed
        if mask.ndim == 3:
            struct = struct[np.newaxis, :, :]

        if iters_v > 0:
            mask_xy = morph.binary_dilation(mask, struct)
        else:
            mask = morph.binary_dilation(mask, struct)

    if iters_v > 0:

        struct = np.ones(iters_v, dtype=bool)
        struct = struct[:, np.newaxis, np.newaxis]

        if iters_xy > 0:
            mask_v = morph.binary_dilation(mask, struct)
        else:
            mask = morph.binary_dilation(mask, struct)

    if iters_v > 0 and iters_xy > 0:
        mask = np.logical_or(mask_v, mask_xy)

    if (iters_v > 0 or iters_xy > 0) and (constraint is not None):
        mask = np.logical_and(mask, constraint)

    if (iters_v == 0 and iters_xy == 0) and (constraint is not None):

        # blob color regions in the constraint
        regions, regct = nd.label(constraint)

        # # get a list of all region assignments that have a True value
        # # in the original mask
        # good_regions = np.unique(regions[mask])

        # # create a new mask that includes only these good new regions
        # mask = np.zeros_like(mask, dtype=bool)
        # for hit in good_regions:
        #     mask[regions == hit] = True

        good_regions = regions[mask]

        # build "keep" table: index = region label, value = keep/discard
        keep = np.zeros(regct + 1, dtype=bool)
        keep[good_regions] = True
        keep[0] = False  # never include background label

        # rebuild mask in one vectorized pass
        mask = keep[regions]

    return(mask)

def same_wcs(cube1, cube2):
    """
    Check if two WCS objects are the same.

    Parameters:

    -----------

    wcs1, wcs2 : astropy.wcs.WCS

    """

    shape_check = cube1.shape == cube2.shape
    wcs_string1 = cube1.wcs.to_header_string()
    wcs_string2 = cube2.wcs.to_header_string()
    wcs_check = wcs_string1 == wcs_string2
    return(shape_check and wcs_check)


def cprops_mask(data, noise=None,
                hi_thresh=5, hi_nchan=2,
                lo_thresh=None, lo_nchan=None,
                min_pix=None, min_area=None,
                min_beams=None, ppbeam=None,
                grow_xy=None, grow_v=None,
                prior_hi = None,
                prior_lo = None,
                invert=False):
    """
    Standard CPROPS masking recipe.

    Parameters:

    -----------

    data : np.array

        Original data

    noise : np.array

        Estimate of the amplitude of the noise at every position in the data
        (or an array that will broadcast to that under division).

    Keywords:
    ---------

    hi_thresh : float
        Threshold for detection (in units of sigma).  Default: 5

    hi_nchan : int
        Number of consecutive channels needed for detection.  Default: 2

    lo_thresh : float
        Threshold for inclusion in mask if connected to a hi_thresh
        detection. Default: None

    lo_nchan : int
        Number of consecutive channels at lo_thresh required for a detection
        if connected to a hi_thresh detection. Default: None

    min_pix : int
        Number of pixels required for a detection.  Default: None

    min_area : int
        Minimum number of pixels required in area projection for a detection.
        Default: None

    min_beams : float
        Minimum number of beams for a pixel to be rejected.  Overrides min_pix.
        Default: None

    ppbeam : float
        Number of pixels per beam.  Required if min_beams is set

    grow_xy : int
        Number of iterations to grow the final mask in the xy plane.

    grow_v : int
        Number of iterations to grow the final mask in velocity

    prior_hi : np. array
        Mask that will be applied to the high significance mask before
        any expansion.

    prior_lo : np. array
        Mask that will be applied to the low significance mask.

    invert : bool
        If True, invert the data before applying masking.

        Used for assessing the number of false positives given masking
        criteria. Default: False.

    """

    # TBD error checking, dimensions, types, etc.

    if noise is None:
        logger.warning("No noise estimate supplied. Taking noise to be unity.")
        noise = 1.0

    # Recase the cube into a signal-to-noise cube
    signif = data / noise

    # If requested, invert the data
    if invert:
        signif *= -1

    # Create a the core mask

    hi_mask = nchan_thresh_mask(
        signif, thresh=hi_thresh, nchan=hi_nchan)

    # If requested, reject small regions from the mask
    if ((min_beams is not None)
        or (min_pix is not None)
        or (min_area is not None)):

        if min_pix is None:
            min_pix = 0

        if min_area is None:
            min_area = 0

        if min_beams is not None:
            assert ppbeam is not None
            min_pix = min_beams * ppbeam

        hi_mask = reject_small_regions(
            hi_mask, min_volume=min_pix, min_area=min_area)

    # If a prior is supplied for the high significane mask, apply it

    if prior_hi is not None:
        hi_mask *= prior_hi

    # If supplied, make a lower significance mask and expand into it

    if (lo_thresh is not None) and (lo_nchan is not None):
        lo_mask = nchan_thresh_mask(
            signif, thresh=lo_thresh, nchan=lo_nchan)

        if prior_lo is not None:
            lo_mask *= prior_lo

        # Now expand the original mask into the lower mask
        mask = grow_mask(hi_mask, constraint=lo_mask)
    else:
        mask = hi_mask

    # If requested, grow the mask in xy and v directions. Sequential
    # calls mean that the xy is applied then the v.

    if grow_xy is not None:
        mask = grow_mask(mask, iters_xy=grow_xy)

    if grow_v is not None:
        mask = grow_mask(mask, iters_v=grow_v)

    return(mask)

def mask_around_value(cube, target=None, delta=None):
    """General function to make a mask that includes only values within
    delta of some target value or cube.

    Parameters:

    -----------

    cube : array

    target : array or float

        The target value

    delta : the tolerance

        When cube is within delta of target, return True

    Keywords:
    ---------

    TBD

    """

    # -------------------------------------------------
    # Generate the mask
    # -------------------------------------------------

    mask = np.abs(cube - target) <= delta

    # -------------------------------------------------
    # Return
    # -------------------------------------------------

    return(mask)

def make_vfield_mask(cube_in, vfield_in, window_in,
                     vfield_hdu=0, window_hdu=0,
                     outfile=None, overwrite=True):
    """Make a mask that includes only pixels within +/- some velocity
    window of a provided velocity field, which can be two-d or one-d.

    Parameters:

    -----------

    cube : string or SpectralCube

        The original data cube.

    vfield : float or two-d array or string

        The velocity field to use as a template. If a string, it's read 
        as a Projection and reprojected onto the cube. If it's a single 
        value then this is broadcast across the whole map. If it's a 
        two-d array it is assumed to be the velocity field.

    window : float

        The velocity window about the velocity field. The window
        is applied +/- half the input window so that the total
        span of the window equals the input window.
        The spectral units should match the velocity field units.


    Keywords:
    ---------

    vfield_hdu : optional refers to the HDU of the velocity field if a
        file name is supplied. Default 0.

    window_hdu : optional refers to the HDU of the velocity window if
        a file name is supplied. Default 0.

    outfile : file name to write output mask to.

    overwrite : flag to allow overwrite to rewrite files.

    """

    # -------------------------------------------------
    # Read the cube if string are supplied
    # -------------------------------------------------

    if type(cube_in) == str:
        cube = SpectralCube.read(cube_in)
    else:
        cube = cube_in

    # Get spectral information from the cube
    spaxis = cube.spectral_axis
    spunit = spaxis.unit
    spvalue = spaxis.value
    nz, ny, nx = cube.shape

    # -------------------------------------------------
    # Now read and align the velocity field
    # -------------------------------------------------
    
    # Read the velocity field to a Projection if a file is fed in
    if type(vfield_in) == str:
        vfield = Projection.from_hdu(fits.open(vfield_in)[vfield_hdu])
    elif type(vfield_in) is Projection:
            # ... NB making a dummy moment here because of issues with
            # reproject and dimensionality. Could instead do header
            # manipulation and feed in "cube"
            dummy_mom0 = cube.moment(order=0)      
            vfield = convert_and_reproject(vfield_in, template=dummy_mom0, unit=spunit)
    else:
        logger.error('Input vfield is unknown type.')
    
    vfield = vfield.to(spunit)

    # Check sizes
    if (cube.shape[1] != vfield.shape[0]) or (cube.shape[2] != vfield.shape[1]):
        return(np.nan)

    # -------------------------------------------------
    # Convert and align the window as needed
    # -------------------------------------------------

    # Read the velocity field to a Projection if a file is fed in
    if type(window_in) == str:
        window = Projection.from_hdu(fits.open(window_in)[window_hdu])
    else:
        window = window_in

    # If window is a Projection reproject it onto the cube and match units
    if type(window) is Projection:
        # ... NB making a dummy moment here because of issues with
        # reproject and dimensionality. Could instead do header
        # manipulation and feed in "cube"
        dummy_mom0 = cube.moment(order=0)        
        window = convert_and_reproject(window, template=dummy_mom0, unit=spunit)
    else:
        window = window.to(spunit)

        # If no units are attached to the window, guess that the
        # units are the same as cube        
        if type(window) != u.quantity.Quantity:
            window = u.quantity.Quantity(window,spunit)
        
        # If a single value is supplied, turn it into a single-valued
        # velocity field
        if np.ndim(window.data) <= 1:
            window = window.to(spunit)
            window = u.quantity.Quantity(np.ones((ny, nx))*window.value, window.unit)
        
        # Just in case convert the units to match the cube
        window = window.to(spunit)
    
    # Check sizes
    if (cube.shape[1] != window.shape[0]) or (cube.shape[2] != window.shape[1]):
        return(np.nan)

    # --------------------------------------------------
    # Generate cubes
    # -------------------------------------------------

    # ... holding the velocity at each pixel
    spaxis_cube = np.ones((ny, nx))[None,:,:] * spvalue[:,None,None]

    # ... the reference velocity at each pixel
    vfield_cube = (vfield.value)[None,:,:] * np.ones(nz)[:,None,None]

    # ... and the window width at each pixel
    window_cube = (window.value)[None,:,:] * np.ones(nz)[:,None,None]

    # -------------------------------------------------
    # Make the mask
    # -------------------------------------------------

    # This checks at each pixel if the value (spaxis) is within delta
    # (the window) of the target value (the local velocity field).
    
    mask = mask_around_value( \
        spaxis_cube,
        target=vfield_cube,
        delta=window_cube/2)

    # -------------------------------------------------
    # Output
    # -------------------------------------------------

    mask = SpectralCube(mask.astype(int), wcs=cube.wcs,
                        header=cube.header,
                        meta={'BUNIT': ' ', 'BTYPE': 'Mask'})

    # Write to disk, if desired
    if outfile is not None:
        header = mask.header
        header['DATAMAX'] = 1
        header['DATAMIN'] = 0
        hdu = fits.PrimaryHDU(np.array(mask.filled_data[:], dtype=np.uint8),
                              header=header)
        hdu.writeto(outfile, overwrite=overwrite)

    return(mask)

def join_masks(orig_mask_in, new_mask_in,
               order='bilinear', operation='or',
               outfile=None,
               thresh=0.5,
               yolo=False,
               ):
    """
    Reproject and combine a new mask

    Parameters:

    -----------

    orig_mask_in : string or SpectralCube

        The original mask.

    new_mask_in : string or SpectralCube

        The new mask

    Keywords:
    ---------

    order : string
        Order of interpolation. Passed to spectral cube.

    operation : string
        method to combine the masks 'or' or 'and'

    outfile : string
        Filename where the mask will be written. The mask is also
        returned.

    thresh : float
        Floating point value above which the mask is considered
        true. Relevant because of interpolation. Default 0.5 .

    yolo : bool
        If True, skip the WCS check and just do the nearest neighbor    
    """

    # TBD - check for two dimensional case

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Read the data
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    if type(orig_mask_in) is str:
        orig_mask = SpectralCube.read(orig_mask_in)
    elif type(orig_mask_in) is SpectralCube:
        orig_mask = orig_mask_in
    else:
        logging.error('Unrecognized input type for orig_mask_in')
        raise NotImplementedError

    # Enable large operations
    orig_mask.allow_huge_operations = True

    if type(new_mask_in) is str:
        new_mask = SpectralCube.read(new_mask_in)
    elif type(new_mask_in) is SpectralCube:
        new_mask = new_mask_in
    else:
        logging.error('Unrecognized input type for new_mask_in')
        raise NotImplementedError

    # Enable large operations
    new_mask.allow_huge_operations = True

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Reproject the new mask onto the original WCS
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # Check if the new mask is already on the same WCS as the original mask. 
    # If so, skip reprojection.

    if same_wcs(orig_mask, new_mask) or yolo:
        logger.warn('New mask is already on the same WCS as the original mask. Skipping reprojection.')
        new_mask_vals = np.array(new_mask._data > thresh, dtype=bool)
    else:
        if order == 'fast_nearest_neighbor':

            logger.warn('Joining masks with nearest neighbor coordinate lookup')
            # Grab WCS out of template mask and map to other mask
            x, y, _ = new_mask.wcs.wcs_world2pix(*(orig_mask.world[0,:,:][::-1]), 0)
            _, _, z = new_mask.wcs.wcs_world2pix(*(orig_mask.world[:,0,0][::-1]), 0)

            x = np.rint(x).astype(int)
            y = np.rint(y).astype(int)
            z = np.rint(z).astype(int)
            new_mask_data = np.array(new_mask.filled_data[:].value > thresh, dtype=bool)

            # Create new mask
            new_mask_vals = np.zeros(orig_mask.shape, dtype=bool)
            # Find all values that are in bounds
            inbounds = reduce((lambda A, B: np.logical_and(A, B)),
                            [(z[:,np.newaxis,np.newaxis] >= 0),
                            (z[:,np.newaxis,np.newaxis] < new_mask.shape[0]),
                            (y[np.newaxis,:,:] >= 0),
                            (y[np.newaxis,:,:] < new_mask.shape[1]),
                            (x[np.newaxis,:,:] >= 0),
                            (x[np.newaxis,:,:] < new_mask.shape[2])])
    #        inbounds = np.logical_and.reduce(
            # Look 'em up in the new_mask
            new_mask_vals[inbounds] = new_mask_data[(z[:,np.newaxis,np.newaxis]*np.ones(inbounds.shape, dtype=int))[inbounds],
                                                    (y[np.newaxis,:,:]*np.ones(inbounds.shape, dtype=int))[inbounds],
                                                    (x[np.newaxis,:,:]*np.ones(inbounds.shape, dtype=int))[inbounds]]

        else:

            logger.warn('Joining masks with reprojection')
            new_mask = new_mask.reproject(orig_mask.header, order=order)
            new_mask = new_mask.spectral_interpolate(orig_mask.spectral_axis)
            new_mask_vals = np.array(new_mask.filled_data[:].value > thresh,
                                    dtype=bool)

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Combine
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    if operation.strip().lower() == 'or':
        mask = np.logical_or(np.array(orig_mask._data > thresh,
                                      dtype=bool),
                             new_mask_vals)
    elif operation.strip().lower() == 'and':
        mask = np.logical_and(np.array(orig_mask._data > thresh,
                                       dtype=bool),
                              new_mask_vals)
    elif operation.strip().lower() == 'sum':
        mask = (orig_mask._data > thresh).astype(int) + new_mask_vals
    else:
        logging.error('Unrecognized operation. Not "and" or "or".')
        raise NotImplementedError

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Write to disk, return output, etc.
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    mask = SpectralCube(mask.astype(int), wcs=orig_mask.wcs,
                        header=orig_mask.header,
                        meta={'BUNIT': ' ', 'BTYPE': 'Mask'})

    # Write to disk, if desired
    if outfile is not None:
        header = mask.header
        header['DATAMAX'] = 1
        header['DATAMIN'] = 0
        hdu = fits.PrimaryHDU(np.array(mask._data, dtype=np.uint8),
                              header=header)
        hdu.writeto(outfile, overwrite=True)

    return(mask)

def recipe_phangs_strict_mask(
    cube_in, innoise, outfile=None,
    coverage=None, coverage_thresh=0.95,
    mask_kwargs=None,
    return_spectral_cube=False,
    overwrite=False):
    """
    Task to create the PHANGS-style "strict" masks.

    Parameters:

    -----------

    cube : string or SpectralCube
        The cube to be masked.

    noise : string or SpectralCube
        The noise estimate.

    Keywords:
    ---------

    outfile : string
        Filename where the mask will be written. The mask is also
        returned.

    masks_kwargs : dictionary
        Parameters to be passed to the cprops masking routine.

    """

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Error checking and work out inputs
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # TBD error checking, dimensions, types, etc.

    if type(cube_in) is SpectralCube:
        cube = cube_in
    elif type(cube_in) == type("hello"):
        cube = SpectralCube.read(cube_in)
    else:
        logger.error("Input cube must be a SpectralCube object or a filename.")

    cube.allow_huge_operations = True

    if type(innoise) is SpectralCube:
        rms = innoise
    elif type(innoise) == type("hello"):
        rms = SpectralCube.read(innoise)
    else:
        logger.error("Input noise must be a SpectralCube object or a filename.")

    rms.allow_huge_operations = True

    if coverage is not None:
        if type(coverage) is SpectralCube:
            coverage_cube = coverage
        elif type(coverage) == type("hello"):
            coverage_cube = SpectralCube.read(coverage)
        else:
            logger.error("Coverage cube must be a SpectralCube object or a filename or None.")

        coverage_cube.allow_huge_operations = True

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Set up the masking kwargs
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # Initialize an empty kwargs dictionary
    if mask_kwargs is None:
        mask_kwargs = {}

    # Fill in strict mask defaults
    if 'hi_thresh' not in mask_kwargs:
        mask_kwargs['hi_thresh'] = 4.0

    if 'hi_nchan' not in mask_kwargs:
        mask_kwargs['hi_nchan'] = 2

    if 'lo_thresh' not in mask_kwargs:
        mask_kwargs['lo_thresh'] = 2.0

    if 'lo_nchan' not in mask_kwargs:
        mask_kwargs['lo_nchan'] = 2

    if 'min_pix' not in mask_kwargs:
        mask_kwargs['min_pix'] = None

    if 'min_area' not in mask_kwargs:
        mask_kwargs['min_area'] = None

    if 'grow_xy' not in mask_kwargs:
        mask_kwargs['grow_xy'] = 0

    if 'grow_v' not in mask_kwargs:
        mask_kwargs['grow_v'] = 0

    if 'min_beams' in mask_kwargs:
        apix = wcs.utils.proj_plane_pixel_area(cube.wcs) * u.deg**2
        ppbeam = (np.pi
                  * cube.beam.major
                  * cube.beam.minor
                  / (4 * np.log(2)) / apix)
        ppbeam = ppbeam.to(u.dimensionless_unscaled).value
        mask_kwargs['ppbeam'] = ppbeam

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Create the mask
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    prior_hi = None
    if coverage is not None:

        prior_hi = coverage_cube.filled_data[:].value > coverage_thresh

    # mask = cprops_mask(cube.filled_data[:].value,
    #                    rms.filled_data[:].value,
    #                    prior_hi = prior_hi,
    #                    **mask_kwargs)
    mask = cprops_mask(cube._data,
                       rms._data,
                       prior_hi = prior_hi,
                       **mask_kwargs)

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Write to disk and return
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # In this case can avoid a recast
    if not return_spectral_cube and (outfile is None):
        return(mask)

    # Recast from numpy array to spectral cube
    mask = SpectralCube(mask*1.0, wcs=cube.wcs, header=cube.header,
                        meta={'BUNIT': ' ', 'BTYPE': 'Mask'})

    # Write to disk, if desired
    if outfile is not None:
        header = mask.header
        header['DATAMAX'] = 1
        header['DATAMIN'] = 0
        hdu = fits.PrimaryHDU(np.array(mask.filled_data[:], dtype=np.uint8),
                              header=header)
        hdu.writeto(outfile, overwrite=overwrite)


    if return_spectral_cube:
        return(mask)
    else:
        return(mask.filled_data[:].value)

def recipe_phangs_broad_mask(
        template_mask, outfile=None, list_of_masks = [],
        grow_xy = None, grow_v = None,
        return_spectral_cube=True, overwrite=False,
        recipe='anyscale', fraction_of_scales=0.25,
        yolo=False,
):
    """Task to create the PHANGS-style "broad" masks from the combination
    of a set of other masks. Optionally also grow the mask at the end.

    Parameters:

    -----------

    template_mask : string or SpectralCube

        The original mask that holds the target WCS. The other masks
        will be reprojected onto this one. This mask is included in
        the final output.

    Keywords:
    ---------

    outfile : string

        Filename where the mask will be written. The mask is also
        returned.

    list_of_masks : list

        List of masks or spectral cubes. These will be reprojected
        onto the template mask and then combined via logical or to
        form the final mask.

    grow_xy : int

        Number of spatial dilations of the mask.

    grow_v : int

        Number of spectralwise dilations of the mask.

    recipe : str

        If 'anyscale', the broad mask include pixels that are included
        in any of the contributing masks.  If 'somescales', this only
        include pixels that are in fraction_of_scales masks of the
        origina list.

    fraction_of_scales : float

        Set to the fraction of scales that is the minimum for
        inclusion in the broadmask.

    """

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Error checking and work out inputs
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # TBD error checking, dimensions, types, etc.

    if type(template_mask) is SpectralCube:
        mask = template_mask
    elif type(template_mask) == str:
        mask = SpectralCube.read(template_mask)
    else:
        logger.error("Input mask must be a SpectralCube object or a filename.")
        return(None)

    mask.allow_huge_operations = True

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Loop over other masks
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    for other_mask in list_of_masks:

        if type(other_mask) is SpectralCube:
            other_mask = other_mask
        elif type(template_mask) == type("hello"):
            other_mask = SpectralCube.read(other_mask)
        else:
            logger.error("Input masks must be SpectralCube objects or filenames.")
            return(None)

        other_mask.allow_huge_operations = True

        mask = join_masks(mask, other_mask, operation='sum'
                          , order='fast_nearest_neighbor'
                          , yolo=yolo)

    if recipe == 'anyscale':
        mask_values = mask.filled_data[:].value > 0
        mask = SpectralCube(mask_values*1.0, wcs=mask.wcs, header=mask.header
                            , meta={'BUNIT': ' ', 'BTYPE': 'Mask'})
        mask.allow_huge_operations = True

    if recipe == 'somescales':
        mask_values = mask.filled_data[:].value > (fraction_of_scales
                                                  * len(list_of_masks))
        mask = SpectralCube(mask_values*1.0, wcs=mask.wcs, header=mask.header
                            , meta={'BUNIT': ' ', 'BTYPE': 'Mask'})
        mask.allow_huge_operations = True

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Grow the mask
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    if (grow_xy is not None) or (grow_v is not None):

        mask_values = mask.filled_data[:].value
        if grow_xy is not None:
            mask_values = grow_mask(mask_values, iters_xy=grow_xy)
        if grow_v is not None:
            mask_values = grow_mask(mask_values, iters_v=grow_v)

        mask = SpectralCube(mask_values*1.0, wcs=mask.wcs, header=mask.header
                            , meta={'BUNIT': ' ', 'BTYPE': 'Mask'})

        mask.allow_huge_operations = True

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Write to disk and return
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # Write to disk, if desired
    if outfile is not None:
        header = mask.header
        header['DATAMAX'] = 1
        header['DATAMIN'] = 0
        header['COMMENT'] = 'Produced with PHANGS-ALMA pipeline version ' + __version__
        hdu = fits.PrimaryHDU(np.array(mask.filled_data[:], dtype=np.uint8),
                              header=header)
        hdu.writeto(outfile, overwrite=overwrite)


    if return_spectral_cube:
        return(mask)
    else:
        return(mask.filled_data[:].value)

def recipe_phangs_flat_mask(
    cube_in=None, 
    vfield_in=None, 
    mask_in=None, 
    vfield_hdu=0, 
    outfile=None,
    coverage=None, 
    mask_kwargs=None,
    return_spectral_cube=False,
    overwrite=False):
    """
    Task to create the PHANGS-style "strict" masks.

    Parameters:

    -----------

    cube_in : string or SpectralCube
        The cube to be masked.


    vfield_in : string or np.array
        Velocity field used to define the velocity window of the 
        flat mask.

    mask_in : string or SpectralCube
        The signal-based mask.

    Keywords:
    ---------
    
    vfield_hdu : optional refers to the HDU of the velocity field if a
        file name is supplied. Default 0.

    outfile : string
        Filename where the mask will be written. The mask is also
        returned.

    masks_kwargs : dictionary
        Parameters to be passed to the cprops masking routine.
        Should contain the velocity window used to define a flat
        mask.

    """

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Error checking and work out inputs
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # TBD error checking, dimensions, types, etc.

    # check input cube
    if type(cube_in) is SpectralCube:
        cube = cube_in
    elif type(cube_in) == str:
        cube = SpectralCube.read(cube_in)
    else:
        logger.error("Input cube must be a SpectralCube object or a filename.")

    cube.allow_huge_operations = True

    # check input mask
    if mask_in is not None:
        if type(mask_in) is SpectralCube:
            mask_signal = mask_in
        elif type(mask_in) == str:
            mask_signal = SpectralCube.read(mask_in)
        else:
            logger.error("Input mask must be a SpectralCube object or a filename.")
        mask_signal.allow_huge_operations = True

    # Read the velocity field to a Projection if a file is fed in
    if type(vfield_in) == str:
        vfield = Projection.from_hdu(fits.open(vfield_in)[vfield_hdu])
    else:
        vfield = vfield_in

    # get dimensions and units
    spaxis = cube.spectral_axis
    spunit = spaxis.unit
    spvalue = spaxis.value
    nz, ny, nx = cube.shape

    # reproject if it's a projection
    if type(vfield) is Projection:
        # ... NB making a dummy moment here because of issues with
        # reproject and dimensionality. Could instead do header
        # manipulation and feed in "cube"
        dummy_mom0 = cube.moment(order=0)
        vfield = convert_and_reproject(vfield, template=dummy_mom0, unit=spunit)

    # check that grids match after regridding
    if mask_in is not None:
        if mask_signal.shape[1] != np.shape(vfield)[0]:
            logger.error("Input velocity vfield map must be a 2D np.array of dimensions Nx, Ny, but Nx does not match..")
        if mask_signal.shape[2] != np.shape(vfield)[1]:
            logger.error("Input velocity vfield map must be a 2D np.array of dimensions Nx, Ny, but Ny does not match..")

    if coverage is not None:
        if type(coverage) is SpectralCube:
            coverage_cube = coverage
        elif type(coverage) == type("hello"):
            coverage_cube = SpectralCube.read(coverage)
        else:
            logger.error("Coverage cube must be a SpectralCube object or a filename or None.")

        coverage_cube.allow_huge_operations = True

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Set up the masking kwargs
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # Initialize an empty kwargs dictionary
    if mask_kwargs is None:
        mask_kwargs = {}

    # Fill in strict mask defaults
    if 'window' not in mask_kwargs:
        mask_kwargs['window'] = '50km/s'


    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Create the mask
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # prior_hi = None
    # if coverage is not None:
    #     prior_hi = coverage_cube.filled_data[:].value > coverage_thresh

    # create mask around velocity centroid with fixed velocity window
    window =u.Quantity(mask_kwargs['window'])
    mask_window = make_vfield_mask(cube, vfield, window,
                                   outfile=None, overwrite=True)

    # if supplied, combine signal and window mask
    if mask_in is not None:
        mask = join_masks(mask_signal, mask_window,
                        order='fast_nearest_neighbor', 
                        operation='or',
                        thresh=0.5, outfile=None)
        mask = mask.filled_data[:].value

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Write to disk and return
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # In this case can avoid a recast
    if not return_spectral_cube and (outfile is None):
        return(mask)

    # Recast from numpy array to spectral cube
    mask = SpectralCube(mask*1.0, wcs=cube.wcs, header=cube.header,
                        meta={'BUNIT': ' ', 'BTYPE': 'Mask'})

    # Write to disk, if desired
    if outfile is not None:
        header = mask.header
        header['DATAMAX'] = 1
        header['DATAMIN'] = 0
        hdu = fits.PrimaryHDU(np.array(mask.filled_data[:], dtype=np.uint8),
                              header=header)
        hdu.writeto(outfile, overwrite=overwrite)


    if return_spectral_cube:
        return(mask)
    else:
        return(mask.filled_data[:].value)
