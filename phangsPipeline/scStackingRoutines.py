import astropy.units as u
import astropy.wcs as wcs
import numpy as np
from astropy.io import fits
from spectral_cube import SpectralCube, Projection

from .scDerivativeRoutines import convert_and_reproject


def channelShiftVec(x, ChanShift):
    """Shift an array of spectra by some number of channels using the FFT.

    Parameters
    ----------

    x : array assumed to be two dimensional with the spectral axis 0.

    ChanShift : the number of channels to shift each spectrum
    by. Should have length equal to axis 1 of x.

    Returns
    -------

    an array that looks like x but with the phase of each spectrum
    shifted.

    """

    # Take the FFT of x along the spectral dimension
    ftx = np.fft.fft(x, axis=0)

    # Calculate frequency ais in units of channels
    m = np.fft.fftfreq(x.shape[0])

    # Calculate the factor to rotate the phase of the FFT
    phase = np.exp(-2 * np.pi * m[:, np.newaxis]
                   * 1j * ChanShift[np.newaxis, :])

    # Transform back along the spectral axis and return
    x2 = np.real(np.fft.ifft(ftx * phase, axis=0))
    
    return(x2)


def ShuffleCube(
        cube_in, vfield_in, vfield_hdu=0,
        outfile=None, overwrite=True,
        chunk=1000):
    """Shuffles cube so that the velocity appearing in the centroid_map is set 
    to the middle channel and velocity centroid.
    
    Parameters
    ----------

    cube : string or SpectralCube

        The original data cube.

    vfield : float or two-d array or string

        The velocity field to use as a reference. If a string, it's read
        as a Projection and reprojected onto the cube. If it's a single
        value then this is broadcast across the whole map. If it's a two-d
        array it is assumed to be the velocity field.
    
    Keywords
    --------

    vfield_hdu : optional refers to the HDU of the velocity field if a
        file name is supplied. Default 0.

    chunk : int
        Number of data points to include in a chunk for processing.

    outfile : file name to write output mask to.

    overwrite : flag to allow overwrite to rewrite files.
    
    Returns
    -------
    OutCube : np.array
        Output SpectralCube cube shuffled so that the emission is at 0.

    """

    # -------------------------------------------------
    # Read the cube
    # -------------------------------------------------

    if type(cube_in) == str:
        cube = SpectralCube.read(cube_in)
    else:
        cube = cube_in

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
    else:
        vfield = vfield_in

    # If vfield is a Projection reproject it onto the cube and match units
    if type(vfield) is Projection:
        # ... NB making a dummy moment here because of issues with
        # reproject and dimensionality. Could instead do header
        # manipulation and feed in "cube"
        dummy_mom0 = cube.moment(order=0)
        vfield = convert_and_reproject(vfield, template=dummy_mom0, unit=spunit)
    else:

        # If no units are attached to the vfield, guess that the
        # units are the same as cube        
        if type(vfield) != u.quantity.Quantity:
            vfield = u.quantity.Quantity(vfield,spunit)
        
        # If a single value is supplied, turn it into a single-valued
        # velocity field
        if np.ndim(vfield.data) <= 1:
            vfield = vfield.to(spunit)
            vfield = u.quantity.Quantity(np.ones((ny, nx))*vfield.value, vfield.unit)
        
        # Just in case convert the units to match the cube
        vfield = vfield.to(spunit)

    # Check sizes
    if (cube.shape[1] != vfield.shape[0]) or (cube.shape[2] != vfield.shape[1]):
        return(np.nan)

    # --------------------------------------------------
    # Identify spectra and channel shifts
    # --------------------------------------------------    

    # This all assumes linear mapping between channel and velocity,
    # which is typically a good assumption for radio data.
    
    # Identify pixels with a reference velocity supplied
    
    y, x = np.where(np.isfinite(vfield))
    centroids = vfield[y, x]

    # Translate velocity reference to a channel shift
    
    # ... define a vector of channel offsets from the central value
    relative_channel = \
        np.array(np.arange(len(spaxis)) - (len(spaxis) // 2), dtype=float)

    # ... sort the spectral axis and apply the same sort to the
    # channel offset to enable interpolation
    
    idx_sorted_by_vel = np.argsort(spaxis)
    sorted_spaxis = spaxis[idx_sorted_by_vel]
    sorted_relative_channel = relative_channel[idx_sorted_by_vel]

    # Interpolate the velocity offset (which should be the new zero)
    # through the spectral axis to identify the channel offset. We
    # want to shift the spectrum at that location by -1 times that
    # amount to bring that velocity to the middle of the spectrum.
    
    channel_shift = -1 * np.interp(centroids, sorted_spaxis,
                                   sorted_relative_channel)

    # --------------------------------------------------
    # Shuffle the spectra
    # --------------------------------------------------    
    
    # Initialize output
    # NOTE: this will cause memory issues for large cubes. Something to address in the future.
    new_cube = np.empty(cube.shape)
    new_cube.fill(np.nan)

    # Identify the number of chunks into which to split the data
    # during FFT-based shifting
    nchunk = (len(x) // chunk)

    print(nchunk, "chunks of size", chunk, "for shuffling. Total of ", len(x), "spectra to shuffle.")
    if nchunk == 0:
        nchunk = 1
        
    for this_x, this_y, this_shift in zip(np.array_split(x, nchunk), 
                                          np.array_split(y, nchunk),
                                          np.array_split(channel_shift, nchunk)):

        # Array of spectra to shift
        this_spectra = cube.filled_data[:, this_y, this_x].value

        # ... mask of NaNs
        missing_data = ~np.isfinite(this_spectra)

        # ... replace NaNs with 0s for shifting
        filled_spectra = np.nan_to_num(this_spectra)

        # ... phase shift the spectrum
        shifted_spectra = \
            channelShiftVec(filled_spectra, np.atleast_1d(this_shift))

        # ... phase shift the mask
        shifted_mask = \
            channelShiftVec(missing_data*1.0, np.atleast_1d(this_shift))

        # ... apply the mask
        shifted_spectra[shifted_mask>0.5] = np.nan

        # ... also mask the region where the spectrum "wrapped" due to the FFT

        max_channel = float(len(spaxis))
        channels = np.arange(max_channel)
        channel_grid = np.tile(channels.reshape((len(spaxis),1)),(1,shifted_spectra.shape[1]))
        shift_grid = np.tile(this_shift,(shifted_spectra.shape[0],1))

        wrap_mask = (shift_grid >= 0.)*(channel_grid <= shift_grid) + \
            (shift_grid < 0)*(channel_grid >= (max_channel + shift_grid - 1))
        
        shifted_spectra[wrap_mask > 0.5] = np.nan        
        
        # ... save in the cube
        new_cube[:, this_y, this_x] = shifted_spectra

    # -------------------------------------------------
    # Output
    # -------------------------------------------------
        
    # Format output into a spectral cube object
    hdr = cube.header.copy()
    hdr['CRVAL3'] = 0.0
    hdr['CRPIX3'] = len(spaxis) // 2 + 1
    new_wcs = wcs.WCS(hdr)

    new_cube = SpectralCube(new_cube, new_wcs, header=hdr)

    # Optionally write to disk
    if outfile is not None:
        new_cube.write(outfile, overwrite=overwrite)    
    
    return(new_cube)


def recipe_phangs_vfield(
    vfield_in,
    vfield_in_hdu=0,
    list_of_vfields=None,
    outfile=None,
    overwrite=False):
    """
    Task to create combined velocity field from a list of velocity maps in
    a hierarchical manner (from first to last).

    Parameters:

    -----------

    vfield_in : string or fits.hdu

        The reference velocity filed that holds the target WCS. The 
        other maps will be reprojected onto this one. This map is 
        included in the final output.

    Keywords:
    ---------

    vfield_in_hdu : 

    list_of_vfields : list or list of fits.hdu

        List of velocity fields or fits.hdus. These will be reprojected
        onto the template vfield and then combined via hierarchical addition.

    outfile : string
        Filename where the vfield will be written. The vfield maps is alsoreturned.

    """

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Read in template velocity field
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # Read the velocity field to a Projection if a file is fed in
    if type(vfield_in) == str:
        template_vfield = Projection.from_hdu(fits.open(vfield_in)[vfield_in_hdu])
    else:
        template_vfield = vfield_in

    # get velocity unit
    spunit = template_vfield.unit

    # initialise combined velocity field
    vfield_combined = np.copy(template_vfield)
    vfield_combined[np.isnan(vfield_combined)] = 0

    # loop over list of velocity fields
    for this_vfield_in in list_of_vfields:

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Read in and align ancillary velocity field
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        # Read the velocity field to a Projection if a file is fed in
        if type(this_vfield_in) == str:
            this_vfield = Projection.from_hdu(fits.open(this_vfield_in)[vfield_in_hdu])
        else:
            this_vfield = this_vfield_in

        # print(this_vfield)

        # If vfield is a Projection reproject it onto the cube and match units
        if type(this_vfield) is Projection:
            this_vfield = convert_and_reproject(this_vfield, template=template_vfield, unit=spunit)
        else:

            # If no units are attached to the vfield, guess that the
            # units are the same as cube        
            if type(this_vfield) != u.quantity.Quantity:
                this_vfield = u.quantity.Quantity(this_vfield, spunit)
            
            # Just in case convert the units to match the cube
            this_vfield = this_vfield.to(spunit)

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Merge velocity fields
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        # fill empty area with current vfield
        mom1_temp = np.copy(this_vfield)
        mom1_temp[vfield_combined != 0] = 0
        vfield_combined += mom1_temp
        vfield_combined[np.isnan(vfield_combined)] = 0

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Write or return as requested
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # prepate velocity field for output
    vfield_combined[vfield_combined == 0] = np.nan

    # Write to disk, if desired
    if outfile is not None:
        fits.writeto(outfile, vfield_combined.value, template_vfield.header, overwrite=overwrite)

    # return combined velocity field
    return(vfield_combined)

def recipe_shuffle_cube(
    cube_in,
    vfield_in,
    vfield_in_hdu=0,
    outfile=None,
    return_spectral_cube=False,
    overwrite=False):
    """
    Task to create velocity-shuffled cubes via input velocity vfield map.

    Parameters:
    -----------

    cube_in : string or SpectralCube
        The cube to be masked.

    vfield_in : 2D numpy.ndarray
        A 2D map of the vfield velocities for the lines to stack of dimensions Nx, Ny.
        Note that DataCube and vfield map must have equivalent (but not necessarily equal) 
        spectral units (e.g., km/s and m/s)
    

    Keywords:
    ---------

    vfield_in_hdu=,

    outfile : string
        Filename where the shuffled cube will be written. The shuffled cube is also
        returned.

    """

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Error checking and work out inputs
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # check input cube
    if type(cube_in) is SpectralCube:
        cube = cube_in
    elif type(cube_in) == type("hello"):
        cube = SpectralCube.read(cube_in)
    else:
        logger.error("Input cube must be a SpectralCube object or a filename.")

    cube.allow_huge_operations = True

    spaxis = cube.spectral_axis
    spunit = spaxis.unit

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Read in and align velocity field
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # Read the velocity field to a Projection if a file is fed in
    if type(vfield_in) == str:
        vfield = Projection.from_hdu(fits.open(vfield_in)[vfield_in_hdu])
    else:
        vfield = vfield_in

    # If vfield is a Projection reproject it onto the cube and match units
    if type(vfield) is Projection:
        # ... NB making a dummy moment here because of issues with
        # reproject and dimensionality. Could instead do header
        # manipulation and feed in "cube"
        dummy_mom0 = cube.moment(order=0)
        vfield = convert_and_reproject(vfield, template=dummy_mom0, unit=spunit)
    else:

        # If no units are attached to the vfield, guess that the
        # units are the same as cube        
        if type(vfield) != u.quantity.Quantity:
            vfield = u.quantity.Quantity(vfield, spunit)
        
        # Just in case convert the units to match the cube
        vfield = vfield.to(spunit)

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Run the shuffling
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # run shuffling
    shuffled_cube = ShuffleCube(cube, vfield, chunk=1000)    

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Write or return as requested
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # prepare cube for output
    # convert data type to 32-bit float and assign WCS and header
    shuffled_cube = SpectralCube(shuffled_cube.filled_data[:].astype(np.float32), wcs=shuffled_cube.wcs, header=shuffled_cube.header)

    # Write to disk, if desired
    if outfile is not None:
        shuffled_cube.write(outfile, overwrite=overwrite)

    # return cube
    if return_spectral_cube:
        return(shuffled_cube)
    else:
        return(shuffled_cube.filled_data[:].value)


def BinByMask(DataCube, mask, vfield, weight_map=None):
    """
    Bin a data cube by a label mask, aligning the data to a common vfield.  Returns an array.

    Parameters
    ----------
    DataCube : SpectralCube
        The original spectral cube with spatial dimensions Nx, Ny and spectral dimension Nv
    Mask : 2D numpy.ndarray
        A 2D map containing boolean values with True indicate where the spectra should be aggregated.
    vfield : 2D numpy.ndarray
        A 2D map of the vfield velocities for the lines to stack of dimensions Nx, Ny.
        Note that DataCube and vfield map must have equivalent spectral units (e.g., km/s)
    weight_map : 2D numpy.ndarray
        Map containing the weight values to be used in averaging
    Returns
    -------
    Spectrum : np.array
        Spectrum of average over mask.
    """
    spaxis = DataCube.spectral_axis.value
    y, x = np.where(mask)
    v0 = spaxis[len(spaxis) // 2] * DataCube.spectral_axis.unit
    relative_channel = np.arange(len(spaxis)) - (len(spaxis) // 2)
    vfields = vfield[y, x].to(DataCube.spectral_axis.unit).value
    sortindex = np.argsort(spaxis)
    channel_shift = -1 * np.interp(vfields, spaxis[sortindex],
                                   np.array(relative_channel[sortindex], dtype=float))
    spectra = DataCube.filled_data[:, y, x].value
    shifted_spectra = channelShiftVec(spectra, channel_shift)
    if weight_map is not None:
        wts = weight_map[y, x]
    else:
        wts = np.ones_like(channel_shift)
    accum_spectrum = np.nansum(wts[np.newaxis, :]
                               * shifted_spectra, axis=1) / np.nansum(wts)
    shifted_spaxis = (DataCube.spectral_axis - v0)
    return(accum_spectrum, shifted_spaxis)


def BinByLabel(DataCube, LabelMap, vfield,
               weight_map=None,
               background_labels=[0]):
    """
    Bin a data cube by a label mask, aligning the data to a common vfield.

    Parameters
    ----------
    DataCube : SpectralCube
        The original spectral cube with spatial dimensions Nx, Ny and spectral dimension Nv
    LabelMap : 2D numpy.ndarray
        A 2D map containing integer labels for each pixel into objects defining the stacking.
    vfield : 2D numpy.ndarray
        A 2D map of the vfield velocities for the lines to stack of dimensions Nx, Ny.
        Note that DataCube and vfield map must have equivalent spectral units to DataCube
    background_labels : list
        List of values in the label map that correspond to background objects and should not
        be processed with the stacking. 

    Returns
    -------
    output_list : list of dict
        List of dict where each entry contains the stacked spectrum for a given label
    unique_labels = array of unique labels in same order as output list
    """
    UniqLabels = np.unique(LabelMap)
    output_list = []
    for ThisLabel in UniqLabels:
        if ThisLabel not in background_labels:
            thisspec, spaxis = BinByMask(DataCube,
                                         (LabelMap == ThisLabel),
                                         vfield,
                                         weight_map=weight_map)
            output_list += [{'label': ThisLabel,
                             'spectrum': thisspec,
                             'spectral_axis': spaxis}]
    return(output_list, UniqLabels)
