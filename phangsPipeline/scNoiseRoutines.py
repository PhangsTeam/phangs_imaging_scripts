import logging

import numpy as np
import scipy.ndimage as nd
import scipy.stats as ss
from astropy.convolution import convolve_fft, Gaussian2DKernel, convolve
from scipy.signal import savgol_coeffs
from spectral_cube import SpectralCube

from . import __version__

np.seterr(divide='ignore', invalid='ignore')

mad_to_std_fac = 1.482602218505602

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def mad_zero_centered(data, mask=None):
    """
    Estimates the noise in a data set using the median absolute
    deviation. Assumes a normal distribution and that the data are
    centered on zero. Excludes any masked regions.

    Parameters:
    -----------

    data : np.array

        Array of data (floats)

    Keywords:
    ---------

    mask : bool

        Boolean array with True indicating where data can be used in
        the noise estimate. (i.e., True is noise). If none is supplied
        a trivial mask is estimated.

    """

    # TBD: Add input error checking

    # Select finite data and exclude not-a-number
    where_data_valid = np.logical_and(~np.isnan(data), np.isfinite(data))

    # Make a trivial mask (to simplify the code) if none has been supplied
    if mask is None:
        mask = np.isfinite(data)

    # Check that we have enough data to estimate the noise
    nData = mask.sum()
    if nData == 0:
        logger.info('No data in mask. Returning NaN.')
        return(np.nan)

    # Estimate the significance threshold likely associated with
    # false positives based on the size of the data.

    sig_false = ss.norm.isf(0.5 / nData)

    # Make a first estimate of the noise based on the negatives.

    data_lt_zero = np.logical_and(
        np.less(data, 0,
                where=where_data_valid,
                out=np.full(where_data_valid.shape,
                            False, dtype=bool)), mask)

    mad1 = mad_to_std_fac * np.abs(np.median(data[data_lt_zero]))

    # Make a second estimate now including the positives less than
    # the false-positive threshold.

    data_lt_mad1 = np.logical_and(
        np.less(data, (sig_false * mad1),
                where=where_data_valid,
                out=np.full(where_data_valid.shape,
                            False, dtype=bool)), mask)
    mad2 = mad_to_std_fac * np.abs(np.median(np.abs(data[data_lt_mad1])))

    # Return this second estimate

    return(mad2)

def noise_cube(data,
               mask=None,
               nThresh=30,
               iterations=1,
               do_map=True,
               do_spec=True,
               box=None,
               spec_box=None,
               substride=None,
               bandpass_smooth_window=None,
               bandpass_smooth_order=3,
               oversample_boundary=False, 
               use_fft=True,
               ):

    """

    Makes an empirical estimate of the noise in a cube assuming that
    it is normally distributed about zero. Treats the spatial and
    spectral dimensions as separable.

    Parameters:
    -----------

    data : np.array
        Array of data (floats)

    Keywords:
    ---------

    mask : bool

        Boolean array with False indicating where data can be
        used in the noise estimate. (i.e., True is signal)

    do_map : bool

        Estimate spatial variations in the noise. Default is True. If
        set to False, all locations in a plane have the same noise
        estimate.

    do_spec : bool

        Estimate spectral variations in the noise. Default is True. If
        set to False, all channels in a spectrum have the same noise
        estimate.

    box : int

        Spatial size of the box over which noise is calculated in
        pixels.  Default: no box, every pixel gets its own noise
        estimte.

    spec_box : int

        Spectral size of the box overwhich the noise is calculated.
        Default: no box, each channel gets its own noise estimate.

    substride : int

        Spatial sub-sampling stride used when calculating noise maps.  Default:
        no sub-sampling.

    nThresh : int
        Minimum number of data to be used in an individual noise estimate.

    iterations : int
        Number of times to iterate the noise solution to force Gaussian
        statistics.  Default: no iterations.

    bandpass_smooth_window : int
        Number of channels used in bandpass smoothing kernel.  Defaults to
        nChan / 4 where nChan number of channels.  Set to zero to suppress
        smoothing. Uses Savitzky-Golay smoothing

    bandpass_smooth_order : int
        Polynomial order used in smoothing kernel.  Defaults to 3.
        
    use_fft: bool
        Use FFT for convolution and infilling (almost as good and much faster)

    """

    # TBD: add error checking

    # Create a mask that identifies voxels to be fitting the noise

    noisemask = np.isfinite(data)
    if mask is not None:
        noisemask[mask] = False

    validdatamap = np.any(noisemask, axis=0)
    pixdim = int(np.ceil(np.sum(validdatamap)**0.5))
    # Default the spatial step size to individual pixels

    step = 1
    halfbox = step // 2

    # If the user has supplied a spatial box size, recast this into a
    # step size that critically samples the box and a halfbox size
    # used for convenience.

    if box is not None:
        step = np.floor(box/2.5).astype(int)
        step = np.min([step, pixdim // 20])
        halfbox = int(box // 2)

    # If step ends up being 0, then set back to 1
    if step == 0:
        step = 1

    if substride is None:
        substride = 1

    # Include all pixels adjacent to the spatial
    # boundary of the data as set by NaNs
    boundary = np.all(np.isnan(data), axis=0)

    if oversample_boundary:
        struct = nd.generate_binary_structure(2, 1)
        struct = nd.iterate_structure(struct, halfbox)
        rind = np.logical_xor(nd.binary_dilation(boundary, struct),
                              boundary)
        extray, extrax = np.where(rind)
    else:
        extray, extrax = None, None

    # If the user has supplied a spectral box size, use this to
    # calculate a spectral step size.

    if spec_box is not None:
        spec_step = np.floor(spec_box / 2).astype(int)
        boxv = int(spec_box // 2)
    else:
        boxv = 0

    # Default the bandpass smoothing window

    if bandpass_smooth_window is None:
        bandpass_smooth_window = 2 * (data.shape[0] // 8) + 1

    # Initialize output to be used in the case of iterative
    # estimation.

    noise_cube_out = np.ones_like(data)

    # Iterate

    for ii in np.arange(iterations):

        if not do_map:

            # If spatial variations are turned off then estimate a
            # single value and fill the noise map with this value.

            noise_value = mad_zero_centered(data, mask=noisemask)
            noise_map = np.zeros(data.shape[1:]) + noise_value

        else:

            # Initialize map to be full of not-a-numbers
            noise_map = np.zeros(data.shape[1:]) + np.nan

            # Make a noise map

            xx = np.arange(data.shape[2])
            yy = np.arange(data.shape[1])

            # Sample starting at halfbox and stepping by step. In the
            # individual pixel limit this just samples every spectrum.

            xsamps = xx[halfbox::step]
            ysamps = yy[halfbox::step]
            xsampsf = (xsamps[np.newaxis,:]
                      * (np.ones_like(ysamps))[:,np.newaxis]).flatten()
            ysampsf = (ysamps[:,np.newaxis] * np.ones_like(xsamps)).flatten()

            for x, y in zip(xsampsf, ysampsf):
                # Extract a minicube and associated mask from the cube

                minicube = data[:, (y-halfbox):(y+halfbox+1):substride,
                               (x-halfbox):(x+halfbox+1):substride]
                minicube_mask = noisemask[:, (y-halfbox):(y+halfbox+1):substride,
                                             (x-halfbox):(x+halfbox+1):substride]

                # If we have enough data, fit a noise value for this entry

                if np.sum(minicube_mask) > nThresh:
                    noise_map[y, x] = mad_zero_centered(minicube,
                                                        mask=minicube_mask)

            if extrax is not None and extray is not None:
                for x, y in zip(extrax, extray):

                    minicube = data[:, (y-halfbox):(y+halfbox+1):substride,
                                    (x-halfbox):(x+halfbox+1):substride]
                    minicube_mask = noisemask[:, (y-halfbox):(y+halfbox+1):substride,
                                              (x-halfbox):(x+halfbox+1):substride]

                    if np.sum(minicube_mask) > nThresh:
                        noise_map[y, x] = mad_zero_centered(minicube,
                                                            mask=minicube_mask)

            noise_map[boundary] = np.nan

            # If we are using a box size greater than an individual pixel
            # interpolate to fill in the noise map.

            if halfbox > 0:

                # Note the location of data, this is the location
                # where we want to fill in noise values.
                data_footprint = np.any(np.isfinite(data), axis=0)

                # Generate a smoothing kernel based on the box size.
                kernel = Gaussian2DKernel(box / np.sqrt(8 * np.log(2)))

                # Make a weight map to be used in the convolution, in
                # this weight map locations with measured values have
                # unity weight. This without measured values have zero
                # weight.

                # wt_map = np.isfinite(noise_map).astype(np.float)
                # wt_map[boundary] = np.nan
                # Take an average weighted by the kernel at each
                # location.
                # noise_map[np.isnan(noise_map)] = 0.0
                # y, x = np.where(np.isfinite(noise_map))
                # import scipy.interpolate as interp
                # func = interp.interp2d(x, y, noise_map[y, x], kind='cubic')

                if use_fft:
                    pad = int(box)
                    noise_map_pad = np.pad(noise_map, pad, 'edge') 
                    noise_map_pad = convolve_fft(noise_map_pad, kernel,
                                                 nan_treatment='interpolate', 
                                                 boundary='wrap')
                    noise_map = noise_map_pad[pad:-pad, pad:-pad]
                else:
                    noise_map = convolve(noise_map, kernel, boundary='extend')
                # yy, xx = np.indices(noise_map.shape)
                # noise_map_beta = interp.griddata((y, x), noise_map[y,x],
                #                                  (yy, xx), method='cubic')
                # noise_map_beta = func(yy, xx)
                # noise_map_beta[boundary] = np.nan
                # noise_map = noise_map_beta
                # wt_map = convolve(wt_map, kernel, boundary='extend')

                # noise_map /= wt_map

                # Set the noise map to not-a-number outside the data
                # footprint.

                noise_map[~data_footprint] = np.nan
        # Initialize spectrum

        noise_spec = np.zeros(data.shape[0]) + np.nan

        if not do_spec:

            # If spectral variations are turned off then assume that
            # the noise_map describes all channels of the cube. In
            # that case the spectrum is just a series of ones.

            noise_spec = np.ones(data.shape[0])
            
        else:

            # Loop over channels

            zz = np.arange(data.shape[0])
            for z in zz:

                # Identify the range of channels to be considered
                # in this estimate.

                lowz = np.clip(z - boxv, 0, data.shape[0])
                hiz = np.clip(z + boxv + 1, 0, data.shape[0])

                # Extract a slab from the cube and normalize it by the
                # noise map. Now any measured noise variations are
                # relative to those in the noise map.

                slab = data[lowz:hiz, :, :] / noise_map[np.newaxis, :, :]
                slab_mask = noisemask[lowz:hiz, :, :]
                noise_spec[z] = mad_zero_centered(slab, mask=slab_mask)

            # Smooth the spectral variations in the noise.

            if bandpass_smooth_window > 0:

                # Initialize a Savitzky-Golay filter then run it over
                # the noise spectrum.

                kernel = savgol_coeffs(int(bandpass_smooth_window),
                                       int(bandpass_smooth_order))

                baddata = np.isnan(noise_spec)
                noise_spec = convolve(noise_spec, kernel,
                                      nan_treatment='interpolate',
                                      boundary='extend')
                noise_spec[baddata] = np.nan

                # Make sure that the noise spectrum is normalized by
                # setting the median to one.

                noise_spec /= np.nanmedian(noise_spec)

        # Combine the spatial and spectral variations into a
        # three-dimensional noise estimate.

        noise_cube = np.ones_like(data)
        noise_cube *= (noise_map[np.newaxis, :]
                       * noise_spec[:, np.newaxis, np.newaxis])

        if iterations == 1:
            return(noise_cube)
        else:

            # If iterating, normalize the data by the current noise
            # estimate and scale the current noise cube by the new
            # estimate.

            data = data / noise_cube
            noise_cube_out *= noise_cube


    # If iterating return the iterated noise cube.

    return(noise_cube_out)

def recipe_phangs_noise(
    incube=None,
    outfile=None,
    mask=None,
    noise_kwargs=None,
    return_spectral_cube=False,
    overwrite=False):
    """

    Wrap noise_cube with a set of preferred parameters for the
    PHANGS-ALMA CO work.

    Parameters:
    -----------

    cube : np.array

        Array of data (floats)

    Keywords:
    ---------

    mask : bool

        Boolean array with False indicating where data can be used in
        the noise estimate. (i.e., True is signal).

    """

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Error checking and work out inputs
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%


    if type(incube) is SpectralCube:
        cube = incube
    elif type(incube) == str:
        cube = SpectralCube.read(incube)
    else:
        logger.error("Input must be a SpectralCube object or a filename.")

    # Initialize an empty kwargs dictionary
    if noise_kwargs is None:
        noise_kwargs = {}

    # If no box is specified, default to one about two beams across
    if 'box' not in noise_kwargs:
        pixels_per_beam = cube.pixels_per_beam
        box = np.ceil(2.5 * pixels_per_beam**0.5)
        noise_kwargs['box'] = box

    if 'substride' not in noise_kwargs:
        pixels_per_beam = cube.pixels_per_beam
        beam_diameter_pixels = np.sqrt(pixels_per_beam * 4.0 / np.pi)
        substride = np.floor(beam_diameter_pixels / 2.5).astype(int)
        if substride < 1:
            substride = 1
        noise_kwargs['substride'] = substride

    # Default to an odd bandpass smothing window
    if 'bandpass_smooth_window' not in noise_kwargs:
        spectral_smooth = np.ceil(cube.shape[0] / 5) // 2 * 2 + 1
        noise_kwargs['bandpass_smooth_window'] = spectral_smooth

    if 'spec_box' not in noise_kwargs:
        noise_kwargs['spec_box'] = 5

    if 'iterations' not in noise_kwargs:
        noise_kwargs['iterations'] = 4

    # Require a valid cube input as a
    if mask is not None:
        if type(mask) is SpectralCube:
            noise_kwargs['mask'] = mask
        elif type(mask) == type("hello"):
            noise_kwargs['mask'] = SpectralCube.read(mask)
        else:
            logger.error("Mask must be a SpectralCube object or a filename or None.")

    # Fill in the mask if it hasn't already been filled in.
    if 'mask' not in noise_kwargs:

        # Check if a non-trivial signal mask is attached to the cube
        if (np.sum(cube.mask.include())
            < np.sum(np.isfinite(cube.filled_data[:].value))):
            noise_kwargs['mask'] = cube.mask.include()
        else:
            noise_kwargs['mask'] = None


    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Run the noise estimate
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    data = cube.filled_data[:].value
    badmask = np.isnan(data)
    badmask = nd.binary_dilation(badmask,
                                 structure=nd.generate_binary_structure(3, 2))
    data[badmask] = np.nan
    
    rms = noise_cube(data, **noise_kwargs)


    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Write or return as requested
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # In this case can avoid a recast
    if not return_spectral_cube and (outfile is None):
        return(rms)

    # Recast from numpy array to spectral cube
    header = cube.header
    datamin = np.nanmin(rms)
    datamax = np.nanmax(rms)
    if np.isfinite(datamin):
        header['DATAMIN'] = datamin
    if np.isfinite(datamax):
        header['DATAMAX'] = datamax
    header['COMMENT'] = 'Produced with PHANGS-ALMA pipeline version ' + __version__
    rms = SpectralCube(rms, wcs=cube.wcs, header=header,
                       meta={'BUNIT':cube.header['BUNIT']})

    # Write to disk, if desired
    if outfile is not None:
        rms.write(outfile, overwrite=overwrite)

    if return_spectral_cube:
        return(rms)
    else:
        return(rms.filled_data[:].value)

