"""
Stand alone routines to carry out basic noise estimation, masking, and
mask manipulation steps in CASA.
"""

import logging
import os

import analysisUtils as au
import astropy.units as u
import numpy as np
import scipy.ndimage as ndimage
from astropy.io import fits
from astropy.wcs.utils import proj_plane_pixel_scales
from radio_beam import Beam
from scipy.special import erfc, ndtri_exp
from scipy.stats import kurtosis, skew
from spectral_cube import Projection, SpectralCube

from . import casaStuff
from . import casaCubeRoutines as ccr

logger = logging.getLogger(__name__)

# region Noise estimation

def mad(
        data=None,
        as_sigma=True
):
    """
    Helper routine to calculate median absolute deviation (MAD). This
    is present already in scipy.stats but not in the version of scipy
    that CASA ships with. The MAD is a use a fast, useful robust noise
    estimator.

    data : the vector of data used to calculate the MAD. The routine
    flattens the array, so no along-axis operations.

    as_sigma (default True) : scale the output so that the returned
    value represents the RMS or 1-sigma value for a normal
    distribution. For Gaussian noise, this implies that the result can
    just be used as a standard noise estimate.
    """

    if data is None:
        logger.error("No data supplied.")

    this_med = np.median(data)
    this_dev = np.abs(data - this_med)
    this_mad = np.median(this_dev)
    if as_sigma:
        return (this_mad / 0.6745)
    else:
        return (this_mad)


def estimate_noise(
        data=None,
        mask=None,
        method='mad',
        niter=None,
):
    """
    Return a noise estimate given a vector and associated mask.

    data : the data used to calculate the noise.

    mask : a mask used to indicate which subset of data to
    consider. In this routine mask values of True will be included in
    the calculation and mask values of False will be excluded. This
    matches the CASA syntax, but might require "inverting" a signal
    mask when the intention is to avoid bright signal.

    method (default "mad") : Method to use. Either "std" for standard
    deviation, "mad" for median absolute deviation, "chauvstd" for
    standard deviation with outlier rejection, or "chauvmad" for mad
    with outlier rejection.

    niter : number of iterations used in outlier rejection.

    Method "mad" is preferred for fast calculation and "chauvmad" for
    accurate calculation. Both should be reasonably robust.
    """

    if data is None:
        logger.error("No data supplied.")
        return (None)

    if mask is not None:
        if len(mask) != len(data):
            logger.error("Mask and data have mismatched sizes.")
            return (None)

    if niter is None:
        niter = 5

    valid_methods = ['std', 'mad', 'chauvstd', 'chauvmad']
    if method not in valid_methods:
        logger.error("Invalid method - " + method + " valid methods are " + str(valid_methods))
        return (None)

    if mask is None:
        use_mask = np.isfinite(data)
    else:
        use_mask = mask * np.isfinite(data)

    # std not defined with less than 2 points. You'd obviously want far more.
    if np.sum(use_mask) < 2:
        logger.error("No valid data. Returning NaN.")
        return (np.nan)

    use_data = data[use_mask]

    if method == 'std':
        this_noise = np.std(use_data)
        return (this_noise)

    if method == 'mad':
        this_noise = mad(use_data, as_sigma=True)
        return (this_noise)

    if method == 'chauvstd' or method == 'chauvmad':
        for ii in range(niter):
            this_mean = np.mean(use_data)
            if method == 'chauvstd':
                this_std = np.std(use_data)
            elif method == 'chauvmad':
                this_std = mad(use_data, as_sigma=True)

            this_dev = np.abs((use_data - this_mean) / this_std) / 2.0 ** 0.5
            this_prob = erfc(this_dev)

            chauv_crit = 1.0 / (2.0 * len(use_data))
            keep = this_prob > chauv_crit
            if np.sum(keep) == 0 or this_std == 0.0:
                logger.error("Rejected all data. Returning NaN.")
                return (np.nan)
            use_data = use_data[keep]
        this_noise = np.std(use_data)
        return (this_noise)

    return (None)


def noise_for_cube(
        infile=None,
        maskfile=None,
        exclude_mask=True,
        method='mad',
        niter=None,
):
    """
    Get a single noise estimate for an image cube.
    """

    if infile is None:
        logger.error('No infile specified.')
        return (None)

    if not os.path.isdir(infile) and not os.path.isfile(infile):
        logger.error('infile specified but not found - ' + infile)
        return (None)

    if maskfile is not None:
        if not os.path.isdir(maskfile) and not os.path.isfile(maskfile):
            logger.error('maskfile specified but not found - ' + maskfile)
            return (None)

    myia = au.createCasaTool(casaStuff.iatool)
    myia.open(infile)

    has_memory_issue, cube_shape = ccr.check_getchunk_putchunk_memory_issue(
        infile, myia=myia, return_shape=True)

    if not has_memory_issue:
        data = myia.getchunk()
        mask = myia.getchunk(getmask=True)
        myia.close()

        if maskfile is not None:
            myia.open(maskfile)
            user_mask = myia.getchunk()
            user_mask_mask = myia.getchunk(getmask=True)
            myia.close()
            if exclude_mask:
                mask = mask * user_mask_mask * (user_mask < 0.5)
            else:
                mask = mask * user_mask_mask * (user_mask >= 0.5)

        this_noise = estimate_noise(
            data=data, mask=mask, method=method, niter=niter)

    else:
        logger.debug('getchunk channel by channel for known memory issue')

        myia_mask = None
        if maskfile is not None:
            myia_mask = au.createCasaTool(casaStuff.iatool)
            myia_mask.open(maskfile)

        per_channel_noise = []

        if len(cube_shape) == 2:
            blc = [0, 0]
            trc = [-1, -1]
            data_slice = myia.getchunk(blc, trc)
            mask_slice = myia.getchunk(blc, trc, getmask=True)
            if myia_mask is not None:
                user_mask_slice = myia_mask.getchunk(blc, trc)
                user_mask_mask_slice = myia_mask.getchunk(blc, trc, getmask=True)
                if exclude_mask:
                    mask_slice = mask_slice * user_mask_mask_slice * (user_mask_slice < 0.5)
                else:
                    mask_slice = mask_slice * user_mask_mask_slice * (user_mask_slice >= 0.5)
            chan_noise = estimate_noise(data=data_slice, mask=mask_slice, method=method, niter=niter)
            if np.isfinite(chan_noise):
                per_channel_noise.append(chan_noise)

        elif len(cube_shape) == 3:
            for ichan in range(cube_shape[2]):
                blc = [0, 0, ichan]
                trc = [-1, -1, ichan]
                data_slice = myia.getchunk(blc, trc)
                mask_slice = myia.getchunk(blc, trc, getmask=True)
                if myia_mask is not None:
                    user_mask_slice = myia_mask.getchunk(blc, trc)
                    user_mask_mask_slice = myia_mask.getchunk(blc, trc, getmask=True)
                    if exclude_mask:
                        mask_slice = mask_slice * user_mask_mask_slice * (user_mask_slice < 0.5)
                    else:
                        mask_slice = mask_slice * user_mask_mask_slice * (user_mask_slice >= 0.5)
                chan_noise = estimate_noise(data=data_slice, mask=mask_slice, method=method, niter=niter)
                if np.isfinite(chan_noise):
                    per_channel_noise.append(chan_noise)

        elif len(cube_shape) == 4:
            for istokes in range(cube_shape[3]):
                for ichan in range(cube_shape[2]):
                    blc = [0, 0, ichan, istokes]
                    trc = [-1, -1, ichan, istokes]
                    data_slice = myia.getchunk(blc, trc)
                    mask_slice = myia.getchunk(blc, trc, getmask=True)
                    if myia_mask is not None:
                        user_mask_slice = myia_mask.getchunk(blc, trc)
                        user_mask_mask_slice = myia_mask.getchunk(blc, trc, getmask=True)
                        if exclude_mask:
                            mask_slice = mask_slice * user_mask_mask_slice * (user_mask_slice < 0.5)
                        else:
                            mask_slice = mask_slice * user_mask_mask_slice * (user_mask_slice >= 0.5)
                    chan_noise = estimate_noise(data=data_slice, mask=mask_slice, method=method, niter=niter)
                    if np.isfinite(chan_noise):
                        per_channel_noise.append(chan_noise)

        else:
            myia.close()
            if myia_mask is not None:
                myia_mask.close()
            raise Exception('Could not proceed with cube dimension ' + str(len(cube_shape)))

        myia.close()
        if myia_mask is not None:
            myia_mask.close()

        if len(per_channel_noise) == 0:
            this_noise = np.nan
        else:
            this_noise = float(np.nanmedian(per_channel_noise))

    if np.isnan(this_noise):
        raise Exception("Returned nan for noise: {}".format(this_noise))

    return (this_noise)


def stat_cube(
        cube_file=None,
):
    """
    Calculate statistics for an image cube. Right now this is a thin
    wrapper to imstat.
    """
    if cube_file == None:
        logger.info("No cube file specified. Returning")
        return

    imstat_dict = casaStuff.imstat(cube_file)

    return imstat_dict


# endregion

# region Mask creation and manipulation

def read_cube(infile, huge_cube_workaround=True):
    """
    Read cube from CASA image file. Includes a switch for large cubes, where getchunk may fail.
    """

    if huge_cube_workaround:
        casaStuff.exportfits(imagename=infile,
                             fitsimage=infile + '.fits',
                             stokeslast=False, overwrite=True)
        hdu = fits.open(infile + '.fits')[0]
        cube = hdu.data.T

        # Remove intermediate fits file
        os.system('rm -rf ' + infile + '.fits')
    else:
        myia = au.createCasaTool(casaStuff.iatool)
        myia.open(infile)
        cube = myia.getchunk()
        myia.close()

    return cube


def write_mask(infile, outfile, mask, huge_cube_workaround=True):
    """
    Write a CASA mask out as a CASA image. Includes a switch for large cubes, where putchunk may fail.
    """

    os.system('rm -rf ' + outfile)
    os.system('cp -r ' + infile + ' ' + outfile)

    if huge_cube_workaround:
        casaStuff.exportfits(imagename=outfile,
                             fitsimage=outfile + '.fits',
                             stokeslast=False, overwrite=True)
        hdu = fits.open(outfile + '.fits')[0]
        hdu.data = mask.T
        hdu.header['BITPIX'] = -32

        # Match up the WCS so tclean doesn't throw an error (this is some rounding to the nth decimal place...)
        header = casaStuff.imhead(infile, mode='list')
        wcs_names = ['cdelt1', 'cdelt2', 'cdelt3', 'cdelt4',
                     'crval1', 'crval2', 'crval3', 'crval4']

        for wcs_name in wcs_names:
            hdu.header[wcs_name.upper()] = header[wcs_name]

        hdu.writeto(outfile + '.fits', overwrite=True)

        casaStuff.importfits(fitsimage=outfile + '.fits',
                             imagename=outfile,
                             overwrite=True)

        # Remove the intermediate fits file
        os.system('rm -rf ' + outfile + '.fits')
    else:
        myia = au.createCasaTool(casaStuff.iatool)
        myia.open(outfile)
        myia.putchunk(mask)
        myia.close()

    return True


def get_beam_fft(
        beam: Beam,
        pixscale: u.Quantity | None = None,
        shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the normalised FFT of a beam kernel

    Args:
        beam: Beam object
        pixscale: u.Quantity: Pixel scale
        shape: tuple[int, int]: Shape for the
            output FFT

    Returns:
        np.ndarray: Beam kernel, FFT of the beam kernel
    """

    if pixscale is None:
        raise ValueError("pixscale must be defined!")

    if shape is None:
        raise ValueError("shape must be defined!")

    # Create a kernel from the beam to convolve with the random field
    k = beam.as_kernel(pixscale=pixscale,
                       x_size=shape[1],
                       y_size=shape[0],
                       ).array

    # Normalise the kernel
    k = k / np.sqrt(np.sum(k ** 2))

    # Get the FFT of the kernel
    k_fft = np.fft.rfft2(
        k,
    )

    return k, k_fft


def convolve_with_fft(
        data: np.ndarray,
        k_fft: np.ndarray,
) -> np.ndarray:
    """Convolve data with a kernel using FFT

    Args:
        data: np.ndarray: Input data to convolve
        k_fft: np.ndarray: FFT of the kernel

    Returns:
        np.ndarray: Convolved data
    """

    # Take FFT of the data
    data_fft = np.fft.rfft2(data)

    # Multiply by the kernel FFT
    convolved_fft = data_fft * k_fft

    # Take inverse FFT to get convolved data
    convolved_data = np.fft.ifftshift(np.fft.irfft2(convolved_fft, s=data.shape))

    return convolved_data


def calculate_departure(
        data: np.ndarray,
        n_eff: float | None = None,
):
    """Calculate departure score for a dataset

    Args:
        data (np.ndarray): Input data
        n_eff (float | None): Effective number of independent measurements.
            If None, will not correct for sample size.

    Returns:
        dict: Dictionary containing skew, kurtosis, observed departure, and corrected departure
    """

    g1 = skew(data, bias=False)
    g2 = kurtosis(data, fisher=True, bias=False)

    # Sample-size-independent departure score
    observed_departure = np.sqrt(g1 ** 2 + g2 ** 2 / 4)

    corrected_departure = None

    # Approximations for Gaussian-beam correlation
    if n_eff is not None:
        n_eff_skew = 1.5 * n_eff
        n_eff_kurt = 2.0 * n_eff

        expected_d2 = (
                6 / n_eff_skew
                + 6 / n_eff_kurt
        )

        corrected_departure = np.sqrt(
            max(0.0, observed_departure ** 2 - expected_d2)
        )

    result = {
        "g1": g1,
        "g2": g2,
        "observed_departure": observed_departure,
        # "expected_gaussian_rms": np.sqrt(expected_d2),
        "corrected_departure": corrected_departure,
    }

    return result


def beam_corrected_gaussianity(
        chan: Projection,
):
    """Calculate beam-corrected statistics for a channel of a cube.

    Args:
        chan (Projection): Channel to calculate statistics for.

    Returns:
        dict: Dictionary containing skew, kurtosis, observed departure, corrected departure,
            Jarque-Bera statistic, p-value, log p-value, log10 p-value, and sigma.
    """

    pix_per_beam = chan.pixels_per_beam

    # Pull out the data, use the underlying data
    # since it's faster
    data = chan._data
    data = data[np.isfinite(data)]

    # If we have no valid data, return None
    if data.size == 0:
        return None

    # Calculate n_beams as the number of independent beams in the image
    n_beams = data.size / pix_per_beam

    result = calculate_departure(data, n_eff=n_beams)

    if result["corrected_departure"] is not None:
        departure = result["corrected_departure"]
    else:
        departure = result["observed_departure"]

    # Convert this to Jarque-Bera statistic
    jb = n_beams * departure ** 2 / 6

    # If we're below the noise floor, set
    # to machine precision
    if jb == 0:
        jb = np.finfo(float).eps

    log_p = -jb / 2
    log10_p = log_p / np.log(10)
    sigma = -ndtri_exp(log_p)

    # May still underflow, but sigma and log_p remain valid
    p_value = np.exp(log_p)

    result.update(
        {
            "jarque_bera": jb,
            "p_value": p_value,
            "log_p_value": log_p,
            "log10_p_value": log10_p,
            "sigma": sigma,
        }
    )

    return result


def calculate_departure_null(
        beam: Beam,
        pixscale: u.Quantity,
        pix_per_beam: float | None = None,
        shape: tuple[int, int] = (100, 100),
        n_draws: int = 1000,
) -> np.ndarray:
    """Calculate departure scores for a null test of Gaussianity, given a beam and pixel scale.

    Args:
        beam (Beam): Beam object to convolve with pure noise
        pixscale (u.Quantity): Pixel scale of the image
        pix_per_beam (float | None): Number of pixels per beam. If None,
            will not correct for the effective number of independent measurements.
        shape (tuple[int, int]): Shape of the random noise field to generate.
            Defaults to (100, 100).
        n_draws (int): Number of random noise fields to generate. Defaults to 1000.

    Returns:
        np.ndarray: Array of departure scores for each random noise field
    """

    _, k_fft = get_beam_fft(
        beam,
        pixscale=pixscale,
        shape=shape,
    )
    departure_null = np.full(n_draws, np.nan)

    rng = np.random.default_rng()

    for n_draw in range(n_draws):

        # Generate a random noise field, convolve and flatten
        noise_field = rng.normal(size=shape)
        noise_field = convolve_with_fft(
            noise_field,
            k_fft=k_fft,
        )
        noise_field = noise_field.flatten()

        # Calculate the effective number of independent measurements
        n_eff = None
        if pix_per_beam is not None:
            n_eff = noise_field.size / pix_per_beam

        result = calculate_departure(noise_field,
                                     n_eff=n_eff,
                                     )

        if result["corrected_departure"] is not None:
            departure = result["corrected_departure"]
        else:
            departure = result["observed_departure"]

        # Don't take the 0s, since they're meaningless
        if departure == 0:
            continue

        departure_null[n_draw] = departure

    return departure_null


def get_noise_only_channels(
        f: str,
        sigma_threshold: float | None = 2,
) -> list[bool]:
    """Check where a cube only has noise channels

    There are two checks that go on here: the first is that the sample-size-independent
    departure score (calculated from skew and kurtosis of the data) is above a sigma-threshold
    to a null test. The second is that the Jarque-Bera statistic is above a sigma-threshold.
    There is a little complication here that statistics need to be corrected for the effective
    number of independent measurements (number of beams).

    Args:
        f (str): Path to the cube
        sigma_threshold (float | None): Threshold in sigma for departure from Gaussianity.
            If None, will not check and just return False.

    Returns:
        bool: True if all channels are consistent with being Gaussian noise, False otherwise
    """

    cube = SpectralCube.read(f)
    cube.allow_huge_operations = True

    # If we're not checking, just return True
    if sigma_threshold is None:
        return [True] * cube.shape[0]

    # Get pixel scale in arcsec. Assume square pixels
    pixscales = proj_plane_pixel_scales(cube.wcs.celestial) * u.deg
    pixscale = [p.to(u.arcsec) for p in pixscales][0]

    # Take the first valid channel of the cube, assuming the beam stays relatively
    # constant. This is a simplification, but should be fine for our purposes.

    mask = cube.get_mask_array()
    valid_mask = list(np.sum(mask, axis=(1, 2)) > 0)
    first_valid_chan = valid_mask.index(True)

    chan = cube[first_valid_chan]
    beam = chan.beam
    pix_per_beam = chan.pixels_per_beam

    departure_null = calculate_departure_null(
        beam=beam,
        pixscale=pixscale,
        pix_per_beam=pix_per_beam,
    )

    # Calculate a "typical" departure from the Gaussian field
    mean_departure_null = np.nanmean(departure_null)
    std_departure_null = np.nanstd(departure_null)

    # Now calculate statistics for each channel in the cube
    sigma = np.full(cube.shape[0], np.nan)
    departure = np.full(cube.shape[0], np.nan)

    for chan_idx in range(cube.shape[0]):

        result = beam_corrected_gaussianity(cube[chan_idx])

        if result is not None:
            sigma[chan_idx] = result["sigma"]
            if result["corrected_departure"] is not None:
                d = result["corrected_departure"]
            else:
                d = result["observed_departure"]
            departure[chan_idx] = d

    # Calculate noise-only channels
    noise_only_channels = np.logical_or(
        sigma < sigma_threshold,
        departure < sigma_threshold * std_departure_null + mean_departure_null,
    )

    # Convert to a strict list of booleans
    noise_only_channels = [bool(x) for x in noise_only_channels]

    return noise_only_channels

def mask_noise_channels(
        imaging_method='tclean',
        cube_root=None,
        suffix_in='',
        suffix_out='',
        operation='AND',
        sigma_threshold=3,
):
    """Mask out channels that are consistent with being noise-only."""

    if imaging_method == 'sdintimaging':
        cube_root += '.joint.cube'

    if not os.path.isdir(cube_root + '.image' + suffix_in):
        logger.error('Data file not found: "' + cube_root + '.image' + suffix_in + '"')
        logger.info('Need CUBE_ROOT.image to be an image file.')
        logger.info('Returning. Generalize the code if you want different syntax.')
        return

    header = casaStuff.imhead(cube_root + '.image' + suffix_in)
    if header['axisnames'][2] == 'Frequency':
        spec_axis = 2
    else:
        spec_axis = 3

    f = cube_root + '.image' + suffix_in

    logger.info('Reading cube.')
    cube = read_cube(f, huge_cube_workaround=True)

    mask = np.ones(cube.shape, dtype=bool)

    logger.info("Finding noise-only channels")
    noise_only_channels = get_noise_only_channels(
        f,
        sigma_threshold=sigma_threshold,
    )
    total_noise_only = sum(noise_only_channels)
    logger.info(f"{total_noise_only}/{len(noise_only_channels)} channels identified as noise-only.")

    for chan_idx in range(cube.shape[spec_axis]):

        if noise_only_channels[chan_idx]:

            # Slice out the channel
            slc = [slice(None)] * len(mask.shape)
            slc[spec_axis] = slice(chan_idx, chan_idx + 1)
            slc = tuple(slc)
            mask[slc] = False

    # Expect to be here with minimal memory footprint and mask
    # created.

    if operation == 'AND' or operation == 'OR':
        if os.path.isdir(cube_root + '.mask' + suffix_out):
            old_mask = read_cube(cube_root + '.mask' + suffix_out, huge_cube_workaround=True)
        else:
            logger.info("Operation AND/OR requested but no previous mask found.")
            logger.info("... will set operation=NEW.")
            operation = 'NEW'

    logger.info('Joining with old mask.')
    if operation == 'AND':
        mask = mask * old_mask
    if operation == 'OR':
        mask = (mask + old_mask) > 0
    if operation == 'NEW':
        mask = mask
    else:
        del old_mask

    logger.info('Recasting as an int.')
    # this might be better: mask.astype(int, copy=False)
    # mask = mask.astype(int)
    mask = mask.astype(np.int32)

    # Export the image to fits, put in the mask and convert back to a CASA image
    logger.info("Writing mask to disk")

    write_mask(
        cube_root + ".image" + suffix_in,
        cube_root + ".mask" + suffix_out,
        mask,
        huge_cube_workaround=True,
    )


def signal_mask(
        imaging_method='tclean',
        cube_root=None,
        out_file=None,
        suffix_in='',
        suffix_out='',
        operation='AND',
        high_snr=4.0,
        low_snr=2.0,
        absolute=False,
        do_roll=True,
):
    """
    A simple signal mask creation routine used to make masks on the
    fly during imaging. Leverages CASA statistics and scipy.
    """

    if imaging_method == 'sdintimaging':
        cube_root += '.joint.cube'

    if not os.path.isdir(cube_root + '.image' + suffix_in):
        logger.error('Data file not found: "' + cube_root + '.image' + suffix_in + '"')
        logger.info('Need CUBE_ROOT.image to be an image file.')
        logger.info('Returning. Generalize the code if you want different syntax.')
        return

    if os.path.isdir(cube_root + '.residual' + suffix_in):
        stats = stat_cube(cube_root + '.residual' + suffix_in)
    else:
        stats = stat_cube(cube_root + '.image' + suffix_in)
    rms = stats['medabsdevmed'][0] / 0.6745
    hi_thresh = high_snr * rms
    low_thresh = low_snr * rms

    header = casaStuff.imhead(cube_root + '.image' + suffix_in)
    if header['axisnames'][2] == 'Frequency':
        spec_axis = 2
    else:
        spec_axis = 3

    logger.info('Reading cube.')
    cube = read_cube(cube_root + '.image' + suffix_in, huge_cube_workaround=True)

    logger.info('Building high mask.')
    if absolute:
        hi_mask = (np.abs(cube) > hi_thresh)
    else:
        hi_mask = (cube > hi_thresh)

    if high_snr > low_snr:
        logger.info('Expanding mask.')
        logger.info('Building low mask.')
        if absolute:
            low_mask = (np.abs(cube) > low_thresh)
        else:
            low_mask = (cube > low_thresh)
        if do_roll:
            logger.info('... rolling.')
            rolled_low_mask = \
                (low_mask + np.roll(low_mask, 1, axis=spec_axis) + \
                 np.roll(low_mask, -1, axis=spec_axis)) >= 1
            low_mask = rolled_low_mask

        logger.info('... joining low mask with high mask via dilation.')
        mask = ndimage.binary_dilation(hi_mask,
                                       mask=low_mask,
                                       iterations=-1)
        del low_mask
        del hi_mask
        if do_roll:
            del rolled_low_mask
    else:
        logger.info('No expansion requested.')
        if do_roll:
            logger.info('... rolling.')
            mask = \
                (hi_mask + np.roll(hi_mask, 1, axis=spec_axis) + \
                 np.roll(hi_mask, -1, axis=spec_axis)) >= 1
            del hi_mask
        else:
            mask = hi_mask

    # Expect to be here with minimal memory footprint and mask
    # created.

    if operation == 'AND' or operation == 'OR':
        if os.path.isdir(cube_root + '.mask' + suffix_out):
            old_mask = read_cube(cube_root + '.mask' + suffix_out, huge_cube_workaround=True)
        else:
            logger.info("Operation AND/OR requested but no previous mask found.")
            logger.info("... will set operation=NEW.")
            operation = 'NEW'

    logger.info('Joining with old mask.')
    if operation == 'AND':
        mask = mask * old_mask
    if operation == 'OR':
        mask = (mask + old_mask) > 0
    if operation == 'NEW':
        mask = mask
    else:
        del old_mask

    logger.info('Recasting as an int.')
    # this might be better: mask.astype(int, copy=False)
    # mask = mask.astype(int)
    mask = mask.astype(np.int32)

    # Export the image to fits, put in the mask and convert back to a CASA image
    logger.info('Writing mask to disk')
    
    write_mask(cube_root + '.image' + suffix_in, cube_root + '.mask' + suffix_out, mask, huge_cube_workaround=True)


def apply_additional_mask(
        old_mask_file=None,
        new_mask_file=None,
        new_thresh=0.0,
        operation='AND'
):
    """
    Combine a mask with another mask on the same grid and some
    threshold. Can run AND/OR operations. Can be used to apply primary
    beam based masks by setting the PB file to new_mask_file and the
    pb_limit as new_thresh.
    """
    myia = au.createCasaTool(casaStuff.iatool)
    myia.open(new_mask_file)
    new_mask = myia.getchunk()
    myia.close()

    myia.open(old_mask_file)
    mask = myia.getchunk()
    if operation == "AND":
        mask *= (new_mask > new_thresh)
    else:
        mask = (mask + (new_mask > new_thresh)) >= 1.0
    myia.putchunk(mask)
    myia.close()

    return


def import_and_align_mask(
        in_file=None,
        out_file=None,
        template=None,
        blank_to_match=False,
):
    """
    Align a mask to a target astrometry. This includes some klugy
    steps (especially related to axes and interpolation) to make this
    work, e.g., for clean masks, most of the time.
    """

    # Import from FITS (could make optional)
    os.system('rm -rf ' + out_file + '.temp_copy' + ' 2>/dev/null')
    logger.debug('Importing mask file: "' + in_file + '"')
    casaStuff.importfits(fitsimage=in_file,
                         imagename=out_file + '.temp_copy',
                         overwrite=True)

    # Prepare analysis utility tool
    myia = au.createCasaTool(casaStuff.iatool)
    myim = au.createCasaTool(casaStuff.imtool)

    # Read mask data
    # myia.open(out_file+'.temp_copy')
    # mask = myia.getchunk(dropdeg=True)
    # myia.close()
    # print('**********************')
    # print('type(mask)', type(mask), 'mask.dtype', mask.dtype, 'mask.shape', mask.shape) # note that here the mask is in F dimension order, not Pythonic.
    # print(np.max(mask), np.min(mask))
    # print('**********************')

    # Read template image header
    hdr = casaStuff.imhead(template)
    # print('hdr', hdr)

    maskhdr = casaStuff.imhead(out_file + '.temp_copy')
    # print('maskhdr', maskhdr)

    # Check if 2D or 3D
    logger.debug('Template data axis names: ' + str(hdr['axisnames']) + ', shape: ' + str(hdr['shape']))
    logger.debug('Mask data axis names: ' + str(maskhdr['axisnames']) + ', shape: ' + str(maskhdr['shape']))
    is_template_2D = (np.prod(list(hdr['shape'])) == np.prod(list(hdr['shape'])[:2]))
    is_mask_2D = (np.prod(list(maskhdr['shape'])) == np.prod(list(maskhdr['shape'])[:2]))
    if is_template_2D and not is_mask_2D:
        logger.debug('Template image is 2D but mask is 3D, collapsing the mask over channel axes: ' + str(
            np.arange(maskhdr['ndim'] - 1, 2 - 1, -1)))
        # read mask array
        myia.open(out_file + '.temp_copy')
        mask = myia.getchunk(dropdeg=True)
        myia.close()
        # print('**********************')
        # print('type(mask)', type(mask), 'mask.dtype', mask.dtype, 'mask.shape', mask.shape) # Note that here array shapes are in F dimension order, i.e., axis 0 is RA, axis 1 is Dec, axis 2 is Frequency, etc.
        # print('**********************')
        #
        # collapse channel and higher axes
        # mask = np.any(mask.astype(int).astype(bool), axis=np.arange(maskhdr['ndim']-1, 2-1, -1)) # Note that here array shapes are in F dimension order, i.e., axis 0 is RA, axis 1 is Dec, axis 2 is Frequency, etc.
        # mask = mask.astype(int)
        # while len(mask.shape) < len(hdr['shape']):
        #    mask = np.expand_dims(mask, axis=len(mask.shape)) # Note that here array shapes are in F dimension order, i.e., axis 0 is RA, axis 1 is Dec, axis 2 is Frequency, etc.
        # print('**********************')
        # print('type(mask)', type(mask), 'mask.dtype', mask.dtype, 'mask.shape', mask.shape) # Note that here array shapes are in F dimension order, i.e., axis 0 is RA, axis 1 is Dec, axis 2 is Frequency, etc.
        # print('**********************')
        # os.system('rm -rf '+out_file+'.temp_collapsed'+' 2>/dev/null')
        ##myia.open(out_file+'.temp_copy')
        # newimage = myia.newimagefromarray(outfile=out_file+'.temp_collapsed', pixels=mask.astype(int), overwrite=True)
        # newimage.done()
        # myia.close()
        #
        # collapse channel and higher axes
        os.system('rm -rf ' + out_file + '.temp_collapsed' + ' 2>/dev/null')
        myia.open(out_file + '.temp_copy')
        collapsed = myia.collapse(outfile=out_file + '.temp_collapsed', function='max',
                                  axes=np.arange(maskhdr['ndim'] - 1, 2 - 1, -1))
        collapsed.done()
        myia.close()
        # ia tools -- https://casa.nrao.edu/docs/CasaRef/image-Tool.html
        os.system('rm -rf ' + out_file + '.temp_copy' + ' 2>/dev/null')
        os.system('cp -r ' + out_file + '.temp_collapsed' + ' ' + out_file + '.temp_copy' + ' 2>/dev/null')
        os.system('rm -rf ' + out_file + '.temp_collapsed' + ' 2>/dev/null')

    # Align to the template grid
    os.system('rm -rf ' + out_file + '.temp_aligned' + ' 2>/dev/null')
    casaStuff.imregrid(imagename=out_file + '.temp_copy',
                       template=template,
                       output=out_file + '.temp_aligned',
                       asvelocity=True,
                       interpolation='nearest',
                       replicate=False,
                       overwrite=True)

    # Make an EXACT copy of the template, avoids various annoying edge cases
    os.system('rm -rf ' + out_file + ' 2>/dev/null')
    myim.mask(image=template, mask=out_file)

    # Pull the data out of the aligned mask and place it in the output file
    myia.open(out_file + '.temp_aligned')
    mask = myia.getchunk(dropdeg=True)
    myia.close()

    # If requested, blank the mask wherever the cube is non-finite.
    if blank_to_match:
        myia.open(template)
        nans = np.invert(myia.getchunk(dropdeg=True, getmask=True))
        myia.close()
        mask[nans] = 0.0

    # Shove the mask into the data set
    if is_template_2D and not is_mask_2D:
        while len(mask.shape) < len(hdr['shape']):
            mask = np.expand_dims(mask, axis=len(mask.shape))
        # print('**********************')
        # print('type(mask)', type(mask), 'mask.dtype', mask.dtype, 'mask.shape', mask.shape) # Note that here array shapes are in F dimension order, i.e., axis 0 is RA, axis 1 is Dec, axis 2 is Frequency, etc.
        # print('**********************')
        myia.open(out_file)
        data = myia.getchunk(dropdeg=False)
        data = mask
        myia.putchunk(data)
        myia.close()
    else:
        if (hdr['axisnames'][3] == 'Frequency') and (hdr['ndim'] == 4):
            myia.open(out_file)
            data = myia.getchunk(dropdeg=False)
            data[:, :, 0, :] = mask.reshape((data.shape[0], data.shape[1], -1))
            myia.putchunk(data)
            myia.close()
        elif (hdr['axisnames'][2] == 'Frequency') and (hdr['ndim'] == 4):
            myia.open(out_file)
            data = myia.getchunk(dropdeg=False)
            data[:, :, :, 0] = mask.reshape((data.shape[0], data.shape[1], -1))
            myia.putchunk(data)
            myia.close()
        else:
            logger.info("ALERT! Did not find a case.")

    os.system('rm -rf ' + out_file + '.temp_copy' + ' 2>/dev/null')
    os.system('rm -rf ' + out_file + '.temp_aligned' + ' 2>/dev/null')
    return ()

# endregion
