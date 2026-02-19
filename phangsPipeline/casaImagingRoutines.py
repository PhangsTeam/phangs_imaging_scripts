"""
Standalone routines related to CASA imaging.
"""

import copy
import inspect
import logging
import os

import analysisUtils as au
import numpy as np

from . import casaMaskingRoutines as cmr
from . import casaStuff
from .clean_call import CleanCall

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# endregion

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Routines to set up imaging
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

# region Setting up imaging

def estimate_cell_and_imsize(
        infile=None,
        clean_call=None,
        oversamp=5,
        force_square=False,
):
    """
    Pick a cell and image size for a measurement set. Requests an
    oversampling factor, which is by default 5. Will pick a good size
    for the FFT and will try to pick a round number for the cell
    size. Returns variables appropriate for cell= and imsize= in
    tclean.
    """

    if not os.path.isdir(infile):
        logger.error('Error! The input file "' + infile + '" was not found!')
        return

    # If supplied, use the clean call to determine pblevel

    if not isinstance(clean_call, CleanCall):
        pblevel = 0.2
    else:
        pblevel = clean_call.get_param('pblimit')

    # These are the CASA-preferred sizes for fast FFTs

    valid_sizes = []
    for ii in range(10):
        for kk in range(3):
            for jj in range(3):
                valid_sizes.append(2 ** (ii + 1) * 5 ** jj * 3 ** kk)
    valid_sizes = sorted(valid_sizes)
    valid_sizes = np.array(valid_sizes)

    # Cell size implied by baseline distribution from analysis
    # utilities.

    au_cellsize, au_imsize, _ = au.pickCellSize(infile,
                                                imsize=True,
                                                npix=oversamp,
                                                intent='',
                                                pblevel=pblevel,
                                                )
    xextent = au_cellsize * au_imsize[0] * 1.2
    yextent = au_cellsize * au_imsize[1] * 1.2

    # Make the cell size a nice round number

    if au_cellsize < 0.01:
        cell_size = au_cellsize
    elif 0.01 <= au_cellsize < 0.05:
        cell_size = np.floor(au_cellsize / 0.005) * 0.005
    elif 0.05 <= au_cellsize < 0.1:
        cell_size = np.floor(au_cellsize / 0.01) * 0.01
    elif 0.1 <= au_cellsize < 0.5:
        cell_size = np.floor(au_cellsize / 0.05) * 0.05
    elif 0.5 <= au_cellsize < 1.0:
        cell_size = np.floor(au_cellsize / 0.1) * 0.1
    elif 1.0 <= au_cellsize < 2.0:
        cell_size = np.floor(au_cellsize / 0.25) * 0.25
    elif 2.0 <= au_cellsize < 5.0:
        cell_size = np.floor(au_cellsize / 0.5) * 0.5
    else:
        cell_size = np.floor(au_cellsize / 1.0) * 0.5

    # Now make the image size a good number for the FFT

    need_cells_x = xextent / cell_size
    need_cells_y = yextent / cell_size

    cells_x = np.min(valid_sizes[valid_sizes > need_cells_x])
    cells_y = np.min(valid_sizes[valid_sizes > need_cells_y])

    # If requested, force the mosaic to be square. This avoids
    # pathologies in CASA versions 5.1 and 5.3.

    if force_square:
        if cells_y < cells_x:
            cells_y = cells_x
        if cells_x < cells_y:
            cells_x = cells_y

    image_size = [int(cells_x), int(cells_y)]
    cell_size_string = str(cell_size) + 'arcsec'

    return cell_size_string, image_size


# endregion

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Routines to set manipulate files associated with imaging
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

# region Input and output of imaging products

def wipe_imaging(
        image_root=None,
        imaging_method='tclean',
):
    """
    Wipe files associated with a cube or continuum imaging. Tries to
    delete all images and supporting products, including the output of
    any MFS imaging.

    Parameters
    ----------
    image_root : str
        Image root name to delete.
    imaging_method : str
        'tclean' or 'sdintimaging'

    """

    if image_root == None:
        return

    allowed_imaging_methods = ['tclean', 'sdintimaging']

    if imaging_method not in allowed_imaging_methods:
        raise ValueError('imaging_method must be one of {0}. Given {1}'
                         .format(allowed_imaging_methods, imaging_method))

    logger.debug('wipe_imaging under "' + os.getcwd() + '"')

    exts = ['image', 'model', 'residual', 'mask', 'pb', 'psf', 'weight', 'sumwt',
            'alpha', 'alpha.error',
            'beta', 'beta.error',
            'image.tt0', 'image.tt1', 'image.tt2',
            'model.tt0', 'model.tt1', 'model.tt2',
            'residual.tt0', 'residual.tt1', 'residual.tt2',
            'mask.tt0', 'mask.tt1', 'mask.tt2',
            'pb.tt0', 'pb.tt1', 'pb.tt2',
            'psf.tt0', 'psf.tt1', 'psf.tt2',
            'weight.tt0', 'weight.tt1', 'weight.tt2',
            'sumwt.tt0', 'sumwt.tt1', 'sumwt.tt2']

    if imaging_method == 'tclean':
        cmd_list = ['rm -rf {0}.{1}'.format(image_root, ext) for ext in exts]
    elif imaging_method == 'sdintimaging':
        cube_exts = ['sd.cube', 'int.cube', 'joint.cube', 'joint.multiterm']
        cmd_list = ['rm -rf {0}.{1}.{2}'.format(image_root, cube_ext, ext)
                    for ext in exts for cube_ext in cube_exts]

    for this_cmd in cmd_list:
        logger.debug(this_cmd)
        os.system(this_cmd + ' 2>/dev/null')

    return ()


def copy_imaging(
        input_root=None,
        output_root=None,
        imaging_method='tclean',
        wipe_first=True,
        chan_num=None):
    """
    Copy all of the files from a cube or continuum imaging output by
    clean to have a new root name. Most commonly used to make a backup
    copy of imaging output during iterative imaging (e.g., clean
    loops, shifting clean modes, selfcal, etc). Overwrites any
    previous imaging with that output name.
    """

    if wipe_first:
        wipe_imaging(output_root, imaging_method=imaging_method)

    logger.debug('Copying imaging from root ' + input_root + ' to root ' + output_root)

    exts = ['image', 'model', 'residual', 'mask', 'pb', 'psf', 'weight', 'sumwt',
            'alpha', 'alpha.error',
            'beta', 'beta.error',
            'image.tt0', 'image.tt1', 'image.tt2',
            'model.tt0', 'model.tt1', 'model.tt2',
            'residual.tt0', 'residual.tt1', 'residual.tt2',
            'mask.tt0', 'mask.tt1', 'mask.tt2',
            'pb.tt0', 'pb.tt1', 'pb.tt2',
            'psf.tt0', 'psf.tt1', 'psf.tt2',
            'weight.tt0', 'weight.tt1', 'weight.tt2',
            'sumwt.tt0', 'sumwt.tt1', 'sumwt.tt2']

    if imaging_method == 'tclean':
        cmd_list = ['cp -r {0}.{1} {2}.{3}'.format(input_root, ext, output_root, ext) for ext in exts]
    elif imaging_method == 'sdintimaging':
        cube_exts = ['sd.cube', 'int.cube', 'joint.cube', 'joint.multiterm']
        cmd_list = ['cp -r {0}.{1}.{2} {3}.{4}.{5}'.format(input_root, cube_ext, ext, output_root, cube_ext, ext)
                    for ext in exts for cube_ext in cube_exts]

    for this_cmd in cmd_list:
        logger.debug(this_cmd)
        os.system(this_cmd + ' 2>/dev/null')


def export_imaging_to_fits(
        image_root=None,
        imaging_method='tclean',
        bitpix=-32,
        just_image=False):
    """
    Export the products associated with a CASA imaging run to FITS.
    """

    exts = ['alpha', 'alpha.error',
            'beta', 'beta.error',
            'image.tt0', 'image.tt1', 'image.tt2',
            'model.tt0', 'model.tt1', 'model.tt2',
            'residual.tt0', 'residual.tt1', 'residual.tt2',
            'mask.tt0', 'mask.tt1', 'mask.tt2',
            'pb.tt0', 'pb.tt1', 'pb.tt2',
            'psf.tt0', 'psf.tt1', 'psf.tt2',
            'weight.tt0', 'weight.tt1', 'weight.tt2',
            'image', 'model', 'residual', 'mask', 'pb', 'psf', 'weight']

    ext_map = {}

    if imaging_method == 'tclean':
        for ext in exts:
            if ext == 'image':
                ext_map['.%s' % ext] = '.fits'
            elif 'image' in ext and 'tt0' in ext:
                ext_map['.%s' % ext] = '.fits'
            elif 'image' in ext:
                ext_map['.%s' % ext] = '%s.fits' % ext.replace('image', '').replace('.', '_')
            elif 'tt0' in ext:
                ext_map['.%s' % ext] = '_%s.fits' % ext.replace('.tt0', '').replace('.', '_')
            else:
                ext_map['.%s' % ext] = '_%s.fits' % ext.replace('.', '_')
    elif imaging_method == 'sdintimaging':
        cube_exts = ['sd.cube', 'int.cube', 'joint.cube', 'joint.multiterm']
        for cube_ext in cube_exts:
            for ext in exts:
                if ext == 'image':
                    ext_map['.%s.%s' % (cube_ext, ext)] = '_%s.fits' % cube_ext.replace('.', '_')
                elif 'image' in ext and 'tt0' in ext:
                    ext_map['.%s.%s' % (cube_ext, ext)] = '_%s.fits' % cube_ext.replace('.', '_')
                elif 'image' in ext:
                    ext_map['.%s.%s' % (cube_ext, ext)] = '_%s_%s.fits' % (cube_ext.replace('.', '_'),
                                                                           ext.replace('image', '').replace('.', '_'))
                elif 'tt0' in ext:
                    ext_map['.%s.%s' % (cube_ext, ext)] = '_%s_%s.fits' % (cube_ext.replace('.', '_'),
                                                                           ext.replace('.tt0', '').replace('.', '_'))
                else:
                    ext_map['.%s.%s' % (cube_ext, ext)] = '_%s_%s.fits' % (cube_ext.replace('.', '_'),
                                                                           ext.replace('.', '_'))

    for this_ext in ext_map.keys():
        if just_image and ((this_ext != '.tt0') and this_ext != '.image'):
            continue

        this_casa_ext = this_ext
        this_fits_ext = ext_map[this_ext]

        casa_image = image_root + this_casa_ext
        if not os.path.isdir(casa_image):
            continue
        fits_image = image_root + this_fits_ext

        logger.debug('exportfits from ' + casa_image + ' to ' + fits_image)
        casaStuff.exportfits(imagename=casa_image,
                             fitsimage=fits_image,
                             velocity=True, overwrite=True, dropstokes=True,
                             dropdeg=True, bitpix=bitpix)

    return ()


# endregion

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Execute a clean call
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

# region clean call execution

def execute_clean_call(
        clean_call=None,
        imaging_method='tclean',
        convergence_fracflux=0.02,
        reset=False,
):
    """
    Execute a clean call object, optionally deleting previous versions
    of the imaging first.

    Set reset to wipe any existing output data.
    """

    if not isinstance(clean_call, CleanCall):
        logger.error("Please input a valid clean call!")
        raise Exception("Please input a valid clean call!")

    if not clean_call.has_param('vis'):
        logger.error("No visibility defined in clean_call. Returning.")
        return

    if not clean_call.has_param('imagename') or clean_call.get_param('imagename') is None:
        logger.error("No imagename defined in clean_call. Returning.")
        return

    # <TODO><DEBUG><DL># this will not overwrite existing data
    # if os.path.isdir(clean_call.get_param('imagename')+'.image') and not os.path.isdir(clean_call.get_param('imagename')+'.image'+'.touch'):
    #    logger.info('Found existing data "'+clean_call.get_param('imagename')+'.image'+'". Will not overwrite.')
    #    return

    if not os.path.isdir(clean_call.get_param('vis')):
        logger.error("Visibility file not found: " + clean_call.get_param('vis'))
        return

    if clean_call.logfile != None:
        oldlogfile = casaStuff.casalog.logfile()
        casaStuff.casalog.setlogfile(clean_call.logfile)

    if reset:
        logger.debug("Wiping previous versions of the cube.")
        wipe_imaging(clean_call.get_param('imagename'), imaging_method=imaging_method)

    # a simple way to slightly solve the compatible issue is to check
    # the list of expected_kwargs and only return keys inside it.
    if imaging_method == 'tclean':
        logger.debug("Running CASA " + str(clean_call))
        expected_kwargs = inspect.getfullargspec(casaStuff.tclean)[0]
    elif imaging_method == 'sdintimaging':
        logger.debug("Running CASA " + str(clean_call).replace('tclean', 'sdintimaging'))
        expected_kwargs = inspect.getfullargspec(casaStuff.sdintimaging)[0]
    else:
        logger.error('Unexpected imaging method %s' % imaging_method)
        raise Exception('Unexpected imaging method %s' % imaging_method)

    clean_kwargs = clean_call.kwargs_for_clean()
    active_kwargs = {}  # kwarg dict

    if expected_kwargs is not None:
        if 'self' in expected_kwargs:
            expected_kwargs.remove('self')
        missing_kwargs = []  # list
        unused_kwargs = []  # list
        for k in expected_kwargs:
            if k in clean_kwargs:
                active_kwargs[k] = clean_kwargs[k]
            else:
                missing_kwargs.append(k)
        for k in clean_kwargs:
            if not (k in expected_kwargs):
                unused_kwargs.append(k)
        if imaging_method == 'sdintimaging':
            # Put in the convergence_fracflux criteria, and make sure we don't throw a warning
            active_kwargs['convergence_fracflux'] = convergence_fracflux
            missing_kwargs.remove('convergence_fracflux')
        if len(unused_kwargs) > 0:
            logger.warning('Unused key arguments for ' + imaging_method + ': ' + str(unused_kwargs))
        if len(missing_kwargs) > 0:
            logger.warning('Missing key arguments for ' + imaging_method + ': ' + str(
                missing_kwargs) + '. Caution that CASA will use some default values depending on the CASA version.')
    else:
        active_kwargs = copy.deepcopy(clean_kwargs)

    # Force pbmask == pblimit
    # For cubes with an empty channel, pbmask < pblimit triggers
    # NaNs in cleaning as of CASA 5.6.1.
    # Force pbmask >= pblimit to avoid this.
    if 'pblimit' in active_kwargs and 'pbmask' in active_kwargs:
        if active_kwargs['pblimit'] > active_kwargs['pbmask']:
            active_kwargs['pbmask'] = active_kwargs['pblimit']

    active_kwargs['fullsummary'] = True

    if imaging_method == 'tclean':
        # os.mkdir(clean_call.get_param('imagename')+'.image'+'.touch') #<TODO><DEBUG><DL>#
        tclean_result = casaStuff.tclean(**active_kwargs)
        # os.rmdir(clean_call.get_param('imagename')+'.image'+'.touch') #<TODO><DEBUG><DL>#
    elif imaging_method == 'sdintimaging':
        tclean_result = casaStuff.sdintimaging(**active_kwargs)
    else:
        logger.error('Unexpected imaging method %s' % imaging_method)
        raise Exception('Unexpected imaging method %s' % imaging_method)

    if clean_call.logfile != None:
        casaStuff.casalog.setlogfile(oldlogfile)

    return tclean_result


# endregion


def check_noise_convergence(tclean_summary, noise, gain, last_n_cycles=3, z_threshold=2.0,
                            verbose=False, n_pix=None):
    """
    Statistical convergence test: are the recent CLEAN components consistent
    with noise rather than signal?

    Under H0 (noise only), CLEAN selects the extremum of the residual image at
    each minor cycle. For n_pix finite pixels the extremum is drawn from both
    tails of the Gaussian, so Var(extremum) ≈ 2·ln(2·n_pix)·(gain·noise)^2.
    The sum S of N_total such extremal draws has
    Std(S) = sqrt(N_total) · gain · noise · sqrt(2·ln(2·n_pix)).
    When n_pix is None the older formula Std(S) = sqrt(N_total)·gain·noise
    is used as a backward-compatible fallback.

    Parameters
    ----------
    tclean_summary : dict
        Return value of tclean with fullsummary=True.
    noise : float
        RMS noise estimate (Jy/beam), e.g. from noise_for_cube().
    gain : float
        CLEAN loop gain (typically 0.1).
    last_n_cycles : int
        Number of trailing major cycles from the tclean call to include.
    z_threshold : float
        |z| < z_threshold declares components consistent with noise (converged).
    verbose : bool
        Print summary info.
    n_pix : int or None
        Number of finite pixels in the primary-beam image. When provided,
        enables the extreme-value variance correction. When None, falls back
        to the simpler random-sampling formula.
    Returns
    -------
    dict with keys: z_score, N_components, flux_sum, std_expected, converged
    None if summary is unavailable or malformed.
    """

    if tclean_summary is None or not isinstance(tclean_summary, dict):
        return None
    if 'summaryminor' not in tclean_summary:
        return None

    summaryminor = tclean_summary['summaryminor']
    total_flux_increment = 0.0
    total_iter_done = 0

    for field in summaryminor:
        for chan in summaryminor[field]:
            for pol in summaryminor[field][chan]:
                data = summaryminor[field][chan][pol]
                try:
                    iter_done = np.array(data['iterDone'])
                    model_flux = np.array(data['modelFlux'])
                    start_flux = np.array(data['startModelFlux'])
                except (KeyError, TypeError):
                    continue
                n_cycles = len(iter_done)
                if n_cycles == 0:
                    continue
                window = min(last_n_cycles, n_cycles)
                total_flux_increment += float(
                    np.sum(model_flux[-window:] - start_flux[-window:]))
                total_iter_done += int(np.sum(iter_done[-window:]))

    if total_iter_done == 0:
        return {'z_score': 0.0, 'converged': True,
                'N_components': 0, 'flux_sum': 0.0, 'std_expected': 0.0}

    if n_pix is not None and n_pix > 1:
        # Extreme-value correction: CLEAN picks the extremum of the noise field.
        # Both positive max and negative min contribute → factor 2·n_pix.
        std_expected = (np.sqrt(total_iter_done) * abs(gain) * float(noise)
                        * np.sqrt(2.0 * np.log(2.0 * n_pix)))
    else:
        # Backward-compatible fallback: random-sampling formula.
        std_expected = np.sqrt(total_iter_done) * abs(gain) * float(noise)

    z_score = total_flux_increment / std_expected if std_expected > 0.0 else 0.0

    if verbose:
        print('z_score', z_score, 'total_iter_done', total_iter_done,
            'total_flux_increment', total_flux_increment,
            'std_expected', std_expected)

    return {
        'z_score': float(z_score),
        'N_components': total_iter_done,
        'flux_sum': total_flux_increment,
        'std_expected': float(std_expected),
        'converged': abs(z_score) < z_threshold,
    }

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Run a clean call with NITER=0 to make a dirty image
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def make_dirty_image(
        clean_call=None,
        imaging_method='tclean'
):
    """
    Create a dirty image using the provided clean call. Forces number
    of iterations to zero before executing the clean call and enforces
    psf and residual calculation but otherwise leaves the clean_call
    unchanged. Making a dirty image also forces a reset, wiping any
    previous version of the imaging. Avoids mutating the clean_call.
    """

    if not isinstance(clean_call, CleanCall):
        logger.error("Please input a valid clean call!")
        raise Exception("Please input a valid clean call!")

    dirty_clean_call = copy.deepcopy(clean_call)

    # TODO: Currently sdintimaging doesn't properly produce a dirty image. Hack around this for now with a low
    #   gain and niter=1

    if imaging_method == 'tclean':
        dirty_clean_call.set_param('niter', 0)
    elif imaging_method == 'sdintimaging':
        dirty_clean_call.set_param('niter', 1)
        dirty_clean_call.set_param('cycleniter', 1)
        dirty_clean_call.set_param('gain', 0.001)

        # Make sure we don't have a PSF set, and use hogbom
        dirty_clean_call.set_param('deconvolver', 'hogbom')
        dirty_clean_call.set_param('sdpsf', '')
    dirty_clean_call.set_param('calcres', True)
    dirty_clean_call.set_param('calcpsf', True)

    execute_clean_call(dirty_clean_call, imaging_method=imaging_method, reset=True)

    return ()


# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Repeated run a clean call, looking for convergence
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def eval_niter(
        loopnum=1,
        baseval=10,
        model='geometric',
        factor=2.0,
        saturation=1000,
        other_input=None,
):
    """
    Helper function to evaluate the number of iterations.
    """
    niter = None

    # Fixed number of iterations

    if model.lower() == 'fixed':
        niter = baseval

    # A geometric model starts at the base value and scales by
    # factor each time.

    if model.lower() == 'geometric':
        niter = baseval * factor ** (loopnum)

    # A linear model starts at the base value and increases by
    # baseval*(factor*loopnum) each time.

    if model.lower() == 'linear':
        niter = baseval * (1.0 + factor * loopnum)

    # Experimental/untested: a sequence of iterations
    if model.lower() == 'sequence':
        if loopnum >= len(other_input):
            index = len(other_input) - 1
        else:
            index = loopnum
        niter = other_input[loopnum]

    # Experimental/untested: an expression to be 'exec'ed
    if model.lower() == 'expr':
        exec(other_input)

    # Cap at the saturation value
    if saturation is not None:
        if niter >= saturation:
            niter = saturation

    return int(niter)


def clean_loop(
        clean_call=None,
        imaging_method='tclean',
        record_file=None,
        suffix='',
        log_ext=None,
        niter_base_perchan=10,
        niter_growth_model='geometric',
        niter_growth_factor=2.0,
        niter_saturation_perchan=1000,
        niter_other_input=None,
        cycleniter_base=100,
        cycleniter_growth_model='linear',
        cycleniter_growth_factor=1.0,
        cycleniter_saturation_value=1000,
        cycleniter_other_input=None,
        threshold_type='snr',
        threshold_value=4.0,
        min_loops=0,
        max_loops=20,
        max_total_niter=None,
        convergence_fracflux=0.02,
        convergence_totalflux=None,
        convergence_fluxperniter=None,
        use_absolute_delta=True,
        stop_at_negative=True,
        remask_each_loop=False,
        force_dirty_image=False,
        convergence_noise_z_threshold=None,
        convergence_noise_window=3,
):
    """
    Carry out an iterative clean until a convergence criteria is
    met. The loop releases progressively more iterations to the
    provided clean call and checks for convergence after each call to
    clean.

    Optional parameters control the growth of major cycle
    (cycle_niter) and total (niter) iterations/components, the
    implementation of absolute or signal-to-noise thresholds, and the
    total number of loops allowed.

    Convergence is checked by looking at the fractional change in the
    integrated model flux during successive iterations. With a strong
    threshold and enough iterations released, this can practically
    reduce to approaching the signal to noise threshold. The user also
    supplies a maximum number of loops, which prevent running away in
    cases of divergence.

    Future improvements could allow the execution of arbitrary
    functions to characterize the data and check for convergence or to
    manipulate the clean mask. Right now "remasking" and "stat_cube"
    are the only option implemented in this step.
    """

    # Some definitions and error checking

    if not isinstance(clean_call, CleanCall):
        logger.error("Please input a valid clean call!")
        raise Exception("Please input a valid clean call!")

    if suffix == '':
        if clean_call.has_param('specmode'):
            if clean_call.get_param('specmode') == 'mfs':
                suffix = '.tt0'

    valid_model_types = ['fixed', 'geometric', 'linear', 'sequence', 'expr']
    for growth_model in [niter_growth_model.lower(), cycleniter_growth_model.lower()]:
        if growth_model not in valid_model_types:
            logger.warning("Growth model not recognized: ", growth_model)
            return ()

    valid_threshold_types = ['snr', 'absolute']
    if threshold_type.lower() not in valid_threshold_types:
        logger.warning("Threshold type not recognized: ", threshold_type)
        return ()

    # Check if a residual image exists. If not, then build the dirty
    # image. Also build the dirty image if the flag to
    # force_dirt_image is set to True.

    missing_image = True

    if imaging_method == 'tclean':
        residual_image_name = clean_call.get_param('imagename') + '.residual' + suffix
    elif imaging_method == 'sdintimaging':
        residual_image_name = clean_call.get_param('imagename') + '.joint.cube.residual' + suffix
    else:
        logger.error('Unexpected imaging method %s' % imaging_method)
        raise Exception('Unexpected imaging method %s' % imaging_method)

    if os.path.isdir(residual_image_name):
        missing_image = False

    if missing_image or force_dirty_image:
        make_dirty_image(clean_call, imaging_method=imaging_method)

    # Copy the clean call so we can manipulate it without changing the
    # input version call.

    working_call = copy.deepcopy(clean_call)
    working_call.set_param('calcres', False)
    working_call.set_param('calcpsf', False)

    # Note the number of channels, which is used in setting the number
    # of iterations that we give to an individual clean call.

    vm = au.ValueMapping(working_call.get_param('vis'))

    # If we have a number of channels defined, then use that
    nchan = working_call.get_param('nchan')

    # Otherwise, pull the total number of channels from the spwInfo
    if nchan is None:
        nchan = -1
    if nchan <= 0:
        nchan = vm.spwInfo[0]['numChannels']

    # Create a text record of progress through successive clean calls.

    record = []
    record.append("loopnum, deconvolver, niter, cycleniter, threshold, noise, model_flux, frac_delta_flux, noise_z_score\n")
    record.append("# column 1: Loop number.\n")
    record.append("# column 2: Deconvolver used in clean.\n")
    record.append("# column 3: Allocated number of iterations.\n")
    record.append("# column 4: Cycleniter used to force major cycles.\n")
    record.append("# column 5: Threshold supplied to clean.\n")
    record.append("# column 6: Noise level measured in residuals.\n")
    record.append("# column 7: Integrated model flux.\n")
    record.append("# column 8: Fractional change in flux from previous loop.\n")
    record.append("# column 9: Noise z-score.\n")

    # Initialize the loop counter and our tracking of the flux in the
    # model (which we use to estimate convergence).

    loop = 0
    cumulative_niter = 0
    previous_flux = 0.0
    current_flux = 0.0

    # Run the main loop

    proceed = True
    while proceed == True:

        # Calculate the number of total iterations (niter) and
        # iterations per major cycle (cycleniter) released to clean
        # during this call.

        niter = eval_niter(loopnum=loop,
                           baseval=niter_base_perchan * nchan,
                           model=niter_growth_model, factor=niter_growth_factor,
                           saturation=niter_saturation_perchan * nchan,
                           other_input=niter_other_input)

        cycleniter = eval_niter(loopnum=loop, baseval=cycleniter_base,
                                model=cycleniter_growth_model, factor=cycleniter_growth_factor,
                                saturation=cycleniter_saturation_value,
                                other_input=cycleniter_other_input)

        working_call.set_param('niter', niter, nowarning=True)
        working_call.set_param('cycleniter', cycleniter, nowarning=True)

        cumulative_niter = cumulative_niter + niter

        # Calculate the current noise in the residual image. Don't
        # exclude the masked region from the noise calculation but do
        # turn on iterative noise estimation (Chauvenet+m.a.d. using 5
        # iterations should be quite robust).

        logger.info("Computing noise cube.")

        if imaging_method == 'tclean':
            infile = working_call.get_param('imagename') + '.residual' + suffix
        elif imaging_method == 'sdintimaging':
            infile = working_call.get_param('imagename') + '.joint.cube.residual' + suffix

        current_noise = cmr.noise_for_cube(
            infile=infile,
            method='chauvmad', niter=5)

        # Set the threshold for the clean call. Clean expects a value
        # in Jy/beam. Switch on the threshold type to make the string
        # and attach it to the clean call.

        if threshold_type == 'snr':
            threshold_string = str(current_noise * threshold_value) + 'Jy/beam'
        elif threshold_type == 'absolute':
            if type(threshold_value) == type(0.0):
                threshold_string = str(threshold_value) + 'Jy/beam'
            else:
                threshold_string = threshold_value
        else:
            threshold_string = '0.0Jy/beam'

        working_call.set_param('threshold', threshold_string, nowarning=True)

        logger.info("Loop %d, niter %d, cycleniter %d, cumulative_niter %d, threshold %s."%(\
            loop, niter, cycleniter, cumulative_niter, threshold_string))

        # If requested mask at each step (this is experimental, we're
        # seeing if it helps to avoid divergence during the deep
        # single scale clean.)

        if remask_each_loop:
            logger.info("")
            logger.info("Remasking.")
            logger.info("")

            if imaging_method == 'tclean':
                out_file = working_call.get_param('imagename') + '.mask' + suffix
            elif imaging_method == 'sdintimaging':
                out_file = working_call.get_param('imagename') + '.joint.cube.mask' + suffix

            cmr.signal_mask(
                cube_root=working_call.get_param('imagename'),
                out_file=out_file,
                suffix_in=suffix,
                suffix_out=suffix,
                operation='AND',
                high_snr=4.0,
                low_snr=2.0,
                absolute=False)
            working_call.usemask = 'user'

        # Set the log file (revisit this)

        if log_ext is not None:
            working_call.logfile = working_call.get_param('imagename') + "_loop_" + str(loop) + "_" + log_ext + ".log"
        else:
            working_call.logfile = None

        # Save the previous version of the imaging for comparison

        copy_imaging(
            input_root=working_call.get_param('imagename'),
            output_root=working_call.get_param('imagename') + '_prev',
            imaging_method=imaging_method)

        # Check user-preset mask parameter, disable it if a *.mask already exists

        if (loop > 0) and \
                (working_call.get_param('usemask') == "user") and \
                (working_call.get_param('mask') is not None) and \
                (working_call.get_param('mask') != ''):

            if imaging_method == 'tclean':
                mask_name = working_call.get_param('imagename') + '.mask' + suffix
            elif imaging_method == 'sdintimaging':
                mask_name = working_call.get_param('imagename') + '.joint.cube.mask' + suffix

            if os.path.isdir(mask_name):
                logger.debug("Found clean mask \"%s\", will not re-use the mask \"%s\" in the clean parameter file." % ( \
                    mask_name,
                    working_call.get_param('mask')))
                working_call.set_param('mask', '')

        # Execute the clean call.

        tclean_summary = execute_clean_call(working_call,
                                            imaging_method=imaging_method,
                                            convergence_fracflux=convergence_fracflux)

        # Calculate the new model flux and the change relative to the
        # previous step, normalized by current flux and by iterations.

        if imaging_method == 'tclean':
            cube_file = working_call.get_param('imagename') + '.model' + suffix
        elif imaging_method == 'sdintimaging':
            cube_file = working_call.get_param('imagename') + '.joint.cube.model' + suffix

        model_stats = cmr.stat_cube(cube_file)

        previous_flux = current_flux
        current_flux = model_stats['sum'][0]

        delta_flux = (current_flux - previous_flux)
        if use_absolute_delta:
            delta_flux = abs(delta_flux)

        flux_per_iter = delta_flux / niter
        frac_delta_flux = delta_flux / previous_flux

        # If we don't have any model flux after iter 1, then catch that here
        if np.isnan(frac_delta_flux) and loop > 0:
            frac_delta_flux = 0

        # Check whether the model flux convergence criteria is met

        if convergence_fracflux is not None:
            if frac_delta_flux < convergence_fracflux:
                proceed = False

        if convergence_totalflux is not None:
            if delta_flux < convergence_totalflux:
                proceed = False

        if convergence_fluxperniter is not None:
            if flux_per_iter < convergence_fluxperniter:
                proceed = False

        if max_total_niter is not None:
            if cumulative_niter >= max_total_niter:
                proceed = False

        # If requested, stop if the integrated model flux becomes
        # negative.

        if stop_at_negative:
            if current_flux < 0.0:
                proceed = False

        # Statistical noise convergence check: are the recent CLEAN components
        # consistent with noise draws rather than signal? Uses the conservative
        # variance bound Var(S) = N*(gain*noise)^2.

        if convergence_noise_z_threshold is not None:
            n_pix = None
            _imagename = working_call.get_param('imagename')
            pb_image = (_imagename + '.pb') if _imagename else None
            if pb_image and os.path.isdir(pb_image):
                try:
                    myia = au.createCasaTool(casaStuff.iatool)
                    myia.open(pb_image)
                    pb_data = myia.getchunk()
                    myia.close()
                    n_pix = int(np.sum(np.isfinite(pb_data)))
                except Exception:
                    n_pix = None
            noise_conv = check_noise_convergence(
                tclean_summary=tclean_summary,
                noise=current_noise,
                gain=working_call.get_param('gain') or 0.1,
                last_n_cycles=convergence_noise_window,
                z_threshold=convergence_noise_z_threshold,
                n_pix=n_pix,
            )
            if noise_conv is not None:
                logger.info(
                    "Noise convergence: z=%.2f, N=%d, n_pix=%s, flux_sum=%.4e Jy, "
                    "std_expected=%.4e Jy, converged=%s" % (
                        noise_conv['z_score'], noise_conv['N_components'],
                        str(n_pix),
                        noise_conv['flux_sum'], noise_conv['std_expected'],
                        str(noise_conv['converged'])))
                if noise_conv['converged']:
                    proceed = False
            else:
                noise_conv = None
        else:
            noise_conv = None

        # Enforce minimum and maximum limits on number of loops. These
        # override other convergence criteria.

        if loop >= max_loops:
            proceed = False
        if loop < min_loops:
            proceed = True

        # Generate a record line and print the current status to the screen

        this_record = ''
        this_record += str(loop) + ', '
        this_record += str(working_call.get_param('deconvolver')) + ', '
        this_record += str(working_call.get_param('niter')) + ', '
        this_record += str(working_call.get_param('cycleniter')) + ', '
        this_record += str(working_call.get_param('threshold')) + ', '
        this_record += str(current_noise) + 'Jy/beam, '
        this_record += str(current_flux) + 'Jy*chan, '
        this_record += str(frac_delta_flux) + ', '
        this_record += str(noise_conv['z_score'] if noise_conv is not None else 'N/A')
        this_record += '\n'

        # Print the current record to the screen

        record.append(this_record)
        for line in record:
            print(line)

        logger.info("... proceeding? " + str(proceed))

        if proceed == False:
            break

        loop += 1

    # ... if requested also write this to a file.

    if record_file != None:
        f = open(record_file, 'w')
        f.writelines(record)
        f.close()

    return ()


# endregion

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Evaluate the output of imaging
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def calc_residual_statistics(
        resid_name=None,
        mask_name=None,
):
    """
    """

    if os.path.isdir(resid_name) == False:
        logger.error('Error! The input file "' + resid_name + '" was not found!')
        return

    if os.path.isdir(mask_name) == False:
        logger.error('Error! The input file "' + mask_name + '" was not found!')
        return

    myia = au.createCasaTool(casaStuff.iatool)

    myia.open(mask_name)
    mask = myia.getchunk()
    myia.close()

    myia.open(resid_name)
    resid = myia.getchunk()
    myia.close()

    vec = resid[((mask == 1) * np.isfinite(resid))]
    del mask
    del resid

    current_noise = cmr.noise_for_cube(
        infile=resid_name,
        method='chauvmad', niter=5)

    out_dict = {
        'cubename':resid_name,
        'maskname':mask_name,
        'max':np.max(vec),
        'p99':np.percentile(vec,99),
        'p95':np.percentile(vec,95),
        'p90':np.percentile(vec,90),
        'noise':current_noise,
        }

    return(out_dict)


