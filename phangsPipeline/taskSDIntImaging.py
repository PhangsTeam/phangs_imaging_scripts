"""
Reworking of sdintimaging for the PHANGS-ALMA Pipeline.

Essentially identical, but includes a tolerance in the
fractional flux change between minor cycles to stop it
hanging indefinitely (a problem for 7m+TP data, in
particular).
"""

import copy
import logging

from casatasks.private.imagerhelpers.imager_base import PySynthesisImager
from casatasks.private.imagerhelpers.input_parameters import ImagerParameters
# from casatasks.private.cleanhelper import write_tclean_history, get_func_params
from casatasks.private.sdint_helper import *

from . import casaMaskingRoutines as cmr
from . import casaStuff

# Pull MPI in, if available

try:
    from casampi.MPIEnvironment import MPIEnvironment
    from casampi import MPIInterface
    mpi_available = True
except ImportError:
    mpi_available = False

sdintlib = SDINT_helper()
synu = casaStuff.synthesisutils()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Setup functions
def setup_imager_obj(param_list=None):
    """
    Setup imaging parameters
    """

    default_constructor = False
    if param_list is not None:
        if not isinstance(param_list, ImagerParameters):
            raise RuntimeError("Internal Error: invalid param_list")
    else:
        default_constructor = True

    if default_constructor:
        return PySynthesisImager
    else:
        return PySynthesisImager(params=param_list)


def setup_imager(imagename, calcres, calcpsf, inparams):
    """
    Initialise cube. Optionally run a major cycle
    """

    # Create a local copy of input params dict so that it can be modified
    locparams = copy.deepcopy(inparams)

    # Cube imaging setup
    locparams['imagename'] = imagename
    locparams['specmode'] = 'cube'
    locparams['niter'] = 0
    locparams['deconvolver'] = 'hogbom'

    params = ImagerParameters(**locparams)

    gridder = locparams['gridder']

    # Major cycle is either PySynthesisImager or PyParallelCubeSynthesisImager
    imagertool = setup_imager_obj(params)

    imagertool.initializeImagers()
    imagertool.initializeNormalizers()
    imagertool.setWeighting()
    if 'psfphasecenter' in locparams:
        psfphasecenter = locparams['psfphasecenter']
    else:
        psfphasecenter = ''

    # Extra one for psfphasecenter...
    imagerInst = None
    if psfphasecenter != '' and gridder == 'mosaic':
        imagerInst = setup_imager_obj()

    if 'restart' in locparams and os.path.exists(imagename + '.residual'):
        restart = locparams['restart']
    else:
        restart = False

    if calcpsf and not restart:
        imagertool.makePSF()
        imagertool.makePB()
        if psfphasecenter != '' and gridder == 'mosaic':
            casaStuff.casalog.post("doing with different phasecenter psf", "INFO")
            imagertool.unlockimages(0)
            psfParameters = param_list.getAllPars()
            psfParameters['phasecenter'] = psfphasecenter
            psfparam_list = ImagerParameters(**psfParameters)
            psfimager = imagerInst(params=psfparam_list)
            psfimager.initializeImagers()
            psfimager.setWeighting()
            psfimager.makeImage('psf', psfParameters['imagename'] + '.psf')

    # Run a major cycle
    if not restart:
        # Make dirty image
        if calcres:
            t0 = time.time()
            imagertool.runMajorCycle()
            t1 = time.time()
            casaStuff.casalog.post("***Time for major cycle (calcres=T): " + "%.2f" % (t1 - t0) + " sec", "INFO3",
                                   "task_tclean")

    if not restart:
        sdintlib.copy_restoringbeam(fromthis=imagename + '.psf', tothis=imagename + '.residual')
    return imagertool


def setup_deconvolver(imagename, inparams):
    """
    Cube or MFS minor cycles.
    """

    inparams['imagename'] = imagename
    params = ImagerParameters(**inparams)
    deconvolvertool = setup_imager_obj(params)

    deconvolvertool.initializeImagers()
    deconvolvertool.initializeNormalizers()
    deconvolvertool.setWeighting()

    if 'restart' in inparams and os.path.exists(imagename + '.residual'):
        restart = inparams['restart']
    else:
        restart = False

    if not restart:
        deconvolvertool.makePSF()
        deconvolvertool.makePB()

    # Initialize deconvolvers.
    deconvolvertool.initializeDeconvolvers()
    deconvolvertool.initializeIterationControl()

    if not restart:
        deconvolvertool.runMajorCycle()

    return deconvolvertool


def setup_sdimaging(template='', output='', inparms=None, sdparms=None):
    """
    Make the SD cube Image and PSF

    Option 1 : Use/Regrid cubes for the observed image and PSF
    Option 2 : Make the SD image and PSF cubes using 'tsdimager's usage of the SD gridder option.

    Currently, only Option 1 is supported.

    """

    sdintlib = SDINT_helper()
    if 'sdpsf' in sdparms:
        sdpsf = sdparms['sdpsf']
    else:
        raise RuntimeError("Internal Error: missing sdpsf parameter")

    if 'sdimage' in sdparms:
        sdimage = sdparms['sdimage']
    else:
        raise RuntimeError("Internal Error: missing sdimage parameter")
    if 'pblimit' in inparms:
        pblimit = inparms['pblimit']

    if 'restart' in inparms and os.path.exists(output + '.residual'):
        restart = inparms['restart']
    else:
        restart = False

    if not restart:

        if sdpsf != "":
            # Check the coordinates of psf with int psf
            sdintlib.checkpsf(sdpsf, template + '.psf')

        # Regrid the input SD image and PSF cubes to the target coordinate system.
        sdintlib.regridimage(imagename=sdimage, template=template + '.residual', outfile=output + '.residual')
        sdintlib.regridimage(imagename=sdimage, template=template + '.residual', outfile=output + '.image')

        if sdpsf != "":
            sdintlib.regridimage(imagename=sdpsf, template=template + '.psf', outfile=output + '.psf')
        else:
            # Make an internal sdpsf image if the user has not supplied one.
            casaStuff.casalog.post(
                "Constructing a SD PSF cube by evaluating Gaussians based on the restoring beam information in the "
                "regridded SD Image Cube")
            sdintlib.create_sd_psf(sdimage=output + '.residual', sdpsfname=output + '.psf')

            # Apply the pbmask from the INT image cube, to the SD cubes.
            sdintlib.addmask(inpimage=output + '.residual', pbimage=template + '.pb', pblimit=pblimit)
            sdintlib.addmask(inpimage=output + '.image', pbimage=template + '.pb', pblimit=pblimit)


def feather_residual(int_cube, sd_cube, joint_cube, applypb, inparm):
    if applypb:
        # Take initial INT_dirty image to flat-sky.
        sdintlib.modify_with_pb(inpcube=int_cube + '.residual',
                                pbcube=int_cube + '.pb',
                                cubewt=int_cube + '.sumwt',
                                chanwt=inparm['chanwt'],
                                action='div',
                                pblimit=inparm['pblimit'],
                                freqdep=True)

    # Feather flat-sky INT dirty image with SD image
    sdintlib.feather_int_sd(sdcube=sd_cube + '.residual',
                            intcube=int_cube + '.residual',
                            jointcube=joint_cube + '.residual',
                            sdgain=inparm['sdgain'],
                            dishdia=inparm['dishdia'],
                            usedata=inparm['usedata'],
                            chanwt=inparm['chanwt'])

    if applypb:
        if inparm['specmode'].count('cube') > 0:
            # Multiply the new JOINT dirty image by the frequency-dependent PB.
            fdep_pb = True
        else:
            # Multiply new JOINT dirty image by a common PB to get the effect of conjbeams.
            fdep_pb = False

        sdintlib.modify_with_pb(inpcube=joint_cube + '.residual',
                                pbcube=int_cube + '.pb',
                                cubewt=int_cube + '.sumwt',
                                chanwt=inparm['chanwt'],
                                action='mult',
                                pblimit=inparm['pblimit'],
                                freqdep=fdep_pb)


def delete_tmp_files():
    if os.path.exists('tmp_intplane'):
        os.system('rm -rf tmp_intplane')
    if os.path.exists('tmp_sdplane'):
        os.system('rm -rf tmp_sdplane')
    if os.path.exists('tmp_jointplane'):
        os.system('rm -rf tmp_jointplane')


def sdintimaging(
        usedata,
        # Single dish input data
        sdimage,
        sdpsf,
        sdgain,
        dishdia,
        # Interfermeter Data Selection
        vis,
        selectdata,
        field,
        spw,
        timerange,
        uvrange,
        antenna,
        scan,
        observation,
        intent,
        datacolumn,
        # Image definition
        imagename,
        imsize,
        cell,
        phasecenter,
        stokes,
        projection,
        startmodel,
        # Spectral parameters
        specmode,
        reffreq,
        nchan,
        start,
        width,
        outframe,
        veltype,
        restfreq,
        interpolation,
        perchanweightdensity,
        # Gridding parameters
        gridder,
        facets,
        psfphasecenter,
        wprojplanes,
        # PB
        vptable,
        mosweight,
        aterm,
        psterm,
        wbawp,
        cfcache,
        usepointing,
        computepastep,
        rotatepastep,
        pointingoffsetsigdev,
        pblimit,
        # Deconvolution parameters
        deconvolver,
        scales,
        nterms,
        smallscalebias,
        # Restoration options
        restoration,
        restoringbeam,
        pbcor,
        # Weighting
        weighting,
        robust,
        noise,
        npixels,
        uvtaper,
        # Iteration control
        niter,
        gain,
        threshold,
        nsigma,
        cycleniter,
        cyclefactor,
        minpsffraction,
        maxpsffraction,
        interactive,
        fullsummary,
        # (new) Mask parameters
        usemask,
        mask,
        pbmask,
        # Automask by multithresh
        sidelobethreshold,
        noisethreshold,
        lownoisethreshold,
        negativethreshold,
        smoothfactor,
        minbeamfrac,
        cutthreshold,
        growiterations,
        dogrowprune,
        minpercentchange,
        verbose,
        fastnoise,
        # Misc
        restart,
        calcres,
        calcpsf,
        convergence_fracflux=None,
):

    # From SDINT.do_reconstruct

    int_cube = imagename + '.int.cube'
    sd_cube = imagename + '.sd.cube'
    joint_cube = imagename + '.joint.cube'
    joint_multiterm = imagename + '.joint.multiterm'

    if specmode == 'mfs':
        decname = joint_multiterm
    else:
        decname = joint_cube

    # Checks and controls

    inpparams = locals().copy()

    # Deal with any parameters that need to change names or shouldn't be included

    inpparams.pop('convergence_fracflux')

    locvis = inpparams.pop('vis')
    if type(locvis) == list:
        llocvis = [v.lstrip() for v in locvis]
    else:
        llocvis = locvis.lstrip()

    inpparams['msname'] = llocvis
    inpparams['timestr'] = inpparams.pop('timerange')
    inpparams['uvdist'] = inpparams.pop('uvrange')
    inpparams['obs'] = inpparams.pop('observation')
    inpparams['state'] = inpparams.pop('intent')
    inpparams['loopgain'] = inpparams.pop('gain')
    inpparams['scalebias'] = inpparams.pop('smallscalebias')

    sdparms = {'sdimage': inpparams['sdimage'], 'sdpsf': inpparams['sdpsf'], 'sdgain': inpparams['sdgain']}

    if specmode == 'cont':
        specmode = 'mfs'
        inpparams['specmode'] = 'mfs'

    # Decide if pb needs to be applied
    if gridder == 'mosaic' or gridder == 'awproject':
        applypb = True
    else:
        applypb = False

    if (deconvolver == "mtmfs") and (specmode != 'mfs') and (specmode != 'cube' or nterms != 1) and (
            specmode != 'cubedata' or nterms != 1):
        casaStuff.casalog.post(
            "The MSMFS algorithm (deconvolver='mtmfs') applies only to specmode='mfs' or specmode='cube' with "
            "nterms=1 or specmode='cubedata' with nterms=1.",
            "WARN", "task_sdintimaging")
        return

    if deconvolver == "mtmfs" and (specmode == 'cube' or specmode == 'cubedata') and nterms == 1:
        casaStuff.casalog.post(
            "The MSMFS algorithm (deconvolver='mtmfs') with specmode='cube', nterms=1 is currently not supported. "
            "Please use deconvolver='multiscale' instead for cubes.",
            "WARN", "task_sdintimaging")
        return

    if specmode == 'mfs' and deconvolver != 'mtmfs':
        casaStuff.casalog.post(
            "Currently, only the multi-term MFS algorithm is supported for specmode=mfs. To make a single plane MFS "
            "image (while retaining the frequency dependence for the cube major cycle stage), please pick nterms=1 "
            "along with deconvolver=mtmfs. The scales parameter is still usable for multi-scale multi-term "
            "deconvolution",
            "WARN", "task_sdintimaging")
        return

    if gridder == 'awproject':
        casaStuff.casalog.post(
            "The awproject gridder is temporarily not supported with cube major cycles. Support will be brought back "
            "in a subsequent release.",
            "WARN", "task_sdintimaging")
        return

    if usedata == 'sd':
        casaStuff.casalog.post(
            "The Single-Dish-Only mode of sdintimaging is better supported via the deconvolve task which supports "
            "spectral cube, mfs and multi-term mfs deconvolution in the image domain alone. The deconvolve task is "
            "the more appropriate version to use for stand-alone image-domain deconvolution, and will not have the "
            "bookkeeping overheads currently present in the sdintimaging task's sd-only mode. Please note that the "
            "'sd' option of the sdintimaging task will be removed in a subsequent release.  Please refer to the task "
            "deconvolve documentation for instructions on how to prepare image and psf cubes for the deconvolve task "
            "for all these modes.",
            "WARN", "task_sdintimaging")

    # Construct ImagerParameters object

    imager = None
    param_list = None
    deconvolvertool = None

    # Put all parameters into dictionaries and check them.

    defparm = dict(
        list(zip(ImagerParameters.__init__.__code__.co_varnames[1:], ImagerParameters.__init__.__defaults__)))

    # Assign values to the ones passed to tclean and if not defined yet in tclean, assign them the default value of the
    # constructor
    bparm = {k: inpparams[k] if k in inpparams else defparm[k] for k in defparm.keys()}

    # Default mosweight=True is tripping other gridders as they are not expecting it to be true
    if bparm['mosweight'] and bparm['gridder'].find("mosaic") == -1:
        bparm['mosweight'] = False

    # Two options have been removed from the interface. Hard-code them here.
    bparm['normtype'] = 'flatnoise'  # Hard-code this since the pbcor steps assume it.
    bparm['conjbeams'] = False

    if len(pointingoffsetsigdev) > 0 and pointingoffsetsigdev[0] != 0.0 and usepointing and gridder.count('awproj') > 1:
        casaStuff.casalog.post("pointingoffsetsigdev will be used for pointing corrections with AWProjection", "WARN")

    # Set the children to load C++ libraries and applicator make workers ready for C++ based mpicommands

    cppparallel = False
    if mpi_available and MPIEnvironment.is_mpi_enabled:
        mint = MPIInterface.MPIInterface()
        cl = mint.getCluster()
        if is_CASA6:
            cl._cluster.pgc("from casatools import synthesisimager", False)
            cl._cluster.pgc("si=synthesisimager()", False)
        else:
            cl._cluster.pgc("from casac import casac", False)
            cl._cluster.pgc("si=casac.synthesisimager()", False)
        cl._cluster.pgc("si.initmpi()", False)
        cppparallel = True
        # Ignore chanchunk
        bparm['chanchunks'] = 1

    retrec = {}

    try:

        sdintlib = SDINT_helper()
        # Init major cycle elements
        casaStuff.casalog.post("INT cube setup ....")
        t0 = time.time()
        imager = setup_imager(imagename=int_cube, calcres=calcres, calcpsf=calcpsf, inparams=bparm)

        t1 = time.time()
        casaStuff.casalog.post("***Time for initializing imager (INT cube) : " + "%.2f" % (t1 - t0) + " sec", "INFO3",
                               "task_sdintimaging")

        # Init minor cycle elements
        if niter > 0 or restoration:
            casaStuff.casalog.post("Combined image setup ....")
            t0 = time.time()
            deconvolvertool = setup_deconvolver(imagename=decname, inparams=bparm)
            t1 = time.time()
            casaStuff.casalog.post("***Time for seting up deconvolver(s): " + "%.2f" % (t1 - t0) + " sec", "INFO3",
                                   "task_sdintimaging")

        if usedata != 'int':
            casaStuff.casalog.post("SD cube setup ....")
            setup_sdimaging(template=int_cube, output=sd_cube, inparms=bparm, sdparms=sdparms)

        # Check estimated memory
        imager.estimatememory()

        # Checks on INT and SD cubes
        validity, inpparams = sdintlib.check_coords(intres=int_cube + '.residual', intpsf=int_cube + '.psf',
                                                    intwt=int_cube + '.sumwt',
                                                    sdres=sd_cube + '.residual', sdpsf=sd_cube + '.psf',
                                                    sdwt='',
                                                    pars=inpparams)

        if not validity:
            casaStuff.casalog.post(
                'Exiting from the sdintimaging task due to inconsistencies found between the interferometer-only and '
                'singledish-only image and psf cubes. Please modify inputs as needed',
                'WARN')
            if imager is not None:
                imager.deleteTools()
            if deconvolvertool is not None:
                deconvolvertool.deleteTools()
            delete_tmp_files()
            return

        # inpparams now has a new parameter "chanwt" with ones and zeros to indicate chans that have data from both INT
        # and SD cubes (they are the 'ones'). This is to be used in feathering and in the cube-to-taylor sum and
        # modify_with_pb.

        # SDINT specific feathering....
        # Feather INT and SD residual images (feather in flat-sky. output has common PB)

        if 'restart' in bparm and os.path.exists(joint_cube + '.image'):
            restart = bparm['restart']
        else:
            restart = False

        if not restart:
            casaStuff.casalog.post("Feathering INT and SD residual images...")
            feather_residual(int_cube, sd_cube, joint_cube, applypb, inpparams)
            sdintlib.feather_int_sd(sdcube=sd_cube + '.psf',
                                    intcube=int_cube + '.psf',
                                    jointcube=joint_cube + '.psf',
                                    sdgain=sdgain,
                                    dishdia=dishdia,
                                    usedata=usedata,
                                    chanwt=inpparams['chanwt'])

        synu.fitPsfBeam(joint_cube)

        if specmode == 'mfs':
            # Calculate Spectral PSFs and Taylor Residuals
            casaStuff.casalog.post("Calculate spectral PSFs and Taylor Residuals...")
            sdintlib.cube_to_taylor_sum(cubename=joint_cube + '.psf',
                                        cubewt=int_cube + '.sumwt',
                                        chanwt=inpparams['chanwt'],
                                        mtname=joint_multiterm + '.psf',
                                        nterms=nterms, reffreq=inpparams['reffreq'], dopsf=True)
            sdintlib.cube_to_taylor_sum(cubename=joint_cube + '.residual',
                                        cubewt=int_cube + '.sumwt',
                                        chanwt=inpparams['chanwt'],
                                        mtname=joint_multiterm + '.residual',
                                        nterms=nterms, reffreq=inpparams['reffreq'], dopsf=False)
            synu.fitPsfBeam(joint_multiterm, nterms=nterms)

        if niter > 0:
            isit = deconvolvertool.hasConverged()
            deconvolvertool.updateMask()

            # Record the flux with each minor cycle to see if it's appreciably changing
            current_flux = 0.0
            proceed = True

            # If the algorithm has converged or within acceptable tolerance, quit out of this loop
            while not deconvolvertool.hasConverged() and proceed:

                t0 = time.time()
                deconvolvertool.runMinorCycle()
                t1 = time.time()
                casaStuff.casalog.post("***Time for minor cycle: " + "%.2f" % (t1 - t0) + " sec", "INFO3",
                                       "task_sdintimaging")

                # sdint specific feathering steps HERE
                # Prepare the joint model cube for INT and SD major cycles
                if specmode == 'mfs':
                    # Convert Taylor model coefficients into a model cube : int_cube.model
                    sdintlib.taylor_model_to_cube(cubename=int_cube,
                                                  mtname=joint_multiterm,
                                                  nterms=nterms,
                                                  reffreq=inpparams['reffreq'])
                else:
                    # Copy the joint_model cube to the int_cube.model
                    shutil.rmtree(int_cube + '.model', ignore_errors=True)
                    shutil.copytree(joint_cube + '.model', int_cube + '.model')
                    hasfile = os.path.exists(joint_cube + '.model')

                if applypb:
                    # Take the int_cube.model to flat sky.
                    if specmode == 'cube':
                        # Divide the model by the frequency-dependent PB to get to flat-sky
                        fdep_pb = True
                    else:
                        # Divide the model by the common PB to get to flat-sky
                        fdep_pb = False
                    sdintlib.modify_with_pb(inpcube=int_cube + '.model',
                                            pbcube=int_cube + '.pb',
                                            cubewt=int_cube + '.sumwt',
                                            chanwt=inpparams['chanwt'],
                                            action='div',
                                            pblimit=pblimit,
                                            freqdep=fdep_pb)

                if usedata != "int":
                    # Copy the int_cube.model to the sd_cube.model
                    shutil.rmtree(sd_cube + '.model', ignore_errors=True)
                    shutil.copytree(int_cube + '.model', sd_cube + '.model')

                if applypb:
                    # Multiply flat-sky model with freq-dep PB
                    sdintlib.modify_with_pb(inpcube=int_cube + '.model',
                                            pbcube=int_cube + '.pb',
                                            cubewt=int_cube + '.sumwt',
                                            chanwt=inpparams['chanwt'],
                                            action='mult',
                                            pblimit=pblimit,
                                            freqdep=True)

                # Major cycle for interferometer data
                t0 = time.time()
                if usedata != "sd":
                    imager.runMajorCycle()
                t1 = time.time()
                casaStuff.casalog.post("***Time for major cycle: " + "%.2f" % (t1 - t0) + " sec", "INFO3",
                                       "task_tclean")

                if usedata != "int":
                    # Major cycle for Single Dish data (uses the flat sky cube model in sd_cube.model)
                    sdintlib.calc_sd_residual(origcube=sd_cube + '.image',
                                              modelcube=sd_cube + '.model',
                                              residualcube=sd_cube + '.residual',
                                              psfcube=sd_cube + '.psf')

                # Feather the residuals
                feather_residual(int_cube, sd_cube, joint_cube, applypb, inpparams)

                if specmode == 'mfs':
                    # Calculate Spectral Taylor Residuals
                    sdintlib.cube_to_taylor_sum(cubename=joint_cube + '.residual',
                                                cubewt=int_cube + '.sumwt',
                                                chanwt=inpparams['chanwt'],
                                                mtname=joint_multiterm + '.residual',
                                                nterms=nterms, reffreq=inpparams['reffreq'], dopsf=False)

                deconvolvertool.updateMask()

                # Get summary from iterbot
                retrec = deconvolvertool.getSummary(fullsummary);
                retrec['nmajordone'] = imager.majorCnt
                if calcres == True:
                    retrec['nmajordone'] = retrec['nmajordone'] + 1  ## To be consistent with tclean. Remove, when we can change the meaning of nmajordone to exclude the initial major cycles.

                # If we have a convergence_fracflux set, check here and break out if the flux change is below threshold
                if convergence_fracflux is not None:

                    model_stats = cmr.stat_cube(joint_cube + '.model')

                    previous_flux = current_flux
                    current_flux = model_stats['sum'][0]
                    delta_flux = abs(current_flux - previous_flux)
                    frac_delta_flux = delta_flux / previous_flux

                    if frac_delta_flux < convergence_fracflux:
                        proceed = False

            # Restore images.
            if restoration:
                t0 = time.time()
                deconvolvertool.restoreImages()
                t1 = time.time()
                casaStuff.casalog.post("***Time for restoring images: " + "%.2f" % (t1 - t0) + " sec", "INFO3",
                                       "task_tclean")
                if pbcor:
                    t0 = time.time()
                    if specmode == 'mfs':
                        sdintlib.pbcor(imagename=decname + '.image.tt0', pbimage=decname + '.pb.tt0', cutoff=pblimit,
                                       outfile=decname + '.image.tt0.pbcor')
                    else:
                        sdintlib.pbcor(imagename=joint_cube + '.image', pbimage=int_cube + '.pb', cutoff=pblimit,
                                       outfile=joint_cube + '.image.pbcor')
                    t1 = time.time()
                    casaStuff.casalog.post("***Time for pb-correcting images: " + "%.2f" % (t1 - t0) + " sec", "INFO3",
                                           "task_tclean")

        # Close tools. Needs to deletetools before concat or lock waits forever
        imager.deleteTools()
        deconvolvertool.deleteTools()

    finally:
        if imager is not None:
            imager.deleteTools()
        if cppparallel:
            # Release workers back to python mpi control
            si = casaStuff.synthesisimager()
            si.releasempi()

        # Clean up tmp files
        delete_tmp_files()

    # Write history at the end, when hopefully all temp files are gone from disk,
    # so they won't be picked up. They need time to disappear on NFS or slow hw.
    # Copied from tclean.
    try:
        from casatasks.private.cleanhelper import write_tclean_history, get_func_params
        params = get_func_params(sdintimaging, locals())
        write_tclean_history(imagename, 'sdintimaging', params, casaStuff.casalog)
    except Exception as exc:
        casaStuff.casalog.post("Error updating history (logtable): {} ".format(exc), 'WARN')

    return retrec
