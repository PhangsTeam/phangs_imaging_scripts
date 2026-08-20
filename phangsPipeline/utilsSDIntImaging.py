import logging
import os
import shutil

import astropy.units as u
import numpy as np
from astropy.utils.console import ProgressBar
from casatasks.private.sdint_helper import SDINT_helper
from casatools import image
from spectral_cube import (
    BooleanArrayMask,
    DaskVaryingResolutionSpectralCube,
    SpectralCube,
)
from uvcombine import feather_simple, feather_simple_cube

from .casaCubeRoutines import check_getchunk_putchunk_memory_issue

logger = logging.getLogger(__name__)


def feather_residual(
    int_cube: str,
    sd_cube: str,
    joint_cube: str,
    applypb: bool,
    inparm: dict,
    mysdintlib = None,
):

    if mysdintlib is None:
        mysdintlib = SDINT_helper()

    if applypb:
        ## Take initial INT_dirty image to flat-sky.
        mysdintlib.modify_with_pb(
            inpcube=f"{int_cube}.residual",
            pbcube=f"{int_cube}.pb",
            cubewt=f"{int_cube}.sumwt",
            chanwt=inparm["chanwt"],
            action="div",
            pblimit=inparm["pblimit"],
            freqdep=True,
        )

    ## Feather flat-sky INT dirty image with SD image
    feather_int_sd(
        sdcube=f"{sd_cube}.residual",
        intcube=f"{int_cube}.residual",
        jointcube=f"{joint_cube}.residual",
        sdgain=inparm["sdgain"],
        usedata=inparm["usedata"],
    )

    if applypb:
        if inparm["specmode"].count("cube") > 0:
            ## Multiply the new JOINT dirty image by the frequency-dependent PB.
            fdep_pb = True
        else:
            ## Multiply new JOINT dirty image by a common PB to get the effect of conjbeams.
            fdep_pb = False

        mysdintlib.modify_with_pb(
            inpcube=f"{joint_cube}.residual",
            pbcube=f"{int_cube}.pb",
            cubewt=f"{int_cube}.sumwt",
            chanwt=inparm["chanwt"],
            action="mult",
            pblimit=inparm["pblimit"],
            freqdep=fdep_pb,
        )


def feather_int_sd(
    sdcube: str = "",
    intcube: str = "",
    jointcube: str = "",
    sdgain: float | int = 1.0,
    usedata: str = "sdint",
):
    """
    Run uvcombine to combine the SD and INT Cubes.
    """

    ### Do the feathering.
    if usedata == "sdint":
        # Read in the cubes. Set fill value to 0 for CASA consistency
        sdcube_sc = SpectralCube.read(
            sdcube,
            format="casa",
        )
        sdcube_sc = sdcube_sc.with_fill_value(0.0)
        sdcube_sc.allow_huge_operations = True
        intcube_sc = SpectralCube.read(
            intcube,
            format="casa",
        )
        intcube_sc = intcube_sc.with_fill_value(0.0)
        intcube_sc.allow_huge_operations = True

        # If the cubes don't have units, these are Jy/beam
        sd_unit = sdcube_sc.unit
        int_unit = intcube_sc.unit
        if sd_unit == "":
            sdcube_sc = sdcube_sc._new_cube_with(unit=u.Jy / u.beam)
        if int_unit == "":
            intcube_sc = intcube_sc._new_cube_with(unit=u.Jy / u.beam)

        # Ensure we have matching WCS. This has already been done in previous steps in sdintimaging,
        # so just check on the shape. If they're not the same, then we need to regrid
        if intcube_sc.shape != sdcube_sc.shape:
            logger.info("Reprojecting the SD cube to match the INT cube")
            sdcube_sc = sdcube_sc.spectral_interpolate(intcube_sc.spectral_axis)
            sdcube_sc = sdcube_sc.reproject(intcube_sc.header)

        # If shape is right but WCS doesn't evaluate as equal, then we need to force the WCS in
        else:
            is_wcs_eq = intcube_sc.wcs.wcs.compare(sdcube_sc.wcs.wcs)
            if not is_wcs_eq:
                # We also need to pull out and replace the mask, else it'll complain
                if sdcube_sc._mask is not None:
                    sdmask = BooleanArrayMask(sdcube_sc.get_mask_array(), wcs=intcube_sc.wcs)
                else:
                    sdmask = None

                sdcube_sc = sdcube_sc._new_cube_with(
                    wcs=intcube_sc.wcs,
                    mask=sdmask,
                )

        # Keep the intcube dtype around for consistency
        intcube_dtype = intcube_sc.unmasked_data[0, 0, 0].dtype

        # Check if we have dask varying spectral resolution, since uvcombine cannot handle this
        is_varying_res = any(
            [isinstance(c, DaskVaryingResolutionSpectralCube) for c in [sdcube_sc, intcube_sc]]
        )

        logger.info("Feathering the INT and SD cubes together")

        # If we do have varying resolution, we have to do this channel-by-channel
        chans = len(intcube_sc.spectral_axis)
        if is_varying_res:
            feathered_cube_data = np.zeros(sdcube_sc.shape, dtype=intcube_dtype)

            with ProgressBar(chans) as bar:
                for chan in range(chans):
                    feathered_cube_data[chan] = feather_simple(
                        hires=intcube_sc[chan],
                        lores=sdcube_sc[chan],
                        lowresscalefactor=sdgain,
                    ).real.astype(intcube_dtype)

                    bar.update()

        # Else just do the whole thing at once
        else:
            feathered_cube = feather_simple_cube(
                cube_hi=intcube_sc,
                cube_lo=sdcube_sc,
                allow_huge_operations=True,
                allow_spectral_resample=False,
                allow_lo_reproj=False,
                lowresscalefactor=sdgain,
            )
            feathered_cube_data = feathered_cube.unitless_filled_data[:].astype(intcube_dtype)

        # We need to transpose the axes for CASA and add in the Stokes
        feathered_cube_data = feathered_cube_data.T
        feathered_cube_data = np.expand_dims(feathered_cube_data, axis=2)

        # Set up the CASA images. We need to load the INT and SD images as well to get
        # pixel masks
        intim = image()
        intim.open(intcube)

        sdim = image()
        sdim.open(sdcube)

        if os.path.exists(jointcube):
            shutil.rmtree(jointcube)
        shutil.copytree(intcube, jointcube)

        jointim = image()
        jointim.open(jointcube)
        jointim.set(0.0)  # Initialize this to zero for all planes

        # Check if getchunk/putchunk will have memory issues. If so, we need to do this channel by channel
        has_memory_issue = check_getchunk_putchunk_memory_issue(
            jointcube,
            myia=jointim,
        )

        # We respect dtypes around here
        jointim_dtype = jointim.pixeltype()

        if has_memory_issue:
            logger.info("Writing out feathered cube channel-by-channel due to memory issue")

            nx = feathered_cube_data.shape[0]
            ny = feathered_cube_data.shape[1]

            blc = [0, 0, 0, 0]
            trc = [nx - 1, ny - 1, 0, 0]

            with ProgressBar(chans) as bar:
                for chan in range(chans):
                    # Channel is the last axis of the cube, so update blc/trc
                    blc[-1] = chan
                    trc[-1] = chan

                    # Create mask, apply to channel slice.
                    int_mask = intim.getchunk(blc=blc, trc=trc, getmask=True)
                    sd_mask = sdim.getchunk(blc=blc, trc=trc, getmask=True)

                    # Slice out the channel axis
                    int_mask = int_mask[:, :, :, 0]
                    sd_mask = sd_mask[:, :, :, 0]

                    mask = int_mask & sd_mask

                    feathered_cube_slice = feathered_cube_data[:, :, :, chan]

                    feathered_cube_slice[~mask] = 0.0

                    jointim.putchunk(
                        feathered_cube_slice.astype(jointim_dtype),
                        blc=blc,
                    )

                    bar.update()

        else:
            logger.info("Writing out feathered cube to CASA image")

            # Combine the INT/SD masks
            int_mask = intim.getchunk(getmask=True)
            sd_mask = sdim.getchunk(getmask=True)
            mask = int_mask & sd_mask

            # Make sure everything outside the mask is set to zero, since CASA expects this
            feathered_cube_data[~mask] = 0.0

            # Write this out back into the CASA image
            jointim.putchunk(feathered_cube_data.astype(jointim_dtype))

        # Close out the images, and we're done!
        intim.close()
        sdim.close()
        jointim.close()

    elif usedata == "sd":
        if os.path.exists(jointcube):
            shutil.rmtree(jointcube)
        shutil.copytree(sdcube, jointcube)
    elif usedata == "int":
        if os.path.exists(jointcube):
            shutil.rmtree(jointcube)
        shutil.copytree(intcube, jointcube)
    else:
        raise ValueError("usedata should be one of sdint, sd, or int")
