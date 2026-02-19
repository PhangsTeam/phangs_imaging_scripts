"""
Standalone routines that take input and output files and manipulate
cubes. These are called as part of the PHANGS post-processing pipeline
but also may be of general utility.
"""

import logging
import os

import analysisUtils as au
import numpy as np
import scipy.ndimage as nd
from astropy.io import fits

from . import casaStuff
from . import __version__

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#region Check getchunk putchunk memory issue

def check_getchunk_putchunk_memory_issue(
    infile,
    myia=None,
    return_myia=False,
    return_data=False,
    return_mask=False,
    return_shape=False,
    huge_cube_workaround=True,
    ):
    """
    Check whether the input data cube is too large to run getchunk/putchunk globally.
    If so, we will need to run getchunk/putchunk channel-by-channel.

    We will first check the cube size. If it is obviously too large, then we will directly
    mark this case as having a memory issue. Otherwise we will try to run `getchunk` and/or
    `getchunk(getmask=True)` to see whether it returns a cube data array. In the case of
    having a memory issue, `getchunk` will return a flattend array whose shape is different
    from the input data cube.

    This memory issue is also noted in https://casa.nrao.edu/docs/casaref/image.putchunk.html
    as: "If all the pixels didn't easily fit in memory, you would iterate through the image
    chunk by chunk to avoid exhausting virtual memory."

    Args:
        infile: input image file.
        myia: optional, if not given then we will run `myia = au.createCasaTool(casaStuff.iatool)` then `myia.open(infile)`.
        return_data: if True then we will return the data if `getchunk()` runs properly withut a memory issue.
        return_mask: if True then we will return the mask if `getchunk(getmask=True)` runs properly withut a memory issue.
        return_shape: if True then we will return the shape of the cube data.
        huge_cube_workaround: historical arg for compatibility. Setting it to False will disable per-channel processing and may cause error.

    Returns:
        Tuple of 1-5 variables:
            check_no_memory_issue (boolean),
            myia object (if return_myia is True),
            data array (if return_data is True),
            mask array (if return_mask is True),
            shape list (if return_shape is True)
    """
    has_memory_issue = False
    cube_data = None
    cube_mask = None
    has_opened_file = False
    if myia is None:
        myia = au.createCasaTool(casaStuff.iatool)
        myia.open(infile)
        has_opened_file = True
    if not myia.isopen():
        myia.open(infile)
        has_opened_file = True
    cube_shape = myia.shape() # [X, Y, CHANNEL, [STOKES]]
    if np.prod(cube_shape) >= 2880*2880*393: # known memory issue
        has_memory_issue = True
    if not has_memory_issue: # try to see if there is a memory issue
        # in which case the getchunk will return a flat array instead of the cube shape
        if return_data or not return_mask:
            cube_data = myia.getchunk() # data.shape = [X, Y, CHANNEL, [STOKES]]
            check_shape = cube_data.shape
        if return_mask:
            cube_mask = myia.getchunk(getmask=True) # mask.shape = [X, Y, CHANNEL, [STOKES]]
            check_shape = cube_mask.shape
        if not np.all(check_shape == cube_shape): # shape does not match, has memory issue
            has_memory_issue = True
            cube_data = None
            cube_mask = None

    if not huge_cube_workaround: # if workaround is not allowed, go the simplest way
        has_memory_issue = False

    #has_memory_issue = True #<DEBUG><DZLIU># uncomment this to always apply per-channel getchunk/putchunk

    list_to_return = [has_memory_issue]
    if return_myia:
        list_to_return.append(myia)
    elif has_opened_file:
        myia.close()
    if return_data:
        list_to_return.append(cube_data)
    if return_mask:
        list_to_return.append(cube_mask)
    if return_shape:
        list_to_return.append(cube_shape)
    return tuple(list_to_return)

#endregion

#region Copying, scaling, etc.

def copy_dropdeg(
    infile=None,
    outfile=None,
    overwrite=False
    ):
    """
    Copy using imsubimage to drop degenerate axes. Optionally handle
    overwriting and importing from FITS.
    """

    if os.path.isdir(outfile):
        if not overwrite:
            logger.error("Output exists and overwrite set to false - "+outfile)
            return(False)
        os.system('rm -rf '+outfile)

    used_temp_outfile = False
    if (infile[-4:] == 'FITS') and os.path.isfile(infile):
        logger.info("Importing from FITS.")
        temp_outfile = outfile+'.temp'
        if os.path.isdir(temp_outfile):
            if not overwrite:
                logger.error("Temp file exists and overwrite set to false - "+temp_outfile)
                return(False)
            os.system('rm -rf '+temp_outfile)

        casaStuff.importfits(fitsimage=infile,
                   imagename=temp_outfile,
                   zeroblanks=False,
                   overwrite=overwrite)
        used_temp_outfile = True

    if used_temp_outfile:
        casaStuff.imsubimage(
            imagename=temp_outfile,
            outfile=outfile,
            dropdeg=True)
        os.system('rm -rf '+temp_outfile)
    else:
        casaStuff.imsubimage(
            imagename=infile,
            outfile=outfile,
            dropdeg=True)

    return(True)


def get_mask(infile, huge_cube_workaround=True):
    """
    Get a mask from a CASA image file. Includes a switch for large cubes, where getchunk can segfault.
    """

    #if huge_cube_workaround:
    #    os.system('rm -rf ' + infile + '.temp_deg_ordered')
    #    casaStuff.imtrans(imagename=infile + '.temp_deg', outfile=infile + '.temp_deg_ordered',
    #                      order='0132')
    #
    #    casaStuff.makemask(mode='copy', inpimage=infile + '.temp_deg_ordered',
    #                       inpmask=infile + '.temp_deg_ordered:mask0',
    #                       output=infile + '.temp_mask', overwrite=True)
    #
    #    casaStuff.exportfits(imagename=infile + '.temp_mask',
    #                         fitsimage=infile + '.temp.fits',
    #                         stokeslast=False, overwrite=True)
    #    hdu = fits.open(infile + '.temp.fits')[0]
    #    mask = hdu.data.T[:, :, 0, :]
    #
    #    os.system('rm -rf ' + infile + '.temp_deg_ordered')
    #    os.system('rm -rf ' + infile + '.temp_mask')
    #    os.system('rm -rf ' + infile + '.temp.fits')
    #
    #else:
    #    myia = au.createCasaTool(casaStuff.iatool)
    #    myia.open(infile+'.temp_deg')
    #    mask = myia.getchunk(getmask=True)
    #    myia.close()
    #
    #os.system('rm -rf ' + infile + '.temp_deg')

    myia = au.createCasaTool(casaStuff.iatool)

    myia.open(infile)

    has_memory_issue, mask, cube_shape = check_getchunk_putchunk_memory_issue(
        infile, myia=myia, return_mask=True, return_shape=True,
        huge_cube_workaround=huge_cube_workaround)

    if has_memory_issue: # getchunk was unsuccessful, has memory issue
        # putchunk channel by channel
        logger.debug('getchunk channel by channel for known memory issue')
        mask = np.full(cube_shape, fill_value=False, dtype=bool)
        if len(cube_shape) == 2:
            blc = [0, 0] # [X, Y]
            trc = [-1, -1] # [X, Y]
            cube_mask_per_chan = myia.getchunk(blc, trc, getmask=True)
            mask[:, :] = cube_mask_per_chan
        elif len(cube_shape) == 3:
            nchan = cube_shape[2]
            for ichan in range(nchan):
                blc = [0, 0, ichan] # [X, Y, CHANNEL]
                trc = [-1, -1, ichan] # [X, Y, CHANNEL]
                cube_mask_per_chan = myia.getchunk(blc, trc, getmask=True)
                mask[:, :, ichan] = cube_mask_per_chan[:, :, 0]
        elif len(cube_shape) == 4:
            nstokes = cube_shape[3]
            nchan = cube_shape[2] # It's okay if Stokes and Spectral axes are swapped.
            for istokes in range(nstokes):
                for ichan in range(nchan):
                    blc = [0, 0, ichan, istokes] # [X, Y, CHANNEL, STOKES]
                    trc = [-1, -1, ichan, istokes] # [X, Y, CHANNEL, STOKES]
                    #logger.debug('get_mask blc {} trc {}'.format(blc, trc))
                    cube_mask_per_chan = myia.getchunk(blc, trc, getmask=True)
                    mask[:, :, ichan, istokes] = cube_mask_per_chan[:, :, 0, 0]
        else:
            raise Exception('Could not proceed with cube dimension ' + str(len(cube_shape)))

    myia.close()

    assert np.all(mask.shape == cube_shape)

    return mask


def copy_mask(infile, outfile, huge_cube_workaround=True):
    """
    Copy a mask from infile to outfile. Includes a switch for large cubes, where getchunk/putchunk can segfault
    """

    #if huge_cube_workaround:
    #    os.system('rm -rf ' + outfile + '/mask0')
    #    os.system('cp -r ' + infile + '/mask0' + ' ' + outfile + '/mask0')
    #else:
    #    myia = au.createCasaTool(casaStuff.iatool)
    #    myia.open(infile)
    #    mask = myia.getchunk(getmask=True)
    #    myia.close()
    #
    #    myia = au.createCasaTool(casaStuff.iatool)
    #    myia.open(outfile)
    #    mask = myia.putregion(pixelmask=mask)
    #    myia.close()

    mask = get_mask(infile)

    # use putregion to update pixel mask

    myia = au.createCasaTool(casaStuff.iatool)

    myia.open(outfile)

    out_shape = myia.shape() # [X, Y, CHANNEL, [STOKES]]

    # input and output shape must be the same
    if not np.all(mask.shape == out_shape):
        myia.close()
        raise Exception('Error! The infile and outfile have different dimensions! Cannot copy mask.')

    has_memory_issue = check_getchunk_putchunk_memory_issue(
        outfile, myia=myia,
        huge_cube_workaround=huge_cube_workaround)

    if not has_memory_issue: # getchunk was successful, no memory issue
        myia.putregion(pixelmask=mask)
    else: # getchunk was unsuccessful, has memory issue
        # putregion channel by channel
        logger.debug('putregion channel by channel for known memory issue')
        myrg = au.createCasaTool(casaStuff.rgtool)
        nx = out_shape[0]
        ny = out_shape[1]
        #logger.debug('copy_mask mask.shape {}'.format(mask.shape))
        if len(out_shape) == 2:
            blc = [0, 0]
            trc = [nx-1, ny-1]
            r1 = myrg.box(blc, trc)
            #logger.debug('copy_mask putregion blc {} trc {}'.format(blc, trc))
            myia.putregion(pixelmask=mask, region=r1)
        elif len(out_shape) == 3:
            specaxis = 2
            nchan = out_shape[specaxis]
            blc = [0, 0, 0]
            trc = [nx-1, ny-1, 0]
            for ichan in range(nchan):
                blc[specaxis] = ichan
                trc[specaxis] = ichan
                r1 = myrg.box(blc, trc)
                #logger.debug('copy_mask putregion blc {} trc {}'.format(blc, trc))
                myia.putregion(pixelmask=np.take(mask, ichan, axis=specaxis), region=r1)
        elif len(out_shape) == 4:
            mycs = myia.coordsys()
            specaxis = mycs.axiscoordinatetypes().index('Spectral')
            stokesaxis = mycs.axiscoordinatetypes().index('Stokes')
            nchan = out_shape[specaxis]
            nstokes = out_shape[stokesaxis]
            blc = [0, 0, 0, 0]
            trc = [nx-1, ny-1, 0, 0]
            for istokes in range(nstokes):
                for ichan in range(nchan):
                    blc[specaxis] = ichan
                    trc[specaxis] = ichan
                    blc[stokesaxis] = istokes
                    trc[stokesaxis] = istokes
                    r1 = myrg.box(blc, trc)
                    #logger.debug('copy_mask putregion blc {} trc {}'.format(blc, trc))
                    myia.putregion(pixelmask=np.take(np.take(mask, blc[3], axis=3), blc[2], axis=2), region=r1)
        else:
            raise Exception('Could not proceed with cube dimension ' + str(len(out_shape)))
        myrg.done()
    myia.close()

    return True


def multiply_cube_by_value(infile, value, brightness_unit, huge_cube_workaround=True):
    """
    Multiply a cube by some value, and update the brightness unit accordingly. Includes a switch for large cubes, where
    getchunk/putchunk may fail.
    """

    #if huge_cube_workaround:
    #    casaStuff.exportfits(imagename=infile,
    #                         fitsimage=infile + '.fits',
    #                         overwrite=True)
    #    hdu = fits.open(infile + '.fits')[0]
    #    hdu.data *= value
    #
    #    hdu.writeto(infile + '.fits', overwrite=True)
    #    casaStuff.importfits(fitsimage=infile + '.fits',
    #                         imagename=infile,
    #                         overwrite=True)
    #    os.system('rm -rf ' + infile + '.fits')
    #else:
    #    myia = au.createCasaTool(casaStuff.iatool)
    #    myia.open(infile)
    #    vals = myia.getchunk()
    #    vals *= value
    #    myia.putchunk(vals)

    myia = au.createCasaTool(casaStuff.iatool)

    myia.open(infile)

    has_memory_issue, vals, cube_shape = check_getchunk_putchunk_memory_issue(
        infile, myia=myia, return_data=True, return_shape=True,
        huge_cube_workaround=huge_cube_workaround)

    if not has_memory_issue: # getchunk was successful, no memory issue
        vals *= value
        myia.putchunk(vals)
    else: # getchunk was unsuccessful, has memory issue
        # putchunk channel by channel
        logger.debug('putchunk channel by channel for known memory issue')
        if len(cube_shape) == 2:
            blc = [0, 0] # [X, Y]
            trc = [-1, -1] # [X, Y]
            vals = myia.getchunk(blc, trc)
            vals *= value
            myia.putchunk(vals, blc)
        elif len(cube_shape) == 3:
            nchan = cube_shape[2]
            for ichan in range(nchan):
                blc = [0, 0, ichan] # [X, Y, CHANNEL]
                trc = [-1, -1, ichan] # [X, Y, CHANNEL]
                vals = myia.getchunk(blc, trc)
                vals *= value
                myia.putchunk(vals, blc)
        elif len(cube_shape) == 4:
            nstokes = cube_shape[3]
            nchan = cube_shape[2] # It's okay if Stokes and Spectral axes are swapped.
            for istokes in range(nstokes):
                for ichan in range(nchan):
                    blc = [0, 0, ichan, istokes] # [X, Y, CHANNEL, STOKES]
                    trc = [-1, -1, ichan, istokes] # [X, Y, CHANNEL, STOKES]
                    vals = myia.getchunk(blc, trc)
                    vals *= value
                    myia.putchunk(vals, blc)
        else:
            raise Exception('Could not proceed with cube dimension ' + str(len(cube_shape)))

    if brightness_unit is not None:
        myia.setbrightnessunit(brightness_unit)

    myia.close()

    return True


def export_and_cleanup(
    infile=None,
    outfile=None,
    overwrite=False,
    remove_cards=[],
    add_cards={},
    add_history=[],
    zap_history=True,
    round_beam=True,
    roundbeam_tol=0.01
    ):
    """
    Export from a CASA image file to a FITS file, in the process
    cleaning up header keywords that are usually confusing or
    useless. Optionally add new keywords to the header and check
    whether the beam is close enough to being round that it makes
    sense to overwrite it.
    """

    if infile is None or outfile is None:
        logger.error("Missing required input.")
        return(False)

    if os.path.isdir(infile) == False:
        logger.error("Input file does not exist - "+infile)
        return(False)

    if os.path.isfile(outfile):
        if not overwrite:
            logger.error("Output exists and overwrite set to false - "+outfile)
            return(False)

    casaStuff.exportfits(imagename=infile,
                    fitsimage=outfile,
                    velocity=True,
                    overwrite=True,
                    dropstokes=True,
                    dropdeg=True,
                    bitpix=-32)

    # Clean up headers

    hdu = fits.open(outfile)

    hdr = hdu[0].header
    data = hdu[0].data

    # Cards to remove by default

    for card in ['BLANK','DATE-OBS','OBSERVER','O_BLANK','O_BSCALE',
                 'O_BZERO','OBSRA','OBSDEC','OBSGEO-X','OBSGEO-Y','OBSGEO-Z',
                 'DISTANCE']:
        if card in hdr.keys():
            hdr.remove(card)

    # User cards to remove

    for card in remove_cards:
        if card in hdr.keys():
            hdr.remove(card)

    # Delete history

    if zap_history:
        while 'HISTORY' in hdr.keys():
            hdr.remove('HISTORY')

    # Add history

    for history_line in add_history:
        add_history(history_line)

    for card in add_cards:
        hdr[card] = add_cards[card]

    # Get the data min and max right

    datamax = np.nanmax(data)
    datamin = np.nanmin(data)
    hdr['DATAMAX'] = datamax
    hdr['DATAMIN'] = datamin


    # Round the beam recorded in the header if it lies within the
    # specified tolerance.

    if round_beam:
        if roundbeam_tol > 0.0:
            bmaj = hdr['BMAJ']
            bmin = hdr['BMIN']
            if bmaj != bmin:
                frac_dev = np.abs(bmaj-bmin)/bmaj
                if frac_dev <= roundbeam_tol:
                    logger.info("Rounding beam.")
                    hdr['BMAJ'] = bmaj
                    hdr['BMIN'] = bmaj
                    hdr['BPA'] = 0.0
                else:
                    logger.info("Beam too asymmetric to round.")
                    logger.info("... fractional deviation: "+str(frac_dev))

    # Never forget where you came from
    hdr['COMMENT'] = 'Produced with PHANGS-ALMA pipeline version ' + __version__

    hdu.writeto(outfile, overwrite=True)

    return()

def trim_cube(
        infile=None,
        outfile=None,
        overwrite=False,
        inplace=False,
        min_pixperbeam=3,
        pad=1,
        rebin=True,
    ):
    """
    Trim empty space from around the edge of a cube. Also rebin the
    cube to smaller size, while ensuring a minimum number of pixels
    across the beam. Used to reduce the volume of cubes.
    """

    if infile is None or outfile is None:
        logger.error("Missing required input.")
        return(False)

    if os.path.isdir(infile) == False:
        logger.error("Input file not found: "+infile)
        return(False)

    if os.path.exists(outfile) and not overwrite:
        logger.info(f"outfile {outfile} already exists and overwrite is False. Skipping")
        return True

    os.system('rm -rf ' + outfile)

    hdr = casaStuff.imhead(infile)
    axis_names = hdr["axisnames"]

    # Get indices for RA/Dec/Frequency axes
    ra_axis = np.where(axis_names == "Right Ascension")[0][0]
    dec_axis = np.where(axis_names == "Declination")[0][0]
    freq_axis = np.where(axis_names == "Frequency")[0][0]

    # First, rebin if needed
    just_copy = True
    if rebin:

        if (hdr['axisunits'][0] != 'rad'):
            logger.error(
                "ERROR: Based on CASA experience. I expected units of radians. I did not find this. Returning.")
            logger.error("Adjust code or investigate file " + infile)
            return (False)

        pixel_as = abs(hdr['incr'][0] / np.pi * 180.0 * 3600.)

        if (hdr['restoringbeam']['major']['unit'] != 'arcsec'):
            logger.error("ERROR: Based on CASA experience. I expected units of arcseconds for the beam. I did not find this. Returning.")
            logger.error("Adjust code or investigate file "+infile)
            return(False)
        bmaj = hdr['restoringbeam']['major']['value']

        pix_per_beam = bmaj*1.0 / pixel_as*1.0

        if pix_per_beam > 6:
            casaStuff.imrebin(
                imagename=infile,
                outfile=outfile+'.temp',
                factor=[2,2,1],
                crop=True,
                dropdeg=True,
                overwrite=overwrite,
                )
            just_copy = False

    if just_copy:
        os.system('cp -r '+infile+' '+outfile+'.temp')

    outfile_ext = ".temp"

    # If we don't have the Stokes I axis, add this in
    if "Stokes" not in hdr["axisnames"]:
        myia = au.createCasaTool(casaStuff.iatool)
        myia.open(outfile + '.temp')
        os.system('rm -rf '+outfile + '.temp_deg')
        deg_im = myia.adddegaxes(outfile=outfile + '.temp_deg', stokes='I', overwrite=True)
        deg_im.done()
        myia.close()
        outfile_ext = ".temp_deg"

    # This should either be .temp or .temp_deg (check if it works for 2d cases)
    mask = get_mask(outfile + outfile_ext, huge_cube_workaround=True)

    # Figure out the extent of the image inside the cube
    mask_spec_x = np.any(mask, axis=tuple([i for i, x in enumerate(list(mask.shape)) if i != ra_axis]))
    xmin = np.max([0,np.min(np.where(mask_spec_x))-pad])
    xmax = np.min([np.max(np.where(mask_spec_x))+pad,mask.shape[ra_axis]-1])

    #mask_spec_y = np.sum(np.sum(mask*1.0,axis=2),axis=0) > 0
    mask_spec_y = np.any(mask, axis=tuple([i for i, x in enumerate(list(mask.shape)) if i != dec_axis]))
    ymin = np.max([0,np.min(np.where(mask_spec_y))-pad])
    ymax = np.min([np.max(np.where(mask_spec_y))+pad,mask.shape[dec_axis]-1])

    #mask_spec_z = np.sum(np.sum(mask*1.0,axis=0),axis=0) > 0
    mask_spec_z = np.any(mask, axis=tuple([i for i, x in enumerate(list(mask.shape)) if i != freq_axis]))
    zmin = np.max([0,np.min(np.where(mask_spec_z))-pad])
    zmax = np.min([np.max(np.where(mask_spec_z))+pad,mask.shape[freq_axis]-1])

    box_string = ''+str(xmin)+','+str(ymin)+','+str(xmax)+','+str(ymax)
    chan_string = ''+str(zmin)+'~'+str(zmax)

    logger.info("... box selection: "+box_string)
    logger.info("... channel selection: "+chan_string)

    casaStuff.imsubimage(
        imagename=outfile+'.temp',
        outfile=outfile,
        box=box_string,
        chans=chan_string,
        )

    os.system('rm -rf '+outfile+'.temp')
    os.system('rm -rf '+outfile+'.temp_deg')

    return(True)

def trim_rind(
        infile=None,
        outfile=None,
        overwrite=False,
        inplace=True,
        pixels=1
        ):

    if infile is None or outfile is None:
        logger.error("Missing required input.")
        return(False)

    if os.path.isdir(infile) == False:
        logger.error("Input file not found: "+infile)
        return(False)

    if inplace == False:
        if os.path.isdir(outfile) or os.path.isfile(outfile):
            if overwrite:
                os.system('rm -rf '+outfile)
            else:
                logger.error("Output file already present: "+outfile)
                return(False)

        os.system('cp -r '+infile+' '+outfile)
        target_file = outfile
    else:
        target_file = infile

    # Figure out the extent of the image inside the cube
    myia = au.createCasaTool(casaStuff.iatool)
    myia.open(target_file)
    mask = myia.getchunk(getmask=True)
    elt = nd.generate_binary_structure(2,1)
    if pixels > 1:
        elt = nd.iterate_structure(elt, pixels-1)
    mask = nd.binary_erosion(mask, elt[:,:,np.newaxis, np.newaxis])
    myia.putregion(pixelmask=mask)
    myia.close()
    return(True)


def trim_coarse_beam_edge_channels(
        infile=None,
        outfile=None,
        inpbfile=None,
        outpbfile=None,
        overwrite=False,
        inplace=True,
    ):
    """Trim the edge channels which have significantly coarser beam sizes in a per-plane beam image cube.
    """

    if infile is None:
        logger.error("Missing required input.")
        return(False)

    if os.path.isdir(infile) == False:
        logger.error("Input file not found: "+infile)
        return(False)

    if inplace == False:
        if os.path.exists(outfile):
            if overwrite:
                os.system('rm -rf '+outfile)
            else:
                logger.warning("Output file already present: "+outfile)
                return(True)

    # get image header
    image_header = casaStuff.imhead(infile)

    # if there is no per-plane beam, return True
    if 'perplanebeams' not in image_header:
        return(True)

    # get per-plane beam array
    nchan = image_header['shape'][-1]
    beam_array = np.array([image_header['perplanebeams']['beams']['*%d'%(i)]['*0']['major']['value'] for i in range(nchan)])

    # get beam threshold value by 3-sigma clipping
    beam_thresh = np.median(beam_array) + 3.0 * np.std(beam_array)
    beam_array2 = beam_array[beam_array<beam_thresh]
    beam_thresh2 = np.median(beam_array2) + 3.0 * np.std(beam_array2)

    # get left and right boundarys
    chan_valid = np.argwhere(beam_array<beam_thresh2).ravel()
    chan_left = chan_valid[0]
    chan_right = chan_valid[-1]

    # run imsubimage
    if chan_left > 0 or chan_right < nchan-1:
        if inplace:
            if os.path.isdir(infile+'.trim.coarse.beam.edge.channels.tmp'):
                os.system('rm -rf '+infile+'.trim.coarse.beam.edge.channels.tmp')
            os.system('mv '+infile+' '+infile+'.trim.coarse.beam.edge.channels.tmp')
            target_infile = infile+'.trim.coarse.beam.edge.channels.tmp'
            target_outfile = infile
        else:
            target_infile = infile
            target_outfile = outfile

        casaStuff.imsubimage(imagename=target_infile, outfile=target_outfile, chans="%d~%d"%(chan_left, chan_right))

        if inplace:
            if os.path.isdir(infile+'.trim.coarse.beam.edge.channels.tmp'):
                os.system('rm -rf '+infile+'.trim.coarse.beam.edge.channels.tmp')

        # also process pbfile
        if inplace:
            if os.path.isdir(inpbfile+'.trim.coarse.beam.edge.channels.tmp'):
                os.system('rm -rf '+inpbfile+'.trim.coarse.beam.edge.channels.tmp')
            os.system('mv '+inpbfile+' '+inpbfile+'.trim.coarse.beam.edge.channels.tmp')
            target_infile = inpbfile+'.trim.coarse.beam.edge.channels.tmp'
            target_outfile = inpbfile
        else:
            target_infile = inpbfile
            target_outfile = outpbfile

        casaStuff.imsubimage(imagename=target_infile, outfile=target_outfile, chans="%d~%d"%(chan_left, chan_right))

        if inplace:
            if os.path.isdir(inpbfile+'.trim.coarse.beam.edge.channels.tmp'):
                os.system('rm -rf '+inpbfile+'.trim.coarse.beam.edge.channels.tmp')

    return(True)

def primary_beam_correct(
    infile=None,
    pbfile=None,
    outfile=None,
    cutoff=0.25,
    overwrite=False
    ):
    """
    Construct a primary-beam corrected image.
    """

    if infile is None or pbfile is None or outfile is None:
        logger.error("Missing required input.")
        return(False)

    if os.path.isdir(infile) == False:
        logger.error("Input file missing - "+infile)
        return(False)

    if os.path.isdir(pbfile) == False:
        logger.error("Primary beam file missing - "+pbfile)
        return(False)

    if os.path.isfile(outfile) or os.path.isdir(outfile):
        if overwrite:
            os.system('rm -rf '+outfile)
        else:
            logger.error("Output exists and overwrite set to false - "+outfile)
            return(False)

    casaStuff.impbcor(imagename=infile, pbimage=pbfile, outfile=outfile, cutoff=cutoff)

    return(True)

#endregion

#region Convolution and alignment

def align_to_target(
    infile=None,
    outfile=None,
    template=None,
    interpolation='cubic',
    asvelocity=True,
    overwrite=False,
    axes=[-1]
    ):
    """
    Align one cube to another, creating a copy. Right now a thin
    wrapper to imregrid used mostly to avoid exposing CASA directly
    into the postprocessHandler. Might evolve in the future.
    """

    if infile is None or template is None or outfile is None:
        logger.error("Missing required input.")
        return(False)

    if os.path.isdir(infile) == False and os.path.isfile(infile) == False:
        logger.error("Input file missing - "+infile)
        return(False)

    if os.path.isdir(template) == False and os.path.isfile(template) == False:
        logger.error("Template file missing - "+template)
        return(False)

    if os.path.isfile(outfile) or os.path.isdir(outfile):
        if overwrite:
            os.system('rm -rf '+outfile)
        else:
            logger.error("Output exists and overwrite set to false - "+outfile)
            return(False)

    # If we have less than 50 (output) pixels along an axis to be regridded, we need to
    # decrease the decimation factor
    decimate = 10
    input_hdr = casaStuff.imhead(infile)
    output_hdr = casaStuff.imhead(template)
    input_shape = input_hdr['shape']
    output_shape = output_hdr['shape']
    regrid_axes = np.where(input_shape != output_shape)[0]
    if np.any(output_shape[regrid_axes] < 50):
        decimate = 1

    casaStuff.imregrid(
        imagename=infile,
        template=template,
        output=outfile,
        interpolation=interpolation,
        decimate=decimate,
        asvelocity=asvelocity,
        axes=axes,
        overwrite=True)

    return(True)

def convolve_to_round_beam(
    infile=None,
    outfile=None,
    force_beam=None,
    overwrite=False
    ):
    """
    Convolve supplied image to have a round beam. Optionally, force
    that beam to some size, else it figures out the beam.
    """

    if infile is None or outfile is None:
        logger.error("Missing required input.")
        return(None)

    if os.path.isdir(infile) == False:
        logger.error("Input file missing - "+infile)
        return(None)

    hdr = casaStuff.imhead(infile)

    if (hdr['axisunits'][0] != 'rad'):
        logger.error("Based on CASA experience. I expected units of radians. I did not find this.")
        logger.error("Adjust code or investigate file "+infile)
        return(None)
    pixel_as = abs(hdr['incr'][0]/np.pi*180.0*3600.)

    if 'perplanebeams' in hdr.keys():
        beamnames = hdr['perplanebeams']['beams'].keys()
        majorlist = [hdr['perplanebeams']['beams'][b]['*0']['major']['value']
                     for b in beamnames]
        bmaj = np.max(majorlist)
    else:
        if (hdr['restoringbeam']['major']['unit'] != 'arcsec'):
            logger.error("Based on CASA experience. I expected units of arcseconds for the beam. I did not find this.")
            logger.error("Adjust code or investigate file "+infile)
            return(None)
        bmaj = hdr['restoringbeam']['major']['value']

    if force_beam is None:
        target_bmaj = np.sqrt((bmaj)**2+(2.0*pixel_as)**2)
    else:
        min_bmaj = np.sqrt((bmaj)**2+(2.0*pixel_as)**2)
        if force_beam < min_bmaj:
            logger.warning("Requested beam is too small for convolution.")
            return(None)
        target_bmaj = force_beam

    casaStuff.imsmooth(imagename=infile,
                       outfile=outfile,
                       targetres=True,
                       major=str(target_bmaj) + 'arcsec',
                       minor=str(target_bmaj) + 'arcsec',
                       pa='0.0deg',
                       overwrite=overwrite
                       )

    # Copy over mask
    copy_mask(infile, outfile)

    return(target_bmaj)

#endregion

#region Units stuff

def calc_jytok(
    hdr=None,
    infile=None
    ):
    """
    Calculate the Jy/beam -> Kelvin conversion. Accepts a header
    already read using imhead or a file name that it will parse.
    """

    c = 2.99792458e10
    kb = 1.380658e-16

    if hdr is None:
        if infile is None:
            logger.error("No header and no infile. Returning.")
            return(None)
        hdr = casaStuff.imhead(infile, mode='list')

    for ii in range(len(hdr['shape'])):
        if hdr['cunit{}'.format(ii+1)] == 'Hz':
            break
        if ii == len(hdr['shape'])-1:
            logger.error("I expected a frequency axis but did not find it. Returning.")
            return(None)
    crpix = hdr['crpix{}'.format(ii+1)]
    cdelt = hdr['cdelt{}'.format(ii+1)]
    crval = hdr['crval{}'.format(ii+1)]
    naxis = hdr['shape'][ii]
    faxis_hz = (np.arange(naxis)+1.-crpix)*cdelt+crval
    freq_hz = np.mean(faxis_hz)

    bmaj_unit = hdr['beammajor']['unit']
    if bmaj_unit != 'arcsec':
        logger.error("Beam unit is not arcsec, which I expected. Returning. Unit instead is "+bmaj_unit)
        return(None)
    bmaj_as = hdr['beammajor']['value']
    bmin_as = hdr['beamminor']['value']
    bmaj_sr = bmaj_as/3600.*np.pi/180.
    bmin_sr = bmin_as/3600.*np.pi/180.
    beam_in_sr = np.pi*(bmaj_sr/2.0*bmin_sr/2.0)/np.log(2)

    jytok = c**2 / beam_in_sr / 1e23 / (2*kb*freq_hz**2)

    return(jytok)

def convert_jytok(
    infile=None,
    outfile=None,
    overwrite=False,
    inplace=False
    ):
    """
    Convert a cube from Jy/beam to K.
    """

    if infile is None or (outfile is None and inplace==False):
        logger.error("Missing required input.")
        return(False)

    if os.path.isdir(infile) == False:
        logger.error("Input file not found: "+infile)
        return(False)

    if inplace == False:
        if os.path.isdir(outfile) or os.path.isfile(outfile):
            if overwrite:
                os.system('rm -rf '+outfile)
            else:
                logger.error("Output file already present: "+outfile)
                return(False)

        os.system('cp -r '+infile+' '+outfile)
        target_file = outfile
    else:
        target_file = infile

    hdr = casaStuff.imhead(target_file, mode='list')
    unit = hdr['bunit']
    if unit != 'Jy/beam':
        logger.error("Input unit is not Jy/beam for file "+target_file+" . Instead found "+unit)
        return(False)

    jytok = calc_jytok(hdr=hdr)

    multiply_cube_by_value(target_file, jytok, brightness_unit='K')

    casaStuff.imhead(target_file, mode='put', hdkey='JYTOK', hdvalue=jytok)

    return(True)

def convert_ktojy(
    infile=None,
    outfile=None,
    overwrite=False,
    inplace=False
    ):
    """
    Convert a cube from K to Jy/beam.
    """

    if infile is None or (outfile is None and inplace==False):
        logger.error("Missing required input.")
        return(False)

    if os.path.isdir(infile) == False:
        logger.error("Input file not found: "+infile)
        return(False)

    if inplace == False:
        if os.path.isdir(outfile) or os.path.isfile(outfile):
            if overwrite:
                os.system('rm -rf '+outfile)
            else:
                logger.error("Output file already present: "+outfile)
                return(False)

        os.system('cp -r '+infile+' '+outfile)
        target_file = outfile
    else:
        target_file = infile

    hdr = casaStuff.imhead(target_file, mode='list')
    unit = hdr['bunit']
    if unit != 'K':
        logger.error("Input unit is not K for file "+target_file+" . Instead found "+unit)
        return(False)

    jytok = calc_jytok(hdr=hdr)

    multiply_cube_by_value(target_file, 1/jytok, 'Jy/beam')

    casaStuff.imhead(target_file, mode='put', hdkey='JYTOK', hdvalue=jytok)

    return(True)

#endregion
