import glob
import logging
import os
import shutil
import tarfile

import analysisUtils as au
import astropy.units as u
import numpy as np
from astropy.table import Table

from . import casaLegacySingleDishRoutines as csdr
from . import casaStuff
from .utilsSingleDish import getTPSampling

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# path constants
path_calibration = '../calibration/'

def extractJyperK(base_path=path_calibration):
    logger.info("Extracting Jy per K conversion factor")

    file_script = f'{base_path}/jyperk.csv'

    if not os.path.isfile(file_script):
        filetgz = glob.glob(f"{base_path}/*auxproducts.tgz")
        if len(filetgz) > 0:
            tar = tarfile.open(filetgz[0])
            tar.extractall(path=base_path)
            tar.close()

        # Check if it's still not there or search for jyperk_query.csv
        if not os.path.exists(file_script):
            alt_file_script = f'{base_path}/jyperk_query.csv'
            if not os.path.exists(alt_file_script):
                raise ValueError("Unable to find the jyperk.csv or jyperk_query.csv files.")
            else:
                file_script = alt_file_script

    # Copying the jyperk.csv file to the working directory
    shutil.copyfile(file_script, 'jyperk.csv')


def SDImaging(filename,
            source,
            name_line,
            vel_cube_range,
            chan_dv_kms,
            restfreq,
            phcenter='',
            freq_Hz=None,
            overwrite=True,
            keep_only_trimmed=True):

    outname = f'ALMA_TP.{source}.{name_line}'
    outimage = f'{outname}.image'
    outweight = f'{outname}.weight'

    # Setup imaging:
    start_vel = min(vel_cube_range)
    vwidth_kms = abs(vel_cube_range[1]-vel_cube_range[0])

    # Force >0 channel width
    chan_dv_kms = abs(chan_dv_kms)

    nchans_vel = int(round(vwidth_kms/chan_dv_kms))

    if freq_Hz is None:
        mymsmd = casaStuff.msmdtool()
        mymsmd.open(filename)
        freq_Hz = mymsmd.meanfreq(0)
        mymsmd.close()
        logger.info("Reading frequency in image: "+str(freq_Hz))

    diameter = 12.
    # Appropriate for ALMA 12m's aperture illumination pattern
    # Specifically, see ALMA Memo #456 (https://library.nrao.edu/public/memos/alma/main/memo456.pdf)
    # and Todd Hunter's page on the getTPSampling function (https://safe.nrao.edu/wiki/bin/view/ALMA/PrimaryBeamArcsec)
    fwhmfactor = 1.13
    c_light = 2.99792458e8 # m/s
    theorybeam = (fwhmfactor * c_light / (freq_Hz * diameter)) * (180/np.pi) * 3600
    cell       = theorybeam/9.0

    xSampling, ySampling, maxsize = getTPSampling(filename)

    imsize = int(round(maxsize/cell)*1.5)

    if phcenter == '':
        coord_phase = csdr.read_source_coordinates(filename, source)
    else:
        coord_phase = phcenter

    logger.info("Start imaging")
    logger.info("Imaging from velocity "+str(start_vel)+", using "+str(nchans_vel)+" channels.")
    logger.info("Rest frequency is "+str(restfreq)+" GHz.")
    logger.info("Cell and image sizes are: "+str(cell)+"arcsec and "+str(imsize))
    casaStuff.tsdimaging(
        infiles=filename,
        mode='velocity',
        field=source,
        nchan=nchans_vel,
        width=str(chan_dv_kms)+'km/s',
        start=str(start_vel)+'km/s',
        veltype ="radio",
        outframe='LSRK',
        restfreq=str(restfreq)+'GHz',
        gridfunction='SF',
        convsupport=6,
        phasecenter=coord_phase,
        imsize=imsize,
        cell=str(cell)+'arcsec',
        overwrite=overwrite,
        outfile=outname)


    # Trim the image:
    myia = casaStuff.iatool()
    myia.open(outimage)
    mask = myia.getchunk(getmask=True)
    myia.close()

    # Finally, trim the image and weight cube down
    mask_spec_x = np.any(mask, axis=tuple([i for i, x in enumerate(list(mask.shape)) if i != 0]))

    # Get bounding box in RA/Dec. Use padding of 1 pixel
    pad = 1
    xmin = np.max([0, np.min(np.where(mask_spec_x))-pad])
    xmax = np.min([np.max(np.where(mask_spec_x))+pad, mask.shape[0]-1])

    #mask_spec_y = np.sum(np.sum(mask*1.0,axis=2),axis=0) > 0
    mask_spec_y = np.any(mask, axis=tuple([i for i, x in enumerate(list(mask.shape)) if i != 1]))
    ymin = np.max([0,np.min(np.where(mask_spec_y))-pad])
    ymax = np.min([np.max(np.where(mask_spec_y))+pad,mask.shape[1]-1])

    box_string = '' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax)

    casaStuff.imsubimage(
        imagename=outimage,
        outfile=outimage+'_trimmed',
        box=box_string,
    )
    casaStuff.imsubimage(
        imagename=outweight,
        outfile=outweight+'_trimmed',
        box=box_string,
    )

    # Move things around
    os.system(f'mv {outimage} {outimage}.orig')
    os.system(f'mv {outimage}_trimmed {outimage}')

    os.system(f"mv {outweight} {outweight}.orig")
    os.system(f"mv {outweight}_trimmed {outweight}")

    if keep_only_trimmed:
        casaStuff.rmtables([outimage+'.orig'])
        casaStuff.rmtables([outweight+'.orig'])

    return outimage

def runALMAPipeline(path_galaxy,
                    in_source='all',
                    baseline_fit_func='poly',
                    baseline_fit_order=1,
                    baseline_linewindowmode='replace',
                    baseline_linewindow=None,
                    product_dict=None,
                    overwrite=True):
    '''
    Wraps the ALMA SD pipeline recipes into a single function. This it to pass
    custom parameters to the pipeline, primarily the baseline correction.

    Parameters
    ----------
    path_galaxy : str
        Path to the galaxy folder.
    baseline_fit_func : str
        Baseline fit function.
    baseline_fit_order : int
        Baseline fit order.
    baseline_linewindowmode : str
        Baseline line window mode.
    baseline_linewindow : dict or list
        Dictionary of baseline line windows. e.g. {"23": ['230.62GHz', '230.78GHz']}
        or List of channel/frequency ranges. e.g. [[230.62e0, 230.78e9]]
    product_dict : dict
        Dictionary of line products information.
    overwrite : bool
        Overwrite previous results.

    '''

    from pipeline.cli import (h_init,
                              hsd_importdata,
                              hsd_flagdata,
                              h_tsyscal,
                              hsd_tsysflag,
                              hsd_skycal,
                              hsd_k2jycal,
                              hsd_applycal,
                              hsd_atmcor,
                              hsd_baseline,
                              hsd_blflag,
                              h_save)

    # Change to working directory:
    workdir = os.path.join(path_galaxy, 'raw')
    logger.info(f"> Changing directory to {workdir}")
    os.chdir(workdir)

    # Defining Execution Blocks (EBS) names
    EBsnames = [f for f in os.listdir(".") if f.startswith('uid_')]

    if len(EBsnames) == 0:
        logger.info("No ASDM files found.")
        raise ValueError("No ASDM files found.")

    # Retrieve the k2jy scaling from the auxiliary calibration products.
    extractJyperK()

    # Extract MS names from the JyperK table
    jyperk_tab = Table.read("jyperk.csv")
    jyperk_ebs = np.unique(jyperk_tab["MS"])
    jyperk_ebs = [str(x.split(".ms")[0]) for x in jyperk_ebs]

    # Drop the '.asdm.sdm' extension. This matches with the naming in the JyperK file
    # and the format in the provided pipeline script.
    newEBnames = []
    for EBname in EBsnames:
        newEBname = EBname.split('.asdm.sdm')[0]

        # Only take EBs that have an entry in the JyperK table
        if newEBname in jyperk_ebs:
            os.rename(EBname, newEBname)

            newEBnames.append(newEBname)

        else:
            logger.warning(f"{newEBname} not found in jyperk.csv. Will not process")

    EBsnames = newEBnames

    logger.info(f"Using these ASDM files: {EBsnames}")

    # Create baseline dict with freq ranges to mask:
    if baseline_linewindow is None:

        # Construct the line window via spw:low~high strings based on the
        # product_dict.
        baseline_linewindow = []
        for this_product in product_dict:

            freq_rest = product_dict[this_product]['freq_rest_MHz']
            vsys = product_dict[this_product]['vsys']
            vel_line_mask = product_dict[this_product]['vel_line_mask']

            # Convert velocity range to frequency range
            freq_line_mask = (vel_line_mask * u.km / u.s).to(u.Hz, u.doppler_radio(freq_rest * u.MHz)).value

            # Giving list of [low, high] frequency ranges. The pipeline will apply the ranges to all valid SPWs.
            # NOTE: the pipeline internally checks the low/high order and correctly applies the range regardless of order.
            baseline_linewindow.append(list(freq_line_mask))

            # Append this to the product dict
            product_dict[this_product]['freq_line_mask'] = freq_line_mask

    # Run pipeline recipe.
    context = h_init()
    try:
        hsd_importdata(vis=EBsnames)

        hsd_flagdata(pipelinemode="automatic")
        h_tsyscal(pipelinemode="automatic")
        hsd_tsysflag(pipelinemode="automatic")
        hsd_skycal(pipelinemode="automatic")
        hsd_k2jycal(dbservice=False)
        hsd_applycal(pipelinemode="automatic")
        hsd_atmcor(pipelinemode="automatic")

        # NOTE: single baseline fit only. The pipeline repeats after 1 flagging step.
        # May need to revisit if flagging is missed.
        hsd_baseline(pipelinemode="automatic",
                     fitfunc=baseline_fit_func,
                     fitorder=baseline_fit_order,
                     linewindowmode=baseline_linewindowmode,
                     linewindow=baseline_linewindow
                     )
        hsd_blflag(pipelinemode="automatic")


    finally:
        h_save()


    this_vis = [f"{filename}.ms.atmcor.atmtype1_bl" for filename in EBsnames
                if os.path.exists(f"{filename}.ms.atmcor.atmtype1_bl")]

    for filename in EBsnames:
        if os.path.exists(f"{filename}.ms.atmcor.atmtype1_bl"):
            logger.info(f"Found {filename}.ms.atmcor.atmtype1_bl")
        else:
            logger.warning(f"Did not find {filename}.ms.atmcor.atmtype1_bl")

    if len(this_vis) == 0:
        logger.info("No valid MSs found for imaging.")
        raise ValueError("No valid MSs found for imaging.")

    # Avoid too many files open issues
    ms_concat_filename = 'ALMA_TP_concat.ms'
    if os.path.exists(ms_concat_filename):
        shutil.rmtree(ms_concat_filename)
    casaStuff.concat(
        vis=this_vis,
        concatvis=ms_concat_filename,
        respectname=True,)
    logger.info('Msnames: %s, N=%d, concatenated: %s'%(this_vis, len(this_vis), ms_concat_filename))

    this_vis = ms_concat_filename

    # read the source name directly from the ms
    source = csdr.get_sourcename(this_vis, source=in_source)

    for this_product in product_dict:


        name_line = product_dict[this_product]['name_line']
        chan_dv_kms = product_dict[this_product]['chan_dv_kms']
        freq_rest_GHz = product_dict[this_product]['freq_rest_MHz'] / 1.e3
        vel_cube = product_dict[this_product]['vel_cube']

        phase_center = product_dict[this_product]['phase_center']

        output_file = product_dict[this_product]['output_file']

        # Setup imaging call:
        outimage = SDImaging(
            filename=this_vis,
            source=source,
            name_line=name_line,
            vel_cube_range=vel_cube,
            chan_dv_kms=chan_dv_kms,
            restfreq=freq_rest_GHz,
            phcenter=phase_center,
            freq_Hz=None,
            overwrite=overwrite
            )

        # Export to fits
        imagefile_fits = f"{outimage}.fits"
        casaStuff.exportfits(imagename=outimage, fitsimage=imagefile_fits, overwrite=True)
        weightimage = outimage.replace(".image", ".weight")
        weightimage_fits = f"{weightimage}.fits"
        casaStuff.exportfits(imagename=weightimage, fitsimage=weightimage_fits, overwrite=True)

        shutil.copy2(imagefile_fits, output_file)
        # And export the weightfile
        weight_output_file = output_file.replace(".fits", '_weight.fits')
        shutil.copy2(weightimage_fits, weight_output_file)

        logger.info('> Copied FITS to "%s"'%(output_file))
        logger.info('> Copied FITS to "%s"'%(weight_output_file))
