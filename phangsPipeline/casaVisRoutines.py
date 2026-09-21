"""
Standalone routines to analyze and manipulate visibilities.
"""

import glob
import logging
import os
import shutil

import astropy.constants as const
import astropy.units as u
import numpy as np
from packaging import version
from scipy.ndimage import label

from . import casaStuff
from . import utilsLines as lines

logger = logging.getLogger(__name__)

# Physical constants
sol_kms = 2.99792458e5


##########################################
# Split, copy, combine measurement sets. #
##########################################


def copy_ms(infile=None, outfile=None, use_symlink=True, overwrite=False):
    """
    Copy a measurement set, optionally using symlink instead of
    actually copying.
    """

    # Check inputs

    if infile is None:
        logging.error("Please specify infile.")
        raise Exception("Please specify infile.")

    if outfile is None:
        logging.error("Please specify outfile.")
        raise Exception("Please specify outfile.")

    if not os.path.isdir(infile):
        logger.error(
            'Error! The input uv data measurement set "'+infile +
            '"does not exist!')
        raise Exception(
            'Error! The input uv data measurement set "'+infile +
            '"does not exist!')

    # Check for presence of existing outfile and abort if it is found
    # without overwrite permission.

    if os.path.isdir(outfile) and not os.path.isdir(outfile+'.touch'):
        if not overwrite:
            logger.warning(
                'Found existing data "'+outfile+'", will not overwrite.')
            return()

    # Delete existing output data.

    for suffix in ['', '.flagversions', '.touch']:
        if os.path.islink(outfile+suffix):
            os.unlink(outfile+suffix)
            logger.debug('os.unlink "'+outfile+'"')

        if os.path.isdir(outfile+suffix):
            shutil.rmtree(outfile+suffix)

    if use_symlink:

        # Make links

        if os.path.isdir(infile):
            os.symlink(infile, outfile)
            logger.debug(
                'os.symlink "'+infile+'", "'+outfile+'"')

        if os.path.isdir(infile+'.flagversions'):
            os.symlink(infile+'.flagversions', outfile+'.flagversions')
            logger.debug(
                'os.symlink "'+infile+'.flagversions'+'", "' +
                outfile+'.flagversions"')

        # Check

        if not os.path.islink(outfile):
            logger.error(
                'Failed to link the uv data to '+os.path.abspath(outfile)+'!')
            logger.error(
                'Please check your file system writing permission or '
                'system breaks.')
            raise Exception(
                'Failed to link the uv data to the imaging directory.')

        return()

    else:

        # Check existing output data

        has_existing_outfile = False
        if os.path.isdir(outfile) and not os.path.isdir(outfile+'.touch'):
            if not overwrite:
                has_existing_outfile = True

        # delete existing copied data if not overwriting

        if not has_existing_outfile:
            for suffix in ['', '.flagversions', '.touch']:
                if os.path.isdir(outfile+suffix):
                    shutil.rmtree(outfile+suffix)
                    logger.debug('shutil.rmtree "'+outfile+suffix+'"')

        # copy the data (.touch directory is a temporary flagpost)

        if not os.path.isdir(outfile+'.touch'):
            os.mkdir(outfile+'.touch')

        if os.path.isdir(infile):
            shutil.copytree(infile, outfile)
            logger.debug(
                'shutil.copytree "'+infile+'", "'+outfile+'"')

        if os.path.isdir(infile+'.flagversions'):
            shutil.copytree(infile+'.flagversions', outfile+'.flagversions')
            logger.debug(
                'shutil.copytree "'+infile+'.flagversions'+'", "' +
                outfile+'.flagversions'+'"')

        if os.path.isdir(outfile+'.touch'):
            os.rmdir(outfile+'.touch')

        # check copied_file, make sure copying was done

        if not os.path.isdir(outfile) or \
                os.path.isdir(outfile+'.touch'):
            logger.error(
                'Failed to copy the uv data to '+os.path.abspath(outfile)+'!')
            logger.error(
                'Please check your file system writing permission or '
                'system breaks.')
            raise Exception(
                'Failed to copy the uv data to the imaging directory.')

        return()

    return()


def split_science_targets(
        infile=None, outfile=None, field='', intent='OBSERVE_TARGET*',
        spw='', timebin='0s', do_statwt=False, overwrite=False):
    """
    Split science targets from the input ALMA measurement set to form
    a new, science-only measurement set. Optionally reweight the data
    using statwt.

    Relatively thin wrapper to split that smooths out some things like
    handling of flagversions and which data tables to use.

    Args:

    infile (str): The input measurement set data.

    outfile (str): The output measurement set data.

    field, spw, intent (str): The field, spw, intent used for selection.

    timebin: The time bin applied.

    overwrite (bool): Set to True to overwrite existing output
    data. The default is False, not overwriting anything.

    Inputs:

    infile: ALMA measurement set data folder.

    Outputs:

    outfile: ALMA measurement set data folder.

    """

    # Check inputs

    if infile is None:
        logging.error("Please specify infile.")
        raise Exception("Please specify infile.")

    if outfile is None:
        logging.error("Please specify outfile.")
        raise Exception("Please specify outfile.")

    if not os.path.isdir(infile):
        logger.error(
            'Error! The input uv data measurement set "'+infile +
            '"does not exist!')
        raise Exception(
            'Error! The input uv data measurement set "'+infile +
            '"does not exist!')

    # Check for presence of existing outfile and abort if it is found
    # without overwrite permission.

    if os.path.isdir(outfile) and not os.path.isdir(outfile+'.touch'):
        if not overwrite:
            logger.warning(
                'Found existing data "'+outfile+'", will not overwrite.')
            return()

    # Delete existing output data.

    for suffix in ['', '.flagversions', '.touch']:
        if os.path.islink(outfile+suffix):
            os.unlink(outfile+suffix)
            logger.debug('os.unlink "'+outfile+'"')

        if os.path.isdir(outfile+suffix):
            shutil.rmtree(outfile+suffix)

    logger.info("")
    logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
    logger.info("I will split out the data.")
    logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
    logger.info("")

    logger.info('Splitting from '+infile+' to '+outfile)

    # Verify the column to use. If present, we use the corrected
    # column. If not, then we use the data column.

    mytb = casaStuff.tbtool()
    mytb.open(infile, nomodify = True)
    colnames = mytb.colnames()
    if 'CORRECTED_DATA' in colnames:
        logger.info("Data has a CORRECTED column. Will use that.")
        use_column = 'CORRECTED'
    else:
        logger.info("Data lacks a CORRECTED column. Will use DATA column.")
        use_column = 'DATA'
    mytb.close()

    logger.info('... intent: '+intent)
    logger.info('... field: '+field)
    logger.info('... spw: '+spw)

    if not os.path.isdir(outfile+'.touch'):
        # mark the beginning of our processing
        os.mkdir(outfile+'.touch')

    split_params = {
        'vis': infile, 'intent': intent, 'field': field, 'spw': spw,
        'datacolumn': use_column, 'outputvis': outfile,
        'keepflags': False, 'timebin': timebin}

    logger.info(
        "... running CASA "+'split(' +
        ', '.join("{!s}={!r}".format(
            k, split_params[k]) for k in split_params.keys()) +
        ')')

    # an MS can have a SPW label for data that is no longer contained in the MS
    # (e.g., it was fully flagged, and keepflags=False was used in a previous split)
    # This try/except should work for CASA 6 and newer versions with CASA's improved
    # exception handling.
    try:
        casaStuff.split(**split_params)
        flag_split_success = True
    except RuntimeError as exc:
        logger.error("Splitting failed with exception: {}".format(exc))
        flag_split_success = False

    # Re-weight the data if desired.
    # Only continue if the split was successful
    if do_statwt and flag_split_success:
            logger.info("Using statwt to re-weight the data.")
            statwt_params = {'vis': outfile, 'datacolumn': 'DATA'}
            logger.info(
                "... running CASA "+'statwt(' +
                ', '.join("{!s}={!r}".format(
                    k, statwt_params[k]) for k in statwt_params.keys())+')')
            casaStuff.statwt(**statwt_params)

    if os.path.isdir(outfile+'.touch'):
        # mark the end of our processing
        os.rmdir(outfile+'.touch')

    return()


def concat_ms(
        infile_list=None, outfile=None, freqtol='', dirtol='',
        copypointing=True, overwrite=False):
    """
    Concatenate a list of measurement sets into one measurement set. A
    thin wrapper to concat. Thin wrapper to concat. Might build out in
    the future.

    Args:
        infile_list (list or str): The input list of measurement sets.
        outfile (str): The output measurement set data with suffix ".ms".

    Inputs:
        infile: ALMA measurement set data folder.

    Outputs:
        outfile: ALMA measurement set data folder.

    """
    # Check inputs

    if infile_list is None:
        logging.error("Please specify infile_list.")
        raise Exception("Please specify infile_list.")

    if outfile is None:
        logging.error("Please specify outfile.")
        raise Exception("Please specify outfile.")

    # make sure the input infile_list is a list
    if np.isscalar(infile_list):
        infile_list = [infile_list]

    # check file existence
    for this_infile in infile_list:
        if not os.path.isdir(this_infile):
            logger.error(
                'Error! The input measurement set "'+this_infile +
                '" not found')
            raise Exception(
                'Error! The input measurement set "'+this_infile +
                '" not found')

    # Quit if output data are present and overwrite is off.
    if os.path.isdir(outfile) and not os.path.isdir(outfile+'.touch'):
        if not overwrite:
            logger.warning(
                'Found existing output data "'+outfile +
                '", will not overwrite it.')
            return()

    # if overwrite or no file present, then delete existing output data.
    for suffix in ['', '.flagversions', '.touch']:
        if os.path.isdir(outfile+suffix):
            shutil.rmtree(outfile+suffix)

    # Concatenate all of the relevant files
    concat_params = {
        'vis': infile_list, 'concatvis': outfile, 'copypointing': copypointing}
    if freqtol is not None and freqtol != '':
        concat_params['freqtol'] = freqtol
    if dirtol is not None and dirtol != '':
        concat_params['dirtol'] = dirtol
    logger.info(
        "... running CASA "+'concat(' +
        ', '.join("{!s}={!r}".format(
            k, concat_params[k]) for k in concat_params.keys()) +
        ')')

    if not os.path.isdir(outfile+'.touch'):
        os.mkdir(outfile+'.touch')  # mark the beginning of our processing

    casaStuff.concat(**concat_params)

    if os.path.isdir(outfile+'.touch'):
        os.rmdir(outfile+'.touch')  # mark the end of our processing

    return()


##########################
# Continuum subtraction. #
##########################


def contsub(
        infile=None, outfile=None,
        ranges_to_exclude=[],
        flag_edge_fraction=0.0,
        solint='int',
        fitorder=0, combine='', overwrite=False):
    """
    Carry out uv continuum subtraction on a measurement set. First
    figures out channels corresponding to spectral lines for a
    provided suite of bright lines.

    Parameters
    ----------
    infile : str
        The input measurement set data folder.
    outfile : str
        The output measurement set data folder.
    ranges_to_exclude : list
        List of frequency ranges to exclude from the fit.
    flag_edge_fraction : float
        Fraction of the data to flag at the beginning and end of the fit.
    solint : str
        The integration time over which to fit the continuum.
    fitorder : int
        The order of the fit. Default is 0.
    combine : str
        The method to combine channels. Default is ''.
    overwrite : bool
        If True, overwrite existing output data.
    """

    # Error and file existence checking

    if infile is None:
        logging.error("Please specify infile.")
        raise Exception("Please specify infile.")

    if outfile is None:
        outfile = infile+'.contsub'

    if not os.path.isdir(infile):
        logger.error(
            'The input uv data measurement set "'+infile+'"does not exist.')
        return()

    # check existing output data in the imaging directory
    if (os.path.isdir(infile+'.contsub') and
            not os.path.isdir(infile+'.contsub'+'.touch')):
        if not overwrite:
            logger.warning(
                'Found existing output data "'+infile+'.contsub' +
                '", will not overwrite it.')
            return
    if os.path.isdir(infile+'.contsub'):
        shutil.rmtree(infile+'.contsub')
    if os.path.isdir(infile+'.contsub'+'.touch'):
        shutil.rmtree(infile+'.contsub'+'.touch')

    # Figure out which channels to exclude from the fit.

    # find_spw_channels_for_lines

    spw_flagging_string = spw_string_for_freq_ranges(
        infile=infile, freq_ranges_ghz=ranges_to_exclude,
        complement=True, # default to complement for new uvcontsub task
        flag_edge_fraction=flag_edge_fraction,
        )

    # uvcontsub, this outputs infile+'.contsub'
    # Pre 6.5.2
    if version.parse(casaStuff.casa_version_str) < version.parse('6.5.2'):
        uvcontsub_params = {
            'vis': infile,
            'fitspw': spw_flagging_string,
            'excludechans': False,  # now uses complement for channel selection.
            'combine': combine,
            'fitorder': fitorder,
            'solint': solint,
            'want_cont': False}
    # Post 6.5.2
    else:
        uvcontsub_params = {
            'vis': infile,
            'outputvis': outfile,
            'fitspec': spw_flagging_string,
            'fitorder': fitorder,
            'fitmethod': 'gsl'}  # or 'casacore'

    logger.info(
        "... running CASA "+'uvcontsub(' +
        ', '.join("{!s}={!r}".format(
            k, uvcontsub_params[k]) for k in uvcontsub_params.keys()) +
        ')')

    if not os.path.isdir(infile+'.contsub'+'.touch'):
        # mark the beginning of our processing
        os.mkdir(infile+'.contsub'+'.touch')

    casaStuff.uvcontsub(**uvcontsub_params)

    if os.path.isdir(infile+'.contsub'+'.touch'):
        # mark the end of our processing
        os.rmdir(infile+'.contsub'+'.touch')

    # Could manipulate outfile names here.

    return()


##########################################################
# Interface between spectral lines and spectral windows. #
##########################################################


def find_spws_for_line(
        infile=None, line=None, restfreq_ghz=None,
        vsys_kms=None, vwidth_kms=None, vlow_kms=None, vhigh_kms=None,
        max_chanwidth_kms=None,
        require_data=False, require_full_line_coverage=False,
        exit_on_error=True, as_list=False):
    """
    List the spectral windows in the input ms data that contains the
    input line, given the line velocity (vsys_kms) and line width
    (vwidth_kms), which are in units of km/s. Defaults to rest frequency
    (with vsys_kms = 0.0 and vwidth_kms = 0.0).
    """

    # Check inputs

    if infile is None:
        logging.error("Please specify infile.")
        raise Exception("Please specify infile.")

    # Verify file existence

    if not os.path.isdir(infile):
        logger.error(
            'Error! The input uv data measurement set "'+infile +
            '"does not exist!')
        raise Exception(
            'Error! The input uv data measurement set "'+infile +
            '"does not exist!')

    # Get the line name and rest-frame frequency in the line_list
    # module for the input line

    if restfreq_ghz is None:
        if line is None:
            logging.error(
                "Specify a line name or provide a rest frequency in GHz.")
            raise Exception("No rest frequency specified.")
        restfreq_ghz = (
            lines.get_line_name_and_frequency(line, exit_on_error=True))[1]

    # Work out the frequencies at the line edes.

    line_low_ghz, line_high_ghz = lines.get_ghz_range_for_line(
        restfreq_ghz=restfreq_ghz,
        vsys_kms=vsys_kms, vwidth_kms=vwidth_kms,
        vlow_kms=vlow_kms, vhigh_kms=vhigh_kms)
    logger.debug(
        "... line: %s, line freq: %.6f - %.6f, rest-freq: %.6f" %
        (line, line_low_ghz, line_high_ghz, restfreq_ghz))

    # If channel width restrictions are in place, calculate the
    # implied channel width requirement in GHz.

    if max_chanwidth_kms is not None:

        # line_freq_ghz = (line_low_ghz+line_high_ghz)*0.5

        # Using RADIO convention for velocities:

        max_chanwidth_ghz = restfreq_ghz*max_chanwidth_kms/sol_kms

        # using the high-z convention used to be this (delete eventually)
        # max_chanwidth_ghz = line_freq_ghz*max_chanwidth_kms/sol_kms

        logger.debug(
            "... max_chanwidth_kms: %.3f, max_chanwidth_ghz: %.6f" %
            (max_chanwidth_kms, max_chanwidth_ghz))

    else:

        max_chanwidth_ghz = None

    # Work out which spectral windows contain the line by looping over
    # SPWs one at a time.

    spw_list = []
    spw_lowest_ghz = None
    spw_highest_ghz = None

    spw_info = get_spw_info(infile)
    scans_for_spw = get_scans_for_spw(infile)

    for this_spw in spw_info.keys():

        spw_high_ghz = np.max(spw_info[this_spw]['edgeChannels'])/1e9
        spw_low_ghz = np.min(spw_info[this_spw]['edgeChannels'])/1e9
        logger.debug(
            "... spw: %s, freq: %.6f - %.6f GHz" %
            (this_spw, spw_low_ghz, spw_high_ghz))

        if spw_high_ghz < line_low_ghz:
            continue

        if spw_low_ghz > line_high_ghz:
            continue

        if max_chanwidth_ghz is not None:
            spw_chanwidth_ghz = abs(spw_info[this_spw]['chanWidth'])/1e9
            if spw_chanwidth_ghz > max_chanwidth_ghz:
                continue

        if require_data:
            if len(scans_for_spw[this_spw]) == 0:
                continue

        if require_full_line_coverage and not (
                spw_high_ghz > line_high_ghz and spw_low_ghz < line_low_ghz):
            continue

        spw_list.append(this_spw)

        if spw_lowest_ghz is None:
            spw_lowest_ghz = spw_low_ghz
        else:
            spw_lowest_ghz = min(spw_lowest_ghz, spw_low_ghz)

        if spw_highest_ghz is None:
            spw_highest_ghz = spw_high_ghz
        else:
            spw_highest_ghz = max(spw_highest_ghz, spw_high_ghz)

    # If we don't find the line in this data set, issue a warning and
    # return.

    if len(spw_list) == 0:

        logger.warning('No spectral windows contain the input line.')
        spw_list = []
        spw_list_string = None # can't be '', that selects all

        if as_list:
            return (spw_list)
        else:
            return (spw_list_string)

    else:

        # sort and remove duplicates
        spw_list = sorted(list(set(spw_list)))

        # make spw_list_string appropriate for use in selection
        spw_list_string = ','.join(np.array(spw_list).astype(str))

    # return
    if as_list:
        return(spw_list)
    else:
        return(spw_list_string)


def find_spws_for_science(
        infile=None, require_data=False, exit_on_error=True, as_list=False):
    """
    List all spectral windows that we judge likely to be used for
    science.
    """

    # Check inputs

    if infile is None:
        logging.error("Please specify infile.")
        raise Exception("Please specify infile.")

    # Verify file existence

    if not os.path.isdir(infile):
        logger.error(
            'Error! The input uv data measurement set "'+infile +
            '"does not exist!')
        raise Exception(
            'Error! The input uv data measurement set "'+infile +
            '"does not exist!')

    # Get science SPWs
    spw_string = get_science_spws(
        vis=infile,
        intent='OBSERVE_TARGET*',
    )
    if spw_string is None or len(spw_string) == 0:
        spw_string = get_science_spws(
            vis=infile,
            intent='OBSERVE_TARGET#ON_SOURCE',
        )

    spw_list = []
    for this_spw_string in spw_string.split(','):
        spw_list.append(int(this_spw_string))

    if require_data:
        scans_for_spw = get_scans_for_spw(infile)

        for spw in spw_list:
            if len(scans_for_spw[spw]) == 0:
                spw_list.remove(spw)
    # Return

    if len(spw_list) == 0:
        logger.warning('No science spectral windows found.')
        spw_list_string = None  # can't be '', that selects all
    else:

        # sort and remove duplicates
        spw_list = sorted(list(set(spw_list)))

        # make spw_list_string appropriate for use in selection
        spw_list_string = ','.join(np.array(spw_list).astype(str))

    if as_list:
        return(spw_list)
    else:
        return(spw_list_string)


def spw_string_for_freq_ranges(
        infile=None,
        freq_ranges_ghz=[],
        just_spw=[],
        flag_edge_fraction=0.0,
        complement=False,
        fail_on_empty=False):
    """
    Given an input measurement set, return the spectral
    List the spectral window and channels corresponding to the input
    lines in the input ms data.  Galaxy system velocity (vsys) and
    velocity width (vwidth) in units of km/s are needed.
    """

    # Check file existence

    if infile is None:
        logging.error("Please specify an input file.")
        raise Exception("Please specify an input file.")

    if not os.path.isdir(infile):
        logger.error(
            'The input measurement set "'+infile+'"does not exist.')
        raise Exception(
            'The input measurement set "'+infile+'"does not exist.')

    # Make sure that we have a list

    if not isinstance(freq_ranges_ghz, list):
        freq_ranges_ghz = [freq_ranges_ghz]

    if not isinstance(just_spw, list):
        just_spw = [just_spw]

    spw_info = get_spw_info(infile)

    # Loop over spectral windows
    spw_flagging_string = ''
    first_string = True
    for this_spw in spw_info:

        if len(just_spw) > 0:
            if this_spw not in just_spw:
                continue

        freq_axis = spw_info[this_spw]['chanFreqs']
        half_chan = abs(freq_axis[1]-freq_axis[0])*0.5
        chan_axis = np.arange(len(freq_axis))
        mask_axis = np.zeros_like(chan_axis, dtype='bool')

        for this_freq_range in freq_ranges_ghz:

            low_freq_hz = this_freq_range[0]*1e9
            high_freq_hz = this_freq_range[1]*1e9

            ind = (
                ((freq_axis-half_chan) >= low_freq_hz) *
                ((freq_axis+half_chan) <= high_freq_hz))
            mask_axis[ind] = True

        if complement:
            mask_axis = np.invert(mask_axis)

        # Additional edge flagging
        if flag_edge_fraction > 0.0:
            low_edge = int(np.ceil(mask_axis.size * flag_edge_fraction))
            high_edge = int(np.floor(mask_axis.size * (1. - flag_edge_fraction)))

            mask_axis[:low_edge] = False
            mask_axis[high_edge:] = False

        if fail_on_empty:
            if np.sum(np.invert(mask_axis)) == 0:
                return(None)

        regions = (label(mask_axis))[0]
        max_reg = np.max(regions)
        for ii in range(1, max_reg+1):
            this_mask = (regions == ii)
            low_chan = np.min(chan_axis[this_mask])
            high_chan = np.max(chan_axis[this_mask])
            this_spw_string = (
                str(this_spw)+':'+str(low_chan)+'~'+str(high_chan))
            if first_string:
                spw_flagging_string += this_spw_string
                first_string = False
            else:
                spw_flagging_string += ','+this_spw_string

    logger.info("... returning SPW selection string:")
    logger.info(spw_flagging_string)

    return(spw_flagging_string)


def compute_common_chanwidth(
        infile_list=None, line=None,
        vsys_kms=None, vwidth_kms=None, vlow_kms=None, vhigh_kms=None,
        require_full_line_coverage=False):
    """
    Calculates the coarsest channel width among all spectral windows
    in the input measurement set that contain the input line.

    Args:

    Returns:

    """

    if infile_list is None:
        logging.error(
            "Please specify one or more input files via infile_list.")
        Exception(
            "Please specify one or more input files via infile_list.")

    if np.isscalar(infile_list):
        infile_list = [infile_list]

    # Get the line name and line center rest-frame frequency
    # in the line_list module for the input line
    line_name, restfreq_ghz = lines.get_line_name_and_frequency(
        line, exit_on_error=True)

    # Work out the frequencies at the line edes and central frequency
    line_low_ghz, line_high_ghz = lines.get_ghz_range_for_line(
        line=line_name, vsys_kms=vsys_kms, vwidth_kms=vwidth_kms,
        vlow_kms=vlow_kms, vhigh_kms=vhigh_kms)

    line_freq_ghz = (line_high_ghz+line_low_ghz)/2.0

    coarsest_channel = None
    for this_infile in infile_list:
        # Find spws for line
        spw_list_string = find_spws_for_line(
            this_infile, line, vsys_kms=vsys_kms, vwidth_kms=vwidth_kms,
            require_full_line_coverage=require_full_line_coverage)

        chan_widths_hz = get_chan_widths(this_infile, spw_list_string)

        # Convert to km/s and return
        for this_chan_width_hz in chan_widths_hz:

            # Using RADIO convention for velocities:
            chan_width_kms = abs(
                this_chan_width_hz / (line_freq_ghz*1e9)*sol_kms)
            if coarsest_channel is None:
                coarsest_channel = chan_width_kms
            else:
                if chan_width_kms > coarsest_channel:
                    coarsest_channel = chan_width_kms

    return(coarsest_channel)


#######################################################
# Extract a single-line, common grid measurement set. #
#######################################################


def batch_extract_line(
        infile_list=[], outfile=None,
        target_chan_kms=None, restfreq_ghz=None, line=None,
        vsys_kms=None, vwidth_kms=None, vlow_kms=None, vhigh_kms=None,
        method='regrid_then_rebin', exact=False, freqtol='',
        allow_freqtol_chanfrac=True, freqtol_chanfrac=0.2,
        clear_pointing=True, require_full_line_coverage=False,
        overwrite=False):
    """
    Run a batch line extraction.
    """

    # Check that we have an output file defined.

    if outfile is None:
        logging.error("Please specify an output file.")
        raise Exception("Please specify an output file.")

    # Check existence of output data and abort if found and overwrite is off

    if os.path.isdir(outfile) and not os.path.isdir(outfile+'.touch'):
        if not overwrite:
            logger.warning(
                '... found existing output data "'+outfile +
                '", will not overwrite it.')
            return()

    # Else, clear all previous files and temporary files

    for suffix in ['', '.flagversions', '.touch', '.temp*']:
        for temp_outfile in glob.glob(outfile+suffix):
            if os.path.isdir(temp_outfile):
                logger.debug('... shutil.rmtree(%r)' % (temp_outfile))
                shutil.rmtree(temp_outfile)

    # Feed directly to generate an extraction scheme. This does a lot
    # of the error checking.

    schemes = suggest_extraction_scheme(
        infile_list=infile_list, target_chan_kms=target_chan_kms,
        method=method, exact=exact, restfreq_ghz=restfreq_ghz, line=line,
        vsys_kms=vsys_kms, vwidth_kms=vwidth_kms,
        vlow_kms=vlow_kms, vhigh_kms=vhigh_kms,
        require_full_line_coverage=require_full_line_coverage)
    for this_infile in schemes.keys():
        logger.info(
            "For this line ({}), I will extract SPWs {} "
            "from infile {}".format(
                line, schemes[this_infile].keys(), this_infile))

    # Execute the extraction scheme
    split_file_list = []
    for this_infile in schemes.keys():

        for this_spw in schemes[this_infile].keys():
            this_scheme = schemes[this_infile][this_spw]

            # Record the channel width in freq for later
            
            chan_width_ghz_final = this_scheme['chan_width_ghz'] * this_scheme['binfactor']

            # Specify output file and check for existence
            this_outfile = this_infile+'.temp_spw'+str(this_spw).strip()

            this_scheme['outfile'] = this_outfile
            this_scheme['overwrite'] = overwrite
            this_scheme['require_full_line_coverage'] = \
                require_full_line_coverage
            split_file_list.append(this_outfile)

            # Execute line extraction

            del this_scheme['chan_width_kms']
            del this_scheme['chan_width_ghz']
            extract_line(**this_scheme)

            # Deal with pointing table - testing shows it to be a
            # duplicate for each SPW here, so we remove all rows for
            # all SPWs except the first one.

            if clear_pointing:
                # This didn't work:
                # os.system('rm -rf '+this_outfile+'/POINTING')

                # This zaps the whole table:
                if os.path.exists(this_outfile+os.sep+'POINTING'):
                    clear_pointing_table(this_outfile)
                else:
                    copy_pointing = False
                    #logger.debug('Warning! Failed to run clear_pointing_table(%r)'%(this_outfile))

    # Allow a small tolerance in the channel width
    if allow_freqtol_chanfrac:
        freqtol_val = freqtol_chanfrac * chan_width_ghz_final
        freqtol = f"{freqtol_val}GHz"

    # Concatenate and combine the output data sets
    concat_ms(
        infile_list=split_file_list, outfile=outfile, freqtol=freqtol,
        overwrite=overwrite, copypointing=(not clear_pointing))

    # Clean up, deleting intermediate files

    for this_file in split_file_list:
        shutil.rmtree(this_file)

    return()


def choose_common_res(
        vals=[], epsilon=1e-4):
    """
    Choose a common resolution given a list and an inflation
    parameter epsilon. Returns max*(1+epsilon).).
    """
    if len(vals) == 0:
        return(None)
    ra = np.array(np.abs(vals))
    common_res = np.max(ra)*(1.+epsilon)
    return(common_res)


def suggest_extraction_scheme(
        infile_list=[], target_chan_kms=None, restfreq_ghz=None, line=None,
        vsys_kms=None, vwidth_kms=None, vlow_kms=None, vhigh_kms=None,
        method='regrid_then_rebin', exact=False,
        require_full_line_coverage=False):
    """
    Recommend extraction parameters given an input list of files, a
    desired target channel width, and a preferred algorithm. Returns a
    dictionary suitable for putting into the extraction routine.
    """

    # Check inputs

    if infile_list is None:
        logging.error("Please specify a list of infiles.")
        raise Exception("Please specify a list of infiles.")

    # make sure the input infile_list is a list

    if np.isscalar(infile_list):
        infile_list = [infile_list]

    # Require a valid method choice

    valid_methods = [
        'regrid_then_rebin', 'rebin_then_regrid', 'just_regrid', 'just_rebin']
    if method.lower().strip() not in valid_methods:
        logger.error("Not a valid line extraction method - "+str(method))
        raise Exception("Please specify a valid line extraction method.")

    # Get the line name and rest-frame frequency in the line_list
    # module for the input line

    if restfreq_ghz is None:
        if line is None:
            logging.error(
                "Specify a line name or provide a rest frequency in GHz.")
            raise Exception("No rest frequency specified.")
        restfreq_ghz = (
            lines.get_line_name_and_frequency(line, exit_on_error=True))[1]

    # # Work out the frequencies at the line edes.
    # line_low_ghz, line_high_ghz = lines.get_ghz_range_for_line(
    #     restfreq_ghz=restfreq_ghz, vsys_kms=vsys_kms, vwidth_kms=vwidth_kms,
    #     vlow_kms=vlow_kms, vhigh_kms=vhigh_kms)
    # line_freq_ghz = 0.5*(line_low_ghz+line_high_ghz)

    # ----------------------------------------------------------------
    # Loop over infiles and spectral windows and record information
    # ----------------------------------------------------------------

    scheme = {}
    chan_width_list = []
    binfactor_list = []
    total_nchans = []

    for this_infile in infile_list:

        spw_info = get_spw_info(this_infile)

        spw_list = find_spws_for_line(
            this_infile, restfreq_ghz=restfreq_ghz,
            vsys_kms=vsys_kms, vwidth_kms=vwidth_kms,
            vlow_kms=vlow_kms, vhigh_kms=vhigh_kms,
            require_data=True, as_list=True,
            require_full_line_coverage=require_full_line_coverage)

        scheme[this_infile] = {}

        for this_spw in spw_list:

            chan_width_ghz = np.abs(spw_info[this_spw]['chanWidth'])/1e9

            # Using RADIO convention:
            chan_width_kms = chan_width_ghz/restfreq_ghz * sol_kms
            # was using the old relative convention (can delete eventually)
            # chan_width_kms = chan_width_ghz/line_freq_ghz * sol_kms

            if chan_width_kms > target_chan_kms:
                logger.warning("Channel too big for SPW "+str(this_spw))
                continue

            # Figure out the binfactor
            this_binfactor = int(np.floor(target_chan_kms/chan_width_kms))
            # clamp to nchan if the binfactor exceeds the number of channels in the spw
            nchan_spw = spw_info[this_spw]['numChannels']
            if this_binfactor > nchan_spw:
                this_binfactor = nchan_spw

            # Figure out the total number of channels we should be expecting
            # for this spw
            total_nchan = int(np.floor(vwidth_kms / (chan_width_kms * this_binfactor)))

            # Inflate target slightly above the native chan width TOPO/LSRK
            # only triggers for cases where desired target channel width is 1 channel and binfactor 
            # is greater than 1
            if total_nchan == 1 and this_binfactor > 1:
                this_binfactor -= 1
                
            # record the values for the scheme
            chan_width_list.append(chan_width_kms)
            binfactor_list.append(this_binfactor)
            total_nchans.append(total_nchan)

            # Record basic file information
            scheme[this_infile][this_spw] = {}
            scheme[this_infile][this_spw]['infile'] = this_infile
            scheme[this_infile][this_spw]['spw'] = str(this_spw)

            # Record the line information
            scheme[this_infile][this_spw]['restfreq_ghz'] = restfreq_ghz
            scheme[this_infile][this_spw]['line'] = line
            scheme[this_infile][this_spw]['vlow_kms'] = vlow_kms
            scheme[this_infile][this_spw]['vhigh_kms'] = vhigh_kms
            scheme[this_infile][this_spw]['vsys_kms'] = vsys_kms
            scheme[this_infile][this_spw]['vwidth_kms'] = vwidth_kms

            # Record method information
            scheme[this_infile][this_spw]['method'] = method
            scheme[this_infile][this_spw]['binfactor'] = this_binfactor
            scheme[this_infile][this_spw]['target_chan_kms'] = None

            # Record channel width information
            scheme[this_infile][this_spw]['chan_width_kms'] = chan_width_kms
            scheme[this_infile][this_spw]['chan_width_ghz'] = chan_width_ghz

    # guard against total_nchans being empty
    if total_nchans:
        total_nchan = np.nanmin(total_nchans)
        for this_infile in scheme.keys():
            for this_spw in scheme[this_infile].keys():
                scheme[this_infile][this_spw]['total_nchan'] = total_nchan
    else:
        logger.warning(
            'No SPW satisfies target_chan_kms for the requested range; '
            'returning empty scheme.'
        )

    # ----------------------------------------------------------------
    # Figure out the strategy
    # ----------------------------------------------------------------

    # ... for rebinning, just do the naive division of floor(target / current)
    if method == 'just_rebin':
        return(scheme)

    # ... for regridding, just regrid to the desired width
    elif method == 'just_regrid':
        for this_infile in scheme.keys():
            for this_spw in scheme[this_infile].keys():
                if exact:
                    scheme[this_infile][this_spw]['target_chan_kms'] = \
                        target_chan_kms
                else:
                    # Could revise this ... not positive of the correct choice
                    scheme[this_infile][this_spw]['target_chan_kms'] = \
                        target_chan_kms

    # ... for rebin-then-regrid, first rebin by the naive amount. Then
    # regrid either to the final value (if exact) or to a common
    # resolution determined by the actual channels and rebinnning.

    elif method == 'rebin_then_regrid':
        common_res = choose_common_res(
            vals=(np.array(chan_width_list) * np.array(binfactor_list)),
            epsilon=3e-4)
        for this_infile in scheme.keys():
            for this_spw in scheme[this_infile].keys():
                if exact:
                    scheme[this_infile][this_spw]['target_chan_kms'] = \
                        target_chan_kms
                else:
                    scheme[this_infile][this_spw]['target_chan_kms'] = \
                        common_res
                    # print("chan_width_list: ", chan_width_list)
                    # print("binfactor_list: ", binfactor_list)
                    # print("common res/target channel: ", common_res)

    # ... for regrid-then-rebin

    elif method == 'regrid_then_rebin':
        common_res = choose_common_res(
            vals=(np.array(chan_width_list) * np.array(binfactor_list)),
            epsilon=3e-4)
        for this_infile in scheme.keys():
            for this_spw in scheme[this_infile].keys():
                if exact:
                    scheme[this_infile][this_spw]['target_chan_kms'] = (
                        target_chan_kms /
                        scheme[this_infile][this_spw]['binfactor'])
                else:
                    this_target_chan_kms = (
                        common_res /
                        scheme[this_infile][this_spw]['binfactor'])
                    scheme[this_infile][this_spw]['target_chan_kms'] = \
                        this_target_chan_kms
                    # print("chan_width_list: ", chan_width_list)
                    # print("binfactor_list: ", binfactor_list)
                    # print("common res: ", common_res)
                    # print("target channel: ", this_target_chan_kms)

    # Return

    return(scheme)


def extract_line(
        infile=None,
        outfile=None,
        spw=None,
        restfreq_ghz=None,
        line='co21',
        vlow_kms=None,
        vhigh_kms=None,
        vsys_kms=None,
        vwidth_kms=None,
        method='regrid_then_rebin',
        target_chan_kms=None,
        nchan=None,
        binfactor=None,
        total_nchan=None,
        require_full_line_coverage=False,
        overwrite=False,
):
    """
    Line extraction routine. Takes infile, outfile, line of interest,
    and algorithm, along with algorithm tuning parameters.
    """

    # Check the method

    valid_methods = [
        'regrid_then_rebin', 'rebin_then_regrid', 'just_regrid', 'just_rebin']
    if method.lower().strip() not in valid_methods:
        logger.error("Not a valid line extraction method - "+str(method))
        raise Exception("Please specify a valid line extraction method.")

    # Check input

    if infile is None:
        logging.error("Please specify an input file.")
        raise Exception("Please specify an input file.")

    if outfile is None:
        logging.error("Please specify an output file.")
        raise Exception("Please specify an output file.")

    if not os.path.isdir(infile):
        logger.error(
            'The input measurement set "'+infile+'"does not exist.')
        raise Exception(
            'The input measurement set "'+infile+'"does not exist.')

    # Check existence of output data and abort if found and overwrite is off

    if os.path.isdir(outfile) and not os.path.isdir(outfile+'.touch'):
        if not overwrite:
            logger.warning(
                '... found existing output data "'+outfile +
                '", will not overwrite it.')
            return()

    # Else, clear all previous files and temporary files

    for suffix in ['', '.flagversions', '.touch', '.temp*']:
        for temp_outfile in glob.glob(outfile+suffix):
            if os.path.isdir(temp_outfile):
                logger.debug('... shutil.rmtree(%r)' % (temp_outfile))
                shutil.rmtree(temp_outfile)

    # Create touch file to mark that we are processing this data
    if not os.path.isdir(outfile+'.touch'):
        os.mkdir(outfile+'.touch')

    # Get the line name and rest-frame frequency in the line_list
    # module for the input line.

    if restfreq_ghz is None:
        if line is None:
            logging.error(
                "Specify a line name or provide a rest frequency in GHz.")
            raise Exception("No rest frequency specified.")

        restfreq_ghz = (
            lines.get_line_name_and_frequency(line, exit_on_error=True))[1]

    # Handle velocity windows, etc.

    if vsys_kms is None and vwidth_kms is None:
        if vlow_kms is not None and vhigh_kms is not None:
            vsys_kms = 0.5*(vhigh_kms+vlow_kms)
            vwidth_kms = (vhigh_kms - vlow_kms)
        else:
            if method == 'just_regrid' or method == 'regrid_then_rebin' or \
                    method == 'rebin_then_regrid':
                logging.error("I need a velocity width for a regridding step.")
                raise Exception("Need velocity width for regridding.")

    # ... should only reach this next block in the "just_rebinning" case

    if (vsys_kms is None) and (vwidth_kms is None) and \
            (vlow_kms is None) and (vhigh_kms is None):
        logging.error("Missing velocities. Setting to zero as a placeholder.")
        vsys_kms = 0.0
        vwidth_kms = 0.0

    # ... if now SPW selection string is provided then note whether we
    # have multiple windows.

    if spw is None:
        spw_list = find_spws_for_line(
            infile=infile, line=line, restfreq_ghz=restfreq_ghz,
            vsys_kms=vsys_kms, vwidth_kms=vwidth_kms,
            vlow_kms=vlow_kms, vhigh_kms=vhigh_kms,
            require_full_line_coverage=require_full_line_coverage,
            require_data=True,
            exit_on_error=True, as_list=True)
        if spw_list is None or len(spw_list) == 0:
            logging.error("No SPWs for selected line and velocity range.")
            return()
        spw = spw_list.join(',')
    else:
        spw_list = spw.split(',')

    multiple_spws = len(spw_list) > 1

    # ............................................
    # Initialize the calls
    # ............................................

    if method == 'just_regrid' or method == 'regrid_then_rebin' or \
            method == 'rebin_then_regrid':

        if target_chan_kms is None:
            logger.warning('Need a target channel width to enable regridding.')
            return()

        vstart_kms = vsys_kms - vwidth_kms/2.0

        regrid_params, regrid_msg = build_mstransform_call(
            infile=infile,
            outfile=outfile,
            restfreq_ghz=restfreq_ghz,
            spw=spw,
            vstart_kms=vstart_kms,
            vwidth_kms=vwidth_kms,
            target_chan_kms=target_chan_kms,
            nchan=nchan,
            binfactor=binfactor,
            total_nchan=total_nchan,
            method='regrid',
            require_full_line_coverage=require_full_line_coverage,
        )

    if method == 'just_rebin' or method == 'regrid_then_rebin' or \
            method == 'rebin_then_regrid':

        if binfactor is None:
            logger.warning('Need a bin factor to enable rebinning.')
            return()

        rebin_params, rebin_msg = build_mstransform_call(
            infile=infile,
            outfile=outfile,
            restfreq_ghz=restfreq_ghz,
            spw=spw,
            binfactor=binfactor,
            method='rebin',
            require_full_line_coverage=require_full_line_coverage,
        )

    if multiple_spws:
        combine_params, combine_msg = build_mstransform_call(
            infile=infile,
            outfile=outfile,
            restfreq_ghz=restfreq_ghz,
            spw=spw,
            method='combine',
            require_full_line_coverage=require_full_line_coverage,
        )

    # ............................................
    # string the calls together in the desired order
    # ............................................

    params_list = []
    msg_list = []
    if method == 'just_regrid':
        params_list.append(regrid_params)
        msg_list.append(regrid_msg)

    if method == 'just_rebin':
        params_list.append(rebin_params)
        msg_list.append(rebin_msg)

    if method == 'rebin_then_regrid':
        params_list.append(rebin_params)
        msg_list.append(rebin_msg)

        params_list.append(regrid_params)
        msg_list.append(regrid_msg)

    if method == 'regrid_then_rebin':
        params_list.append(regrid_params)
        msg_list.append(regrid_msg)

        params_list.append(rebin_params)
        msg_list.append(rebin_msg)

    if multiple_spws:
        params_list.append(combine_params)
        msg_list.append(combine_msg)

    # ............................................
    # Execute the list of mstransform calls
    # ............................................

    n_calls = len(params_list)
    logger.info('... we will have '+str(n_calls)+' mstransform calls')

    for kk in range(n_calls):
        this_params = params_list[kk]
        this_msg = msg_list[kk]
        if kk == 0:
            this_params['vis'] = infile
            this_params['outputvis'] = outfile+'.temp%d' % (kk+1)
        elif kk == n_calls-1:
            this_params['vis'] = outfile+'.temp%d' % (kk)
            this_params['outputvis'] = outfile
        else:
            this_params['vis'] = outfile+'.temp%d' % (kk)
            this_params['outputvis'] = outfile+'.temp%d' % (kk+1)

        if os.path.isdir(this_params['outputvis']):
            shutil.rmtree(this_params['outputvis'])

        logger.info("... "+this_msg)

        # in the case where we are in subsequent split, we expect a
        # single SPW and to use the data column.

        if kk > 0:
            this_params['spw'] = ''
            this_params['datacolumn'] = 'DATA'

        logger.info(
            "... running CASA "+'mstransform(' +
            ', '.join("{!s}={!r}".format(
                t, this_params[t]) for t in this_params.keys()) +
            ')')

        if not os.path.isdir(this_params['outputvis']+'.touch'):
            # mark the beginning of our processing
            os.mkdir(this_params['outputvis']+'.touch')

        casaStuff.mstransform(**this_params)

        if os.path.isdir(this_params['outputvis']+'.touch') and \
           this_params['outputvis'] != outfile:
            # mark the end of our processing
            os.rmdir(this_params['outputvis']+'.touch')

    # ............................................
    # Clean up leftover files
    # ............................................

    # TBD revisit

    if os.path.isdir(outfile):
        logger.info("... deleting temporary files")
        for kk in range(len(params_list)):
            for suffix in [
                    '.temp%d' % (kk),
                    '.temp%d.flagversions' % (kk),
                    '.temp%d.touch' % (kk)]:
                if os.path.isdir(outfile+suffix):
                    shutil.rmtree(outfile+suffix)

    # Remove touch file to mark that we are have done the processing
    if os.path.isdir(outfile+'.touch'):
        os.rmdir(outfile+'.touch')

    return()


def build_mstransform_call(
        infile=None,
        outfile=None,
        restfreq_ghz=None,
        spw=None,
        vstart_kms=None,
        vwidth_kms=None,
        datacolumn=None,
        method='regrid',
        target_chan_kms=None,
        nchan=None,
        binfactor=None,
        total_nchan=None,
        require_full_line_coverage=False,
        overwrite=False,
):
    """
    Extract a spectral line from a measurement set and regrid onto a
    new velocity grid with the desired spacing. There are some minor
    subtleties here related to regridding and rebinning.
    """

    # ............................................
    # Error checking and setup
    # ............................................

    # Check that the requested method is understood

    valid_methods = ['rebin', 'regrid', 'combine']
    if method.lower().strip() not in valid_methods:
        logger.error("Not a valid line extraction method - "+str(method))
        raise Exception("Please specify a valid line extraction method.")

    # Check input

    if infile is None:
        logging.error("Please specify an input file.")
        raise Exception("Please specify an input file.")

    if outfile is None:
        logging.error("Please specify an output file.")
        raise Exception("Please specify an output file.")

    # If not supplied by the user, find which SPWs should be included
    # in the processing.

    if spw is None:
        if restfreq_ghz is not None:
            spw = find_spws_for_line(
                infile=infile, restfreq_ghz=restfreq_ghz,
                vlow_kms=vstart_kms, vhigh_kms=vstart_kms+vwidth_kms,
                require_full_line_coverage=require_full_line_coverage)
            # Exit if no SPWs contain the line.
            if spw is None:
                # there has already a warning message inside
                # find_spws_for_line()
                return()
        else:
            logger.info("... Defaulting to all SPW selections.")
            spw = ''

    # Determine the column to use

    if datacolumn is None:
        mytb = casaStuff.tbtool()
        mytb.open(infile, nomodify = True)
        colnames = mytb.colnames()
        if 'CORRECTED_DATA' in colnames:
            logger.info("... Data has a CORRECTED column. Will use that.")
            datacolumn = 'CORRECTED'
        else:
            logger.info(
                "... Data lacks a CORRECTED column. Will use DATA column.")
            datacolumn = 'DATA'
        mytb.close()

    # ............................................
    # Common parameters
    # ............................................

    params = {
        'vis': infile,
        'outputvis': outfile,
        'datacolumn': datacolumn,
        'spw': spw,
    }

    # ............................................
    # Regridding
    # ............................................

    if method == 'regrid':

        # Check that we are provided a rest frequency or line name

        if restfreq_ghz is None:
            logger.error("Please specify a rest frequency in GHz.")
            raise Exception("Please specify a rest frequency in GHz.")
        restfreq_string = ("{:12.8f}".format(restfreq_ghz)+'GHz').strip()

        # Check that we have a velocity start and width

        if vstart_kms is None:
            logger.error("Please specify a starting velocity in km/s.")
            raise Exception("Please specify a starting velocity in km/s.")
        start_vel_string = ("{:12.8f}".format(vstart_kms)+'km/s').strip()

        # Check that we have a velocity width

        if vwidth_kms is None:
            if nchan is None:
                logger.error(
                    "Please specify a velocity width in km/s or number of channels.")
                raise Exception("Please specify a velocity width in km/s or number of channels.")
            else:
                vwidth_kms = nchan*target_chan_kms

        # Figure out the current channel spacing

        line_low_ghz, line_high_ghz = lines.get_ghz_range_for_line(
            restfreq_ghz=restfreq_ghz,
            vlow_kms=vstart_kms, vhigh_kms=vstart_kms+vwidth_kms)
        # line_freq_ghz = (line_low_ghz+line_high_ghz)*0.5

        max_chan_ghz = np.max(np.abs(get_chan_widths(infile, spw)))/1e9

        # Using RADIO velocity convention:
        current_chan_kms = max_chan_ghz/restfreq_ghz*sol_kms

        # Old approach using the relative/high-z convention was:
        # current_chan_kms = max_chan_ghz/line_freq_ghz*sol_kms

        skip_width = False
        if target_chan_kms is None:
            logger.warning('Target channel not set. Using current channel.')
            target_chan_kms = current_chan_kms
            skip_width = True
        elif current_chan_kms > target_chan_kms:
            logger.warning('Target channel less than current channel:')
            logger.warning(
                'Asked for '+str(target_chan_kms)+' current ' +
                str(current_chan_kms))
            target_chan_kms = current_chan_kms
            skip_width = True

        chanwidth_string = ("{:12.8f}".format(target_chan_kms)+'km/s').strip()

        # Figure the number of channels if not supplied
        if nchan is None:
            nchan = int(np.max(np.ceil(vwidth_kms / target_chan_kms)))

        # Make sure that we won't lose anything in the rebinning stage
        add_chans = get_add_chans(nchan=nchan,
                                  binfactor=binfactor,
                                  total_nchan=total_nchan,
                                  )
        nchan += add_chans

        params.update(
            {'combinespws': False, 'regridms': True, 'chanaverage': False,
             'mode': 'velocity', 'interpolation': 'cubic',
             'outframe': 'lsrk', 'veltype': 'radio', 'restfreq': restfreq_string,
             'start': start_vel_string, 'nchan': nchan, 'width': chanwidth_string })

        if skip_width:
            del params['width']

        message = (
            '... regrid channel width '+chanwidth_string +
            ' and nchan '+str(nchan))

    # ............................................
    # Rebin
    # ............................................

    if method == 'rebin':

        params.update({
            'combinespws': False, 'regridms': False,
            'chanaverage': True, 'chanbin': binfactor,
        })

        if binfactor == 1:
            params['chanaverage'] = False

        message = '... rebin by a factor of '+str(binfactor)

    # ............................................
    # Combine SPWs
    # ............................................

    if method == 'combine':

        params.update({
            'combinespws': True, 'regridms': False,
            'chanaverage': False, 'keepflags': False
        })

        message = '... combine attempting to merge spectral windows.'

    return(params, message)


def get_add_chans(
        nchan: int,
        binfactor: int = None,
        total_nchan: int = None,
):
    """Get number of additional channels to add or subtract in the mstransform regrid

    Args:
        nchan (int): Initial number of channels
        binfactor (int, optional): Bin factor for rebinning. Defaults to None.
        total_nchan (int, optional): Total number of channels. Defaults to None.

    Returns:
        int: Number of additional channels to add or subtract
    """

    add_chans = 0

    # Start with the binfactor
    if binfactor is not None:

        # If we don't perfectly divide, then bin up
        bin_chans = nchan % binfactor
        if bin_chans != 0:
            add_chans += (binfactor - bin_chans)

    # Then the total number of channels we should be expecting
    if total_nchan is not None:

        # Get the total numbers of channels we should expect *before* rebinning
        total_nchan_unbinned = total_nchan * binfactor

        # If this doesn't match up, then correct
        if nchan + add_chans != total_nchan_unbinned:
            add_chans += total_nchan_unbinned - (nchan + add_chans)

    # Cast back to integer
    add_chans = int(add_chans)

    return add_chans


def reweight_data(
        infile=None, edge_kms=None, edge_chans=None,
        overwrite=False, datacolumn=None):
    """
    Use statwt to empirically re-weight data.
    Accepts an "edge" definition in either channels or km/s.
    """

    # Check input

    if infile is None:
        logging.error("Please specify an input file.")
        raise Exception("Please specify an input file.")

    if not os.path.isdir(infile):
        logger.error(
            'The input measurement set "'+infile+'"does not exist.')
        raise Exception(
            'The input measurement set "'+infile+'"does not exist.')

    # Determine column to use

    if datacolumn is None:
        mytb = casaStuff.tbtool()
        mytb.open(infile, nomodify = True)
        colnames = mytb.colnames()
        if 'CORRECTED_DATA' in colnames:
            logger.info("... Data has a CORRECTED column. Will use that.")
            datacolumn = 'CORRECTED'
        else:
            logger.info(
                "... Data lacks a CORRECTED column. Will use DATA column.")
            datacolumn = 'DATA'
        mytb.close()

    # Figure out the channel selection string

    exclude_str = ''
    if (edge_chans is not None) or (edge_kms is not None):

        spw_info = get_spw_info(infile)

        first = True
        for this_spw in spw_info.keys():

            if edge_kms is not None:
                spw_high_ghz = np.max(spw_info[this_spw]['edgeChannels'])/1e9
                spw_low_ghz = np.min(spw_info[this_spw]['edgeChannels'])/1e9
                spw_chanwidth_ghz = abs(spw_info[this_spw]['chanWidth'])/1e9

                mean_freq_ghz = 0.5*(spw_high_ghz+spw_low_ghz)

                # Here we COULD convert to RADIO convention:

                # mean_chanwidth_kms = spw_chanwidth_ghz/restfreq_ghz*sol_kms

                # BUT the rest frequency is not defined and this is an
                # approximate case, so would propose to leave this.

                # Note use of high redshift convention.

                logger.warning(
                    "... be aware that (only) statwt uses the "
                    "high-z velocity convention.")
                logger.warning(
                    "... the rest of the pipleine uses radio convention.")
                mean_chanwidth_kms = spw_chanwidth_ghz/mean_freq_ghz*sol_kms

                edge_chans = int(np.ceil(edge_kms / mean_chanwidth_kms))

            nchan = spw_info[this_spw]['numChannels']

            if edge_chans*2 > nchan:
                logger.warning(
                    "... Too many edge channels for given spw: "+str(this_spw))
                logger.warning(
                    "... By default we will not exclude ANY channels.")
                continue

            low = int(np.ceil(edge_chans-1))
            if low < 0:
                low = 0

            high = int(np.floor(nchan-edge_chans-2))
            if high > nchan-1:
                high = nchan-1

            if high == low:
                logger.warning(
                    "Too many edge channels for given spw: "+str(this_spw))
                logger.warning("By default we will not exclude ANY channels.")
                continue

            if first:
                exclude_str += str(this_spw)+':'+str(low)+'~'+str(high)
                first = False
            else:
                exclude_str += ','+str(this_spw)+':'+str(low)+'~'+str(high)

    if exclude_str != '':
        logger.info("... running statwt with exclusion: "+exclude_str)

    # Build the statwt call
    if exclude_str == '':
        excludechans = False
    else:
        excludechans = True
    statwt_params = {
        'vis': infile, 'timebin': '0.001s', 'slidetimebin': False,
        'chanbin': 'spw', 'statalg': 'classic', 'datacolumn': datacolumn,
        'fitspw': exclude_str, 'excludechans': excludechans,
    }

    # Run the call
    if not os.path.isdir(infile+'.touch'):
        os.mkdir(infile+'.touch')  # no overwrite checks here

    logger.info(
        "... running CASA "+'statwt(' +
        ', '.join("{!s}={!r}".format(
            t, statwt_params[t]) for t in statwt_params.keys()) +
        ')')

    casaStuff.statwt(**statwt_params)

    logger.info("... statwt done")

    if os.path.isdir(infile+'.touch'):
        os.rmdir(infile+'.touch')  # no overwrite checks here

    return()


########################################
# Extract a continuum measurement set. #
########################################


def batch_extract_continuum(
        infile_list=[], outfile=None,
        ranges_to_extract=None, target_chan_ghz=None, lines_to_flag=None,
        vlow_kms=None, vhigh_kms=None, vsys_kms=None, vwidth_kms=None,
        method='regrid_then_rebin', exact=False, freqtol='',
        clear_pointing=True, require_full_cont_coverage=False,
        overwrite=False):
    """
    Run a batch continuum extraction.
    """

    # Check that we have an output file defined.
    if outfile is None:
        logging.error("Please specify an output file.")
        raise Exception("Please specify an output file.")

    # Check existence of output data and abort if found and overwrite is off
    if os.path.isdir(outfile) and not os.path.isdir(outfile+'.touch'):
        if not overwrite:
            logger.warning(
                'Found existing output data "'+outfile +
                '", will not overwrite it.')
            return()

    # Else, clear all previous files and temporary files
    for suffix in ['', '.flagversions', '.touch', '.temp*']:
        for temp_outfile in glob.glob(outfile+suffix):
            if os.path.isdir(temp_outfile):
                shutil.rmtree(temp_outfile)
                logger.debug('... shutil.rmtree(%r)' % (temp_outfile))

    if ranges_to_extract is None and target_chan_ghz is None:

        split_file_list = []
        for this_infile in infile_list:
            # Specify output file and check for existence
            this_outfile = this_infile + '.temp_cont'

            split_file_list.append(this_outfile)

            extract_continuum(
                infile=this_infile,
                outfile=this_outfile,
                lines_to_flag=lines_to_flag,
                vsys_kms=vsys_kms,
                vwidth_kms=vwidth_kms,
                vlow_kms=vlow_kms,
                vhigh_kms=vhigh_kms,
                overwrite=overwrite,
            )

    else:

        # Check input parameters
        if ranges_to_extract is None:
            logging.error(
                "Please specify frequency ranges for continuum extraction.")
            raise Exception(
                "Please specify frequency ranges for continuum extraction.")
        for ii, range_i in enumerate(ranges_to_extract):
            max_i, min_i = np.max(range_i), np.min(range_i)
            for jj, range_j in enumerate(ranges_to_extract):
                max_j, min_j = np.max(range_j), np.min(range_j)
                if ii < jj and max_i > min_j and min_i < max_j:
                    logging.error(
                        "Overlapping frequency ranges for continuum extraction: "
                        "{} vs {}".format(range_i, range_j))
                    raise Exception(
                        "Overlapping frequency ranges for continuum extraction: "
                        "{} vs {}".format(range_i, range_j))

        split_file_list = []
        for kk, this_range in enumerate(ranges_to_extract):

            reffreq_ghz = np.mean(this_range)
            refvsys_kms = 0.0
            refvwidth_kms = (
                abs(np.diff(this_range).item()) / reffreq_ghz * sol_kms)

            if target_chan_ghz is None:
                # use a single channel to cover this entire frequency range
                target_chan_kms = refvwidth_kms
                exact = True
            else:
                target_chan_kms = target_chan_ghz / reffreq_ghz * sol_kms

            # Generate extraction schemes for this frequency range
            if exact:
                schemes = suggest_extraction_scheme(
                    infile_list=infile_list, target_chan_kms=target_chan_kms,
                    method=method, exact=exact, restfreq_ghz=reffreq_ghz,
                    vsys_kms=refvsys_kms, vwidth_kms=refvwidth_kms,
                    require_full_line_coverage=require_full_cont_coverage)
            else:
                # slightly adjust the channel size so that this frequency range
                # contains an integer number of channels
                target_chan_kms = (
                    refvwidth_kms / np.ceil(refvwidth_kms / target_chan_kms))
                schemes = suggest_extraction_scheme(
                    infile_list=infile_list,
                    target_chan_kms=target_chan_kms,
                    method=method, exact=True, restfreq_ghz=reffreq_ghz,
                    vsys_kms=refvsys_kms, vwidth_kms=refvwidth_kms,
                    require_full_line_coverage=require_full_cont_coverage)
            for this_infile in schemes.keys():
                logger.info(
                    "For this frequency range ({:.6f} - {:.6f}), "
                    "I will extract SPWs {} from infile {}".format(
                        np.min(this_range), np.max(this_range),
                        schemes[this_infile].keys(), this_infile))

            # Execute the extraction scheme
            for this_infile in schemes.keys():
                for this_spw in schemes[this_infile].keys():
                    this_scheme = schemes[this_infile][this_spw]
                    # Extract relevant parameters for continuum extraction
                    this_binfactor = this_scheme['binfactor']
                    this_target_chan_ghz = \
                        this_scheme['target_chan_kms'] / sol_kms * reffreq_ghz
                    this_nchan = int(np.floor(
                        refvwidth_kms * (1. + 1e-6) /
                        this_scheme['target_chan_kms']))
                    # Specify output file and check for existence
                    this_outfile = this_infile+'.temp{:.0f}_spw{:.0f}'.format(
                        kk, this_spw)
                    split_file_list.append(this_outfile)

                    # Execute continuum extraction
                    extract_continuum(
                        infile=this_infile, outfile=this_outfile,
                        spw=str(this_spw), range_to_extract=this_range,
                        lines_to_flag=lines_to_flag,
                        vlow_kms=vlow_kms, vhigh_kms=vhigh_kms,
                        vsys_kms=vsys_kms, vwidth_kms=vwidth_kms,
                        method=method, target_chan_ghz=this_target_chan_ghz,
                        nchan=this_nchan, binfactor=this_binfactor,
                        require_full_cont_coverage=require_full_cont_coverage,
                        overwrite=overwrite)

                    # Deal with pointing table - testing shows it to be a
                    # duplicate for each SPW here, so we remove all rows for
                    # all SPWs except the first one.

                    if clear_pointing:
                        clear_pointing_table(this_outfile)

    # Concatenate and combine the output data sets

    concat_ms(
        infile_list=split_file_list, outfile=outfile, freqtol=freqtol,
        overwrite=overwrite, copypointing=(not clear_pointing))

    # Clean up, deleting intermediate files

    for this_file in split_file_list:
        shutil.rmtree(this_file)

    # Remove touch file to mark that we are have done the processing of this data
    if os.path.isdir(outfile+'.touch'):
        os.rmdir(outfile+'.touch')

    return()


########################################
# Extract a continuum measurement set. #
########################################


def extract_continuum(
        infile=None, outfile=None, spw=None,
        range_to_extract=None, method='regrid_then_rebin',
        target_chan_ghz=None, binfactor=None, nchan=None, lines_to_flag=None,
        vlow_kms=None, vhigh_kms=None, vsys_kms=None, vwidth_kms=None,
        require_full_cont_coverage=False, overwrite=False):
    """
    Continuum extraction routine. Takes infile, outfile, ranges of interest,
    lines to flag, and algorithm, along with algorithm tuning parameters.
    """

    # Check the method
    valid_methods = [
        'regrid_then_rebin', 'rebin_then_regrid', 'just_regrid', 'just_rebin']
    if method.lower().strip() not in valid_methods:
        logger.error("Not a valid continuum extraction method - "+str(method))
        raise Exception("Please specify a valid continuum extraction method.")

    # Check input
    if infile is None:
        logging.error("Please specify an input file.")
        raise Exception("Please specify an input file.")

    if not os.path.isdir(infile):
        logger.error(
            'The input measurement set "'+infile+'"does not exist.')
        raise Exception(
            'The input measurement set "'+infile+'"does not exist.')

    if outfile is None:
        logging.error("Please specify an output file.")
        raise Exception("Please specify an output file.")

    # Check existing output data and abort if found and overwrite is off
    if os.path.isdir(outfile) and not os.path.isdir(outfile+'.touch'):
        if not overwrite:
            logger.warning(
                'Found existing output data "'+outfile +
                '", will not overwrite it.')
            return()

    # Else, clear all previous files and temporary files
    for suffix in ['', '.flagversions', '.touch', '.temp*']:
        for temp_outfile in glob.glob(outfile+suffix):
            if os.path.isdir(temp_outfile):
                logger.debug('... shutil.rmtree(%r)' % (temp_outfile))
                shutil.rmtree(temp_outfile)

    # Create touch file to mark that we are processing this data
    if not os.path.isdir(outfile+'.touch'):
        os.mkdir(outfile+'.touch')

    # Make copy of the input data
    shutil.copytree(infile, infile+'.temp_copy')

    # Evaluate line flagging logic. Use line and velocity data to generate
    # frequency ranges to flag.
    spw_flagging_string = ''
    if lines_to_flag is not None:
        vsys_method = (vsys_kms is not None) and (vwidth_kms is not None)
        vlow_method = (vlow_kms is not None) and (vhigh_kms is not None)
        if vsys_method or vlow_method:
            ranges_to_exclude = lines.get_ghz_range_for_list(
                line_list=lines_to_flag,
                vsys_kms=vsys_kms, vwidth_kms=vwidth_kms,
                vlow_kms=vlow_kms, vhigh_kms=vhigh_kms)
            spw_flagging_string = spw_string_for_freq_ranges(
                infile=infile, freq_ranges_ghz=ranges_to_exclude)

    # Flag the relevant line-affected channels.
    if spw_flagging_string != '':
        casaStuff.flagdata(
            vis=infile+'.temp_copy', spw=spw_flagging_string)

    if range_to_extract is None:

        # Collapse the SPW down to a single channel

        logger.info("... Collapsing each continuum SPW to a single channel.")

        if not os.path.isdir(outfile + '.touch'):
            os.mkdir(outfile + '.touch')

        mytb = casaStuff.tbtool()
        mytb.open(infile + '.temp_copy', nomodify=True)
        colnames = mytb.colnames()
        if 'CORRECTED_DATA' in colnames:
            logger.info("... Data has a CORRECTED column. Will use that.")
            datacolumn = 'CORRECTED'
        else:
            logger.info("... Data lacks a CORRECTED column. Will use DATA column.")
            datacolumn = 'DATA'
        mytb.close()

        casaStuff.split(vis=infile + '.temp_copy',
                        outputvis=outfile,
                        width=100000,
                        datacolumn=datacolumn,
                        keepflags=False)

    else:

        # Calculate reference frequencies and nominal velocity ranges
        reffreq_ghz = np.mean(range_to_extract)
        refvsys_kms = 0.0
        refvwidth_kms = \
            abs(np.diff(range_to_extract).item()) / reffreq_ghz * sol_kms

        # ... if no SPW selection string is provided then note whether we
        # have multiple windows.
        if spw is None:
            spw_list = find_spws_for_line(
                infile=infile+'.temp_copy', restfreq_ghz=reffreq_ghz,
                vsys_kms=refvsys_kms, vwidth_kms=refvwidth_kms,
                require_full_line_coverage=require_full_cont_coverage,
                require_data=True, exit_on_error=True, as_list=True)
            if spw_list is None or len(spw_list) == 0:
                logging.error(
                    "No SPWs for a selected frequency range: "
                    "{}--{}".format(*range_to_extract))
                return()
            spw = spw_list.join(',')
        else:
            spw_list = spw.split(',')

        multiple_spws = len(spw_list) > 1

        # ............................................
        # Initialize the calls
        # ............................................

        if method in ('just_regrid', 'regrid_then_rebin', 'rebin_then_regrid'):

            if target_chan_ghz is None:
                logger.error(
                    'Need a target channel width to enable regridding.')
                raise Exception(
                    "Need a target channel width to enable regridding.")

            target_chan_kms = target_chan_ghz / reffreq_ghz * sol_kms
            if nchan is None:
                nchan = int(np.floor(refvwidth_kms / target_chan_kms))
            refvstart_kms = refvsys_kms - refvwidth_kms/2. + target_chan_kms/2.

            regrid_params, regrid_msg = build_mstransform_call(
                infile=infile+'.temp_copy', outfile=outfile,
                restfreq_ghz=reffreq_ghz, spw=spw,
                method='regrid', vstart_kms=refvstart_kms,
                target_chan_kms=target_chan_kms, nchan=nchan,
                require_full_line_coverage=require_full_cont_coverage)

        if method in ('just_rebin', 'regrid_then_rebin', 'rebin_then_regrid'):

            if binfactor is None:
                logger.warning('Need a bin factor to enable rebinning.')
                return()

            rebin_params, rebin_msg = build_mstransform_call(
                infile=infile+'.temp_copy', outfile=outfile,
                restfreq_ghz=reffreq_ghz, spw=spw,
                method='rebin', binfactor=binfactor,
                require_full_line_coverage=require_full_cont_coverage)

        if multiple_spws:

            combine_params, combine_msg = build_mstransform_call(
                infile=infile+'.temp_copy', outfile=outfile,
                restfreq_ghz=reffreq_ghz, spw=spw,
                method='combine',
                require_full_line_coverage=require_full_cont_coverage)

        # ............................................
        # string the calls together in the desired order
        # ............................................

        params_list = []
        msg_list = []

        if method == 'just_regrid':
            params_list.append(regrid_params)
            msg_list.append(regrid_msg)
        elif method == 'just_rebin':
            params_list.append(rebin_params)
            msg_list.append(rebin_msg)
        elif method == 'rebin_then_regrid':
            params_list.append(rebin_params)
            msg_list.append(rebin_msg)
            params_list.append(regrid_params)
            msg_list.append(regrid_msg)
        else:
            params_list.append(regrid_params)
            msg_list.append(regrid_msg)
            params_list.append(rebin_params)
            msg_list.append(rebin_msg)

        if multiple_spws:
            params_list.append(combine_params)
            msg_list.append(combine_msg)

        # ............................................
        # Execute the list of mstransform calls
        # ............................................

        n_calls = len(params_list)
        logger.info('... we will have '+str(n_calls)+' mstransform calls')

        for kk, (this_params, this_msg) in enumerate(zip(
                params_list, msg_list)):

            if kk == 0:
                this_params['vis'] = infile+'.temp_copy'
                this_params['outputvis'] = \
                    outfile+'.temp{:.0f}'.format(kk+1)
            elif kk == n_calls-1:
                this_params['vis'] = \
                    outfile+'.temp{:.0f}'.format(kk)
                this_params['outputvis'] = outfile
            else:
                this_params['vis'] = \
                    outfile+'.temp{:.0f}'.format(kk)
                this_params['outputvis'] = \
                    outfile+'.temp{:.0f}'.format(kk+1)

            if os.path.isdir(this_params['outputvis']):
                shutil.rmtree(this_params['outputvis'])

            logger.info("... "+this_msg)

            if kk > 0:
                this_params['spw'] = ''
                this_params['datacolumn'] = 'DATA'

            logger.info(
                "... running CASA "+'mstransform(' +
                ', '.join("{!s}={!r}".format(
                    t, this_params[t]) for t in this_params.keys()) +
                ')')

            if not os.path.isdir(this_params['outputvis']+'.touch'):
                # mark the beginning of our processing
                os.mkdir(this_params['outputvis']+'.touch')

            casaStuff.mstransform(**this_params)

            if os.path.isdir(this_params['outputvis']+'.touch') and \
               this_params['outputvis'] != outfile:
                # mark the end of our processing
                os.rmdir(this_params['outputvis']+'.touch')

    # ............................................
    # Clean up leftover files
    # ............................................

    # TBD revisit
    if os.path.isdir(outfile):
        logger.info("... deleting temporary files")
        for suffix in ['.temp_copy', '.temp_copy.flagversions']:
            if os.path.isdir(infile+suffix):
                shutil.rmtree(infile+suffix)
        if range_to_extract is not None:
            for kk in range(n_calls):
                for suffix in [
                        '.temp%d' % (kk),
                        '.temp%d.flagversions' % (kk),
                        '.temp%d.touch' % (kk)]:
                    if os.path.isdir(outfile+suffix):
                        shutil.rmtree(outfile+suffix)

    # Remove touch file to mark that we have done the processing
    if os.path.isdir(outfile+'.touch'):
        os.rmdir(outfile+'.touch')

    return()


##################
# Analysis tasks #
##################


def noise_spectrum(
        vis=None, stat_name="medabsdevmed", start_chan=None, stop_chan=None):
    """
    Calculates the u-v based noise spectrum and returns it as an array.
    """

    # This function is not used for now.

    if vis is None:
        return None

    # Note the number of channels in SPW 0
    spw_info = get_spw_info(vis)

    nchan = spw_info[0]['numChannels']
    spec = np.zeros(nchan)
    for ii in range(nchan):
        if start_chan is not None:
            if ii < start_chan:
                continue
        if stop_chan is not None:
            if ii > stop_chan:
                continue
        logger.debug("Channel "+str(ii)+" / "+str(nchan))
        result = casaStuff.visstat(
            vis=vis, axis='amp', spw='0:'+str(ii))
        if result is None:
            logger.debug("Skipping channel.")
            continue
        spec[ii] = result[result.keys()[0]][stat_name]

    return spec

def estimate_mrs(
    vis: str,
    baseline_percentile: float = 5,
    mrs_factor: float = 0.983,
    use_first_field: bool = True,
    chunk_size: int = 100_000,
) -> dict:
    """Estimate the MRS for a given measurement set.

    This function is specifically designed to sidestep the biases that can arise
    from concatenating measurement sets. It calculates some minimum baseline percentile
    from each unique observation ID, and then uses the minimum of these to calculate the MRS.
    Because including 12m data to your 7m dataset shouldn't shrink the MRS, right?

    This is set up to by default use the working equation in the ALMA handbook.

    Args:
        vis (str): Path to the measurement set.
        baseline_percentile (float, optional): The percentile of the baseline distribution to use.
            Defaults to 5.
        mrs_factor (float, optional): Factor to multiply the MRS by. Defaults to
            0.983.
        use_first_field (bool, optional): If True, will just use the first field of each observation
            ID for the calculation. Speeds things up and requires less loading of data into memory.
            Defaults to True.
        chunk_size (int, optional): Number of rows to process in each chunk, to limit memory usage.
            Defaults to 100,000.

    Returns:
        dict: Dictionary containing the representative frequency, baseline for MRS,
            the MRS in arcseconds, and the individual calculations for each observation ID.
    """

    rep_freq = get_representative_freq(vis)
    logger.debug(f"Representative frequency: {rep_freq}")

    # Get a list of the observation IDs
    obs_ids = get_obs_ids(vis=vis)
    logger.debug(f"Found {len(obs_ids)} observation IDs")

    tb = casaStuff.tbtool()
    tb.open(vis)

    # Set up a dictionary to hold all the calculations
    result = {
        "representative_frequency": rep_freq.to(u.GHz).value,
        "baseline_for_mrs_per_obs_id": {},
        "mrs_per_obs_id": {},
    }

    for obs_id in obs_ids:

        # Downselect on observation ID
        subset_conditions = [
            f"OBSERVATION_ID == {obs_id}",
        ]
        tb_subset = tb.query(" && ".join(subset_conditions))

        # If we're only using first field, get the first field ID for further downselecting
        if use_first_field:
            field_id = tb_subset.getcell("FIELD_ID", 0)
            subset_conditions.append(f"FIELD_ID == {field_id}")

        tb_subset.close()

        # Get UV distances in meters
        uv_distance_m = get_uv_distance_m(
            tb=tb,
            subset_conditions=subset_conditions,
            chunk_size=chunk_size,
        )

        # Now we loop over, take the 5th percentile baseline
        baseline_percentiles = []
        bp = np.nanpercentile(uv_distance_m, baseline_percentile)
        baseline_percentiles.append(bp)

        # Take the minimum of these baseline percentiles to calculate MRS
        baseline_for_mrs = np.nanmin(baseline_percentiles) * u.m

        # Calculate the MRS in arcsec
        mrs = (
            mrs_factor
            * const.c.to(u.m * u.Hz)
            / (rep_freq.to(u.Hz) * baseline_for_mrs.to(u.m))
            * u.rad
        )
        mrs = mrs.to(u.arcsec).value

        result["baseline_for_mrs_per_obs_id"][obs_id] = baseline_for_mrs.to(u.m).value
        result["mrs_per_obs_id"][obs_id] = mrs

        logger.debug(f"Calculated MRS of {mrs} for obs ID {obs_id}")

    tb.close()

    # Finally, get the minimum baseline/maximum MRS from this to return
    result["baseline_for_mrs"] = np.nanmin([r[-1] for r in result["baseline_for_mrs_per_obs_id"].items()])
    result["mrs"] = np.nanmax([r[-1] for r in result["mrs_per_obs_id"].items()])

    return result

def estimate_synthesised_beam(
    vis: str,
    baseline_percentile: float = 80.0,
    beam_factor: float = 0.574,
    use_first_field: bool = True,
    chunk_size: int = 100_000,
) -> float:
    """
    Estimate the synthesised beam for a measurement set.

    This function is specifically designed to sidestep the biases that can arise
    from concatenating measurement sets. It calculates some minimum baseline percentile
    from each unique observation ID, and then uses the minimum of these to for the beam size.
    By default, this function uses 0.574 * 80th baseline percentile, to match the analysisUtils default.

    Args:
        vis (str): Path to measurement set
        baseline_percentile (float): The percentile of the baseline distribution to use.
            Defaults to 80.
        beam_factor (float): Factor to convert from baseline percentile to synthesised beam.
            Defaults to 0.574.
        use_first_field (bool): Whether to only use the first field.
            Defaults to True.
        chunk_size (int): Number of rows to process in each chunk, to limit memory usage.
            Defaults to 100,000.

    Returns:
        float: The estimated synthesised beam in arcseconds
    """

    # Get the representative frequency
    rep_freq = get_representative_freq(vis)
    logger.debug(f"Representative frequency: {rep_freq}")

    # Get a list of the observation IDs
    obs_ids = get_obs_ids(vis=vis)
    logger.debug(f"Found {len(obs_ids)} observation IDs")

    tb = casaStuff.tbtool()
    tb.open(vis)

    # Set up a dictionary to hold all the calculations
    result = {
        "representative_frequency": rep_freq.to(u.GHz).value,
        "synthesised_beam_per_obs_id": {},
    }

    for obs_id in obs_ids:

        # Downselect on observation ID
        subset_conditions = [
            f"OBSERVATION_ID == {obs_id}",
        ]
        tb_subset = tb.query(" && ".join(subset_conditions))

        # If we're only using first field, get the first field ID for further downselecting
        if use_first_field:
            field_id = tb_subset.getcell("FIELD_ID", 0)
            subset_conditions.append(f"FIELD_ID == {field_id}")

        tb_subset.close()

        # Get UV distances in meters
        uv_distance_m = get_uv_distance_m(
            tb=tb,
            subset_conditions=subset_conditions,
            chunk_size=chunk_size,
        )
        baseline_for_beam = np.nanpercentile(uv_distance_m, baseline_percentile) * u.m

        # Calculate the synthesised beam in arcsec
        synthesised_beam = (
            beam_factor
            * const.c.to(u.m * u.Hz)
            / (rep_freq.to(u.Hz) * baseline_for_beam.to(u.m))
            * u.rad
        )
        synthesised_beam = synthesised_beam.to(u.arcsec).value

        result["synthesised_beam_per_obs_id"][obs_id] = synthesised_beam

        logger.debug(f"Calculated synthesised beam of {synthesised_beam} for obs ID {obs_id}")

    tb.close()

    # The synthesised beam is the minimum of the synthesised beams from each observation ID
    synthesised_beam = np.nanmin([r[-1] for r in result["synthesised_beam_per_obs_id"].items()])

    return synthesised_beam


def pick_cell_and_im_size(
    vis: str,
    npix: float = 5.0,
    baseline_percentile: float = 80.0,
    cellstring: bool = False,
    roundcell: int = 2,
    pblevel: float = 0.2,
    beam_size_factor: float = 0.574,
    beam_size_use_first_field: bool = True,
    beam_size_chunk_size: int = 100_000,
) -> tuple[float | str, list[int]]:
    """Pick a cell size and image size for a measurement set.

    Args:
        vis (str): Path to measurement set.
        npix (float): Number of pixels across the synthesised beam.
            Defaults to 5.0.
        baseline_percentile (float): The percentile of the baseline distribution to use for estimating the synthesised
            beam. Defaults to 80.0.
        cellstring (bool): Whether to return the cell size as a string with units.
            Defaults to False.
        roundcell (int): Number of decimal places to round the cell size to.
            Defaults to 2.
        pblevel (float): The primary beam level for calculating image size.
            Defaults to 0.2.
        beam_size_factor (float): Factor to convert from baseline percentile to synthesised beam.
            Defaults to 0.574.
        beam_size_use_first_field (bool): Whether to only use the first field for estimating the synthesised beam.
            Defaults to True.
        beam_size_chunk_size (int): Number of rows to process in each chunk when estimating the synthesised beam.
            Defaults to 100,000.

    Returns:
        float or str, list: Tuple of the cell size (potentially as a string) and an associated list of imsize.
    """

    # Get cell size from synthesised beam and oversample factor
    synthesised_beam = estimate_synthesised_beam(
        vis=vis,
        baseline_percentile=baseline_percentile,
        beam_factor=beam_size_factor,
        use_first_field=beam_size_use_first_field,
        chunk_size=beam_size_chunk_size,
    )
    cell_size = synthesised_beam / npix

    # If selected, round to requested number of significant figures
    if roundcell > 0:
        cell_size = float(f"{cell_size:.{roundcell}g}")

    # Convert to a cell size with units
    cell_size_unit = cell_size * u.arcsec
        
    if cellstring:
        cell_size = f"{cell_size}arcsec"

    # Get dish diameter for image size calculation. Use max to get minimum FOV
    tb = casaStuff.table()
    tb.open(os.path.join(f"{vis}/ANTENNA"))
    dish_dia = tb.getcol("DISH_DIAMETER").max()
    dish_dia = dish_dia * u.m
    tb.close()

    # Get representative frequency
    rep_freq = get_representative_freq(vis)

    # Get the source name from the MS metadata
    msmd = casaStuff.msmdtool()
    msmd.open(vis)

    # Get a list of all science field IDs, turn into a source name by taking the first
    # in the MS
    intents = msmd.intents()
    field_ids = msmd.fieldsforintent(intents[0])
    sourcename = msmd.namesforfields(field_ids)[0]
    logger.debug(f"Using source name {sourcename} for image size calculation")
    msmd.close()
    
    tb = casaStuff.table()
    tb.open(f"{vis}/FIELD")
    tb_subset = tb.query(f"NAME == '{sourcename}'")
    phase_dirs = tb_subset.getcol("PHASE_DIR")

    tb_subset.close()
    tb.close()

    # Extract RA and Dec arrays (in radians)
    ra_fields = phase_dirs[0, 0, :] * u.rad
    dec_fields = phase_dirs[1, 0, :] * u.rad
    
    # Get primary beam FWHM
    fwhm = (1.14 * 1.22 * const.c.to(u.m * u.Hz) / rep_freq.to(u.Hz) / dish_dia) * u.rad
    
    # Apply the PB cutoff
    radius_pb = (fwhm / 2.0) * np.sqrt(np.log(pblevel) / np.log(0.5))

    ra_mins = ra_fields - radius_pb
    ra_maxs = ra_fields + radius_pb
    dec_mins = dec_fields - radius_pb
    dec_maxs = dec_fields + radius_pb

    ra_extent = ra_maxs.max() - ra_mins.min()
    dec_extent = dec_maxs.max() - dec_mins.min()

    x_extent = ra_extent / cell_size_unit
    y_extent = dec_extent / cell_size_unit

    # Round this up to a good FFT number for the FFT,
    # cast to int
    x_extent = int(1.2 * x_extent)
    y_extent = int(1.2 * y_extent)

    su = casaStuff.synthesisutils()
    x_extent = int(su.getOptimumSize(x_extent))
    y_extent = int(su.getOptimumSize(y_extent))
    su.done()

    im_size = [x_extent, y_extent]

    return cell_size, im_size

def get_representative_freq(
        vis: str,
) -> u.Quantity:
    """Get the representative frequency for a measurement set

    Will either pull this out from ASDM_SBSUMMARY (quick) if that
    table exists, or fall back to calculating from the channel frequencies
    in SPECTRAL_WINDOW.

    Args:
        vis (str): Path to measurement set

    Returns:
        u.Quantity: The representative frequency
    """

    tb = casaStuff.tbtool()

    # If we have a representative frequency in the table, just use that
    if os.path.exists(vis + "/ASDM_SBSUMMARY"):
        
        logger.debug("Using ASDM_SBSUMMARY to calculate representative frequency")
        
        tb.open(vis + "/ASDM_SBSUMMARY")

        colnames = tb.colnames()

        # The representative frequency can have different names, so loop
        # over the possibilities until we find it
        for frequency_row_name in ["frequency", "representativeFrequency"]:
            if frequency_row_name in colnames:

                frequencies = np.asarray(
                    tb.getcol("frequency"),
                    dtype=float,
                )
                rep_freq = frequencies[0]

                # Convert to units of GHz
                rep_freq = rep_freq * u.GHz

                break
        else:
            raise KeyError("Could not find representative frequency within table")

    # Otherwise, obtain representative frequency from the average of the frequencies.
    else:
        logger.debug("Using SPECTRAL_WINDOW to calculate representative frequency")
        
        tb.open(vis + "/SPECTRAL_WINDOW")
        frequencies = np.asarray(tb.getcell("CHAN_FREQ"), dtype=float)
        frequencies = frequencies[np.isfinite(frequencies) & (frequencies > 0)]
        rep_freq = np.median(frequencies) * u.Hz
    tb.close()

    return rep_freq


def get_obs_ids(vis):
    """Get a list of the observation IDs from a MS

    Args:
        vis (str): Path to measurement set.

    Returns:
        list: List of the observation ID numbers
    """

    tb = casaStuff.tbtool()
    tb.open(vis + "/OBSERVATION")
    nrows = tb.nrows()
    tb.close()

    obs_ids = list(range(nrows))

    return obs_ids


def get_chan_widths(
    vis: str,
    spw: str | int | list = "",
    velocity: bool = False,
) -> np.ndarray:
    """Get channel widths for a given measurement set and SPW selection.

    Args:
        vis (str): Path to measurement set.
        spw (str | int | list, optional): SPW selection. Can be a string, integer, or list of integers.
            Defaults to "" (all SPWs).
        velocity (bool, optional): If True, return channel widths in km/s.
            Defaults to False.

    Returns:
        np.ndarray: Array of channel widths for the selected SPWs. If velocity is True,
            the widths are in velocity units; otherwise, they are in Hz.
    """

    if not os.path.exists(vis):
        raise FileNotFoundError(f"Measurement set '{vis}' not found.")

    msmd = casaStuff.msmdtool()
    msmd.open(vis)

    # If an empty string or list is passed, selected all SPWs
    if spw in ["", []]:
        spw = list(range(msmd.nspw()))

    # If we still have a string, turn to int
    if isinstance(spw, str):
        spw = [int(i) for i in spw.split(",")]

    # Make sure spw is a list
    if not isinstance(spw, list):
        spw = [spw]

    chan_widths = []
    mean_freqs = []
    for s in spw:
        chan_width = np.median(msmd.chanwidths(s))
        mean_freq = msmd.meanfreq(s)

        chan_widths.append(chan_width)
        mean_freqs.append(mean_freq)

    chan_widths = chan_widths * u.Hz

    # Convert to velocity units if requested
    if velocity:
        mean_freqs = mean_freqs * u.Hz
        chan_widths *= const.c.to(u.m * u.Hz) / mean_freqs
        chan_widths = chan_widths.to(u.km / u.s).value
    else:
        chan_widths = chan_widths.to(u.Hz).value

    msmd.close()

    return chan_widths

def get_uv_distance_m(
    tb: casaStuff.tbtool,
    subset_conditions: list,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Get the UV distance in meters for a given table and subset conditions.

    Args:
        tb (casaStuff.tbtool): The CASA table tool object.
        subset_conditions (list): List of conditions to subset the table.
        chunk_size (int): Number of rows to process in each chunk, to limit memory usage.
        
    Returns:
        np.ndarray: Array of UV distances in meters.
    """

    tb_subset = tb.query(" && ".join(subset_conditions))
    total_rows = tb_subset.nrows()

    uv_distance_m = []

    # Loop over in chunks to reduce memory cost
    for startrow in range(0, total_rows, chunk_size):
        nrow = min(chunk_size, total_rows - startrow)

        uvw = np.asarray(
            tb_subset.getcol(
                "UVW",
                startrow=startrow,
                nrow=nrow,
                rowincr=1,
            )
        )
        antenna1 = np.asarray(
            tb_subset.getcol(
                "ANTENNA1",
                startrow=startrow,
                nrow=nrow,
                rowincr=1,
            )
        )
        antenna2 = np.asarray(
            tb_subset.getcol(
                "ANTENNA2",
                startrow=startrow,
                nrow=nrow,
                rowincr=1,
            )
        )
        ddid = np.asarray(
            tb_subset.getcol(
                "DATA_DESC_ID",
                startrow=startrow,
                nrow=nrow,
                rowincr=1,
            )
        )
        flag_row = np.asarray(
            tb_subset.getcol(
                "FLAG_ROW",
                startrow=startrow,
                nrow=nrow,
                rowincr=1,
            ),
            dtype=bool,
        )
        flags = np.asarray(
            tb_subset.getcol(
                "FLAG",
                startrow=startrow,
                nrow=nrow,
                rowincr=1,
            ),
            dtype=bool,
        )

        # CASA normally returns UVW as (3, nrow).
        if uvw.shape[0] != 3 and uvw.shape[-1] == 3:
            uvw = uvw.T

        uvdm = np.hypot(uvw[0], uvw[1])

        # FLAG normally has dimensions (ncorr, nchan, nrow).
        # Keep a row if at least one correlation/channel is unflagged.
        flag_axes = tuple(range(flags.ndim - 1))
        completely_flagged = np.all(flags, axis=flag_axes)

        valid = (
            ~flag_row
            & ~completely_flagged
            & (antenna1 != antenna2)
            & np.isfinite(uvdm)
            & (uvdm > 0)
            & (ddid >= 0)
        )

        # Only take valid values
        uv_distance_m.extend(uvdm[valid])

    tb_subset.close()

    uv_distance_m = np.asarray(uv_distance_m)

    return uv_distance_m

def get_spw_info(
    vis: str,
    ignore_wvr: bool = True,
) -> dict:
    """Get SPW information from a measurement set.

    Args:
        vis (str): Path to measurement set.
        ignore_wvr (bool): Whether to ignore WVR SPWs. Defaults to True.

    Returns:
        dict: Dictionary containing SPW information, including bandwidth, channel frequencies,
            channel width, edge channels, sideband, mean frequency, and number of channels.
    """

    spw_info = {}

    mytb = casaStuff.tbtool()
    mytb.open(f"{vis}/SPECTRAL_WINDOW")

    # Keep track of number of rows in the table
    nrows = range(mytb.nrows())

    for i in nrows:

        # Get out useful info
        bandwidth = mytb.getcell("TOTAL_BANDWIDTH", i)
        chan_freqs = mytb.getcell("CHAN_FREQ", i)
        min_freq = min(chan_freqs)
        max_freq = max(chan_freqs)
        mean_freq = chan_freqs.mean()
        num_channels = chan_freqs.shape[0]
        chan_width = mytb.getcell("CHAN_WIDTH", i)[0]
        net_sideband = mytb.getcell("NET_SIDEBAND", i)

        # Put this all into a dictionary
        spw_info[i] = {}
        spw_info[i]["bandwidth"] = bandwidth
        spw_info[i]["chanFreqs"] = chan_freqs
        spw_info[i]["chanWidth"] = chan_width
        spw_info[i]["edgeChannels"] = [min_freq, max_freq]
        if net_sideband == 2:
            spw_info[i]["sideband"] = 1
        else:
            spw_info[i]["sideband"] = -1
        spw_info[i]["meanFreq"] = mean_freq
        spw_info[i]["numChannels"] = num_channels

        # If ignoring WVR, then remove and log this
        if ignore_wvr and (num_channels == 4):
            logger.debug(f"Ignoring spectral window {i} because it is WVR related")
            spw_info.pop(i)

    mytb.close()

    return spw_info

def get_scans_for_spw(
        vis: str,
) -> dict:
    """Get the unique scan numbers for each spectral window in a measurement set.

    Args:
        vis (str): Path to measurement set.

    Returns:
        dict: Dictionary mapping SPW IDs to unique scan numbers.
    """

    # Get out SPWs for each data description ID
    tb = casaStuff.tbtool()
    tb.open(f"{vis}/DATA_DESCRIPTION")
    spw_for_data_desc_id = tb.getcol("SPECTRAL_WINDOW_ID")
    tb.close()

    tb = casaStuff.tbtool()
    tb.open(vis)

    data_desc_id = tb.getcol('DATA_DESC_ID')
    scans = tb.getcol('SCAN_NUMBER')

    scans_for_spw = {}
    for i in spw_for_data_desc_id:
        spw = spw_for_data_desc_id[i]
        indices = np.where(data_desc_id == i)
        scans_for_spw[spw] = np.unique(scans[indices])
        
    tb.close()

    return scans_for_spw

def get_science_spws(
    vis: str,
    intent: str = "OBSERVE_TARGET#ON_SOURCE",
    return_string: bool = True,
    return_list_of_strings: bool = False,
    return_freq_ranges: bool = False,
    tdm: bool = True,
    fdm: bool = True,
    sqld: bool = False,
    chavg: bool = False,
) -> str | list[int | str] | dict:
    """ Get science SPWs from a measurement set based.

    Will select SPWs based on specified intent. For ALMA data,
    it also can ignore channel-averaged and SQLD SPWs.

    Args:
        vis (str): Path to measurement set.
        intent (str): The intent to filter by.
            Defaults to "OBSERVE_TARGET#ON_SOURCE".
        return_string (bool): If True, return a comma-separated string of SPWs.
            Defaults to True.
        return_list_of_strings (bool): If True, return a list of SPWs as strings.
            Defaults to False.
        return_freq_ranges (bool): If True, return a dictionary with frequency ranges.
            Defaults to False.
        tdm (bool): If True, include TDM SPWs.
            Defaults to True.
        fdm (bool): If True, include FDM SPWs.
            Defaults to True.
        sqld (bool): If True, include SQLD SPWs.
            Defaults to False.
        chavg (bool): If True, include channel-averaged SPWs.
            Defaults to False.

    Returns:
        str: Comma-separated string of science SPWs if return_string is True.
        list[int]: List of science SPWs as integers if return_list_of_strings is False.
        list[str]: List of science SPWs as strings if return_list_of_strings is True.
        dict: Dictionary of science SPWs with frequency ranges if return_freq_ranges is True.
    """

    if return_string and return_list_of_strings:
        raise ValueError("You can only specify one of: return_string, return_list_of_strings")

    msmd = casaStuff.msmdtool()
    msmd.open(vis)

    all_intents = msmd.intents()

    if intent not in all_intents and intent != "":
        for i in all_intents:
            if i.find(intent) >= 0:
                intent = i
                logger.debug(f"Translated intent to {i}")
                break

    # Minimum match OBSERVE_TARGET to OBSERVE_TARGET#UNSPECIFIED
    value = [i.find(intent.replace("*", "")) for i in all_intents]

    # If any intent gives a match, the mean value of the location list will be > -1
    if np.mean(value) == -1 and intent != "":
        logger.warning(f"{intent} not found in this dataset. Available intents: {all_intents}")

        if return_string:
            science_spws = ""
        else:
            science_spws = []

    else:
        # If we don't have an intent, match on wildcards
        if intent == "":
            intent = "*"

        # Get SPWs for intent, cast back to int
        spws = msmd.spwsforintent(intent)
        spws = [int(i) for i in spws]

        # Get observatory name, if ALMA data then perform further filtering
        observatory_name = get_observatory_name(vis)

        if observatory_name.find("ALMA") >= 0 or observatory_name.find("OSF") >= 0:

            logger.debug(
                f"Calling almaspws with:\n  chavg={chavg}\n  tdm={tdm}\n  fdm={fdm}\n  sqld={sqld}"
            )
            alma_spws = msmd.almaspws(
                chavg=chavg,
                tdm=tdm,
                fdm=fdm,
                sqld=sqld,
            )
            alma_spws = [int(i) for i in alma_spws]
            
            if chavg and not sqld:
                sqld_spws = msmd.almaspws(sqld=True)
                sqld_spws = [int(i) for i in sqld_spws]

                logger.debug(f"removing SQLD ({sqld_spws}) from list")
                alma_spws = list(set(alma_spws) - set(sqld_spws))
            if len(spws) == 0 or len(alma_spws) == 0:
                science_spws = []
            else:
                science_spws = [int(i) for i in np.intersect1d(spws, alma_spws)]
        else:
            science_spws = spws

        science_spws_dict = {}
        for spw in science_spws:
            science_spws_dict[spw] = sorted([msmd.chanfreqs(spw)[0], msmd.chanfreqs(spw)[-1]])

        msmd.close()

        if return_freq_ranges:
            science_spws = science_spws_dict
        if return_string:
            science_spws = ",".join(str(i) for i in science_spws)
        elif return_list_of_strings:
            science_spws = list([str(i) for i in science_spws])
        else:
            science_spws = list(science_spws)

    return science_spws

def get_observatory_name(
        vis: str,
) -> str:
    """Get observatory name for given measurement set.

    Args:
        vis (str): The path to the measurement set.

    Returns:
        str: The observatory name, if found; otherwise, an empty string.
    """

    tb = casaStuff.tbtool()
    tb.open(f"{vis}/OBSERVATION")

    try:
        observatory_name = tb.getcell("TELESCOPE_NAME")
    except RuntimeError:
        observatory_name = ""

    tb.close()

    return observatory_name

def clear_pointing_table(
    vis: str,
):
    """Removes all rows from the POINTING table of a measurement set.

    Args:
        vis (str): Path to the measurement set.

    Returns:
        bool: True if the POINTING table was cleared successfully.
    """

    tb = casaStuff.tbtool()
    tb.open(
        f"{vis}/POINTING",
        nomodify=False,
    )
    row_numbers = tb.rownumbers()
    tb.removerows(row_numbers)
    tb.close()

    logger.info(f"Cleared the POINTING table for {vis}")

    return True
