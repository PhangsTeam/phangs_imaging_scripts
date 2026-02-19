"""
Standalone routines to analyze and manipulate visibilities.
"""

import glob
import logging
import os
import shutil

import analysisUtils as au
import numpy as np
from packaging import version
from scipy.ndimage import label

from . import casaStuff
from . import utilsLines as lines

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

    mytb = au.createCasaTool(casaStuff.tbtool)
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

    logger.debug("... vm = au.ValueMapping(infile) ...")
    vm = au.ValueMapping(infile)
    logger.debug("... vm = au.ValueMapping(infile) done")

    for this_spw in vm.spwInfo.keys():

        spw_high_ghz = np.max(vm.spwInfo[this_spw]['edgeChannels'])/1e9
        spw_low_ghz = np.min(vm.spwInfo[this_spw]['edgeChannels'])/1e9
        logger.debug(
            "... spw: %s, freq: %.6f - %.6f GHz" %
            (this_spw, spw_low_ghz, spw_high_ghz))

        if spw_high_ghz < line_low_ghz:
            continue

        if spw_low_ghz > line_high_ghz:
            continue

        if max_chanwidth_ghz is not None:
            spw_chanwidth_ghz = abs(vm.spwInfo[this_spw]['chanWidth'])/1e9
            if spw_chanwidth_ghz > max_chanwidth_ghz:
                continue

        if require_data:
            if len(vm.scansForSpw[this_spw]) == 0:
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
    science. Mostly wraps analysisUtils rather than reinventing the
    wheel.
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

    # Call the analysisUtil version.

    spw_string = au.getScienceSpws(
        infile, intent='OBSERVE_TARGET*')
    if spw_string is None or len(spw_string) == 0:
        spw_string = au.getScienceSpws(
            infile, intent='OBSERVE_TARGET#ON_SOURCE')

    spw_list = []
    for this_spw_string in spw_string.split(','):
        spw_list.append(int(this_spw_string))

    # Shouldn't get here, I think, because of the analysisUtils logic

    if require_data:
        vm = au.ValueMapping(infile)

        for spw in spw_list:
            if len(vm.scansForSpw[spw]) == 0:
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

    vm = au.ValueMapping(infile)

    # Loop over spectral windows
    spw_flagging_string = ''
    first_string = True
    for this_spw in vm.spwInfo:

        if len(just_spw) > 0:
            if this_spw not in just_spw:
                continue

        freq_axis = vm.spwInfo[this_spw]['chanFreqs']
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

        chan_widths_hz = au.getChanWidths(this_infile, spw_list_string)

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
                    au.clearPointingTable(this_outfile)
                else:
                    copy_pointing = False
                    #logger.debug('Warning! Failed to run au.clearPointingTable(%r)'%(this_outfile))

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

    for this_infile in infile_list:

        vm = au.ValueMapping(this_infile)

        spw_list = find_spws_for_line(
            this_infile, restfreq_ghz=restfreq_ghz,
            vsys_kms=vsys_kms, vwidth_kms=vwidth_kms,
            vlow_kms=vlow_kms, vhigh_kms=vhigh_kms,
            require_data=True, as_list=True,
            require_full_line_coverage=require_full_line_coverage)

        scheme[this_infile] = {}

        for this_spw in spw_list:

            chan_width_ghz = np.abs(vm.spwInfo[this_spw]['chanWidth'])/1e9

            # Using RADIO convention:
            chan_width_kms = chan_width_ghz/restfreq_ghz * sol_kms
            # was using the old relative convention (can delete eventually)
            # chan_width_kms = chan_width_ghz/line_freq_ghz * sol_kms

            if chan_width_kms > target_chan_kms:
                logger.warning("Channel too big for SPW "+str(this_spw))
                continue

            chan_width_list.append(chan_width_kms)
            this_binfactor = int(np.floor(target_chan_kms/chan_width_kms))
            binfactor_list.append(this_binfactor)

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
        infile=None, outfile=None, spw=None, restfreq_ghz=None, line='co21',
        vlow_kms=None, vhigh_kms=None, vsys_kms=None, vwidth_kms=None,
        method='regrid_then_rebin', target_chan_kms=None, nchan=None,
        binfactor=None, require_full_line_coverage=False, overwrite=False):
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
            infile=infile, outfile=outfile, restfreq_ghz=restfreq_ghz, spw=spw,
            vstart_kms=vstart_kms, vwidth_kms=vwidth_kms,
            target_chan_kms=target_chan_kms, nchan=nchan, binfactor=binfactor,
            method='regrid',
            require_full_line_coverage=require_full_line_coverage)

    if method == 'just_rebin' or method == 'regrid_then_rebin' or \
            method == 'rebin_then_regrid':

        if binfactor is None:
            logger.warning('Need a bin factor to enable rebinning.')
            return()

        rebin_params, rebin_msg = build_mstransform_call(
            infile=infile, outfile=outfile, restfreq_ghz=restfreq_ghz, spw=spw,
            binfactor=binfactor, method='rebin',
            require_full_line_coverage=require_full_line_coverage)

    if multiple_spws:
        combine_params, combine_msg = build_mstransform_call(
            infile=infile, outfile=outfile, restfreq_ghz=restfreq_ghz, spw=spw,
            method='combine',
            require_full_line_coverage=require_full_line_coverage)

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
        infile=None, outfile=None, restfreq_ghz=None, spw=None,
        vstart_kms=None, vwidth_kms=None, datacolumn=None, method='regrid',
        target_chan_kms=None, nchan=None, binfactor=None,
        require_full_line_coverage=False, overwrite=False):
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
        mytb = au.createCasaTool(casaStuff.tbtool)
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

        max_chan_ghz = np.max(np.abs(au.getChanWidths(infile, spw)))/1e9

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

        if binfactor is not None:
            # Make sure that we won't lose anything in the rebinning stage
            add_chans = nchan % binfactor
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
        mytb = au.createCasaTool(casaStuff.tbtool)
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

        vm = au.ValueMapping(infile)

        first = True
        for this_spw in vm.spwInfo.keys():

            if edge_kms is not None:
                spw_high_ghz = np.max(vm.spwInfo[this_spw]['edgeChannels'])/1e9
                spw_low_ghz = np.min(vm.spwInfo[this_spw]['edgeChannels'])/1e9
                spw_chanwidth_ghz = abs(vm.spwInfo[this_spw]['chanWidth'])/1e9

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

            nchan = vm.spwInfo[this_spw]['numChannels']

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
                vsys_kms=vsys_kms, vwidth_kms=vwidth_kms,
                vlow_kms=vlow_kms, vhigh_kms=vhigh_kms,
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
                        au.clearPointingTable(this_outfile)

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

        mytb = au.createCasaTool(casaStuff.tbtool)
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

    vm = au.ValueMapping(vis)
    nchan = vm.spwInfo[0]['numChannels']
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
