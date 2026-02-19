"""
handlerVis (VisHandler object)

The PHANGS pipeline to handle staging and pre-processing of uv data
before imaging. Works through a single big class (the
VisHandler). This needs to be attached to a keyHandler to access the
target, product, and configuration keys and locations of the uv data.

To run the individual routines, this code needs to be run inside
CASA. See an example application inside stage_7m_co21.py .

Example:

    $ casa
    from phangsPipeline import handlerKeys as kh
    from phangsPipeline import uvdataHandler as uvh
    this_kh = kh.KeyHandler(master_key = 'config_keys/master_key.txt')
    this_uvh = uvh.VisHandler(key_handler = this_kh, dry_run = False)
    # Set which data to process
    this_uvh.set_line_products(only=['co21'])
    this_uvh.set_interf_configs(only=['12m+7m'])
    this_uvh.set_targets(only=['ngc3621'])
    # Process
    this_uvh.loop_stage_uvdata()

"""

import logging
import os

# Check casa environment by importing CASA-only packages
from .check_imports import is_casa_installed

casa_enabled = is_casa_installed()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if casa_enabled:
    logger.debug('casa_enabled = True')
else:
    logger.debug('casa_enabled = False')

if casa_enabled:

    from . import casaVisRoutines as cvr
    from . import handlerTemplate
    from . import utilsFilenames as fnames
    from . import utilsLines as lines

    class VisHandler(handlerTemplate.HandlerTemplate):
        """
        Class to manipulate calibrated ALMA visibility data (measurement
        sets), extracting lines, combining multiple data sets, and
        carrying out other steps in prepration for imaging.
        """

        ############
        # __init__ #
        ############

        def __init__(self, key_handler=None, dry_run=False):
            """
            """
            # Can't use super and keep python2/3 agnostic
            handlerTemplate.HandlerTemplate.__init__(
                self, key_handler=key_handler, dry_run=dry_run)

        # region Loops

        ######################################
        # Loop through all steps and targets #
        ######################################

        def loop_stage_uvdata(
                self,
                do_all=False,
                do_copy=False,
                do_remove_staging=False,
                do_custom=False,
                do_contsub=False,
                do_extract_line=False,
                do_extract_cont=False,
                extra_ext='',
                make_directories=True,
                statwt_line=True,
                statwt_cont=True,
                intent=None,
                timebin=None,
                just_projects=None,
                strict_config=True,
                require_full_line_coverage=False,
                require_full_cont_coverage=False,
                overwrite=False):
            """
            Loops over the full set of targets, products, and configurations
            to run the uv data processing. Toggle the parts of the loop
            using the do_XXX booleans. Other choices affect the algorithms
            used.

            The strict_config option sets whether to require that a target has data
            from ALL arrays that make up the configuration (True) or not (False).

            The require_full_line_coverage option sets whether to require a
            measurement set to completely cover a given line's frequency range
            (True) or not (False).
            """

            if make_directories:
                self._kh.make_missing_directories(imaging=True)

            if do_all:
                do_copy = True
                do_contsub = True
                do_custom = True
                do_extract_line = True
                do_extract_cont = True
                do_remove_staging = True

            target_list = self.get_targets()
            product_list = self.get_all_products()
            config_list = self.get_interf_configs()

            # Our first loop goes over the individual measurement sets,
            # splits, and continuum subtracts the data. At this stage we
            # have no knowledge of configs except that selection may
            # reduce the number of input measurement sets.

            for this_target, this_project, this_array_tag, this_obsnum in \
                    self._kh.loop_over_input_ms(
                        target=target_list,
                        config=config_list,
                        project=just_projects,
                        strict_config=strict_config):

                for this_product in product_list:

                    # Our first step uses CASA's split to extract the relevant
                    # fields and spectral windows from each input data set.

                    if do_copy:
                        self.task_split(
                            target=this_target,
                            project=this_project,
                            array_tag=this_array_tag,
                            obsnum=this_obsnum,
                            product=this_product,
                            intent=intent,
                            timebin=timebin,
                            require_full_line_coverage=require_full_line_coverage,
                            overwrite=overwrite,
                        )

                    # Run custom processing. Not currently used.

                    if do_custom:
                        pass

                    # Next we apply uv continuum subtraction. We may later offer
                    # an algorithm choice to do this before or after
                    # regridding. The correct choice depends on some details of
                    # the observation setup.

                    if this_product in self._kh.get_line_products() and do_contsub:
                        self.task_contsub(
                            target=this_target,
                            project=this_project,
                            array_tag=this_array_tag,
                            obsnum=this_obsnum,
                            product=this_product,
                            # could add algorithm flags here
                            overwrite=overwrite)

            # Now we reprocess the data to have the desired spectral
            # setup(s). This involves rebinning and regridding for line
            # products and flagging and integration for continuum
            # products. This requires cross-talk among the different
            # measurement sets.

            for this_target, this_product, this_config in \
                    self.looper(
                        do_targets=True,
                        do_products=True,
                        do_configs=True,
                        just_line=True,
                        just_interf=True):

                if strict_config:
                    # this seems like it doesn't do anything - do we
                    # actually want a test here and if we do shouldn't
                    # it be checking if this is false then
                    # continuing? In theory this is checked above.
                    self._kh.has_data_for_config(
                        target=this_target,
                        config=this_config,
                        strict=True)

                if do_extract_line:

                    if this_product in self._kh.get_line_products():
                        self.task_extract_line(
                            target=this_target,
                            config=this_config,
                            product=this_product,
                            exact=False,
                            do_statwt=statwt_line,
                            extra_ext_in="",
                            contsub="prefer",
                            # could add algorithm flags here
                            require_full_line_coverage=require_full_line_coverage,
                            overwrite=overwrite,
                            strict_config=strict_config)

            for this_target, this_product, this_config in \
                    self.looper(
                        do_targets=True,
                        do_products=True,
                        do_configs=True,
                        just_cont=True,
                        just_interf=True):

                # Same as above - check / revise
                if strict_config:
                    self._kh.has_data_for_config(
                        target=this_target,
                        config=this_config,
                        strict=True)

                if do_extract_cont:

                    if this_product in self._kh.get_continuum_products():
                        self.task_extract_continuum(
                            target=this_target,
                            product=this_product,
                            config=this_config,
                            exact=False,
                            do_statwt=statwt_cont,
                            require_full_cont_coverage=require_full_cont_coverage,
                            overwrite=overwrite,
                            strict_config=strict_config)

            # Clean up the staged measurement sets. They cost time to
            # re-split, but have a huge disk imprint and are redundant
            # with the concatenated data and original data.

            for this_target, this_project, this_array_tag, this_obsnum in \
                    self._kh.loop_over_input_ms(
                        target=target_list,
                        config=config_list,
                        project=just_projects,
                        strict_config=strict_config):

                for this_product in product_list:

                    if do_remove_staging:
                        self.remove_staged_products(
                            target=this_target,
                            project=this_project,
                            array_tag=this_array_tag,
                            obsnum=this_obsnum,
                            product=this_product,
                            strict_config=strict_config)

            return ()

        # endregion

        # region Tasks

        ##########################################
        # Tasks - individual operations on data. #
        ##########################################

        def task_split(
                self,
                target=None,
                project=None,
                array_tag=None,
                obsnum=None,
                product=None,
                intent=None,
                extra_ext_out='',
                do_statwt=False,
                timebin=None,
                use_symlink=True,
                require_full_line_coverage=False,
                overwrite=False):
            """
            Copy visibility data for one target, project, array_tag,
            obsnum combination from their original location to the imaging
            directory for the target. Then optionally split out only the
            science targets.
            """

            if target is None:
                logger.error("Please specify a target.")
                raise Exception("Please specify a target.")

            if project is None:
                logger.error("Please specify a project.")
                raise Exception("Please specify a project.")

            if array_tag is None:
                logger.error("Please specify an array_tag.")
                raise Exception("Please specify an array_tag.")

            if obsnum is None:
                logger.error("Please specify an obsnum.")
                raise Exception("Please specify an obsnum.")

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Splitting u-v data for")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            infile = self._kh.get_file_for_input_ms(
                target=target, project=project, array_tag=array_tag, obsnum=obsnum)

            logger.info("... file: " + infile)

            if infile is None:
                logger.error("Infile not found. Returning.")
                return ()

            if not os.path.isdir(infile):
                logger.error("Infile not found. Returning.")
                return ()

            field = self._kh.get_field_for_input_ms(
                target=target, project=project, array_tag=array_tag, obsnum=obsnum)
            if (field.lower()).strip() == 'all':
                field = ''

            if intent is None:
                intent='OBSERVE_TARGET*'

            outfile = fnames.get_staged_msname(
                target=target, project=project, array_tag=array_tag, obsnum=obsnum,
                product=product, ext=extra_ext_out)

            logger.info("... output: " + outfile)

            # Check existence of output data and abort if found and overwrite is off

            if os.path.isdir(outfile) and not os.path.isdir(outfile + '.touch'):
                if not overwrite:
                    logger.warning('... found existing output data "' + outfile + '", will not overwrite it.')
                    return ()

            # If the user doesn't override the time bin, get it from the
            # key handler.

            if timebin is None:
                timebin = self._kh.get_timebin_for_array_tag(array_tag=array_tag)
                logger.info("... timebin: " + str(timebin))

            # If requested, select on SPW for the product

            spw = ''
            if product is not None:

                logger.info("... product: " + str(product))

                if product in self._kh.get_line_products():

                    this_line = self._kh.get_line_tag_for_line_product(product)
                    vsys, vwidth = \
                        self._kh.get_system_velocity_and_velocity_width_for_target(
                            target, check_parent=False)
                    max_chanwidth_kms = \
                        self._kh.get_channel_width_for_line_product(product)

                    combinespw = self._kh.get_contsub_combinespw(product=product)
                    if combinespw is None:
                        combinespw = False

                    logger.info("... combinespw: " + str(combinespw))

                    if not self._dry_run:
                        if combinespw:
                            spw = cvr.find_spws_for_science(infile=infile)
                        else:
                            spw = cvr.find_spws_for_line(
                                infile=infile, line=this_line,
                                max_chanwidth_kms=max_chanwidth_kms,
                                vsys_kms=vsys, vwidth_kms=vwidth,
                                require_full_line_coverage=require_full_line_coverage)

                        if spw is None or len(spw) == 0:
                            logger.warning(
                                "... No SPWs meet the selection criteria. "
                                "Skipping.")

                            return ()

                if product in self._kh.get_continuum_products():
                    spw = cvr.find_spws_for_science(infile=infile)

            logger.info("... extracting spws :" + str(spw))

            # Change to the imaging directory for the target

            _ = self._kh.get_imaging_dir_for_target(target, changeto=True)

            if not self._dry_run:
                cvr.split_science_targets(
                    infile=infile,
                    outfile=outfile,
                    field=field,
                    intent=intent,
                    spw=spw,
                    timebin=timebin,
                    do_statwt=do_statwt,
                    overwrite=overwrite)

            return ()

        def remove_staged_products(
                self,
                target=None,
                project=None,
                array_tag=None,
                obsnum=None,
                product=None,
                extra_ext='',
                strict_config=True):
            """
            Remove 'staged' visibility products, which are intermediate
            between the calibrated data and the concated measurement sets
            that we begin processing on. Run this step after concat to
            reduct the disk footprint of the pipeline.
            """

            if target is None:
                logger.error("Please specify a target.")
                raise Exception("Please specify a target.")

            if project is None:
                logger.error("Please specify a project.")
                raise Exception("Please specify a project.")

            if array_tag is None:
                logger.error("Please specify an array_tag.")
                raise Exception("Please specify an array_tag.")

            if obsnum is None:
                logger.error("Please specify an obsnum.")
                raise Exception("Please specify an obsnum.")

            infile = fnames.get_staged_msname(
                target=target, project=project, array_tag=array_tag, obsnum=obsnum,
                product=product, ext=extra_ext)

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Clearing intermediate staged u-v data for " + infile)
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            # Change to the imaging directory for the target

            _ = self._kh.get_imaging_dir_for_target(target, changeto=True)

            if not self._dry_run:
                os.system('rm -rf ' + infile)
                os.system('rm -rf ' + infile + '.contsub')

            return ()

        def task_concat_uvdata(
                self,
                target=None,
                product=None,
                config=None,
                just_projects=None,
                extra_ext_in='',
                extra_ext_out='',
                overwrite=False,
                strict_config=True):
            """
            Concatenate all measurement sets for the supplied
            target+config+product combination.
            """

            if target is None:
                logger.error("Please specify a target.")
                raise Exception("Please specify a target.")

            if product is None:
                logger.error("Please specify a product.")
                raise Exception("Please specify a product.")

            if config is None:
                logger.error("Please specify a config.")
                raise Exception("Please specify a config.")

            # Change to the imaging directory for the target

            _ = self._kh.get_imaging_dir_for_target(target, changeto=True)

            # Generate the list of staged measurement sets to combine

            staged_ms_list = []
            for this_target, this_project, this_array_tag, this_obsnum in \
                    self._kh.loop_over_input_ms(
                        target=target,
                        config=config,
                        project=just_projects,
                        strict_config=strict_config):

                this_staged_ms = fnames.get_staged_msname(
                    target=this_target, project=this_project,
                    array_tag=this_array_tag, obsnum=this_obsnum, product=product,
                    ext=extra_ext_in)
                if os.path.isdir(this_staged_ms):
                    staged_ms_list.append(this_staged_ms)
                else:
                    logger.warning(
                        "MS not found and will be dropped from concat: " +
                        str(this_staged_ms))
                    logger.warning("This might or might not be a problem.")

            if len(staged_ms_list) == 0:
                logger.warning("No measurement sets to concatenate, returning.")
                return ()

            # Generate the outfile name

            outfile = fnames.get_vis_filename(
                target=target, config=config, product=product,
                ext=extra_ext_out, suffix=None)

            # Revise here

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Concatenating staged and split u-v data for:")
            logger.info("... target: " + target)
            logger.info("... product: " + product)
            logger.info("... config: " + config)
            logger.info("... files: " + str(staged_ms_list))
            logger.info("... output: " + str(outfile))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            # Concatenate the measurement sets

            if not self._dry_run:
                cvr.concat_ms(
                    infile_list=staged_ms_list,
                    outfile=outfile,
                    overwrite=overwrite,
                    copypointing=False)  # come back later

            return ()

        def task_contsub(
                self,
                target=None,
                project=None,
                array_tag=None,
                obsnum=None,
                product=None,
                extra_ext_in='',
                overwrite=False):
            """
            Run u-v plane continuum subtraction on an individual input
            measurement set.
            """

            if target is None:
                logger.error("Please specify a target.")
                raise Exception("Please specify a target.")

            if project is None:
                logger.error("Please specify a project.")
                raise Exception("Please specify a project.")

            if array_tag is None:
                logger.error("Please specify an array_tag.")
                raise Exception("Please specify an array_tag.")

            if obsnum is None:
                logger.error("Please specify an obsnum.")
                raise Exception("Please specify an obsnum.")

            infile = fnames.get_staged_msname(
                target=target, project=project,
                array_tag=array_tag, obsnum=obsnum, product=product,
                ext=extra_ext_in)

            # get target vsys and vwidth
            # if part of linear mosaic, use vsys, vwidth of the parent target
            vsys, vwidth = \
                self._kh.get_system_velocity_and_velocity_width_for_target(
                    target, check_parent=True)

            # Get lines to exclude.

            lines_to_exclude = self._kh.get_lines_to_flag(product=product)
            this_line_tag = self._kh.get_line_tag_for_line_product(product)
            if len(lines_to_exclude) == 0:
                lines_to_exclude = [this_line_tag]

            # Translate these into frequency ranges

            ranges_to_exclude = lines.get_ghz_range_for_list(
                line_list=lines_to_exclude, vsys_kms=vsys, vwidth_kms=vwidth)

            # Check for manually defined frequency windows:
            manual_range_to_exclude = self._kh.get_contsub_excludefreqrange(product=product)
            if manual_range_to_exclude is not None:
                # This needs to incorporate it into the range, and should look for overlap
                # to extend the already excluded region.

                distinct_ranges = []

                for this_range in manual_range_to_exclude:
                    if not len(this_range) == 2:
                        raise ValueError("Parameter `exclude_freq_ranges_ghz` in target_definitions.txt must be a list"
                                         " of 2 element lists with a low and high frequency. Given: "+str(this_range))
                    freq_low = min(this_range)
                    freq_high = max(this_range)

                    # NOTE: this assumes only 1 defined are from ranges_to_exclude
                    existfreq_low = min(ranges_to_exclude[0])
                    existfreq_high = max(ranges_to_exclude[0])

                    # Does this fall completely within the existing range for a single target.
                    if existfreq_low <= freq_low and existfreq_high >= freq_high:
                        # No need to adjust. Keep original
                        distinct_ranges.append(list(ranges_to_exclude[0]))
                        continue
                    # Is it completely separate?
                    elif existfreq_low >= freq_high or existfreq_high <= freq_low:
                        # Include the original and new
                        distinct_ranges.append(list(ranges_to_exclude[0]))
                        distinct_ranges.append(this_range)
                    # Does it extend the lower side?
                    elif existfreq_low >= freq_low and existfreq_high >= freq_high:
                        distinct_ranges.append([freq_low, existfreq_high])
                    # Does it extend the upper side?
                    elif existfreq_low <= freq_low and existfreq_high <= freq_high:
                        distinct_ranges.append([existfreq_low, freq_high])
                    # Does it completely enclose the range?
                    elif existfreq_low >= freq_low and existfreq_high <= freq_high:
                        distinct_ranges.append([freq_low, freq_high])

                # Reduce the range list to unique ranges.
                ranges_to_exclude = list(set([tuple(this_range) for this_range in distinct_ranges]))

            # Query the keyhandler for the details of continuum subtraction

            fitorder = self._kh.get_contsub_fitorder(product=product)
            if fitorder is None:
                fitorder = 0

            combinespw = self._kh.get_contsub_combinespw(product=product)
            if combinespw is None:
                combinespw = False

            combine = ''
            if combinespw:
                combine = 'spw'

            flag_edge_fraction = self._kh.get_contsub_flagedgefraction(product=product)

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("u-v continuum subtraction for")
            logger.info("... file: " + infile)
            logger.info("... output: " + infile + '.contsub')
            logger.info("... excluding frequency ranges: " + str(ranges_to_exclude))
            logger.info("... fitorder: " + str(fitorder))
            logger.info("... combine: " + str(combine))
            logger.info("... flag_edge_fraction: " + str(flag_edge_fraction))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            # Change to the imaging directory for the target

            _ = self._kh.get_imaging_dir_for_target(target, changeto=True)

            if not self._dry_run:
                cvr.contsub(
                    infile=infile,
                    # outfile is not an option right now, comes out ".contsub"
                    ranges_to_exclude=ranges_to_exclude,
                    overwrite=overwrite,
                    fitorder=fitorder,
                    combine=combine,
                    flag_edge_fraction=flag_edge_fraction)

            return ()

        def task_run_custom_scripts(
                self,
                target=None,
                product=None,
                config=None,
                extra_ext=''):
            """
            """
            pass

        def task_extract_line(
                self,
                target=None,
                product=None,
                config=None,
                exact=False,
                contsub="prefer",
                extra_ext_in='',
                extra_ext_out='',
                do_statwt=True,
                edge_for_statwt=None,
                method="regrid_then_rebin",
                require_full_line_coverage=False,
                overwrite=False,
                strict_config=True):
            """
            Extract spectral line data from ms data for the input target,
            config and product.
            """

            # Error checking

            if target is None:
                logger.error("Please specify a target.")
                raise Exception("Please specify a target.")

            if product is None:
                logger.error("Please specify a product.")
                raise Exception("Please specify a product.")

            if config is None:
                logger.error("Please specify a config.")
                raise Exception("Please specify a config.")

            # If the user wants statwt but doesn't override the edge
            # value, get it from the key handler.

            if do_statwt:
                if edge_for_statwt is None:
                    edge_for_statwt = \
                        self._kh.get_statwt_edge_for_line_product(product=product)

            # Option re: continuum subtraction

            valid_contsub_options = ['prefer', 'require', 'none']
            if contsub.lower().strip() not in valid_contsub_options:
                logger.error("Please choose a valid contsub option.")
                logger.error("Valid options are:" + str(valid_contsub_options))
                raise Exception("Please choose a valid contsub option.")

            # Compile a list of input files, looping over the staged
            # measurement sets.

            _ = self._kh.get_imaging_dir_for_target(target, changeto=True)

            logger.debug('')
            logger.debug('task_extract_line')
            logger.debug('loop_over_input_ms')
            logger.debug('target=%s' % (str([target])))
            logger.debug('config=%s' % (str([config])))
            # <TODO># we have not excluded the combined interf config '12m+7m'

            infile_dict = {}
            for this_target, this_project, this_array_tag, this_obsnum in \
                    self._kh.loop_over_input_ms(target=[target],
                                                config=[config],
                                                project=None,
                                                strict_config=strict_config):

                # The name of the staged measurement set with this
                # combination of target, project, array, obsnum.

                this_infile = fnames.get_staged_msname(
                    target=this_target, project=this_project,
                    array_tag=this_array_tag, obsnum=this_obsnum,
                    product=product, ext=extra_ext_in)

                # Check for existence of original data and continuum
                # subtraction.

                infile_dict[this_infile] = {}
                infile_dict[this_infile]['present'] = \
                    os.path.isdir(this_infile)
                infile_dict[this_infile]['contsub'] = \
                    os.path.isdir(this_infile + '.contsub')

            # Implement the logic related to continuum
            # subtraction. Options are "require" (use only data with
            # continuum subtraction), "prefer" (if any data are missing
            # continuum subtraction but are present then skip continuum
            # subtraction), or "none" (use original data).

            infile_list = []

            if contsub == 'prefer':

                all_have_contsub = True

                for this_infile in infile_dict.keys():
                    if not infile_dict[this_infile]['present']:
                        continue
                    if not infile_dict[this_infile]['contsub']:
                        all_have_contsub = False

                if all_have_contsub:
                    logger.info(
                        "All files have continuum subtraction. Using that.")
                    contsub = 'require'
                else:
                    logger.info(
                        "Some files missing continuum subtraction. Skipping.")
                    contsub = 'none'

            if contsub == 'require':

                logger.warning(infile_dict)

                for this_infile in infile_dict.keys():
                    if infile_dict[this_infile]['contsub']:
                        infile_list.append(this_infile + '.contsub')
                        logger.warning("In file: {}".format(this_infile))
                    else:
                        logger.warning(
                            "File lacks contsub, skipping: " + str(this_infile))

            if contsub == 'none':

                for this_infile in infile_dict.keys():
                    if infile_dict[this_infile]['present']:
                        infile_list.append(this_infile)
                    else:
                        logger.warning("File missing, skipping: " + str(this_infile))

            if len(infile_list) == 0:
                logger.warning("No files to process.")
                return ()

            # Define the output file. Line extraction has a concatenation
            # step, so the individual measurement sets will be combined
            # after extraction.

            outfile = fnames.get_vis_filename(
                target=target, config=config, product=product,
                ext=extra_ext_out, suffix=None)

            # Extract the spectral information needed for the regrid

            vsys_kms, vwidth_kms = \
                self._kh.get_system_velocity_and_velocity_width_for_target(
                    target, check_parent=False)
            line_to_extract = self._kh.get_line_tag_for_line_product(product)

            valid_methods = [
                'regrid_then_rebin', 'rebin_then_regrid',
                'just_regrid', 'just_rebin']
            if method.lower().strip() not in valid_methods:
                logger.error("Not a valid line extraction method - " + str(method))
                raise Exception("Please specify a valid line extraction method.")

            target_chanwidth = self._kh.get_channel_width_for_line_product(product)

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Extracting spectral product:")
            logger.info("... Line: " + str(line_to_extract))
            logger.info("... Vsys [km/s]: " + str(vsys_kms))
            logger.info("... Vwidth [km/s]: " + str(vwidth_kms))
            logger.info("... Method: " + str(method))
            logger.info("... From files: " + str(infile_list))
            logger.info("... To file: " + str(outfile))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            # Check existence of output data and abort if found and overwrite is off

            if os.path.isdir(outfile) and not os.path.isdir(outfile + '.touch'):
                if not overwrite:
                    logger.warning('... found existing output data "' + outfile + '", will not overwrite it.')
                    return ()

            if not self._dry_run:

                cvr.batch_extract_line(
                    infile_list=infile_list,
                    outfile=outfile,
                    target_chan_kms=target_chanwidth,
                    line=line_to_extract,
                    vsys_kms=vsys_kms, vwidth_kms=vwidth_kms,
                    method=method,
                    exact=exact,
                    overwrite=overwrite,
                    clear_pointing=False,
                    require_full_line_coverage=require_full_line_coverage)

                if do_statwt:
                    cvr.reweight_data(
                        infile=outfile,
                        edge_kms=edge_for_statwt,
                        overwrite=overwrite)

            return ()

        def task_extract_continuum(
                self,
                target=None,
                product=None,
                config=None,
                exact=False,
                extra_ext_in='',
                extra_ext_out='',
                do_statwt=True,
                method="regrid_then_rebin",
                require_full_cont_coverage=False,
                overwrite=False,
                strict_config=True):
            """
            Extract continuum data from ms data for the input target, config,
            and product.
            """

            # Error checking

            if target is None:
                logger.error("Please specify a target.")
                raise Exception("Please specify a target.")

            if product is None:
                logger.error("Please specify a product.")
                raise Exception("Please specify a product.")

            if config is None:
                logger.error("Please specify a config.")
                raise Exception("Please specify a config.")

            _ = self._kh.get_imaging_dir_for_target(target, changeto=True)

            logger.debug('')
            logger.debug('task_extract_continuum')
            logger.debug('loop_over_input_ms')
            logger.debug('target=%s' % (str([target])))
            logger.debug('config=%s' % (str([config])))
            logger.debug('product=%s' % (product))
            # <TODO># we have not excluded the combined interf config '12m+7m'

            infile_dict = {}
            for this_target, this_project, this_array_tag, this_obsnum in \
                    self._kh.loop_over_input_ms(target=[target],
                                                config=[config],
                                                project=None,
                                                strict_config=strict_config):
                # The name of the staged measurement set with this
                # combination of target, project, array, obsnum.

                this_infile = fnames.get_staged_msname(
                    target=this_target, project=this_project,
                    array_tag=this_array_tag, obsnum=this_obsnum,
                    product=product, ext=extra_ext_in)

                # Check for existence of original data and continuum
                # subtraction.

                infile_dict[this_infile] = {}
                infile_dict[this_infile]['present'] = \
                    os.path.isdir(this_infile)

            # If no ms data found for the given target, then just return.
            # This could happen if the target name is a mosaic target,
            # and each ms data will be named by the mosaic parts.

            if len(infile_dict) == 0:
                return

            infile_list = []
            for this_infile in infile_dict.keys():
                if infile_dict[this_infile]['present']:
                    infile_list.append(this_infile)

            # Note that there is a concatenation step

            outfile = fnames.get_vis_filename(
                target=target, config=config, product=product,
                ext=extra_ext_out, suffix=None)

            # Extract necessary information for flagging lines
            # get target vsys and vwidth use the parent vsys, vwidth
            # when part of a linear mosaic. useful when spectral chunks are defined
            vsys_kms, vwidth_kms = \
                self._kh.get_system_velocity_and_velocity_width_for_target(
                    target, check_parent=True)
            lines_to_flag = self._kh.get_lines_to_flag(product=product)

            # Extract the spectral information needed for the regrid/rebin
            ranges_to_extract = self._kh.get_freq_ranges_for_cont_product(product)
            target_chanwidth = self._kh.get_channel_width_for_cont_product(product)

            valid_methods = [
                'regrid_then_rebin', 'rebin_then_regrid',
                'just_regrid', 'just_rebin']
            if method.lower().strip() not in valid_methods:
                logger.error(
                    "Not a valid continuum extraction method - " + str(method))
                raise Exception(
                    "Please specify a valid continuum extraction method.")

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Extracting continuum product:")
            logger.info("... Extracting ranges: " + str(ranges_to_extract))
            logger.info("... Lines to flag: " + str(lines_to_flag))
            logger.info("... Target channel width: " + str(target_chanwidth))
            logger.info("... Method: " + str(method))
            logger.info("... From files: " + str(infile_list))
            logger.info("... To file: " + str(outfile))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            if not self._dry_run:

                cvr.batch_extract_continuum(
                    infile_list=infile_list,
                    outfile=outfile,
                    ranges_to_extract=ranges_to_extract,
                    target_chan_ghz=target_chanwidth,
                    lines_to_flag=lines_to_flag,
                    vsys_kms=vsys_kms,
                    vwidth_kms=vwidth_kms,
                    method=method,
                    exact=exact,
                    overwrite=overwrite,
                    clear_pointing=False,
                    require_full_cont_coverage=require_full_cont_coverage)

                if do_statwt:
                    # not sure if we need this here
                    pass

            return ()

        def task_remove_concat(
                self,
                target=None,
                product=None,
                config=None,
                extra_ext_in='',
                suffixes=None):
            """
            Remove any concatenated measurement sets. These are
            intermediate (though time consuming) products not needed for
            imaging. This procedure wipes them and saves disk space.
            """

            # Error checking

            if target is None:
                logger.error("Please specify a target.")
                raise Exception("Please specify a target.")

            if product is None:
                logger.error("Please specify a product.")
                raise Exception("Please specify a product.")

            if config is None:
                logger.error("Please specify a config.")
                raise Exception("Please specify a config.")

            _ = self._kh.get_imaging_dir_for_target(target, changeto=True)

            if suffixes is None:
                suffixes = ['']
            if isinstance(suffixes, list):
                suffixes = [suffixes]

            for this_suffix in suffixes:
                if this_suffix == '':
                    this_suffix = None

                infile = fnames.get_vis_filename(
                    target=target, config=config, product=product,
                    ext=extra_ext_in, suffix=this_suffix)

                logger.info('Removing ' + infile)

                if not self._dry_run:
                    os.system('rm -rf ' + infile)

    # endregion
