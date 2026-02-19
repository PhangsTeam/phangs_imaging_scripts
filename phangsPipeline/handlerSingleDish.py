"""
SingleDishHandler

The PHANGS pipeline to handle single dish data reduction.

There should not be any direct calls to CASA in here. Eventually, this
should be able to run without CASA enabled (though it won't be able to
call any of the CASA-specific routines). Right now, just avoid direct
calls to CASA from this class.
"""

import glob
import logging
import os
import shutil

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

    from . import casaLegacySingleDishRoutines as csdr
    from . import casaSingleDishALMAWrapper as sdalma
    from . import handlerTemplate
    from . import utilsLines

    class SingleDishHandler(handlerTemplate.HandlerTemplate):
        """
        Class to handle single dish data.
        """

        def __init__(
            self,
            key_handler = None,
            dry_run = False,
            use_legacy_pipeline=False,
            ):
            handlerTemplate.HandlerTemplate.__init__(self, key_handler = key_handler, dry_run = dry_run)

            self.use_legacy_pipeline = use_legacy_pipeline

    #region File name routines

        ###########################################
        # Defined file names for various products #
        ###########################################

        def _fname_dict(
            self,
            target=None,
            product=None,
            extra_ext='',
            ):
            """
            Make the file name dictionary for all files used
            in the process.
            """

            # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
            # Error checking
            # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

            if target is None:
                logger.error("Need a target.")
                return

            if product is None:
                logger.error("Need a product.")
                return

            # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
            # Initialize
            # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

            fname_dict = {}

            # single dish file

            has_sd = self._kh.has_singledish(target=target, product=product)
            tag = 'sd_file'
            fname_dict[tag] = ''
            if has_sd:
                sd_file = self._kh.get_sd_filename(target = target, product = product, nocheck = True)
                if sd_file is not None:
                    fname_dict[tag] = sd_file

            fname_dict['source'] = []

            tag = 'sd_raw_data_list'
            fname_dict[tag] = []
            if has_sd:
                for this_target, this_project, this_arraytag, this_obsnum in self._kh.loop_over_input_ms(target=target, config='tp'):
                    sd_file = self._kh.get_file_for_input_ms(target=this_target,
                                                             project=this_project,
                                                             array_tag=this_arraytag,
                                                             obsnum=this_obsnum)


                    source = self._kh.get_field_for_input_ms(target=this_target,
                                                             project=this_project,
                                                             array_tag=this_arraytag,
                                                             obsnum=this_obsnum,
                                                             )

                    fname_dict[tag].append(sd_file)
                    fname_dict['source'].append(source)

            # Return

            return(fname_dict)

    #endregion

    #region "Tasks" : Individual steps

        def task_execute_single_dish_pipeline(
            self,
            target,
            product='all',
            source='all',
            extra_ext_in='',
            extra_ext_out='',
            line_wing_kms=200.0,
            ):
            """
            Execute single dish data reduction for one target.
            """

            if product == 'all':
                product_list = self.get_line_products()
            else:
                product_list = [product]

            fname_dict = self._fname_dict(
                target=target,
                product=product_list[0],
                )

            if fname_dict['sd_file'] == '':
                logger.info("Target "+target+" product "+product+" has no single dish data in the singledish_key file.")

            if len(fname_dict['sd_raw_data_list']) > 1:
                logger.warning('Warning! Multiple single dish raw data entries are found in the ms_file_key! We will only process the first one! [TODO]')
                #<TODO># We can only process one single dish raw data for now. Not sure how to combine those. Unless we specify line_product in the ms_file_key?

            input_raw_data = fname_dict['sd_raw_data_list'][0]

            # Legacy pipeline doesn't handle multiple line products.
            if self.use_legacy_pipeline:
                logger.warning('Using legacy single dish pipeline')
                logger.warning(f'Only the first line product will be processed: {product_list[0]}')

                product_list = [product_list[0]]

            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Executing single dish pipeline")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("  target: "+str(target))
            logger.info("  product: "+str(product_list))
            logger.info("  raw data: "+str(input_raw_data))
            # logger.info("  output file: "+str(output_file))

            # if os.path.isfile(output_file):
            #     logger.info('Output file already exists: '+str(output_file)+'. Will not re-process it.')
            #     return

            for product in product_list:
                if product not in self._kh.get_line_products():
                    logger.error('Error! Product '+str(product)+' is not defined in the config_definitions key?!')
                    return

            parameters = self._kh.get_params_for_singledish(singledish_config='tp')

            product_dict = {}
            for product in product_list:

                fname_dict = self._fname_dict(
                    target=target,
                    product=product,
                    )
                output_file = fname_dict['sd_file']

                product_dict[product] = self._kh._config_dict['line_product'][product]

                this_line = self._kh.get_line_tag_for_line_product(product)
                vsys, vwidth = self._kh.get_system_velocity_and_velocity_width_for_target(target, check_parent=False)
                max_chanwidth_kms = self._kh.get_channel_width_for_line_product(product)

                #joint_imaging_dirs = self._kh.get_joint_imaging_dirs_for_singledish_config()
                #joint_imaging_suffix = self._kh.get_joint_imaging_suffix_for_singledish_config()

                line_wing = line_wing_kms # km/s
                vlow1 = vsys - vwidth/2.0
                vhigh1 = vsys + vwidth/2.0
                vlow2 = vsys - vwidth/2.0 - line_wing
                vhigh2 = vsys + vwidth/2.0 + line_wing

                line_name, freq_rest_GHz = utilsLines.get_line_name_and_frequency(this_line)
                freq_rest_MHz = freq_rest_GHz * 1e3 # MHz

                rastring, decstring = self._kh.get_phasecenter_for_target(target=target)
                phase_center = 'J2000 '+rastring+' '+decstring

                name_line = line_name.upper() + '_%.0fkmsres'%(max_chanwidth_kms)

                # Add to product dict:
                product_dict[product]['line_name'] = line_name
                product_dict[product]['freq_rest_MHz'] = freq_rest_MHz
                product_dict[product]['vel_line_mask'] = [vlow2, vhigh2]
                product_dict[product]['vel_cube'] = [vlow1, vhigh1]
                product_dict[product]['name_line'] = name_line
                product_dict[product]['phase_center'] = phase_center

                product_dict[product]['max_chanwidth_kms'] = max_chanwidth_kms
                product_dict[product]['vsys'] = vsys
                product_dict[product]['vwidth'] = vwidth

                product_dict[product]['output_file'] = output_file

                for key in parameters:
                    if key not in product_dict[product]:
                        product_dict[product][key] = parameters[key]


            # copy raw data over
            path_galaxy = self._kh.get_singledish_dir_for_target(target=target, changeto=False) + os.sep + 'processing_singledish_'+target + os.sep
            path_galaxy = os.path.abspath(path_galaxy) + os.sep
            input_raw_data = os.path.abspath(input_raw_data) + os.sep
            if not os.path.isdir(path_galaxy):
                os.makedirs(path_galaxy)
            for dir_to_copy in ['calibration', 'raw', 'script', 'qa']:
                if os.path.isdir(os.path.join(path_galaxy, dir_to_copy)):
                    input_raw_data_files = glob.glob(os.path.join(input_raw_data, dir_to_copy))
                    copied_raw_data_files = glob.glob(os.path.join(path_galaxy, dir_to_copy))
                    if len(input_raw_data_files) > len(copied_raw_data_files):
                        logger.info("  cleaning up: "+str(os.path.join(path_galaxy, dir_to_copy)))
                        shutil.rmtree(glob.glob(os.path.join(path_galaxy, dir_to_copy)))
                if not os.path.isdir(os.path.join(path_galaxy, dir_to_copy)):
                    logger.info("  copying raw data: "+str(os.path.join(input_raw_data, dir_to_copy)))
                    logger.info("  to processing dir: "+str(os.path.join(path_galaxy, dir_to_copy)))
                    shutil.copytree(os.path.join(input_raw_data, dir_to_copy), \
                                    os.path.join(path_galaxy, dir_to_copy))


            if self.use_legacy_pipeline:
                logger.info("  Using legacy pipeline")

                product = product_list[0]

                vlow2 = product_dict[product]['vel_cube'][0]
                vhigh2 = product_dict[product]['vel_cube'][1]

                vlow1 = product_dict[product]['vel_line_mask'][0]
                vhigh1 = product_dict[product]['vel_line_mask'][1]

                kwargs = {}
                kwargs['path_galaxy'] = path_galaxy                            #
                kwargs['flag_file']  = ''                                      #
                kwargs['doplots']    = False                                    # Do non-interactive. additional plots (plots will be saved in "calibration/plots" folder)
                kwargs['bl_order']   = 1                                       # Order for the baseline fitting
                kwargs['max_flag_frac'] = 0.9                                  # Remove antennae with significant amounts of flagged data
                kwargs['in_source']     = source                               # Source name. This comes from the field name in the MS file keys
                kwargs['freq_rest']  = product_dict[product]['freq_rest_MHz']                           # Rest frequency of requested line in MHz (ex: "freq_rest  = 230538" for CO(2-1))
                kwargs['vel_cube']   = vel_cube = '%.3f~%.3f'%(vlow2, vhigh2)  # Range in velocity in km/s to extract the line cube.
                kwargs['vel_line']   = vel_line = '%.3f~%.3f'%(vlow1, vhigh1)  # Range in velocity in km/s to exclude the line emission from the baseline fit.
                kwargs['phase_center']   = phase_center                        # Provide coordinate of phase center, otherwise set to "False" and coordinates will be read from the data
                kwargs['source_vel_kms'] = product_dict[product]['vsys']                                # Provide velocity of the source, otherwise set to "False" and coordinates will be read from the data
                kwargs['vwidth_kms']     = product_dict[product]['vwidth']                              # width in velocity and velocity resolution in km/s
                kwargs['chan_dv_kms']    = product_dict[product]['max_chanwidth_kms']                   #
                kwargs['freq_rest_im']   = product_dict[product]['freq_rest_GHz']                       # rest frequency in GHz for imaging
                kwargs['name_line']      = product_dict[product]['line_name']                           # Name of the line, to be used for naming the files -- will not be used anymore
                kwargs['output_file']    = product_dict[product]['output_file']                         # Output file path
                #kwargs['joint_imaging_dirs'] = joint_imaging_dirs              # Do a joint imaging by including *.cal.jy in joint_imaging_dir
                #kwargs['joint_imaging_suffix'] = joint_imaging_suffix          # Suffix after name_line in the output file name.
                kwargs['do_step'] = []

                # see if there is anything defined in the config_definitions key
                parameters = self._kh.get_params_for_singledish(singledish_config='tp')

                if parameters is not None:
                    for key in parameters:
                        kwargs[key] = parameters[key]

                logger.info("  kwargs: "+str(kwargs))

                csdr.run_ALMA_TP_tools(**kwargs)

            else:
                # Run the modified version of the ALMA pipeline w/ custom imaging routine

                sdalma.runALMAPipeline(path_galaxy=path_galaxy,
                                       in_source=fname_dict['source'][0],
                                       baseline_fit_func='poly',
                                       baseline_fit_order=parameters['bl_order'] if 'bl_order' in parameters else 1,
                                       baseline_linewindowmode='replace',
                                       baseline_linewindow=None,
                                       product_dict=product_dict,
                                       )


    #endregion

    #region Recipes execute a set of linked tasks for one data set.

        def recipe_process_one_target(
            self,
            target = None,
            product = None,
            ):
            """
            """

            # Work out file names and note whether the target is part of a
            # mosaic, has single dish data, etc.

            logger.warning('recipe_process_one_target is only used for the legacy pipeline!')

            fname_dict = self._fname_dict(
                target=target,
                product=product,
            )

            if fname_dict['sd_file'] == '':
                logger.info("Target "+target+" product "+product+" has no single dish data in the singledish_key file.")
                return

            # Call tasks

            if len(fname_dict['sd_raw_data_list']) > 1:
                logger.warning('Warning! Multiple single dish raw data entries are found in the ms_file_key! We will only process the first one! [TODO]')
                #<TODO># We can only process one single dish raw data for now. Not sure how to combine those. Unless we specify line_product in the ms_file_key?

            for idx in range(len(fname_dict['sd_raw_data_list'])):
                self.task_execute_single_dish_pipeline(
                    target = target,
                    product = product,
                    source = fname_dict['source'],
                    input_raw_data = fname_dict['sd_raw_data_list'][idx],
                    output_file = fname_dict['sd_file'],
                    )
                if idx > 0:
                    break
                    #<TODO># We can only process one single dish raw data for now. Not sure how to combine those. Unless we specify line_product in the ms_file_key?

            return

    #endregion

    #region Loops

        def loop_singledish(
            self,
            do_all=True,
            make_directories=True,
            ):
            """
            Loops over the full set of targets, products, and
            configurations to run the postprocessing. Toggle the parts of
            the loop using the do_XXX booleans. Other choices affect the
            algorithms used.
            """

            if do_all:
                do_step = True

            if len(self.get_targets()) == 0:
                logger.error("Need a target list.")
                return(None)

            if len(self.get_all_products()) == 0:
                logger.error("Need a products list.")
                return(None)

            if make_directories:
                self._kh.make_missing_directories(postprocess=True)

            #

            if do_step:

                if self.use_legacy_pipeline:

                    for this_target, this_product in self.looper(do_targets=True,
                                                                do_products=True,
                                                                do_configs=False):

                        self.recipe_process_one_target(
                            target=this_target,
                            product=this_product,
                            )

                else:
                    for this_target in self.get_targets():

                        self.task_execute_single_dish_pipeline(
                            target=this_target,
                            product='all',
                            )
