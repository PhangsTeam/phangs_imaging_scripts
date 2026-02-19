"""imagingHandler

This module makes images and cubes out of the uv data products created by uvdataHandler.

This code needs to be run inside CASA.

Example:
    $ casa
    from phangsPipeline import handlerKeys as kh
    from phangsPipeline import handlerImaging as imh
    this_kh = kh.KeyHandler(master_key = 'config_keys/master_key.txt')
    this_imh = uvh.ImagingHandler(key_handler = this_kh)
    this_imh.set_targets(only = ['ngc3627'])
    this_imh.loop_imaging()

Args:
    make_dirty_image (boolean): If `True` then make dirty image cube from ms data. Default is `True`.
    revert_to_dirty (boolean): If `True` then reset current image cube by the dirty image cube (mainly for debugging). Default is `False`.
    read_in_clean_mask (boolean): If `True` then read in clean mask as defined in "cleanmask_keys.txt". Default is `True`.
    run_multiscale_clean (boolean): If `True` then run multiscale clean. Default is `True`.
    revert_to_multiscale (boolean): If `True` then reset current image cube by the multiscale cleaned image cube (mainly for debugging). Default is `False`.
    make_singlescale_mask (boolean): If `True` then make singlescale clean mask based on current image cube. Default is `True`.
    run_singlescale_clean (boolean): If `True` then run singlescale clean. Default is `True`.
    do_export_to_fits (boolean): If `True` then export ms data folders into fits-format image cube files. Default is `True`.

Notes:
    This code calls following functions:
        import casaImagingRoutines as imr
        import casaMaskingRoutines as msr
        if make_dirty_image: imr.make_dirty_map
        if revert_to_dirty: imr.replace_cube_with_copy
        if read_in_clean_mask: msr.import_and_align_mask
        if run_multiscale_clean: imr.multiscale_loop
        if revert_to_multiscale: imr.replace_cube_with_copy
        if make_singlescale_mask: msr.signal_mask
        if run_singlescale_clean: imr.singlescale_loop
        if do_export_to_fits: imr.export_to_fits

"""

import logging
import os

# Check casa environment by importing CASA-only packages
from .check_imports import is_casa_installed

casa_enabled = is_casa_installed()

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if casa_enabled:
    logger.debug('casa_enabled = True')
else:
    logger.debug('casa_enabled = False')

if casa_enabled:

    # Analysis utilities
    import analysisUtils as au

    from .clean_call import CleanCall, CleanCallFunctionDecorator

    from . import casaImagingRoutines as imr
    from . import casaMaskingRoutines as msr
    from . import casaStuff
    from . import handlerTemplate
    from . import utilsLines as lines
    from . import utilsFilenames

    class ImagingHandler(handlerTemplate.HandlerTemplate):
        """
        Class to makes image cubes out of uv data from each spectral line and continuum of each galaxy.
        """

        ############
        # __init__ #
        ############

        def __init__(
            self,
            key_handler = None,
            dry_run = False,
            ):

            # inherit template class
            handlerTemplate.HandlerTemplate.__init__(self,key_handler = key_handler, dry_run = dry_run)

        ###############
        # _fname_dict #
        ###############

        def _fname_dict(
            self,
            product,
            imagename,
            imaging_method='tclean',
            ):
            """
            Handles file names used in the imaging processes.
            """
            fname_dict = {}
            is_line_product = product in self._kh.get_line_products()
            if is_line_product:
                fname_dict['root'] = imagename
                fname_dict['suffix'] = ''
                if imaging_method == 'tclean':
                    fname_dict['image'] = imagename + '.image'
                    fname_dict['model'] = imagename + '.model'
                    fname_dict['residual'] = imagename + '.residual'
                    fname_dict['mask'] = imagename + '.mask'
                    fname_dict['pb'] = imagename + '.pb'
                    fname_dict['psf'] = imagename + '.psf'
                    fname_dict['weight'] = imagename + '.weight'
                    fname_dict['sumwt'] = imagename + '.sumwt'
                elif imaging_method == 'sdintimaging':
                    fname_dict['image'] = imagename + '.joint.cube.image'
                    fname_dict['model'] = imagename + '.joint.cube.model'
                    fname_dict['residual'] = imagename + '.joint.cube.residual'
                    fname_dict['mask'] = imagename + '.joint.cube.mask'
                    fname_dict['pb'] = imagename + '.joint.cube.pb'
                    fname_dict['psf'] = imagename + '.joint.cube.psf'
                    fname_dict['weight'] = imagename + '.joint.cube.weight'
                    fname_dict['sumwt'] = imagename + '.joint.cube.sumwt'
            else:
                fname_dict['root'] = imagename
                fname_dict['suffix'] = '.tt0'
                if imaging_method == 'tclean':
                    fname_dict['image'] = imagename + '.image.tt0'
                    fname_dict['model'] = imagename + '.model.tt0'
                    fname_dict['residual'] = imagename + '.residual.tt0'
                    fname_dict['mask'] = imagename + '.mask'  # not '.mask.tt0'
                    fname_dict['pb'] = imagename + '.pb.tt0'
                    fname_dict['psf'] = imagename + '.psf.tt0'
                    fname_dict['weight'] = imagename + '.weight.tt0'
                    fname_dict['sumwt'] = imagename + '.sumwt.tt0'
                    fname_dict['alpha'] = imagename + '.alpha'
                    fname_dict['beta'] = imagename + '.beta'
                elif imaging_method == 'sdintimaging':
                    fname_dict['image'] = imagename + '.joint.cube.image.tt0'
                    fname_dict['model'] = imagename + '.joint.cube.model.tt0'
                    fname_dict['residual'] = imagename + '.joint.cube.residual.tt0'
                    fname_dict['mask'] = imagename + '.joint.cube.mask'  # not '.mask.tt0'
                    fname_dict['pb'] = imagename + '.joint.cube.pb.tt0'
                    fname_dict['psf'] = imagename + '.joint.cube.psf.tt0'
                    fname_dict['weight'] = imagename + '.joint.cube.weight.tt0'
                    fname_dict['sumwt'] = imagename + '.joint.cube.sumwt.tt0'
                    fname_dict['alpha'] = imagename + '.joint.cube.alpha'
                    fname_dict['beta'] = imagename + '.joint.cube.beta'
            return fname_dict


        ################
        # loop_imaging #
        ################

        def loop_imaging(
                self,
                do_all=False,
                imaging_method='tclean',
                imaging_method_override=None,
                do_dirty_image=False,
                do_revert_to_dirty=False,
                do_read_clean_mask=False,
                do_multiscale_clean=False,
                do_revert_to_multiscale=False,
                do_singlescale_mask=False,
                singlescale_mask_high_snr=None,
                singlescale_mask_low_snr=None,
                singlescale_mask_absolute=False,
                do_singlescale_clean=False,
                do_revert_to_singlescale=False,
                do_export_to_fits=False,
                convergence_fracflux=0.01,
                convergence_noise_z_threshold=None,
                singlescale_threshold_value=1.0,
                extra_ext_in=None,
                suffix_in=None,
                extra_ext_out=None,
                recipe='phangsalma',
                make_directories=True,
                dynamic_sizing=True,
                force_square=False,
                export_dirty=False,
                export_multiscale=False,
                overwrite=False,
        ):
            """
            Loops over the full set of targets, products, and
            configurations to do the imaging. Toggle the parts of the loop
            using the do_XXX booleans. Other choices affect algorithms
            used.

            The clean convergence is set by the convergence_fracflux and
            convergence_noise_z_threshold parameters.
            - convergence_fracflux is the fractional flux threshold between imaging loops.
                Convergence is reached when the fractional flux change is less than this value.
            - convergence_noise_z_threshold is the noise threshold in z-scores.
                Convergence is reached when the z-score test is less than this value. By default,
                this is disabled by setting to None. We suggest setting this to 2.0 when enabled
                as a conservative choice.

            Parameters
            ----------
            do_all : bool
                If True, do all steps.
            imaging_method : str
                'tclean' or 'sdintimaging'
            do_dirty_image : bool
                If True, make a dirty image.
            do_revert_to_dirty : bool
                If True, revert to dirty image.
            do_read_clean_mask : bool
                If True, read clean mask.
            do_multiscale_clean : bool
                If True, multiscale clean.
            do_revert_to_multiscale : bool
                If True, revert to multiscale clean.
            do_singlescale_mask : bool
                If True, make singlescale mask.
            singlescale_mask_high_snr : float
                High SNR threshold for singlescale mask.
            singlescale_mask_low_snr : float
                Low SNR threshold for singlescale mask.
            singlescale_mask_absolute : bool
                If True, use absolute threshold for singlescale mask.
            do_singlescale_clean : bool
                If True, singlescale clean.
            do_revert_to_singlescale : bool
                If True, revert to singlescale clean.
            do_export_to_fits : bool
                If True, export ms data folders into fits-format image cube files.
            convergence_fracflux : float
                Fractional flux convergence threshold for multiscale clean. Default is 0.01.
            convergence_noise_z_threshold : float
                Noise convergence threshold for multiscale clean. Default is None. A suggested
                value to use is 2.0, i.e. the null hypothesis is rejected if the z-score is
                greater than 2.0.
            singlescale_threshold_value : float
                Threshold value for singlescale clean. Default is 1.0.
            extra_ext_in : str
                Extra extension for input files.
            suffix_in : str
                Suffix for input files.
            extra_ext_out : str
                Extra extension for output files.
            recipe : str
                The recipe. Default is 'phangsalma'.
            make_directories : bool
                Create missing directories.
            dynamic_sizing : bool
                Use dynamic sizing.
            force_square : bool
                Force the image size to be square.
            export_dirty : bool
                If True, export dirty images.
            export_multiscale : bool
                If True, export multiscale images.
            overwrite : bool
                Overwrite existing files.
            """

            if do_all:
                do_dirty_image = True
                # debateable ...
                do_revert_to_dirty = True
                do_read_clean_mask = True
                do_multiscale_clean = True
                # debateable ...
                do_revert_to_multiscale = True
                do_singlescale_mask = True
                do_singlescale_clean = True
                # debateable ...
                do_revert_to_singlescale = True
                do_export_to_fits = True

            if imaging_method not in ['tclean', 'sdintimaging']:
                logger.error('imaging_method should be either tclean or sdintimaging')
                raise Exception('imaging_method should be either tclean or sdintimaging')

            if len(self.get_targets()) == 0:
                logger.error("Need a target list.")
                return (None)

            if len(self.get_all_products()) == 0:
                logger.error("Need a products list.")
                return (None)

            known_recipes = ['phangsalma']
            if recipe not in known_recipes:
                logger.error("Recipe not known " + recipe)
                return (None)

            if make_directories:
                self._kh.make_missing_directories(imaging=True)

            # Loop over targets, products, and interferometric configs

            for this_target, this_product, this_config in \
                    self.looper(do_targets=True, do_products=True, do_configs=True, just_interf=True):

                # Change to the relevant directory

                this_imaging_dir = self._kh.get_imaging_dir_for_target(this_target, changeto=True)

                # print starting message
                logger.info("")
                logger.info("--------------------------------------------------------")
                logger.info("START: Imaging target " + this_target + " config " + this_config + " product " + this_product)
                logger.info("--------------------------------------------------------")
                logger.info('Imaging recipe: ' + recipe)

                #<20241113><DZLIU>#
                skip = False
                if not overwrite:
                    do_dirty_image = True
                    do_revert_to_dirty = True
                    do_multiscale_clean = True
                    do_revert_to_multiscale = True
                    do_singlescale_mask = True
                    do_singlescale_clean = True
                    do_revert_to_singlescale = True
                    do_export_to_fits = True
                    export_dirty = True
                    export_dirty = True
                    export_multiscale = True
                    if os.path.exists(this_target+'_' +this_config+'_'+this_product+'_dirty.image'):
                        do_dirty_image = False
                        do_revert_to_dirty = True
                    if os.path.exists(this_target+'_' +this_config+'_'+this_product+'_multiscale.image'):
                        do_dirty_image = False
                        do_revert_to_dirty = False
                        do_multiscale_clean = False
                        do_revert_to_multiscale = True
                    if os.path.exists(this_target+'_' +this_config+'_'+this_product+'_singlescale.image'):
                        do_dirty_image = False
                        do_revert_to_dirty = False
                        do_multiscale_clean = False
                        do_revert_to_multiscale = False
                        do_singlescale_clean = False
                        do_revert_to_singlescale = True
                    if os.path.exists(this_target+'_' +this_config+'_'+this_product+'.fits'):
                        do_dirty_image = False
                        do_revert_to_dirty = False
                        do_multiscale_clean = False
                        do_revert_to_multiscale = False
                        do_singlescale_clean = False
                        do_revert_to_singlescale = False
                        skip = True

                if recipe == 'phangsalma' and not skip:
                    self.recipe_phangsalma_imaging(
                        target=this_target,
                        product=this_product,
                        config=this_config,
                        extra_ext_in=extra_ext_in,
                        suffix_in=suffix_in,
                        extra_ext_out=extra_ext_out,
                        imaging_method=imaging_method,
                        imaging_method_override=imaging_method_override,
                        do_dirty_image=do_dirty_image,
                        do_revert_to_dirty=do_revert_to_dirty,
                        do_read_clean_mask=do_read_clean_mask,
                        do_multiscale_clean=do_multiscale_clean,
                        do_revert_to_multiscale=do_revert_to_multiscale,
                        do_singlescale_mask=do_singlescale_mask,
                        singlescale_mask_high_snr=singlescale_mask_high_snr,
                        singlescale_mask_low_snr=singlescale_mask_low_snr,
                        singlescale_mask_absolute=singlescale_mask_absolute,
                        do_singlescale_clean=do_singlescale_clean,
                        do_revert_to_singlescale=do_revert_to_singlescale,
                        do_export_to_fits=do_export_to_fits,
                        convergence_fracflux=convergence_fracflux,
                        convergence_noise_z_threshold=convergence_noise_z_threshold,
                        singlescale_threshold_value=singlescale_threshold_value,
                        dynamic_sizing=dynamic_sizing,
                        force_square=force_square,
                        export_dirty=export_dirty,
                        export_multiscale=export_multiscale,
                        overwrite=overwrite,
                    )

                # print ending message
                logger.info("--------------------------------------------------------")
                logger.info("END: Imaging target " + this_target + " config " + this_config + " product " + this_product)
                logger.info("--------------------------------------------------------")
            #
            # end of for looper

        #
        # end of loop_imaging()

        ####################
        # File name lookup #
        ####################

        ##################################################################
        # Tasks - discrete steps on target, product, config combinations #
        ##################################################################

        def task_initialize_clean_call(
                self,
                target=None,
                config=None,
                product=None,
                imaging_method='tclean',
                extra_ext_in=None,
                suffix_in=None,
                extra_ext_out=None,
                stage='dirty',
        ):
            """
            Initialize a clean call object for a target, config, product
            combination and an imaging stage.
            """

            # check user input.
            if target is None or config is None or product is None:
                logger.error('Please input target, config and product!')
                logger.error('e.g., target = "ngc3627_1", config = "12m+7m", product = "co21"')
                raise Exception('Please input target, config and product!')

            # Look up the recipes for this case
            recipe_list = self._kh.get_imaging_recipes(config=config, product=product)
            # logger.debug('self._kh._imaging_dict = ' + str(self._kh._imaging_dict))
            # print(self._kh._imaging_dict)
            if recipe_list is None:
                logger.error(
                    'Error! Could not get imaging recipe for config ' + config + ' product ' + product + '. Please check your "imaging_recipes.txt".')
                raise Exception(
                    'Error! Could not get imaging recipe for config ' + config + ' product ' + product + '. Please check your "imaging_recipes.txt".')

            # Initialize the clean call with the appropriate recipe list
            clean_call = CleanCall(recipe_list, use_chunks=False)

            # Get the visibility name
            vis_file = utilsFilenames.get_vis_filename(
                target=target, product=product, config=config,
                ext=extra_ext_in, suffix=suffix_in)

            # Test existence
            full_vis_file = self._kh.get_imaging_dir_for_target(target=target) + vis_file
            if not os.path.isdir(full_vis_file):
                logger.error('Visibility file not found: ' + full_vis_file + " returning.")
                # raise Exception('Missing visibility needed for imaging.')
                return (None)

            # Set the visibility file (note that we assume we are in the working directory)

            clean_call.set_param('vis', vis_file, nowarning=True)

            # Set the output image file name (note no suffix for imaging root)

            image_root = utilsFilenames.get_cube_filename(
                target=target, product=product, config=config,
                ext=extra_ext_out, casa=True, casaext='')

            clean_call.set_param('imagename', image_root, nowarning=True)

            # Get the phase center associated with the target
            rastring, decstring = self._kh.get_phasecenter_for_target(target=target)
            phasecenter = 'J2000 ' + rastring + ' ' + decstring

            clean_call.set_param('phasecenter', phasecenter)

            # Set the rest frequency or reference frequency

            is_line_product = product in self._kh.get_line_products()
            if is_line_product:
                if clean_call.get_param('specmode') != 'cube':
                    logger.error('Line product detected but specmode is not cube.')
                    raise Exception('Malformed clean call.')
                if stage == 'dirty':
                    clean_call.set_param('deconvolver', 'hogbom')
                elif stage == 'multiscale':
                    clean_call.set_param('deconvolver', 'multiscale')
                elif stage == 'singlescale':
                    clean_call.set_param('deconvolver', 'hogbom')
            else:
                if clean_call.get_param('specmode') != 'mfs':
                    logger.error('Continuum product detected but specmode is not msf.')
                    raise Exception('Malformed clean call.')
                if stage == 'dirty':
                    clean_call.set_param('deconvolver', 'mtmfs')
                elif stage == 'multiscale':
                    clean_call.set_param('deconvolver', 'mtmfs')
                elif stage == 'singlescale':
                    clean_call.set_param('deconvolver', 'mtmfs')

            if is_line_product:
                this_line_tag = self._kh.get_line_tag_for_line_product(product=product)
                if this_line_tag not in lines.line_list.keys():
                    logger.error("Did not find line in line_list " + this_line_tag)
                    raise Exception('Malformed clean call.')
                rest_freq_ghz = lines.line_list[this_line_tag]
                clean_call.set_restfreq_ghz(rest_freq_ghz)
            else:
                clean_call.set_reffreq_ghz(None)

            if imaging_method == 'sdintimaging':
                # Put in sdintimaging specific parameters to point at the SD image and PSF
                image_name = clean_call.get_param('imagename')
                clean_call.set_param('usedata', 'sdint')

                sdimage_name = image_name + '.sd.cube.image'
                sdpsf_name = image_name + '.sd.cube.psf'
                if not os.path.exists(sdimage_name):
                    sdimage_name = image_name + '.sd'
                    sdpsf_name = ''

                clean_call.set_param('sdimage', sdimage_name)
                clean_call.set_param('sdpsf', sdpsf_name)

            return (clean_call)

        @CleanCallFunctionDecorator
        def task_pick_cell_and_imsize(
                self,
                clean_call=None,
                oversamp=5,
                force_square=False,
                check_files=True,
                check_overrides=True,
                target=None,
                config=None,
                product=None,
        ):
            """
            Pick a cell size and imsize for a clean call by inspecting the
            visibility data, then attach these values to the clean call.

            This will call casaImagingRoutines.estimate_cell_and_imsize()
            first, then apply custom overrides if they exist.
            """

            if clean_call is None:
                logger.warning("Require a clean_call object. Returning.")
                return ()

            # Look up the file name
            vis_file = clean_call.get_param('vis')

            # Check visibility existence
            if check_files:
                if not (os.path.isdir(vis_file)):
                    logger.warning("Missing visibility data " + vis_file)
                    return ()

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Calculating target cell size and image size for:")
            logger.info(vis_file)
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            # Call the estimating routine
            if not self._dry_run:
                cell, imsize = imr.estimate_cell_and_imsize(
                    vis_file, oversamp, force_square=force_square)
            else:
                cell, imsize = '0.1arcsec', [1000, 1000]
                logger.info('DRY RUN skips calling imr.estimate_cell_and_imsize()')

            # Print info
            logger.info('cell='+cell+', imsize='+str(imsize))

            # Check user-set overriding parameters: cell deltara deltadec
            # (pixel size and width/height of the field of view in units of arcsec)
            if check_overrides:
                # first try to find 'target_config_product' key 'cell/deltara/deltadec' in the override file,
                # if not found, then find 'target_config' key 'cell/deltara/deltadec',
                # if also not found, then find 'target' key 'cell/deltara/deltadec',
                # finally, fall back to 'all' key 'cell/deltara/deltadec' if it is not None.
                override_keys = []
                if target is not None and config is not None and product is not None:
                    override_keys.append(target+'_'+config+'_'+product)
                if target is not None and config is not None:
                    override_keys.append(target+'_'+config)
                if target is not None:
                    override_keys.append(target)
                override_keys.append('all')
                override_cell = None
                override_deltara = None
                override_deltadec = None
                for override_key in override_keys:
                    if override_cell is None:
                        override_cell = self._kh.get_overrides(override_key, 'cell')
                    if override_deltara is None:
                        override_deltara = self._kh.get_overrides(override_key, 'deltara')
                    if override_deltadec is None:
                        override_deltadec = self._kh.get_overrides(override_key, 'deltadec')
                orig_cell_arcsec = float(str(cell).replace('arcsec',''))
                if override_cell is not None:
                    logger.info('applying overriding cell '+str(override_cell)+' arcsec')
                    cell_arcsec = np.abs(float(override_cell))
                    cell = str(cell_arcsec)+'arcsec'
                else:
                    cell_arcsec = orig_cell_arcsec
                orig_imsize = imsize.copy()
                if override_deltara is not None:
                    logger.info('applying overriding deltara '+str(override_deltara)+' arcsec')
                    override_deltara = np.abs(float(override_deltara))
                    imsize[0] = int(np.ceil(override_deltara/cell_arcsec/2.0)*2) # need an even number
                else:
                    orig_deltara = orig_imsize[0] * orig_cell_arcsec
                    imsize[0] = int(np.ceil(orig_deltara/cell_arcsec/2.0)*2)  # ensure even number
                if override_deltadec is not None:
                    logger.info('applying overriding deltadec '+str(override_deltadec)+' arcsec')
                    override_deltadec = np.abs(float(override_deltadec))
                    imsize[1] = int(np.ceil(override_deltadec/cell_arcsec/2.0)*2) # need an even number
                else:
                    orig_deltadec = orig_imsize[1] * orig_cell_arcsec
                    imsize[1] = int(np.ceil(orig_deltadec/cell_arcsec/2.0)*2)  # ensure even number

                # Print info
                if override_cell is not None or override_deltara is not None or override_deltadec is not None:
                    logger.info('cell='+cell+', imsize='+str(imsize))

            # Set clean call parameters
            clean_call.set_param('cell', cell, nowarning=True)
            clean_call.set_param('imsize', imsize, nowarning=True)

            # Return
            return (cell, imsize)

        @CleanCallFunctionDecorator
        def task_setup_sdintimaging(
                self,
                clean_call=None,
                target=None,
                product=None,
                asvelocity=True,
                overwrite=False
        ):
            """
            Regrid existing TP image to the staged MS
            """

            if clean_call is None:
                logger.warning("Require a clean_call object. Returning.")
                return None

            if target is None:
                logger.warning("Require a target. Returning.")
                return None

            if product is None:
                logger.warning("Require a product. Returning.")
                return None

            # Convert the fits file to an MS
            sd_fits_file = self._kh.get_sd_filename(target=target, product=product)
            sd_image_file = clean_call.get_param('imagename') + '.sd'

            if not os.path.exists(sd_image_file) or overwrite:
                os.system('rm -rf ' + sd_image_file)
                casaStuff.importfits(fitsimage=sd_fits_file, imagename=sd_image_file, overwrite=True)

                # Make sure the image axes are the right way round
                os.system('rm -rf ' + sd_image_file + '_reorder')
                order = ['rig', 'declin', 'stok', 'frequ']
                casaStuff.imtrans(imagename=sd_image_file, outfile=sd_image_file + '_reorder',
                                  order=order)
                os.system('rm -rf ' + sd_image_file)
                os.system('mv -f ' + sd_image_file + '_reorder ' + sd_image_file)

                # Regrid this to the input measurement set to avoid any weirdness with overlap.
                mytb = au.createCasaTool(casaStuff.tbtool)
                mytb.open(clean_call.get_param('vis') + '/SPECTRAL_WINDOW')
                freq = mytb.getcol('CHAN_FREQ')
                n_chan = mytb.getcol('NUM_CHAN')[0]
                d_freq = mytb.getcol('CHAN_WIDTH')[0, 0]
                first_freq = freq[0, 0]
                mytb.close()

                template_hdr = casaStuff.imregrid(sd_image_file, template='get')

                spec_shap = template_hdr['shap'][-1]
                spec_crpix = template_hdr['csys']['spectral2']['wcs']['crpix']
                spec_cdelt = template_hdr['csys']['spectral2']['wcs']['cdelt']
                spec_crval = template_hdr['csys']['spectral2']['wcs']['crval']

                if not (spec_shap == n_chan and spec_crpix == 0.0 and spec_cdelt == d_freq and spec_crval == first_freq):

                    template_hdr['shap'][-1] = n_chan
                    template_hdr['csys']['spectral2']['wcs']['crpix'] = 0.0
                    template_hdr['csys']['spectral2']['wcs']['cdelt'] = d_freq
                    template_hdr['csys']['spectral2']['wcs']['crval'] = first_freq

                    casaStuff.imregrid(imagename=sd_image_file, output=sd_image_file + '_regrid',
                                       template=template_hdr, asvelocity=asvelocity, overwrite=True)
                    os.system('rm -rf ' + sd_image_file)
                    os.system('mv -f ' + sd_image_file + '_regrid ' + sd_image_file)

                # Make sure the cube has per-plane restoring beans, both in channels and polarizations
                cube_info = casaStuff.imhead(sd_image_file, mode='list')
                n_chan = cube_info['shape'][-1]
                n_pol = cube_info['shape'][-2]

                myia = au.createCasaTool(casaStuff.iatool)
                myia.open(sd_image_file)
                restoring_beam = myia.restoringbeam()
                myia.setrestoringbeam(remove=True)
                for c in range(n_chan):
                    for p in range(n_pol):
                        myia.setrestoringbeam(beam=restoring_beam, channel=c, polarization=p)
                myia.close()

            return sd_image_file

        @CleanCallFunctionDecorator
        def task_assign_multiscales(
                self,
                clean_call=None,
                config=None,
        ):
            """
            Look up the appropriate scales for multi-scale clean
            associated with this config.
            """

            if clean_call is None:
                logger.warning("Require a clean_call object. Returning.")
                return ()

            scales_to_clean = self._kh.get_clean_scales_for_config(config=config)
            clean_call.set_multiscale_arcsec(scales=scales_to_clean)

            return ()

        @CleanCallFunctionDecorator
        def task_make_dirty_image(
                self,
                clean_call=None,
                imaging_method='tclean',
                backup=True,
        ):
            """
            Execute the attached clean call with zero iterations to create
            a dirty image.

            Set backup to True will make a copy of the dirty image as
            {imagename}_dirty.image
            """

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Making dirty image:")
            logger.info(str(clean_call.get_param('imagename')))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            if not self._dry_run:
                imr.make_dirty_image(clean_call, imaging_method=imaging_method)
                if backup:
                    imr.copy_imaging(
                        input_root=clean_call.get_param('imagename'),
                        output_root=clean_call.get_param('imagename') + '_dirty',
                        imaging_method=imaging_method,
                        wipe_first=True)

            return ()

        @CleanCallFunctionDecorator
        def task_revert_to_imaging(
                self,
                clean_call=None,
                imaging_method='tclean',
                tag='dirty',
        ):
            """
            Reset the current imaging stack to an earlier imaging
            state. Requires a tag that will be used to define the old
            imaging moved into place. Wipes the current imaging.
            """

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Resetting to " + tag + " imaging:")
            logger.info(str(clean_call.get_param('imagename')))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            if not self._dry_run:
                imr.copy_imaging(
                    input_root=clean_call.get_param('imagename') + '_' + tag,
                    output_root=clean_call.get_param('imagename'),
                    imaging_method=imaging_method,
                    wipe_first=True)

            return ()

        @CleanCallFunctionDecorator
        def task_read_clean_mask(
                self,
                clean_call=None,
                target=None,
                config=None,
                product=None,
                imaging_method='tclean'
        ):
            """
            Identify the clean mask associated with the target and
            product, read it from disk (in FITS form) and align it to the
            astrometry and grid of the current imaging.
            """

            if target is None:
                logger.warning("Require a target. Returning.")
                return ()

            # if config is None:
            #    logger.warning("Require a config. Returning.")
            #    return()

            if product is None:
                logger.warning("Require a product. Returning.")
                return ()

            # Could add an error check here that the template imaging exists

            this_cleanmask = self._kh.get_cleanmask_filename(target=target, product=product)
            if this_cleanmask is None:
                logger.info("No clean mask found for target " + target + " product " + product)
                # AKL - propose to deprecate
                # clean_call.set_param('usemask','pb')
                return ()

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Reading and aligning clean mask.")
            logger.info('From target ' + target + ' product ' + product)
            logger.info('To ' + clean_call.get_param('imagename'))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            if self._dry_run:
                return ()

            # Get fname dict
            fname_dict = self._fname_dict(product=product, imagename=clean_call.get_param('imagename'),
                                          imaging_method=imaging_method)

            # import_and_align_mask
            msr.import_and_align_mask(in_file=this_cleanmask,
                                      out_file=fname_dict['mask'],
                                      template=fname_dict['image'],
                                      blank_to_match=True)
            # AKL - propose to deprecate
            # clean_call.set_param('usemask','user')

            # if imaging_method == 'sdintimaging':
            #     os.system('cp -r %s %s' % (out_file, out_file.replace('.joint.cube', '.int.cube')))

            return ()

        @CleanCallFunctionDecorator
        def task_multiscale_clean(
                self,
                clean_call=None,
                imaging_method='tclean',
                convergence_fracflux=0.01,
                convergence_noise_z_threshold=None,
                backup=True,
        ):
            """
            Run a multiscale clean loop to convergence. This task
            currently hardcodes some of the PHANGS-ALMA best choice
            parameters.

            Set backup to True will make a copy of the multiscale cleaned
            image as {imagename}_multiscale.image
            """

            if clean_call.get_param('deconvolver') not in ['multiscale', 'mtmfs']:
                logger.warning("I expected a multiscale or mtmfs deconvolver but got " + str(
                    clean_call.get_param('deconvolver')) + ".")
                raise Exception("Incorrect clean call! Should have a multiscale or mtmfs deconvolver.")

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Running clean call to convergence for:")
            logger.info(clean_call.get_param('imagename'))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            if self._dry_run:
                return ()

            imr.clean_loop(clean_call=clean_call,
                           imaging_method=imaging_method,
                           record_file=clean_call.get_param('imagename') + '_multiscale_record.txt',
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
                           min_loops=3,
                           max_loops=20,
                           max_total_niter=None,
                           convergence_fracflux=convergence_fracflux,
                           convergence_totalflux=None,
                           convergence_fluxperniter=None,
                           use_absolute_delta=True,
                           stop_at_negative=True,
                           remask_each_loop=False,
                           force_dirty_image=False,
                           convergence_noise_z_threshold=convergence_noise_z_threshold,
                           )
            # log_ext='multiscale',
            if backup:
                imr.copy_imaging(
                    input_root=clean_call.get_param('imagename'),
                    output_root=clean_call.get_param('imagename') + '_multiscale',
                    imaging_method=imaging_method,
                    wipe_first=True)

            return ()

        @CleanCallFunctionDecorator
        def task_singlescale_mask(
                self,
                clean_call=None,
                product=None,
                imaging_method='tclean',
                high_snr=None,
                low_snr=None,
                absolute=False,
        ):
            """
            Create a signal-to-noise based mask within the existing clean
            mask for deep cleaning. Used before running a deep single
            scale clean.

            If high_snr is None, will default to 4.0 (line imaging), or 2.5 (continuum imaging)

            If low_snr is None, will default to 2.0 (line imaging), or 1.0 (continuum imaging)
            """

            if product is None:
                logger.error("Require a product. Returning.")
                raise Exception("Require a product. Returning.")
                return ()

            # Get fname dict
            fname_dict = self._fname_dict(product=product, imagename=clean_call.get_param('imagename'),
                                          imaging_method=imaging_method)

            # get imagename

            imagename = fname_dict['image']
            if not os.path.isdir(imagename):
                logger.error("Image not found: " + imagename)
                return ()

            # print message
            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Creating signal-to-noise based clean mask for:")
            logger.info(str(imagename))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            if self._dry_run:
                return()

            # check if line product
            is_line_product = product in self._kh.get_line_products()

            if high_snr is None:
                if is_line_product:
                    high_snr = 4.0
                else:
                    high_snr = 2.5
            if low_snr is None:
                if is_line_product:
                    low_snr = 2.0
                else:
                    low_snr = 1.0

            if is_line_product:

                # signal_mask
                msr.signal_mask(imaging_method=imaging_method,
                                cube_root=fname_dict['root'],
                                out_file=fname_dict['mask'],
                                suffix_in=fname_dict['suffix'],
                                suffix_out='',
                                operation='AND',
                                high_snr=high_snr,
                                low_snr=low_snr,
                                absolute=absolute)

            else:

                # for continuum product, we need to make 2D mask
                # and note that the mask has a suffix .mask.tt0

                # here we also use a lower high_snr and low_snr
                # for continuum cleaning

                # signal_mask
                msr.signal_mask(imaging_method=imaging_method,
                                cube_root=fname_dict['root'],
                                out_file=fname_dict['mask'],
                                suffix_in=fname_dict['suffix'],
                                suffix_out='',
                                operation='AND',
                                high_snr=high_snr,
                                low_snr=low_snr,
                                absolute=absolute,
                                do_roll=False)

            clean_call.set_param('usemask', 'user')

            return ()

        @CleanCallFunctionDecorator
        def task_singlescale_clean(
                self,
                clean_call=None,
                imaging_method='tclean',
                convergence_fracflux=0.01,
                threshold_value=1.0,
                convergence_noise_z_threshold=None,
                backup=True,
        ):
            """
            Run a singlescale clean loop to convergence. This task
            currently hardcodes some of the PHANGS-ALMA best choice
            parameters.

            Set backup to True will make a copy of the singlescale cleaned
            image as {imagename}_singlescale.image
            """

            if not (clean_call.get_param('deconvolver') in ['hogbom','mtmfs']):
                logger.warning("I expected a singlescale or mtmfs deconvolver but got "+str(clean_call.get_param('deconvolver'))+".")
                raise Exception("Incorrect clean call! Should have a hogbom or mtmfs deconvolver.")
                return ()

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Running clean call to convergence for:")
            logger.info(clean_call.get_param('imagename'))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            if self._dry_run:
                return ()

            imr.clean_loop(clean_call=clean_call,
                           imaging_method=imaging_method,
                           record_file=clean_call.get_param('imagename') + '_singlescale_record.txt',
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
                           threshold_value=threshold_value,
                           min_loops=3,
                           max_loops=20,
                           max_total_niter=None,
                           convergence_fracflux=convergence_fracflux,
                           convergence_totalflux=None,
                           convergence_fluxperniter=None,
                           use_absolute_delta=True,
                           stop_at_negative=False,
                           remask_each_loop=False,
                           force_dirty_image=False,
                           convergence_noise_z_threshold=convergence_noise_z_threshold,
                           )
            # log_ext='singlescale',
            if backup:
                imr.copy_imaging(
                    input_root=clean_call.get_param('imagename'),
                    output_root=clean_call.get_param('imagename') + '_singlescale',
                    imaging_method=imaging_method,
                    wipe_first=True)

            return ()

        @CleanCallFunctionDecorator
        def task_export_to_fits(
                self,
                clean_call=None,
                imaging_method='tclean',
                tag=None,
        ):
            """
            Export the results of a clean_call to FITS files. Optionally
            append _tag to the file.
            """

            image_root = str(clean_call.get_param('imagename'))
            if tag is not None:
                image_root += '_' + tag

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Exporting to FITS:" + image_root)
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            if not self._dry_run:
                imr.export_imaging_to_fits(image_root, imaging_method=imaging_method)

            return ()

        #############################
        # recipe_imaging_one_target #
        #############################

        def recipe_phangsalma_imaging(
                self,
                target=None,
                product=None,
                config=None,
                extra_ext_in=None,
                suffix_in=None,
                extra_ext_out=None,
                imaging_method='tclean',
                imaging_method_override=None,
                do_dirty_image=True,
                do_revert_to_dirty=True,
                do_read_clean_mask=True,
                do_multiscale_clean=True,
                do_revert_to_multiscale=True,
                do_singlescale_mask=True,
                singlescale_mask_high_snr=None,
                singlescale_mask_low_snr=None,
                singlescale_mask_absolute=False,
                do_singlescale_clean=True,
                do_revert_to_singlescale=True,
                do_export_to_fits=True,
                convergence_fracflux=0.01,
                convergence_noise_z_threshold=None,
                singlescale_threshold_value=1.0,
                dynamic_sizing=True,
                force_square=False,
                export_dirty=False,
                export_multiscale=False,
                overwrite=False,
        ):
            """
            PHANGS-ALMA basic imaging recipe.

            Major steps:

            (1) Dirty imaging
            (2) Align a broad clean mask (if present)
            (3) Lightly masked multiscale clean to S/N ~ 4
            (4) Heavily masked single scale clean until convergence
            (5) Export to FITS

            Operates by passing a "clean_call" object between steps. Can
            restart from any individual step.

            Input file names: {target}_{config}_{product}_{extra_ext_in}.ms{.suffix_in}

            Output file names: {target}_{config}_{product}_{extra_ext_out}.image

            imaging_method_override allows for some switches if you don't want to use a
            particular algorithm for a particular setup. This should be supplied as a
            dictionary containing 'target', 'config', and 'product' keys that match up
            with a particular setup (these can be lists or 'all'), and a 'new_imaging_method'
            key that the target/product/config setup should switch to (either tclean or sdintimaging)
            """

            cell = None
            imsize = None

            # Allow for manual overriding of configs if we need it
            if imaging_method_override is not None:
                target_override = imaging_method_override['target']
                config_override = imaging_method_override['config']
                product_override = imaging_method_override['product']

                if type(target_override) == str:
                    target_override = [target_override]
                if type(config_override) == str:
                    config_override = [config_override]
                if type(product_override) == str:
                    product_override = [product_override]

                if target in target_override or 'all' in target_override:
                    if config in config_override or 'all' in config_override:
                        if product_override in product_override or 'all' in product_override:
                            imaging_method = imaging_method_override['new_imaging_method']

            # These calls instantiate a clean call object with default
            # clean parameters. For the next few steps, we use the clean
            # call appropriate for stage "dirty" imaging.

            clean_call = self.task_initialize_clean_call(
                target=target, config=config, product=product,
                extra_ext_in=extra_ext_in, suffix_in=suffix_in,
                extra_ext_out=extra_ext_out,
                stage='dirty')

            if clean_call is None:
                logger.warning("I could not make a well-formed clean call.")
                return ()

            if dynamic_sizing:
                if cell is None or imsize is None:
                    cell, imsize = self.task_pick_cell_and_imsize(
                        clean_call=clean_call,
                        force_square=force_square,
                        check_overrides=True,
                        target=target, config=config, product=product,
                        )
                else:
                    clean_call.set_param('cell', cell, nowarning=True)
                    clean_call.set_param('imsize', imsize, nowarning=True)

            if imaging_method == 'sdintimaging':
                sd_fits_file = self._kh.get_sd_filename(target=target, product=product)
                feather_config = self._kh.get_feather_config_for_interf_config(interf_config=config)
                if not sd_fits_file or not feather_config:
                    logger.warning('No singledish setup for %s, %s, %s, reverting to standard tclean' %
                                   (target, product, config))
                    imaging_method = 'tclean'
                else:
                    sd_image_file = self.task_setup_sdintimaging(clean_call, target=target, product=product,
                                                                 overwrite=overwrite)
                    if not sd_image_file:
                        logger.error('Error in setting up singledish for sdintimaging')

                    # Set the clean call parameters as necessary.
                    clean_call.set_param('usedata', 'sdint')
                    clean_call.set_param('sdimage', sd_image_file)

                    # Catch the case where the frequency axis might go the wrong way round by specifying this exactly to
                    # the SD parameters
                    sdintlib = casaStuff.sdint_helper.SDINT_helper()
                    cube_params = sdintlib.setup_cube_params(sdcube=sd_image_file)
                    clean_call.set_param('nchan', cube_params['nchan'])
                    clean_call.set_param('start', cube_params['start'])
                    clean_call.set_param('width', cube_params['width'])

            # Make a dirty image (niter=0)

            if do_dirty_image:
                self.task_make_dirty_image(clean_call=clean_call, imaging_method=imaging_method)

            if do_export_to_fits and export_dirty:
                self.task_export_to_fits(clean_call=clean_call, imaging_method=imaging_method, tag='dirty')

            # Reset the current imaging to the dirty image.

            if do_revert_to_dirty:
                self.task_revert_to_imaging(clean_call=clean_call, imaging_method=imaging_method, tag='dirty')

            # Read and align the clean mask to the astrometry of the image.

            if do_read_clean_mask:
                self.task_read_clean_mask(
                    # AKL - propose to deprecate interaction with the clean_call here
                    clean_call=clean_call,
                    target=target, config=config, product=product,
                    imaging_method=imaging_method)

            # For the next few steps, we use the clean call appropriate
            # for stage "multiscale" imaging.

            clean_call = self.task_initialize_clean_call(
                target=target, config=config, product=product,
                imaging_method=imaging_method,
                extra_ext_in=extra_ext_in, suffix_in=suffix_in,
                extra_ext_out=extra_ext_out,
                stage='multiscale')

            # AKL - adding a check on the mask existence here

            # Check if we have a mask in place. This might happen if we
            # didn't find a clean mask or if we did not supply one. Either
            # way, now turn on a primary beam based mask.

            fname_dict = self._fname_dict(product=product, imagename=clean_call.get_param('imagename'),
                                          imaging_method=imaging_method)

            mask_name = fname_dict['mask']
            if not os.path.isdir(mask_name):
                logger.warning("I looked for this and did not find it: " + mask_name)
                if (clean_call.get_param('usemask') == "user") and (clean_call.get_param('mask') is not None) and (
                        clean_call.get_param('mask') != ''):
                    logger.debug("Using the user mask \"%s\" as defined in the clean parameter file." % (
                        clean_call.get_param('mask')))
                else:
                    logger.warning("No mask found, we will use a pblimit mask.")
                    clean_call.set_param('usemask', 'pb')
                    if clean_call.get_param('pblimit') is None:
                        clean_call.set_param('pbmask', 0.5)
                    else:
                        clean_call.set_param('pbmask', clean_call.get_param('pblimit'))

            # AKL - propose to deprecate
            # if not do_read_clean_mask:
            #    clean_call.set_param('usemask', 'pb')
            #    clean_call.set_param('pbmask', 0.2)

            # Dynamic sizing

            if dynamic_sizing:
                clean_call.set_param('cell', cell, nowarning=True)
                clean_call.set_param('imsize', imsize, nowarning=True)

            # Cube params for sdintimaging

            if imaging_method == 'sdintimaging':
                clean_call.set_param('nchan', cube_params['nchan'])
                clean_call.set_param('start', cube_params['start'])
                clean_call.set_param('width', cube_params['width'])

            # Look up the angular scales to clean for this config

            self.task_assign_multiscales(config=config, clean_call=clean_call)

            # Run a multiscale clean until it converges.

            if do_multiscale_clean:
                self.task_multiscale_clean(clean_call=clean_call,
                                           imaging_method=imaging_method,
                                           convergence_fracflux=convergence_fracflux,
                                           convergence_noise_z_threshold=convergence_noise_z_threshold,
                                           )

            if do_export_to_fits and export_multiscale:
                self.task_export_to_fits(clean_call=clean_call, imaging_method=imaging_method, tag='multiscale')

            # Reset the current imaging to the results of the multiscale clean.

            if do_revert_to_multiscale:
                self.task_revert_to_imaging(clean_call=clean_call, imaging_method=imaging_method, tag='multiscale')

            # For the next few steps, we use the clean call appropriate
            # for stage "singlescale" imaging.

            clean_call = self.task_initialize_clean_call(
                target=target, config=config, product=product,
                imaging_method=imaging_method,
                extra_ext_in=extra_ext_in, suffix_in=suffix_in,
                extra_ext_out=extra_ext_out,
                stage='singlescale')

            if dynamic_sizing:
                clean_call.set_param('cell', cell, nowarning=True)
                clean_call.set_param('imsize', imsize, nowarning=True)

            # Cube params for sdintimaging

            if imaging_method == 'sdintimaging':
                clean_call.set_param('nchan', cube_params['nchan'])
                clean_call.set_param('start', cube_params['start'])
                clean_call.set_param('width', cube_params['width'])

            # Make a signal-to-noise based mask for use in singlescale clean.

            if do_singlescale_mask:
                self.task_singlescale_mask(clean_call=clean_call, product=product, imaging_method=imaging_method,
                                           high_snr=singlescale_mask_high_snr, low_snr=singlescale_mask_low_snr,
                                           absolute=singlescale_mask_absolute)

            # Run a singlescale clean until it converges.

            if do_singlescale_clean:
                self.task_singlescale_clean(clean_call=clean_call,
                                            imaging_method=imaging_method,
                                            convergence_fracflux=convergence_fracflux,
                                            convergence_noise_z_threshold=convergence_noise_z_threshold,
                                            threshold_value=singlescale_threshold_value,
                                            )

            # Reset the current imaging to the results of the singlescale clean.

            if do_revert_to_singlescale:
                self.task_revert_to_imaging(clean_call=clean_call, imaging_method=imaging_method, tag='singlescale')

            # Export the products of the current clean to FITS files.

            if do_export_to_fits:
                self.task_export_to_fits(clean_call=clean_call, imaging_method=imaging_method)

            # Return

            return

        # end of recipe_phangsalma_imaging()


