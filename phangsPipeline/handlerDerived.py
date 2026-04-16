"""DerivativeHandler

This module creates signal masks based on image cubes, and then applies
the masks to make moment maps. This is done for each galaxy at multiple
spatial scales.

Example:
    $ ipython
    from phangsPipeline import handlerKeys as kh
    from phangsPipeline import handlerDerived as dh
    this_kh = kh.KeyHandler(master_key = 'phangsalma_keys/master_key.txt')
    this_dh = dh.DerivedHandler(key_handler = this_kh)
    this_dh.set_targets(only = ['ngc0628', 'ngc2997', 'ngc4321'])
    this_dh.set_no_interf_configs(no_interf = False)
    this_dh.set_interf_configs(only = ['7m'])
    this_dh.set_feather_configs(only = ['7m+tp'])
    this_dh.set_line_products(only = ['co21'])
    this_dh.set_no_cont_products(no_cont = True)
    this_dh.loop_make_products()

"""

import logging
import os

import astropy.units as u
import numpy as np

from . import handlerTemplate
from . import utilsFilenames
from .scConvolution import smooth_cube
from .scMaskingRoutines import recipe_phangs_strict_mask, recipe_phangs_broad_mask
from .scMoments import moment_generator
from .scNoiseRoutines import recipe_phangs_noise
from .scStackingRoutines import recipe_phangs_vfield, recipe_shuffle_cube

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DerivedHandler(handlerTemplate.HandlerTemplate):
    """
    Class to create signal masks based on image cubes, and then apply
    the masks to make moment maps. This is done for each galaxy at
    multiple spatial/angular scales.
    """

    ############
    # __init__ #
    ############

    def __init__(
            self,
            key_handler=None,
            dry_run=False,
    ):
        # inherit template class
        handlerTemplate.HandlerTemplate.__init__(self,
                                                 key_handler=key_handler,
                                                 dry_run=dry_run)

    ########################
    # Main processing loop #
    ########################

    def loop_derive_products(
            self,
            do_all=False,
            do_convolve=False,
            do_noise=False,
            do_strictmask=False,
            do_broadmask=False,
            do_moments=False,
            do_secondary=False,
            do_vfield=False,
            do_shuffling=False,
            do_flatmask=False,
            do_flatmaps=False,
            make_directories=True,
            extra_ext_in='',
            extra_ext_out='',
            overwrite=True,
    ):
        """
        Loops over the full set of targets, spectral products (note
        the dual definition of "product" here), and configurations to
        do the imaging. Toggle the parts of the loop using the do_XXX
        booleans. Other choices affect algorithms used.
        """

        if do_all:
            do_convolve = True
            do_noise = True
            do_strictmask = True
            do_broadmask = True
            do_moments = True
            do_secondary = True
            do_vfield = True,
            do_shuffling = True
            do_flatmask = True
            do_flatmaps = True

        # Error checking

        if len(self.get_targets()) == 0:
            logger.error("Need a target list.")
            return (None)

        if len(self.get_all_products()) == 0:
            logger.error("Need a products list.")
            return (None)

        # If requested, make the directories

        if make_directories:
            self._kh.make_missing_directories(derived=True)

        # Convolve the data to all requested angular and physical resolutions.

        if do_convolve:

            for this_target, this_product, this_config in \
                    self.looper(do_targets=True, do_products=True, do_configs=True):

                # Always start with the native resolution

                self.task_convolve(
                    target=this_target, config=this_config, product=this_product,
                    just_copy=True, overwrite=overwrite)

                # Loop over all angular and physical resolutions.

                res_dict = self._kh.get_ang_res_dict(
                    config=this_config, product=this_product)
                res_list = list(res_dict)
                if len(res_list) > 0:
                    res_list.sort()
                for this_res_tag in res_list:
                    this_res_value = res_dict[this_res_tag]
                    self.task_convolve(
                        target=this_target, config=this_config, product=this_product,
                        res_tag=this_res_tag, res_value=this_res_value, res_type='ang',
                        overwrite=overwrite)

                res_dict = self._kh.get_phys_res_dict(
                    config=this_config, product=this_product)
                res_list = list(res_dict)
                if len(res_list) > 0:
                    res_list.sort()
                for this_res_tag in res_list:
                    this_res_value = res_dict[this_res_tag]
                    self.task_convolve(
                        target=this_target, config=this_config, product=this_product,
                        res_tag=this_res_tag, res_value=this_res_value, res_type='phys',
                        overwrite=overwrite)

        # Estimate the noise for each cube.

        if do_noise:

            for this_target, this_product, this_config in \
                    self.looper(do_targets=True, do_products=True, do_configs=True):

                # Always start with the native resolution

                self.task_estimate_noise(
                    target=this_target, config=this_config, product=this_product,
                    overwrite=overwrite)

                # Loop over all angular and physical resolutions.

                res_dict = self._kh.get_ang_res_dict(
                    config=this_config, product=this_product)
                res_list = list(res_dict)
                if len(res_list) > 0:
                    res_list.sort()
                for this_res_tag in res_list:
                    self.task_estimate_noise(
                        target=this_target, config=this_config, product=this_product,
                        res_tag=this_res_tag, overwrite=overwrite)

                res_dict = self._kh.get_phys_res_dict(
                    config=this_config, product=this_product)
                res_list = list(res_dict)
                if len(res_list) > 0:
                    res_list.sort()
                for this_res_tag in res_list:
                    self.task_estimate_noise(
                        target=this_target, config=this_config, product=this_product,
                        res_tag=this_res_tag, overwrite=overwrite)

        # Make "strict" signal masks for each cube

        if do_strictmask:

            for this_target, this_product, this_config in \
                    self.looper(do_targets=True, do_products=True, do_configs=True):

                # Always start with the native resolution

                self.task_build_strict_mask(
                    target=this_target, config=this_config, product=this_product,
                    overwrite=overwrite, res_tag=None)

                # Loop over all angular and physical resolutions.

                for this_res in self._kh.get_ang_res_dict(
                        config=this_config, product=this_product):
                    self.task_build_strict_mask(
                        target=this_target, config=this_config, product=this_product,
                        overwrite=overwrite, res_tag=this_res)

                for this_res in self._kh.get_phys_res_dict(
                        config=this_config, product=this_product):
                    self.task_build_strict_mask(
                        target=this_target, config=this_config, product=this_product,
                        overwrite=overwrite, res_tag=this_res)

        # Make "broad" combination masks.

        if do_broadmask:

            for this_target, this_product, this_config in \
                    self.looper(do_targets=True, do_products=True, do_configs=True):
                # Only build one broad mask that covers all resolutions

                self.task_build_broad_mask(
                    target=this_target, config=this_config, product=this_product,
                    overwrite=overwrite, res_tag=None)

        # Make "moments" - derived data products.

        if do_moments:

            for this_target, this_product, this_config in \
                    self.looper(do_targets=True, do_products=True, do_configs=True):

                # Always start with the native resolution

                self.task_generate_moments(
                    target=this_target, product=this_product, config=this_config,
                    res_tag=None, overwrite=overwrite)

                # Loop over all angular and physical resolutions.

                for this_res in self._kh.get_ang_res_dict(
                        config=this_config, product=this_product):
                    self.task_generate_moments(
                        target=this_target, product=this_product, config=this_config,
                        res_tag=this_res, overwrite=overwrite)

                for this_res in self._kh.get_phys_res_dict(
                        config=this_config, product=this_product):
                    self.task_generate_moments(
                        target=this_target, product=this_product, config=this_config,
                        res_tag=this_res, overwrite=overwrite)

        if do_secondary:
            for this_target, this_product, this_config in \
                    self.looper(do_targets=True, do_products=True, do_configs=True):

                # Loop over all angular and physical resolutions.

                self.task_generate_secondary_moments(
                    target=this_target, product=this_product, config=this_config,
                    res_tag=None, overwrite=overwrite)

                for this_res in self._kh.get_ang_res_dict(
                        config=this_config, product=this_product):
                    self.task_generate_secondary_moments(
                        target=this_target, product=this_product, config=this_config,
                        res_tag=this_res, overwrite=overwrite)

                for this_res in self._kh.get_phys_res_dict(
                        config=this_config, product=this_product):
                    self.task_generate_secondary_moments(
                        target=this_target, product=this_product, config=this_config,
                        res_tag=this_res, overwrite=overwrite)

        # Create velocity field.

        if do_vfield:

            for this_target, this_product, this_config in \
                    self.looper(do_targets=True, do_products=True, do_configs=True):
                # Only build one velocity field that covers all resolutions

                self.task_build_vfield(
                    target=this_target, config=this_config, product=this_product,
                    overwrite=overwrite, res_tag=None)

        # Make shuffled cube.

        if do_shuffling:

            for this_target, this_product, this_config in \
                    self.looper(do_targets=True, do_products=True, do_configs=True):
                # Always start with the native resolution

                self.task_shuffle_cube(
                    target=this_target, config=this_config, product=this_product,
                    overwrite=overwrite)

                # Loop over all angular and physical resolutions.
                # N.B. we only want native resolution shuffled cubes;
                # feel free to uncomment the lines below for other resolutions

                # res_dict = self._kh.get_ang_res_dict(
                #     config=this_config,product=this_product)
                # res_list = list(res_dict)
                # if len(res_list) > 0:
                #     res_list.sort()
                # for this_res_tag in res_list:

                #     self.task_shuffle_cube(
                #         target=this_target, config=this_config, product=this_product,
                #         res_tag=this_res_tag, overwrite=overwrite)

                # res_dict = self._kh.get_phys_res_dict(
                #     config=this_config,product=this_product)
                # res_list = list(res_dict)
                # if len(res_list) > 0:
                #     res_list.sort()
                # for this_res_tag in res_list:

                #     self.task_shuffle_cube(
                #         target=this_target, config=this_config, product=this_product,
                #         res_tag=this_res_tag, overwrite=overwrite)

        # Make "flat masks" for each cube

        if do_flatmask:

            for this_target, this_product, this_config in \
                    self.looper(do_targets=True, do_products=True, do_configs=True):

                #
                # Flat Strict Mask

                # Always start with the native resolution

                self.task_build_flat_strict_mask(
                    target=this_target, config=this_config, product=this_product,
                    overwrite=overwrite, res_tag=None)

                # Loop over all angular and physical resolutions.

                for this_res in self._kh.get_ang_res_dict(
                        config=this_config, product=this_product):
                    self.task_build_flat_strict_mask(
                        target=this_target, config=this_config, product=this_product,
                        overwrite=overwrite, res_tag=this_res)

                for this_res in self._kh.get_phys_res_dict(
                        config=this_config, product=this_product):
                    self.task_build_flat_strict_mask(
                        target=this_target, config=this_config, product=this_product,
                        overwrite=overwrite, res_tag=this_res)

                #
                # Flat Broad Mask

                # Always start with the native resolution

                self.task_build_flat_broad_mask(
                    target=this_target, config=this_config, product=this_product,
                    overwrite=overwrite, res_tag=None)

                # Loop over all angular and physical resolutions.

                for this_res in self._kh.get_ang_res_dict(
                        config=this_config, product=this_product):
                    self.task_build_flat_broad_mask(
                        target=this_target, config=this_config, product=this_product,
                        overwrite=overwrite, res_tag=this_res)

                for this_res in self._kh.get_phys_res_dict(
                        config=this_config, product=this_product):
                    self.task_build_flat_broad_mask(
                        target=this_target, config=this_config, product=this_product,
                        overwrite=overwrite, res_tag=this_res)

        # Make "flat maps" - derived data products.

        if do_flatmaps:

            for this_target, this_product, this_config in \
                    self.looper(do_targets=True, do_products=True, do_configs=True):

                # Always start with the native resolution

                self.task_generate_flatmaps(
                    target=this_target, product=this_product, config=this_config,
                    res_tag=None, overwrite=overwrite)

                # Loop over all angular and physical resolutions.

                for this_res in self._kh.get_ang_res_dict(
                        config=this_config, product=this_product):
                    self.task_generate_flatmaps(
                        target=this_target, product=this_product, config=this_config,
                        res_tag=this_res, overwrite=overwrite)

                for this_res in self._kh.get_phys_res_dict(
                        config=this_config, product=this_product):
                    self.task_generate_flatmaps(
                        target=this_target, product=this_product, config=this_config,
                        res_tag=this_res, overwrite=overwrite)

    # end of loop

    ###########################################
    # Defined file names for various products #
    ###########################################

    def _fname_dict(
            self,
            target=None,
            config=None,
            product=None,
            res_tag=None,
            extra_ext_in='',
            extra_ext_out='',
    ):
        """
        Function to define file names used in other functions.
        """

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Error checking
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        if target is None:
            raise Exception("Need a target.")
        if product is None:
            raise Exception("Need a product.")
        if config is None:
            raise Exception("Need a config.")

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Initialize
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        # The output is a nested dictionary structure, for each cube
        # resolution (res_tag)

        fname_dict = {}

        # Resolution string (if any, not required)
        fname_dict['res_tag'] = res_tag
        if res_tag is None:
            res_tag = ''

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Original files
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        orig_filename = utilsFilenames.get_cube_filename(
            target=target, config=config, product=product,
            ext='pbcorr_trimmed_k' + extra_ext_in,
            casa=False)

        fname_dict['orig'] = orig_filename

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Output Convolved Cubes
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        cube_filename = utilsFilenames.get_cube_filename(
            target=target, config=config, product=product,
            ext=res_tag + extra_ext_out,
            casa=False)

        coverage_filename = utilsFilenames.get_cube_filename(
            target=target, config=config, product=product,
            ext=res_tag + '_coverage' + extra_ext_out,
            casa=False)

        coverage2d_filename = utilsFilenames.get_cube_filename(
            target=target, config=config, product=product,
            ext=res_tag + '_coverage2d' + extra_ext_out,
            casa=False)

        fname_dict['cube'] = cube_filename
        fname_dict['coverage'] = coverage_filename
        fname_dict['coverage2d'] = coverage2d_filename

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Velocity field
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        vfield_filename = self._kh.get_vfield_file_for_target(target=target)

        fname_dict['vfield'] = vfield_filename

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Shuffled Cubes
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        shuffled_filename = utilsFilenames.get_cube_filename(
            target=target, config=config, product=product,
            ext=res_tag + extra_ext_out + '_shuffled',
            casa=False)

        fname_dict['shuffled'] = shuffled_filename

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Noise Cubes
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        noise_filename = utilsFilenames.get_cube_filename(
            target=target, config=config, product=product,
            ext=res_tag + extra_ext_out + '_noise',
            casa=False)

        fname_dict['noise'] = noise_filename

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Signal Mask
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        # This differs by resolution

        strictmask_filename = utilsFilenames.get_cube_filename(
            target=target, config=config, product=product,
            ext=res_tag + extra_ext_out + '_strictmask',
            casa=False)

        fname_dict['strictmask'] = strictmask_filename

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Broad / Hybrid Mask
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        # Note that the broadmask is the same across all resolutions.

        broadmask_filename = utilsFilenames.get_cube_filename(
            target=target, config=config, product=product,
            ext=extra_ext_out + '_broadmask',
            casa=False)

        fname_dict['broadmask'] = broadmask_filename

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Flat Strict Mask
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        flatstrictmask_filename = utilsFilenames.get_cube_filename(
            target=target, config=config, product=product,
            ext=extra_ext_out + '_flatstrictmask',
            casa=False)

        fname_dict['flatstrictmask'] = flatstrictmask_filename

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Flat Broad Mask
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        flatbroadmask_filename = utilsFilenames.get_cube_filename(
            target=target, config=config, product=product,
            ext=extra_ext_out + '_flatbroadmask',
            casa=False)

        fname_dict['flatbroadmask'] = flatbroadmask_filename

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Moments
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        # Just note the root of the moment file name. There are a lot
        # of extensions that will be filled in by the programs.

        moment_root = utilsFilenames.get_cube_filename(
            target=target, config=config, product=product,
            ext=res_tag + extra_ext_out)
        moment_root = moment_root.replace('.fits', '')
        fname_dict['momentroot'] = moment_root

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Return
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        return (fname_dict)

    ##################################################################
    # Tasks - discrete steps on target, product, config combinations #
    ##################################################################

    # region convolve

    def task_convolve(
            self,
            target=None,
            config=None,
            product=None,
            res_tag=None,
            res_value=None,
            res_type='ang',
            just_copy=False,
            extra_ext_in='',
            extra_ext_out='',
            overwrite=False,
            tol=0.1,
            nan_treatment='interpolate',
    ):
        """
        Convolve data to lower resolutions. Defaults to copying in some cases.
        """

        # Parse the input resolution
        if not just_copy:

            if res_value is None:
                logger.warning("Need an input resolution.")
                logger.warning("Defaulting to copy mode.")
                just_copy = True

            if res_type.lower() not in ['ang', 'phys']:
                logger.warning("Input resolution can be angular or physical, ang or phys .")
                logger.warning("Defaulting to copy mode.")
                just_copy = True

            if res_tag is None:
                logger.warning("Need a resolution tag to avoid overlapping file names.")
                logger.warning("Defaulting to copy mode.")
                just_copy = True

        # Generate file names

        indir = self._kh.get_postprocess_dir_for_target(target=target, changeto=False)
        indir = os.path.abspath(indir) + '/'

        outdir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        outdir = os.path.abspath(outdir) + '/'

        fname_dict_in = self._fname_dict(
            target=target, config=config, product=product, res_tag=None,
            extra_ext_in=extra_ext_in)

        if just_copy:
            fname_dict_out = self._fname_dict(
                target=target, config=config, product=product, res_tag=None,
                extra_ext_out=extra_ext_out)
        else:
            fname_dict_out = self._fname_dict(
                target=target, config=config, product=product, res_tag=res_tag,
                extra_ext_out=extra_ext_out)

        input_file = fname_dict_in['orig']
        outfile = fname_dict_out['cube']
        coveragefile = fname_dict_out['coverage']
        coverage2dfile = fname_dict_out['coverage2d']

        # Check input file existence

        if not (os.path.isfile(indir + input_file)):
            logger.warning("Missing " + indir + input_file)
            return ()

        # Access keywords for mask generation

        convolve_kwargs = self._kh.get_derived_kwargs(
            config=config, product=product, kwarg_type='convolve_kw'
        )

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Copying or convolving cube for:")
        logger.info(str(target) + " , " + str(product) + " , " + str(config))
        if just_copy:
            logger.info("... mode is just copying.")
        else:
            logger.info("... mode is convolving.")
            logger.info("... to resolution tag: " + res_tag)
            logger.info("... which is resolution type: " + res_type)
            logger.info("... and value: " + str(res_value))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        logger.info("Input file: " + input_file)
        logger.info("Target file: " + outfile)
        logger.info("Coverage file: " + coveragefile)
        logger.info("Keyword arguments: " + str(convolve_kwargs))

        if (not self._dry_run):

            if just_copy:

                if overwrite:
                    os.system('rm -rf ' + outdir + outfile)

                if (os.path.isfile(outdir + outfile)):
                    logger.warning("Target file already present " + outdir + outfile)
                    return ()

                os.system('cp -r ' + indir + input_file + ' ' + outdir + outfile)

            else:
                huge_cube = True
                logger.debug(f"Setting huge_cube to {huge_cube} for channel-wise processing")

                if 'tol' in convolve_kwargs:
                    tol = convolve_kwargs['tol']

                if 'nan_treatment' in convolve_kwargs:
                    nan_treatment = convolve_kwargs['nan_treatment']

                if res_type == 'ang':
                    input_res_value = res_value * u.arcsec
                    smooth_cube(incube=indir + input_file, outfile=outdir + outfile,
                                angular_resolution=input_res_value,
                                tol=tol, nan_treatment=nan_treatment,
                                make_coverage_cube=True, coveragefile=outdir + coveragefile,
                                collapse_coverage=True, coverage2dfile=outdir + coverage2dfile,
                                overwrite=overwrite, 
                                huge_cube=huge_cube)

                if res_type == 'phys':
                    this_distance = self._kh.get_distance_for_target(target)
                    if this_distance is None:
                        logger.error("No distance for target " + target)
                        return ()
                    this_distance = this_distance * 1e6 * u.pc
                    input_res_value = res_value * u.pc
                    smooth_cube(incube=indir + input_file, outfile=outdir + outfile,
                                linear_resolution=input_res_value, distance=this_distance,
                                tol=tol, nan_treatment=nan_treatment,
                                make_coverage_cube=True, coveragefile=outdir + coveragefile,
                                collapse_coverage=True,
                                overwrite=overwrite, 
                                huge_cube=huge_cube)

        return ()

    # endregion

    # region noise

    def task_estimate_noise(
            self,
            target=None,
            config=None,
            product=None,
            res_tag=None,
            extra_ext='',
            overwrite=False,
    ):
        """
        Estimate the noise associated with a data cube and save it to disk.
        """

        # Generate file names

        indir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        indir = os.path.abspath(indir) + '/'

        outdir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        outdir = os.path.abspath(outdir) + '/'

        fname_dict = self._fname_dict(
            target=target, config=config, product=product, res_tag=res_tag,
            extra_ext_in=extra_ext)

        input_file = fname_dict['cube']
        outfile = fname_dict['noise']

        # Check input file existence

        if not (os.path.isfile(indir + input_file)):
            logger.warning("Missing " + indir + input_file)
            return ()

        # Access keywords for noise generation

        noise_kwargs = self._kh.get_derived_kwargs(
            config=config, product=product, kwarg_type='noise_kw')

        # Report

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Running a noise estimate for:")
        logger.info(str(target) + " , " + str(product) + " , " + str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        logger.info("Input file: " + input_file)
        logger.info("Target file: " + outfile)
        logger.info("Keyword arguments: " + str(noise_kwargs))

        # Call noise routines

        if (not self._dry_run):
            recipe_phangs_noise(
                incube=indir + input_file,
                outfile=outdir + outfile,
                noise_kwargs=noise_kwargs,
                return_spectral_cube=False,
                overwrite=overwrite)

    # endregion

    # region strict mask

    def task_build_strict_mask(
            self,
            target=None,
            config=None,
            product=None,
            res_tag=None,
            extra_ext='',
            overwrite=False,
    ):
        """
        Estimate the noise associated with a data cube and save it to disk.
        """

        # Generate file names

        indir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        indir = os.path.abspath(indir) + '/'

        outdir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        outdir = os.path.abspath(outdir) + '/'

        fname_dict = self._fname_dict(
            target=target, config=config, product=product, res_tag=res_tag,
            extra_ext_in=extra_ext)

        input_file = fname_dict['cube']
        noise_file = fname_dict['noise']
        coverage_file = fname_dict['coverage']
        coverage2d_file = fname_dict['coverage2d']

        outfile = fname_dict['strictmask']

        # Check input file existence

        if not (os.path.isfile(indir + input_file)):
            logger.warning("Missing cube: " + indir + input_file)
            return ()

        if not (os.path.isfile(indir + noise_file)):
            logger.warning("Missing noise estimate: " + indir + noise_file)
            return ()

        # Coverage

        if not (os.path.isfile(indir + coverage_file)):
            logger.warning("Missing coverage estimate: " + indir + coverage_file)
            logger.warning("This may be fine. Proceeding.")
            coverage_file = None

        # Access keywords for mask generation

        strictmask_kwargs = self._kh.get_derived_kwargs(
            config=config, product=product, kwarg_type='strictmask_kw'
        )

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Report
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Creating a strict mask for:")
        logger.info(str(target) + " , " + str(product) + " , " + str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        logger.info("Input file: " + input_file)
        logger.info("Noise file: " + noise_file)
        if coverage_file is not None:
            logger.info("Coverage file: " + coverage_file)
        logger.info("Target file: " + outfile)
        logger.info("Keyword arguments: " + str(strictmask_kwargs))

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Call the masking routines
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        if (not self._dry_run):

            # ... put the directory into the name to allow it to remain
            # None when missing.
            if coverage_file is not None:
                coverage_file_in = indir + coverage_file
            else:
                coverage_file_in = None

            recipe_phangs_strict_mask(
                indir + input_file,
                innoise=indir + noise_file,
                coverage=coverage_file_in,
                outfile=outdir + outfile,
                mask_kwargs=strictmask_kwargs,
                return_spectral_cube=False,
                overwrite=overwrite)

    # endregion

    # region broad mask

    def task_build_broad_mask(
            self,
            target=None,
            config=None,
            product=None,
            res_tag=None,
            extra_ext='',
            overwrite=False,
    ):
        """
        Estimate the noise associated with a data cube and save it to disk.
        """

        # Generate file names

        indir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        indir = os.path.abspath(indir) + '/'

        outdir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        outdir = os.path.abspath(outdir) + '/'

        fname_dict = self._fname_dict(
            target=target, config=config, product=product, res_tag=res_tag,
            extra_ext_in=extra_ext)

        input_file = fname_dict['strictmask']
        outfile = fname_dict['broadmask']

        # Check input file existence

        if not (os.path.isfile(indir + input_file)):
            logger.warning("Missing cube: " + indir + input_file)
            return ()

        # Access keywords for mask generation

        broadmask_kwargs = self._kh.get_derived_kwargs(
            config=config, product=product, kwarg_type='broadmask_kw'
        )

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Create the list of masks to combine
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        list_of_masks = []

        linked_configs = self._kh.get_linked_mask_configs(
            config=config, product=product)

        if config not in linked_configs:
            linked_configs.append(config)

        for cross_config in linked_configs:

            fname_dict = self._fname_dict(
                target=target, config=cross_config, product=product, res_tag=None,
                extra_ext_in=extra_ext)

            this_mask = fname_dict['strictmask']
            if this_mask not in list_of_masks:
                if os.path.isfile(indir + this_mask):
                    list_of_masks.append(indir + this_mask)

            # Loop over all angular and physical resolutions.

            for this_res in self._kh.get_ang_res_dict(
                    config=cross_config, product=product):

                fname_dict = self._fname_dict(
                    target=target, config=cross_config, product=product, res_tag=this_res,
                    extra_ext_in=extra_ext)

                this_mask = fname_dict['strictmask']
                if this_mask not in list_of_masks:
                    if os.path.isfile(indir + this_mask):
                        list_of_masks.append(indir + this_mask)

            for this_res in self._kh.get_phys_res_dict(
                    config=cross_config, product=product):

                fname_dict = self._fname_dict(
                    target=target, config=cross_config, product=product, res_tag=this_res,
                    extra_ext_in=extra_ext)

                this_mask = fname_dict['strictmask']
                if this_mask not in list_of_masks:
                    if os.path.isfile(indir + this_mask):
                        list_of_masks.append(indir + this_mask)

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Report
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Creating a broad mask for:")
        logger.info(str(target) + " , " + str(product) + " , " + str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        logger.info("Input file: " + input_file)
        logger.info("List of other masks " + str(list_of_masks))
        logger.info("Target file: " + outfile)
        logger.info("Keyword arguments: " + str(broadmask_kwargs))

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Call the mask combining routine
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        if (not self._dry_run):
            recipe_phangs_broad_mask(
                indir + input_file,
                list_of_masks=list_of_masks,
                outfile=outdir + outfile,
                # mask_kwargs=broadmask_kwargs,
                # return_spectral_cube=False,
                overwrite=overwrite)

    # endregion

    # region moments

    def task_generate_moments(
            self,
            target=None,
            config=None,
            product=None,
            res_tag=None,
            extra_ext='',
            overwrite=False,
    ):
        """
        Generate moment maps.
        """
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Look up filenames, list of moments, etc.
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        # Generate file names

        indir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        indir = os.path.abspath(indir) + '/'

        outdir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        outdir = os.path.abspath(outdir) + '/'

        # Filenames

        fname_dict_nores = self._fname_dict(
            target=target, config=config, product=product, res_tag=None,
            extra_ext_in=extra_ext)

        fname_dict = self._fname_dict(
            target=target, config=config, product=product, res_tag=res_tag,
            extra_ext_in=extra_ext)

        # ... broad mask never has a resolution tag

        broadmask_file = fname_dict_nores['broadmask']

        # ... files with resolution tag

        input_file = fname_dict['cube']
        noise_file = fname_dict['noise']
        strictmask_file = fname_dict['strictmask']

        outroot = fname_dict['momentroot']

        # Check input file and mask existence

        if not (os.path.isfile(indir + input_file)):
            logger.warning("Missing cube: " + indir + input_file)
            return ()

        found_broadmask = (os.path.isfile(indir + broadmask_file))
        found_strictmask = (os.path.isfile(indir + strictmask_file))

        # Look up which moments to calculate

        list_of_moments = self._kh.get_moment_list(config=config, product=product)

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Report
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Generating moment maps for:")
        logger.info(str(target) + " , " + str(product) + " , " + str(config))
        if res_tag is not None:
            logger.info("Resolution " + str(res_tag))
        logger.info("Found a strict mask? " + str(found_strictmask))
        logger.info("Found a broad mask? " + str(found_broadmask))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        logger.info("... input file: " + input_file)
        logger.info("... noise file: " + noise_file)
        logger.info("... strict mask file: " + strictmask_file)
        logger.info("... broad mask file: " + broadmask_file)
        logger.info("... list of moments: " + str(list_of_moments))
        logger.info("... output root: " + outroot)

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Execute
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        rounds = []
        for this_mom in list_of_moments:
            rounds.append(self._kh.get_params_for_moment(this_mom)['round'])

        uniqrounds = sorted(list(set(rounds)))

        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        logger.info("... Total stages of moment calculation: {0}".format(len(uniqrounds)))
        logger.info("... First, calculate all moments in stage {0}".format(np.min(uniqrounds)))
        logger.info("")

        sublist_of_moments = [this_mom for this_mom in list_of_moments
                              if self._kh.get_params_for_moment(this_mom)['round'] == uniqrounds[0]]

        if (not self._dry_run):
            for this_mom in sublist_of_moments:
                logger.info('... generating moment: ' + str(this_mom))

                mom_params = self._kh.get_params_for_moment(this_mom)

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # Look up mask
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                if mom_params['mask'] is None:
                    mask_file = None
                elif mom_params['mask'].strip().lower() == 'none':
                    mask_file = None
                elif mom_params['mask'] == 'strictmask':
                    if not found_strictmask:
                        logger.warning("Strict mask needed but not found. Skipping.")
                        continue
                    mask_file = indir + strictmask_file
                elif mom_params['mask'] == 'broadmask':
                    if not found_broadmask:
                        logger.warning("Broad mask needed but not found. Skipping.")
                        continue
                    mask_file = indir + broadmask_file
                else:
                    logger.warning("Mask choice not recognized for moment: " + str(this_mom))
                    logger.warning("Skipping.")
                    continue

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # Check noise
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                if not (os.path.isfile(indir + noise_file)):
                    logger.warning("Missing noise: " + indir + noise_file)
                    noise_in = None
                    errorfile = None
                else:
                    noise_in = indir + noise_file
                    errorfile = outdir + outroot + mom_params['ext_error'] + '.fits'

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # Set up output file
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                outfile = outdir + outroot + mom_params['ext'] + '.fits'

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # In the first round, just call the moment generator
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                moment_generator(
                    indir + input_file, mask=mask_file, noise=noise_in,
                    moment=mom_params['algorithm'], momkwargs=mom_params['kwargs'],
                    outfile=outfile, errorfile=errorfile,
                    channel_correlation=None)

    # endregion

    # region secondary moments

    def task_generate_secondary_moments(
            self,
            target=None,
            config=None,
            product=None,
            res_tag=None,
            extra_ext='',
            overwrite=False,
    ):
        """
        Generate moment maps.
        """
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Look up filenames, list of moments, etc.
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        # Generate file names

        indir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        indir = os.path.abspath(indir) + '/'

        outdir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        outdir = os.path.abspath(outdir) + '/'

        # Filenames

        fname_dict_nores = self._fname_dict(
            target=target, config=config, product=product, res_tag=None,
            extra_ext_in=extra_ext)

        fname_dict = self._fname_dict(
            target=target, config=config, product=product, res_tag=res_tag,
            extra_ext_in=extra_ext)

        # ... broad mask never has a resolution tag

        broadmask_file = fname_dict_nores['broadmask']

        # ... files with resolution tag

        input_file = fname_dict['cube']
        noise_file = fname_dict['noise']
        strictmask_file = fname_dict['strictmask']

        outroot = fname_dict['momentroot']

        # Check input file and mask existence

        if not (os.path.isfile(indir + input_file)):
            logger.warning("Missing cube: " + indir + input_file)
            return ()

        found_broadmask = (os.path.isfile(indir + broadmask_file))
        found_strictmask = (os.path.isfile(indir + strictmask_file))

        # Look up which moments to calculate

        list_of_moments = self._kh.get_moment_list(config=config, product=product)

        rounds = []
        for this_mom in list_of_moments:
            rounds.append(self._kh.get_params_for_moment(this_mom)['round'])
        uniqrounds = sorted(list(set(rounds)))

        if len(uniqrounds) == 1:
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("")
            logger.info("... Total stages of moment calculation: {0}".format(len(uniqrounds)))
            logger.info("... Secondary moments requested but not specified in moment keys")
            logger.info("... Returning to main loop")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("")
            return ()

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Report
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Generating moment maps for:")
        logger.info(str(target) + " , " + str(product) + " , " + str(config))
        if res_tag is not None:
            logger.info("Resolution " + str(res_tag))
        logger.info("Found a strict mask? " + str(found_strictmask))
        logger.info("Found a broad mask? " + str(found_broadmask))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        logger.info("... input file: " + input_file)
        logger.info("... noise file: " + noise_file)
        logger.info("... strict mask file: " + strictmask_file)
        logger.info("... broad mask file: " + broadmask_file)
        logger.info("... list of moments: " + str(list_of_moments))
        logger.info("... output root: " + outroot)

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Execute
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        logger.info("... Total stages of moment calculation: {0}".format(len(uniqrounds)))

        for thisround in uniqrounds[1:2]:
            logger.info("... Now, calculate all moments in stage {0}".format(thisround))
            logger.info("")

            sublist_of_moments = [this_mom for this_mom in list_of_moments
                                  if self._kh.get_params_for_moment(this_mom)['round'] == thisround]

            if (not self._dry_run):
                for this_mom in sublist_of_moments:
                    logger.info('... generating moment: ' + str(this_mom))

                    proceed = True

                    mom_params = self._kh.get_params_for_moment(this_mom)

                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # Look up mask
                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    if mom_params['mask'] is None:
                        mask_file = None
                    elif mom_params['mask'].strip().lower() == 'none':
                        mask_file = None
                    elif mom_params['mask'] == 'strictmask':
                        if not found_strictmask:
                            logger.warning("Strict mask needed but not found. Skipping.")
                            continue
                        mask_file = indir + strictmask_file
                    elif mom_params['mask'] == 'broadmask':
                        if not found_broadmask:
                            logger.warning("Broad mask needed but not found. Skipping.")
                            continue
                        mask_file = indir + broadmask_file
                    else:
                        logger.warning("Mask choice not recognized for moment: " + str(this_mom))
                        logger.warning("Skipping.")
                        continue

                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # Check noise
                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    if not (os.path.isfile(indir + noise_file)):
                        logger.warning("Missing noise: " + indir + noise_file)
                        noise_in = None
                        errorfile = None
                    else:
                        noise_in = indir + noise_file
                        errorfile = outdir + outroot + mom_params['ext_error'] + '.fits'

                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # Set up output file
                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    outfile = outdir + outroot + mom_params['ext'] + '.fits'

                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # Look up maps to pass and build the kwarg list
                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    kwargs_dict = mom_params['kwargs']

                    maps_to_pass = mom_params['maps_to_pass']
                    for map_ext in maps_to_pass:

                        # File name for this extension
                        this_map_file = outroot + '_' + map_ext + '.fits'

                        # Verify file existence
                        if not (os.path.isfile(indir + this_map_file)):
                            logger.warning("Missing needed context file: " + indir + this_map_file)
                            proceed = False

                        # Add as param to kwarg dict
                        kwargs_dict[map_ext] = indir + this_map_file

                    other_exts = mom_params['other_exts']
                    for param_name in other_exts.keys():

                        # File name for this extension
                        this_ext = other_exts[param_name]

                        # Code to look up file name
                        this_ext_file = str(target) + this_ext

                        # Verify file existence
                        if not (os.path.isfile(indir + this_ext_file)):
                            logger.warning("Missing needed context file: " + indir + this_ext_file)
                            proceed = False

                        # Add as param to kwarg dict
                        kwargs_dict[param_name] = indir + this_ext_file

                    if proceed == False:
                        logger.warning("Missing some needed information. Skipping this calculation.")
                        continue

                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # Call the moment generator
                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    # print(kwargs_dict)

                    moment_generator(
                        indir + input_file, mask=mask_file, noise=noise_in,
                        outfile=outfile, errorfile=errorfile,
                        channel_correlation=None,
                        moment=mom_params['algorithm'],
                        momkwargs=kwargs_dict,
                        # Deprecated context
                    )

    # endregion

    # region vfield
    def task_build_vfield(
            self,
            target=None,
            config=None,
            product=None,
            res_tag=None,
            extra_ext='',
            overwrite=False,
    ):
        """
        Generate a combined velocity field from a list of mom1 maps.
        """

        # Generate file names

        indir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        indir = os.path.abspath(indir) + '/'

        outdir = self._kh.get_vfield_dir_for_target(target=target, changeto=False)
        outdir = os.path.abspath(outdir) + '/'

        fname_dict = self._fname_dict(
            target=target, config=config, product=product, res_tag=res_tag,
            extra_ext_in=extra_ext)

        input_file = fname_dict['momentroot'] + '_mom1wprior.fits'
        outfile = fname_dict['vfield']

        # Check input file existence

        if not (os.path.isfile(indir + input_file)):
            logger.warning("Missing cube: " + indir + input_file)
            return ()

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Create the list of maps to combine
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        list_of_vfiels = []

        linked_configs = self._kh.get_linked_mask_configs(
            config=config, product=product)

        if config not in linked_configs:
            linked_configs.append(config)

        for cross_config in linked_configs:

            fname_dict = self._fname_dict(
                target=target, config=cross_config, product=product, res_tag=None,
                extra_ext_in=extra_ext)

            this_vfield = fname_dict['momentroot'] + '_mom1wprior.fits'
            if this_vfield not in list_of_vfiels:
                if os.path.isfile(indir + this_vfield):
                    list_of_vfiels.append(indir + this_vfield)

            # Loop over all angular and physical resolutions.

            for this_res in self._kh.get_ang_res_dict(
                    config=cross_config, product=product):

                fname_dict = self._fname_dict(
                    target=target, config=cross_config, product=product, res_tag=this_res,
                    extra_ext_in=extra_ext)

                this_vfield = fname_dict['momentroot'] + '_mom1wprior.fits'
                if this_vfield not in list_of_vfiels:
                    if os.path.isfile(indir + this_vfield):
                        list_of_vfiels.append(indir + this_vfield)

            for this_res in self._kh.get_phys_res_dict(
                    config=cross_config, product=product):

                fname_dict = self._fname_dict(
                    target=target, config=cross_config, product=product, res_tag=this_res,
                    extra_ext_in=extra_ext)

                this_vfield = fname_dict['momentroot'] + '_mom1wprior.fits'
                if this_vfield not in list_of_vfiels:
                    if os.path.isfile(indir + this_vfield):
                        list_of_vfiels.append(indir + this_vfield)

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Report
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Creating a velocity field for:")
        logger.info(str(target) + " , " + str(product) + " , " + str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        logger.info("Input file: " + input_file)
        logger.info("List of other velocity fields:")
        for this_vfield in list_of_vfiels:
            logger.info(str(this_vfield))
        logger.info("Target file: " + outfile)

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Call the vfield combining routine
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        if (not self._dry_run):
            recipe_phangs_vfield(
                indir + input_file,
                list_of_vfields=list_of_vfiels,
                outfile=outdir + outfile,
                overwrite=overwrite)

    # endregion

    # region shuffle cube

    def task_shuffle_cube(
            self,
            target=None,
            config=None,
            product=None,
            res_tag=None,
            extra_ext='',
            overwrite=False,
    ):
        """
        Construct shuffled cube and save it to disk.
        """

        # Generate file names

        indir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        indir = os.path.abspath(indir) + '/'

        outdir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        outdir = os.path.abspath(outdir) + '/'

        vfield_dir = self._kh.get_vfield_dir_for_target(target=target, changeto=False)
        vfield_dir = os.path.abspath(vfield_dir) + '/'

        fname_dict = self._fname_dict(
            target=target, config=config, product=product, res_tag=res_tag,
            extra_ext_in=extra_ext)

        input_file = fname_dict['cube']
        vfield_file = fname_dict['vfield']
        outfile = fname_dict['shuffled']

        # Check input file existence

        if not (os.path.isfile(indir + input_file)):
            logger.warning("Missing cube: " + indir + input_file)
            return ()

        if not (os.path.isfile(vfield_dir + vfield_file)):
            logger.warning("Missing velocity field: " + vfield_dir + vfield_file)
            return ()

        # Access keywords for noise generation

        shuffle_kwargs = self._kh.get_derived_kwargs(
            config=config, product=product, kwarg_type='shuffle_kw')

        # Report

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Running shuffling for:")
        logger.info(str(target) + " , " + str(product) + " , " + str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        logger.info("Input file: " + input_file)
        logger.info("Target file: " + outfile)
        logger.info("Keyword arguments: " + str(shuffle_kwargs))

        # Call shuffling routine

        recipe_shuffle_cube(
            cube_in=indir + input_file,
            vfield_in=vfield_dir + vfield_file,
            outfile=outdir + outfile,
            overwrite=overwrite)

    # endregion

    # region flat strict mask

    def task_build_flat_strict_mask(
            self,
            target=None,
            config=None,
            product=None,
            res_tag=None,
            extra_ext='',
            overwrite=False,
    ):
        """
        Construct the flat strict mask and save it to disk.
        """

        # Generate file names

        indir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        indir = os.path.abspath(indir) + '/'

        outdir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        outdir = os.path.abspath(outdir) + '/'

        vfield_dir = self._kh.get_vfield_dir_for_target(target=target, changeto=False)
        vfield_dir = os.path.abspath(vfield_dir) + '/'

        fname_dict = self._fname_dict(
            target=target, config=config, product=product, res_tag=res_tag,
            extra_ext_in=extra_ext)

        input_file = fname_dict['cube']
        vfield_file = fname_dict['vfield']
        mask_file = fname_dict['strictmask']
        coverage_file = fname_dict['coverage']
        coverage2d_file = fname_dict['coverage2d']

        outfile = fname_dict['flatstrictmask']

        # Check input file existence

        if not (os.path.isfile(indir + input_file)):
            logger.warning("Missing cube: " + indir + input_file)
            return ()

        if not (os.path.isfile(vfield_dir + vfield_file)):
            logger.warning("Missing velocity field: " + vfield_dir + vfield_file)
            return ()

        if not (os.path.isfile(indir + mask_file)):
            logger.warning("Missing strict mask: " + indir + mask_file)
            return ()

        # Coverage

        if not (os.path.isfile(indir + coverage_file)):
            logger.warning("Missing coverage estimate: " + indir + coverage_file)
            logger.warning("This may be fine. .")
            coverage_file = None

        # Access keywords for mask generation

        flatmask_kwargs = self._kh.get_derived_kwargs(
            config=config, product=product, kwarg_type='flatstrictmask_kw')

        # get velocity window from separate key if not provided from derived key
        if not flatmask_kwargs:
            this_window = self._kh.get_window_for_target(target)
            if this_window is None:
                logger.error("No velocity window for target " + target)
                return ()
            flatmask_kwargs = {'v_window': this_window}

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Report
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Creating a flat strict mask for:")
        logger.info(str(target) + " , " + str(product) + " , " + str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        logger.info("Input file: " + input_file)
        logger.info("Velocity field file: " + vfield_file)
        logger.info("Strict mask file: " + mask_file)
        if coverage_file is not None:
            logger.info("Coverage file: " + coverage_file)
        logger.info("Target file: " + outfile)
        logger.info("Keyword arguments: " + str(flatmask_kwargs))

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Call the masking routines
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        if (not self._dry_run):

            # ... put the directory into the name to allow it to remain
            # None when missing.
            if coverage_file is not None:
                coverage_file_in = indir + coverage_file
            else:
                coverage_file_in = None

            # run flat mask routine
            recipe_phangs_flat_mask(
                cube_in=indir + input_file,
                vfield_in=vfield_dir + vfield_file,
                mask_in=indir + mask_file,
                coverage=coverage_file_in,
                outfile=outdir + outfile,
                mask_kwargs=flatmask_kwargs,
                return_spectral_cube=False,
                overwrite=overwrite)

    # region flat broad mask

    def task_build_flat_broad_mask(
            self,
            target=None,
            config=None,
            product=None,
            res_tag=None,
            extra_ext='',
            overwrite=False,
    ):
        """
        Construct the flat broad mask and save it to disk.
        """

        # Generate file names

        indir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        indir = os.path.abspath(indir) + '/'

        outdir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        outdir = os.path.abspath(outdir) + '/'

        vfield_dir = self._kh.get_vfield_dir_for_target(target=target, changeto=False)
        vfield_dir = os.path.abspath(vfield_dir) + '/'

        fname_dict = self._fname_dict(
            target=target, config=config, product=product, res_tag=res_tag,
            extra_ext_in=extra_ext)

        input_file = fname_dict['cube']
        vfield_file = fname_dict['vfield']
        mask_file = fname_dict['broadmask']
        coverage_file = fname_dict['coverage']
        coverage2d_file = fname_dict['coverage2d']

        outfile = fname_dict['flatbroadmask']

        # Check input file existence

        if not (os.path.isfile(indir + input_file)):
            logger.warning("Missing cube: " + indir + input_file)
            return ()

        if not (os.path.isfile(vfield_dir + vfield_file)):
            logger.warning("Missing velocity field: " + vfield_dir + vfield_file)
            return ()

        if not (os.path.isfile(indir + mask_file)):
            logger.warning("Missing broad mask: " + indir + mask_file)
            return ()

        # Coverage

        if not (os.path.isfile(indir + coverage_file)):
            logger.warning("Missing coverage estimate: " + indir + coverage_file)
            logger.warning("This may be fine. Proceeding.")
            coverage_file = None

        # Access keywords for mask generation

        flatmask_kwargs = self._kh.get_derived_kwargs(
            config=config, product=product, kwarg_type='flatbroadmask_kw')

        # get velocity window from separate key if not provided from derived key
        if not flatmask_kwargs:
            this_window = self._kh.get_window_for_target(target)
            if this_window is None:
                logger.error("No velocity window for target " + target)
                return ()
            flatmask_kwargs = {'window': this_window}

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Report
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Creating a flat broad mask for:")
        logger.info(str(target) + " , " + str(product) + " , " + str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        logger.info("Input file: " + input_file)
        logger.info("Velocity field file: " + vfield_file)
        logger.info("Broad mask file: " + mask_file)
        if coverage_file is not None:
            logger.info("Coverage file: " + coverage_file)
        logger.info("Target file: " + outfile)
        logger.info("Keyword arguments: " + str(flatmask_kwargs))

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Call the masking routines
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        if (not self._dry_run):

            # ... put the directory into the name to allow it to remain
            # None when missing.
            if coverage_file is not None:
                coverage_file_in = indir + coverage_file
            else:
                coverage_file_in = None

            # run flat mask routine
            recipe_phangs_flat_mask(
                cube_in=indir + input_file,
                vfield_in=vfield_dir + vfield_file,
                mask_in=indir + mask_file,
                coverage=coverage_file_in,
                outfile=outdir + outfile,
                mask_kwargs=flatmask_kwargs,
                return_spectral_cube=False,
                overwrite=overwrite)

    # end regions

    # region flat maps

    def task_generate_flatmaps(
            self,
            target=None,
            config=None,
            product=None,
            res_tag=None,
            extra_ext='',
            overwrite=False,
    ):
        """
        Generate moment maps.
        """
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Look up filenames, list of moments, etc.
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        # Generate file names

        indir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        indir = os.path.abspath(indir) + '/'

        outdir = self._kh.get_derived_dir_for_target(target=target, changeto=False)
        outdir = os.path.abspath(outdir) + '/'

        # Filenames

        fname_dict_nores = self._fname_dict(
            target=target, config=config, product=product, res_tag=None,
            extra_ext_in=extra_ext)

        fname_dict = self._fname_dict(
            target=target, config=config, product=product, res_tag=res_tag,
            extra_ext_in=extra_ext)

        # ... broad mask never has a resolution tag

        broadmask_file = fname_dict_nores['flatbroadmask']

        # ... files with resolution tag

        input_file = fname_dict['cube']
        noise_file = fname_dict['noise']
        strictmask_file = fname_dict['flatstrictmask']

        outroot = fname_dict['momentroot']

        # Check input file and mask existence

        if not (os.path.isfile(indir + input_file)):
            logger.warning("Missing cube: " + indir + input_file)
            return ()

        found_broadmask = (os.path.isfile(indir + broadmask_file))
        found_strictmask = (os.path.isfile(indir + strictmask_file))

        # Look up which moments to calculate

        list_of_moments = self._kh.get_moment_list(config=config, product=product)

        rounds = []
        for this_mom in list_of_moments:
            rounds.append(self._kh.get_params_for_moment(this_mom)['round'])
        uniqrounds = sorted(list(set(rounds)))

        if len(uniqrounds) == 1:
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("")
            logger.info("... Total stages of moment calculation: {0}".format(len(uniqrounds)))
            logger.info("... Secondary moments requested but not specified in moment keys")
            logger.info("... Returning to main loop")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("")
            return ()

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Report
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Generating moment maps for:")
        logger.info(str(target) + " , " + str(product) + " , " + str(config))
        if res_tag is not None:
            logger.info("Resolution " + str(res_tag))
        logger.info("Found a flat strict mask? " + str(found_strictmask))
        logger.info("Found a flat broad mask? " + str(found_broadmask))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        logger.info("... input file: " + input_file)
        logger.info("... noise file: " + noise_file)
        logger.info("... flat strict mask file: " + strictmask_file)
        logger.info("... flat broad mask file: " + broadmask_file)
        logger.info("... list of moments: " + str(list_of_moments))
        logger.info("... output root: " + outroot)

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Execute
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        logger.info("... Total stages of moment calculation: {0}".format(len(uniqrounds)))

        for thisround in uniqrounds[2:3]:
            logger.info("... Now, calculate all moments in stage {0}".format(thisround))
            logger.info("")

            sublist_of_moments = [this_mom for this_mom in list_of_moments
                                  if self._kh.get_params_for_moment(this_mom)['round'] == thisround]

            if (not self._dry_run):
                for this_mom in sublist_of_moments:
                    logger.info('... generating moment: ' + str(this_mom))

                    proceed = True

                    mom_params = self._kh.get_params_for_moment(this_mom)

                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # Look up mask
                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    if mom_params['mask'] is None:
                        mask_file = None
                    elif mom_params['mask'].strip().lower() == 'none':
                        mask_file = None
                    elif mom_params['mask'] == 'flatstrictmask':
                        if not found_strictmask:
                            logger.warning("Flat strict mask needed but not found. Skipping.")
                            continue
                        mask_file = indir + strictmask_file
                    elif mom_params['mask'] == 'flatbroadmask':
                        if not found_broadmask:
                            logger.warning("Flat broad mask needed but not found. Skipping.")
                            continue
                        mask_file = indir + broadmask_file
                    else:
                        logger.warning("Mask choice not recognized for moment: " + str(this_mom))
                        logger.warning("Skipping.")
                        continue

                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # Check noise
                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    if not (os.path.isfile(indir + noise_file)):
                        logger.warning("Missing noise: " + indir + noise_file)
                        noise_in = None
                        errorfile = None
                    else:
                        noise_in = indir + noise_file
                        errorfile = outdir + outroot + mom_params['ext_error'] + '.fits'

                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # Set up output file
                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    outfile = outdir + outroot + mom_params['ext'] + '.fits'

                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # Look up maps to pass and build the kwarg list
                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    kwargs_dict = mom_params['kwargs']

                    maps_to_pass = mom_params['maps_to_pass']
                    for map_ext in maps_to_pass:

                        # File name for this extension
                        this_map_file = outroot + '_' + map_ext + '.fits'

                        # Verify file existence
                        if not (os.path.isfile(indir + this_map_file)):
                            logger.warning("Missing needed context file: " + indir + this_map_file)
                            proceed = False

                        # Add as param to kwarg dict
                        kwargs_dict[map_ext] = indir + this_map_file

                    other_exts = mom_params['other_exts']
                    for param_name in other_exts.keys():

                        # File name for this extension
                        this_ext = other_exts[param_name]

                        # Code to look up file name
                        this_ext_file = str(target) + this_ext

                        # Verify file existence
                        if not (os.path.isfile(indir + this_ext_file)):
                            logger.warning("Missing needed context file: " + indir + this_ext_file)
                            proceed = False

                        # Add as param to kwarg dict
                        kwargs_dict[param_name] = indir + this_ext_file

                    if proceed == False:
                        logger.warning("Missing some needed information. Skipping this calculation.")
                        continue

                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # Call the moment generator
                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    moment_generator(
                        indir + input_file, mask=mask_file, noise=noise_in,
                        moment=mom_params['algorithm'],
                        momkwargs=mom_params['kwargs'],
                        outfile=outfile, errorfile=errorfile,
                        channel_correlation=None)
    # endregion
