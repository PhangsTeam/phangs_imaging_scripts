"""
PostProcessHandler

The PHANGS pipeline to handle post-processing of cubes. Works through
a single big class (the PostProcessHandler). This needs to be attached
to a keyHandler to access the target, product, and configuration
keys.

There should not be any direct calls to CASA in here. Eventually, this
should be able to run without CASA enabled (though it won't be able to
call any of the CASA-specific routines). Right now, just avoid direct
calls to CASA from this class.
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

    from . import casaCubeRoutines as ccr
    from . import casaFeatherRoutines as cfr
    from . import casaMosaicRoutines as cmr
    from . import handlerTemplate
    from . import utilsFilenames
    from . import utilsResolutions
    from .clean_call import CleanCall

    class PostProcessHandler(handlerTemplate.HandlerTemplate):
        """
        Class to handle post-processing of ALMA data. Post-processing here
        begins with the results of imaging and proceeds through reduced,
        science-ready data cubes.
        """

        def __init__(
                self,
                key_handler=None,
                dry_run=False,
                raise_exception_mosaic_part_missing=False,
        ):

            # inherit template class
            handlerTemplate.HandlerTemplate.__init__(self, key_handler=key_handler, dry_run=dry_run)
            self.raise_exception_mosaic_part_missing = raise_exception_mosaic_part_missing

        # region File name routines

        ###########################################
        # Defined file names for various products #
        ###########################################

        def _fname_dict(
                self,
                target=None,
                config=None,
                product=None,
                imaging_method='tclean',
                extra_ext='',
        ):
            """
            Make the file name dictionary for all postprocess files. This
            will give the a big dictionary of names where one can look up
            each type of file (e.g., primary beam corrected, single dish
            aligned, etc.) given some target, config, and product. This
            routine has a lot of hard-coded knowledge about our
            postprocessing conventions.
            """

            # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
            # Error checking
            # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

            if target is None:
                logger.error("Need a target.")
                return ()

            if product is None:
                logger.error("Need a product.")
                return ()

            if config is None:
                logger.error("Need a config.")
                return ()

            # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
            # Initialize
            # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

            fname_dict = {}

            # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
            # Original files
            # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

            # Original cube

            tag = 'orig'

            if imaging_method == 'tclean':
                casaext = '.image'
            elif imaging_method == 'sdintimaging':
                casaext = '.joint.cube.image'
            else:
                logger.error('imaging_method %s not recognised' % imaging_method)
                raise Exception('imaging_method %s not recognised' % imaging_method)

            orig_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext=None,
                casa=True,
                casaext=casaext)
            fname_dict[tag] = orig_file

            # Original primary beam file

            if imaging_method == 'tclean':
                casaext = '.pb'
            elif imaging_method == 'sdintimaging':
                casaext = '.joint.cube.pb'

            tag = 'pb'
            pb_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext=None,
                casa=True,
                casaext=casaext)
            fname_dict[tag] = pb_file

            # Original single dish file (note that this comes with a
            # directory)

            has_sd = self._kh.has_singledish(target=target, product=product)
            tag = 'orig_sd'
            if has_sd:
                orig_sd_file = self._kh.get_sd_filename(
                    target=target, product=product)
                fname_dict[tag] = orig_sd_file
            else:
                fname_dict[tag] = ''

            # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
            # Processed files (apply the extra_ext tag here)
            # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

            # Primary beam corrected file

            tag = 'pbcorr'
            pbcorr_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='pbcorr' + extra_ext,
                casa=True,
                casaext='.image')
            fname_dict[tag] = pbcorr_file

            # Files with round beams

            tag = 'round'
            round_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='round' + extra_ext,
                casa=True,
                casaext='.image')
            fname_dict[tag] = round_file

            tag = 'pbcorr_round'
            pbcorr_round_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='pbcorr_round' + extra_ext,
                casa=True,
                casaext='.image')
            fname_dict[tag] = pbcorr_round_file

            # Weight file for use in linear mosaicking

            tag = 'weight'
            weight_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='weight' + extra_ext,
                casa=True,
                casaext='.image')
            fname_dict[tag] = weight_file

            tag = 'weight_aligned'
            weight_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='weight_aligned' + extra_ext,
                casa=True,
                casaext='.image')
            fname_dict[tag] = weight_file

            # Common resolution parts for mosaic

            tag = 'linmos_commonres'
            commonres_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='linmos_commonres' + extra_ext,
                casa=True,
                casaext='.image')
            fname_dict[tag] = commonres_file

            # Aligned parts for mosaic

            tag = 'linmos_aligned'
            aligned_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='linmos_aligned' + extra_ext,
                casa=True,
                casaext='.image')
            fname_dict[tag] = aligned_file

            # Imported single dish file aligned to the interfometer data

            tag = 'prepped_sd'
            prepped_sd_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='singledish' + extra_ext,
                casa=True,
                casaext='.image')
            fname_dict[tag] = prepped_sd_file

            # Singledish weight for use in linear mosaicking

            tag = 'sd_weight'
            sd_weight_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='singledish_weight' + extra_ext,
                casa=True,
                casaext='.image')
            fname_dict[tag] = sd_weight_file

            # Singledish data aliged to a common grid for mosaicking

            tag = 'sd_aligned'
            sd_align_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='singledish_aligned' + extra_ext,
                casa=True,
                casaext='.image')
            fname_dict[tag] = sd_align_file

            # Singledish weight for use in linear mosaicking now on a
            # common astrometric grid

            tag = 'sd_weight_aligned'
            sd_weight_aligned_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='singledish_weight_aligned' + extra_ext,
                casa=True,
                casaext='.image')
            fname_dict[tag] = sd_weight_aligned_file

            # Compressed files with edges trimmed off and smallest
            # reasonable pixel size.

            tag = 'trimmed'
            trimmed_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='trimmed' + extra_ext,
                casa=True,
                casaext='.image')
            fname_dict[tag] = trimmed_file

            tag = 'pbcorr_trimmed'
            pbcorr_trimmed_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='pbcorr_trimmed' + extra_ext,
                casa=True,
                casaext='.image')
            fname_dict[tag] = pbcorr_trimmed_file

            tag = 'trimmed_pb'
            trimmed_pb_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='trimmed' + extra_ext,
                casa=True,
                casaext='.pb')
            fname_dict[tag] = trimmed_pb_file

            # Files converted to Kelvin, including FITS output files

            tag = 'trimmed_k'
            trimmed_k_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='trimmed_k' + extra_ext,
                casa=True,
                casaext='.image')
            fname_dict[tag] = trimmed_k_file

            tag = 'trimmed_k_fits'
            trimmed_k_fits = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='trimmed_k' + extra_ext,
                casa=False)
            fname_dict[tag] = trimmed_k_fits

            tag = 'pbcorr_trimmed_k'
            pbcorr_trimmed_k_file = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='pbcorr_trimmed_k' + extra_ext,
                casa=True,
                casaext='.image')
            fname_dict[tag] = pbcorr_trimmed_k_file

            tag = 'pbcorr_trimmed_k_fits'
            pbcorr_trimmed_k_fits = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='pbcorr_trimmed_k' + extra_ext,
                casa=False)
            fname_dict[tag] = pbcorr_trimmed_k_fits

            tag = 'trimmed_pb_fits'
            trimmed_pb_fits = utilsFilenames.get_cube_filename(
                target=target, config=config, product=product,
                ext='trimmed_pb' + extra_ext,
                casa=False)
            fname_dict[tag] = trimmed_pb_fits

            # Return

            return (fname_dict)

        # endregion

        # region "Tasks" : Individual postprocessing steps

        def task_stage_interf_data(
                self,
                target=None,
                product=None,
                config=None,
                imaging_method='tclean',
                extra_ext_in='',
                extra_ext_out='',
                check_files=True,
                trim_coarse_beam_edge_channels=False,
        ):
            """
            For one target, product, config combination copy the
            interferometric cube and primary beam file to the working
            postprocessing directory.
            """

            # Generate file names

            indir = self._kh.get_imaging_dir_for_target(target)
            outdir = self._kh.get_postprocess_dir_for_target(target)
            fname_dict_in = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_in, imaging_method=imaging_method)
            fname_dict_out = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_out, imaging_method=imaging_method)

            # Copy the primary beam and the interferometric imaging

            for this_tag in ['orig', 'pb']:

                infile = fname_dict_in[this_tag]
                outfile = fname_dict_out[this_tag]

                # Check input file existence
                if check_files:
                    if not (os.path.isdir(indir + infile)):
                        logger.warning("Missing " + indir + infile)
                        continue

                logger.info("")
                logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
                logger.info("Staging data for:")
                logger.info(str(target) + " , " + str(product) + " , " + str(config))
                logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
                logger.info("")
                # logger.info("Using ccr.copy_dropdeg.")
                logger.info("Staging " + outfile)

                # Move the cubes to the postprocess directory, trimming along the way
                # (though not rebinning)
                if not self._dry_run:
                    os.system('rm -rf ' + outdir + outfile)
                    os.system('rm -rf ' + outdir + outfile + ".temp")
                    os.system('rm -rf ' + outdir + outfile + ".temp_deg")
                    # os.system('cp -r ' + indir + infile + ' ' + outdir + outfile)

                    ccr.trim_cube(
                        infile=indir + infile,
                        outfile=outdir + outfile,
                        overwrite=True,
                        inplace=False,
                        pad=1,
                        rebin=False,
                    )
                    # ccr.copy_dropdeg(
                    #     infile=indir+infile,
                    #     outfile=outdir+outfile,
                    #     overwrite=True)

            # in case of merged datasets with non-identical frequency setups imaged with per-plane beam,
            # some edge channels will have much coarser beam, we trim these edge channels here.
            if trim_coarse_beam_edge_channels:
                ccr.trim_coarse_beam_edge_channels(
                    infile=outdir + fname_dict_out['orig'],
                    inpbfile=outdir + fname_dict_out['pb'],
                    inplace=True,
                )

            return ()

        def task_remove_degenerate_axes(
                self,
                target=None,
                product=None,
                config=None,
                imaging_method='tclean',
                extra_ext='',
                check_files=True,
        ):
            """
            Remove
            degenerate axes for target, product, config combination in postprocessing directory.
            """

            # Generate file names

            file_dir = self._kh.get_postprocess_dir_for_target(target)
            fname_dict = self._fname_dict(target=target, config=config, product=product, extra_ext=extra_ext,
                                          imaging_method=imaging_method)

            # Copy the primary beam and the interferometric imaging

            logger.info("Dropping degenerate axes from postprocess image/pb files.")

            for this_tag in ['orig', 'pb', 'pbcorr', 'pbcorr_round']:

                file_name = fname_dict[this_tag]

                # Check input file existence
                if check_files:
                    if not (os.path.isdir(file_dir + file_name)):
                        logger.warning("Missing " + file_dir + file_name)
                        continue

                if not self._dry_run:
                    ccr.copy_dropdeg(file_dir + file_name, file_dir + file_name + '_nodeg', overwrite=True)

                    os.system('rm -rf ' + file_dir + file_name)
                    os.system('cp -r ' + file_dir + file_name + '_nodeg ' + file_dir + file_name)
                    os.system('rm -rf ' + file_dir + file_name + '_nodeg')

            return ()

        def task_pbcorr(
                self,
                target=None,
                product=None,
                config=None,
                imaging_method='tclean',
                in_tag='orig',
                out_tag='pbcorr',
                extra_ext_in='',
                extra_ext_out='',
                check_files=True,
        ):
            """
            For one target, product, config combination primary beam
            correct the interferometer data.
            """

            # Generate file names

            indir = self._kh.get_postprocess_dir_for_target(target)
            outdir = self._kh.get_postprocess_dir_for_target(target)
            fname_dict_in = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_in, imaging_method=imaging_method)
            fname_dict_out = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_out, imaging_method=imaging_method)

            # Pull in the pblimit for setting the cutoff
            recipe_list = self._kh.get_imaging_recipes(config=config, product=product)
            clean_call = CleanCall(recipe_list)
            cutoff = clean_call.get_param('pblimit')

            infile = fname_dict_in[in_tag]
            outfile = fname_dict_out[out_tag]
            pbfile = fname_dict_in['pb']

            # Check input file existence

            if check_files:
                if not (os.path.isdir(indir + infile)):
                    logger.warning("Missing " + indir + infile)
                    return ()
                if not (os.path.isdir(indir + pbfile)):
                    logger.warning("Missing " + indir + pbfile)
                    return ()

            # Apply the primary beam correction to the data.

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Primary beam correction for:")
            logger.info(str(target) + " , " + str(product) + " , " + str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            logger.info("Using ccr.primary_beam_correct")
            logger.info("Correcting to " + outfile)
            logger.info("Correcting from " + infile)
            logger.info("Correcting using " + pbfile)

            if not self._dry_run:
                ccr.primary_beam_correct(
                    infile=indir + infile,
                    outfile=outdir + outfile,
                    pbfile=indir + pbfile,
                    cutoff=cutoff,
                    overwrite=True)

            return ()

        def task_round_beam(
                self,
                target=None,
                product=None,
                config=None,
                imaging_method='tclean',
                in_tag='pbcorr',
                out_tag='pbcorr_round',
                extra_ext_in='',
                extra_ext_out='',
                force_beam_as=None,
                check_files=True,
        ):
            """
            For one target, product, config combination, convolve the cube
            to have a round beam. Note that via the force_beam_as keyword
            this task can also be used to convolve data to a fixed (round)
            angular resolution.
            """

            # Generate file names

            indir = self._kh.get_postprocess_dir_for_target(target)
            outdir = self._kh.get_postprocess_dir_for_target(target)
            fname_dict_in = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_in, imaging_method=imaging_method)
            fname_dict_out = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_out, imaging_method=imaging_method)

            infile = fname_dict_in[in_tag]
            outfile = fname_dict_out[out_tag]

            # Check input file existence

            if check_files:
                if not (os.path.isdir(indir + infile)):
                    logger.warning("Missing " + infile)
                    return ()

            # Convolve the data to have a round beam.

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Convolving to a round beam for:")
            logger.info(str(target) + " , " + str(product) + " , " + str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            logger.info("Using ccr.convolve_to_round_beam")
            logger.info("Convolving from " + infile)
            logger.info("Convolving to " + outfile)
            if force_beam_as is not None:
                logger.info("Forcing beam to " + str(force_beam_as))

            if not self._dry_run:
                ccr.convolve_to_round_beam(
                    infile=indir + infile,
                    outfile=outdir + outfile,
                    force_beam=force_beam_as,
                    overwrite=True)

            return ()

        def task_stage_singledish(
                self,
                target=None,
                product=None,
                config=None,
                template_tag='pbcorr_round',
                out_tag='prepped_sd',
                extra_ext_in='',
                extra_ext_out='',
                check_files=True,
        ):
            """
            For one target, product, config combination, copy the single
            dish data and align it to the interferometric grid.
            """

            # Generate file names

            indir = ''
            outdir = self._kh.get_postprocess_dir_for_target(target)
            tempdir = self._kh.get_postprocess_dir_for_target(target)
            fname_dict_in = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_in)
            fname_dict_out = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_out)

            template = fname_dict_in[template_tag]
            infile = fname_dict_in['orig_sd']
            outfile = fname_dict_out[out_tag]

            # Check input file existence

            if check_files:
                if (not (os.path.isdir(indir + infile))) and \
                        (not (os.path.isfile(indir + infile))):
                    logger.warning("Missing " + infile)
                    return ()
                if not (os.path.isdir(tempdir + template)):
                    logger.warning("Missing " + tempdir + template)
                    return ()

            # Stage the singledish data for feathering

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Preparing single dish data for:")
            logger.info(str(target) + " , " + str(product) + " , " + str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            logger.info("Using cfr.prep_sd_for_feather.")
            logger.info("Prepping " + outfile)
            logger.info("Original file " + infile)
            logger.info("Using interferometric template " + template)

            if not self._dry_run:
                cfr.prep_sd_for_feather(
                    sdfile_in=indir + infile,
                    sdfile_out=outdir + outfile,
                    interf_file=tempdir + template,
                    do_import=True,
                    do_dropdeg=True,
                    do_align=True,
                    do_checkunits=True,
                    overwrite=True)

            return ()

        def task_make_interf_weight(
                self,
                target=None,
                product=None,
                config=None,
                imaging_method='tclean',
                image_tag='pbcorr_round',
                in_tag='pb',
                input_type='pb',
                scale_by_noise=True,
                out_tag='weight',
                extra_ext_in='',
                extra_ext_out='',
                check_files=True,
        ):
            """
            For one target, product, config combination, make a 'weight'
            image for use in linearly mosaicking the cube with other,
            overlapping cubes. This task targets interferometric dish
            data.
            """

            # Generate file names

            indir = self._kh.get_postprocess_dir_for_target(target)
            outdir = self._kh.get_postprocess_dir_for_target(target)
            fname_dict_in = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_in, imaging_method=imaging_method)
            fname_dict_out = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_out, imaging_method=imaging_method)

            image_file = fname_dict_in[image_tag]
            infile = fname_dict_in[in_tag]
            outfile = fname_dict_out[out_tag]

            # Check input file existence

            if check_files:
                if not (os.path.isdir(indir + infile)):
                    logger.warning("Missing " + infile)
                    return ()
                if not (os.path.isdir(indir + image_file)):
                    logger.warning("Missing " + image_file)
                    return ()

            # Create a weight image for use linear mosaicking targets that
            # are part of a linear mosaic

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("Making weight file for:")
            logger.info(str(target) + " , " + str(product) + " , " + str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("")

            logger.info("Using cmr.generate_weight_file.")
            logger.info("Making weight file " + outfile)
            logger.info("Based off of primary beam file " + infile)
            logger.info("Measuring noise from file " + image_file)

            if not self._dry_run:
                cmr.generate_weight_file(
                    image_file=indir + image_file,
                    input_file=indir + infile,
                    input_type=input_type,
                    outfile=indir + outfile,
                    scale_by_noise=scale_by_noise,
                    overwrite=True)

            return ()

        def task_make_singledish_weight(
                self,
                target=None,
                product=None,
                config=None,
                image_tag='prepped_sd',
                out_tag='sd_weight',
                extra_ext_in='',
                extra_ext_out='',
                check_files=True,
        ):
            """
            For one target, product, config combination, make a 'weight'
            image for use in linearly mosaicking the cube with other,
            overlapping cubes. This task targets single dish data.
            """

            # Generate file names

            indir = self._kh.get_postprocess_dir_for_target(target)
            outdir = self._kh.get_postprocess_dir_for_target(target)
            fname_dict_in = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_in)
            fname_dict_out = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_out)

            image_file = fname_dict_in[image_tag]
            outfile = fname_dict_out[out_tag]

            # Check input file existence

            if check_files:
                if not (os.path.isdir(indir + image_file)):
                    logger.warning("Missing " + image_file)
                    return ()

            # Make a weight file for single dish targets that
            # are part of a linear mosaic

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("Making single dish weight file for:")
            logger.info(str(target) + " , " + str(product) + " , " + str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("")

            logger.info("Using cmr.generate_weight_file.")
            logger.info("Making weight file " + outfile)
            logger.info("Measuring noise from file " + image_file)

            if not self._dry_run:
                cmr.generate_weight_file(
                    image_file=indir + image_file,
                    input_value=1.0,
                    input_type='weight',
                    outfile=indir + outfile,
                    scale_by_noise=True,
                    overwrite=True)

            return ()

        def task_feather(
                self,
                target=None,
                product=None,
                config=None,
                interf_tag='pbcorr_round',
                sd_tag='prepped_sd',
                out_tag='pbcorr_round',
                extra_ext_in='',
                extra_ext_out='',
                apodize=False,
                apod_ext='pb',
                copy_weights=True,
                check_files=True,
        ):
            """
            For one target, product, config combination, feather together
            a single dish and interferometric data set. Note that
            apodization is exposed as an option. Also note that the
            configuration of the input and output will differ (an
            interferometric configuration comes in, a feather
            configuration comes out). Optionally, propagate the weights
            from the interferometric side to become the weights for the
            new feathered data.
            """

            # Generate file names

            indir = self._kh.get_postprocess_dir_for_target(target)
            outdir = self._kh.get_postprocess_dir_for_target(target)
            fname_dict_in = self._fname_dict(
                target=target, config=config, product=product,
                extra_ext=extra_ext_in)

            # Note that feather changes the config

            feather_config = self._kh.get_feather_config_for_interf_config(
                interf_config=config)

            fname_dict_out = self._fname_dict(
                target=target, config=feather_config, product=product,
                extra_ext=extra_ext_out)

            interf_file = fname_dict_in[interf_tag]
            sd_file = fname_dict_in[sd_tag]
            if len(fname_dict_out) == 0:
                logger.info("No feather config found for:")
                logger.info(str(target) + " , " + str(product) + " , " + str(config))
                return ()
            outfile = fname_dict_out[out_tag]

            # Error checking

            # Check input file existence

            # Feather the single dish and interferometer data

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Feathering interferometer and single dish data for:")
            logger.info(str(target) + " , " + str(product) + " , " + str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            logger.info("Using cfr.feather_two_cubes.")
            logger.info("Feathering " + outfile)
            logger.info("Feathering interferometric data " + interf_file)
            logger.info("Feathering single dish data " + sd_file)

            # Feather has a couple of algorithmic choices
            # associated with it. Run the method that the
            # user has selected.

            if apodize:

                apod_file = fname_dict_in[apod_ext]

                logger.info("Apodizing using file " + apod_file)

                if not self._dry_run:
                    cfr.feather_two_cubes(
                        interf_file=indir + interf_file,
                        sd_file=indir + sd_file,
                        out_file=outdir + outfile,
                        do_blank=True,
                        do_apodize=True,
                        apod_file=indir + apod_file,
                        apod_cutoff=0.0,
                        overwrite=True)

            else:

                if not self._dry_run:
                    cfr.feather_two_cubes(
                        interf_file=indir + interf_file,
                        sd_file=indir + sd_file,
                        out_file=outdir + outfile,
                        do_blank=True,
                        do_apodize=False,
                        apod_file=None,
                        apod_cutoff=-1.0,
                        overwrite=True)

            if copy_weights:

                interf_weight_exists = False
                interf_weight_file = fname_dict_in['weight']
                if os.path.isdir(indir + interf_weight_file):
                    interf_weight_exists = True
                else:
                    logger.info("Interferometric weight file not found " + interf_weight_file)

                if interf_weight_exists:
                    logger.info("")
                    logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
                    logger.info("Copying weights for:")
                    logger.info(str(target) + " , " + str(product) + " , " + str(config))
                    logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
                    logger.info("")

                    out_weight_file = fname_dict_out['weight']

                    logger.info("Copying from " + interf_weight_file)
                    logger.info("Copying to " + out_weight_file)
                    if not self._dry_run:
                        ccr.copy_dropdeg(infile=indir + interf_weight_file,
                                         outfile=outdir + out_weight_file,
                                         overwrite=True)
            return ()

        def task_rename_sdintimaging(self,
                                     target=None,
                                     product=None,
                                     config=None,
                                     imaging_method='sdintimaging'):

            if target is None:
                logger.warning('Missing target')
                return
            if product is None:
                logger.warning('Missing product')
                return
            if config is None:
                logger.warning('Missing config')
                return

            if imaging_method != 'sdintimaging':
                logger.warning('This should only be run for sdintimaging')
                return

            fname_dict_in = self._fname_dict(target=target, product=product, config=config,
                                             imaging_method=imaging_method)

            imaging_dir = self._kh.get_imaging_dir_for_target(target)
            using_sdint = os.path.isdir(imaging_dir + fname_dict_in['orig'])
            is_mosaic = self._kh.is_target_linmos(target)

            # If not a mosaic and we're not sdintimaging, skip

            if not using_sdint and not is_mosaic:
                return

            if is_mosaic:

                # In the case where we have a mosaic, look for the telltale sdint files

                mosaic_parts = self._kh.get_parts_for_linmos(target)
                for mosaic_part in mosaic_parts:
                    fname_dict_mosaic = self._fname_dict(
                        target=mosaic_part, product=product, config=config,
                        imaging_method=imaging_method)
                    imaging_dir = self._kh.get_imaging_dir_for_target(mosaic_part)
                    using_sdint = os.path.isdir(imaging_dir + fname_dict_mosaic['orig'])
                    if using_sdint:
                        break
                if not using_sdint:
                    return

            outdir = self._kh.get_postprocess_dir_for_target(target)
            feather_config = self._kh.get_feather_config_for_interf_config(interf_config=config)
            fname_dict_out = self._fname_dict(target=target, product=product, config=feather_config)

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Renaming sdintimaging outputs for:")
            logger.info(str(target) + " , " + str(product) + " , " + str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            for key in fname_dict_in.keys():
                item = fname_dict_in[key]

                # Make sure we don't just delete the whole postprocess folder
                if item == '':
                    continue

                file_name = outdir + item
                if os.path.exists(file_name):
                    new_file_name = outdir + fname_dict_out[key]
                    command = 'mv -f %s %s' % (file_name, new_file_name)
                    os.system('rm -rf %s' % new_file_name)
                    os.system(command)

            return

        def task_compress(
                self,
                target=None,
                product=None,
                config=None,
                imaging_method='tclean',
                in_tag='pbcorr_round',
                out_tag='pbcorr_trimmed',
                do_trimrind=True,
                do_pb_too=True,
                in_pb_tag='pb',
                out_pb_tag='pb_trimmed',
                extra_ext_in='',
                extra_ext_out='',
                check_files=True
        ):
            """
            For one target, product, config combination, compress the cube
            to the smallest reasonable volume. Also align the primary beam
            file out onto this grid.
            """

            # Generate file names

            indir = self._kh.get_postprocess_dir_for_target(target)
            outdir = self._kh.get_postprocess_dir_for_target(target)
            fname_dict_in = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_in, imaging_method=imaging_method)
            fname_dict_out = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_out, imaging_method=imaging_method)

            infile = fname_dict_in['pbcorr_round']
            outfile = fname_dict_out['pbcorr_trimmed']

            infile_pb = fname_dict_in['pb']
            outfile_pb = fname_dict_out['trimmed_pb']

            # Check input file existence

            if check_files:
                if not (os.path.isdir(indir + infile)):
                    logger.warning("Missing " + infile)
                    return ()

            # Compress, reducing cube volume.

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Trimming cube for:")
            logger.info(str(target) + " , " + str(product) + " , " + str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            logger.info("Producing " + outfile + " using ccr.trim_cube.")
            logger.info("Trimming from original file " + infile)

            if not self._dry_run:
                ccr.trim_cube(
                    infile=indir + infile,
                    outfile=outdir + outfile,
                    overwrite=True,
                    inplace=False,
                    min_pixperbeam=3,
                    pad=1)

                if do_trimrind:
                    ccr.trim_rind(
                        infile=outdir + outfile,
                        inplace=True,
                        pixels=1)

            if do_pb_too is False:
                return ()

            if check_files:
                if not (os.path.isdir(indir + infile_pb)):
                    logger.warning("Missing " + infile_pb)
                    return ()

            template = fname_dict_out['pbcorr_trimmed']

            if check_files:
                if not (os.path.isdir(outdir + template)):
                    logger.warning("Missing " + template)
                    return ()

            logger.info("Aligning primary beam image to new astrometry")
            logger.info("Using ccr.align_to_target.")
            logger.info("Aligning original file " + infile_pb)
            logger.info("Aligning to produce output file " + outfile_pb)
            logger.info("Aligning to template " + template)

            if not self._dry_run:
                ccr.align_to_target(
                    infile=indir + infile_pb,
                    outfile=outdir + outfile_pb,
                    template=outdir + template,
                    interpolation='cubic',
                    overwrite=True,
                )

            return ()

        def task_convert_units(
                self,
                target=None,
                product=None,
                config=None,
                imaging_method='tclean',
                in_tag='pbcorr_trimmed',
                out_tag='pbcorr_trimmed_k',
                extra_ext_in='',
                extra_ext_out='',
                check_files=True,
        ):
            """
            For one target, config, product combination convert the units
            from Jy/beam to Kelvin.
            """

            # Generate file names

            indir = self._kh.get_postprocess_dir_for_target(target)
            outdir = self._kh.get_postprocess_dir_for_target(target)
            fname_dict_in = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_in, imaging_method=imaging_method)
            fname_dict_out = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_out, imaging_method=imaging_method)

            infile = fname_dict_in[in_tag]
            outfile = fname_dict_out[out_tag]

            # Check input file existence

            if check_files:
                if not (os.path.isdir(indir + infile)):
                    logger.warning("Missing " + infile)
                    return ()

            # Change units from Jy/beam to Kelvin.

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Converting units for:")
            logger.info(str(target) + " , " + str(product) + " , " + str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            logger.info("Using ccr.convert_jytok")
            logger.info("Creating " + outfile)
            logger.info("Converting from original file " + infile)

            if not self._dry_run:
                ccr.convert_jytok(
                    infile=indir + infile,
                    outfile=outdir + outfile,
                    overwrite=True,
                    inplace=False,
                )

            return ()

        def task_export_to_fits(
                self,
                target=None,
                product=None,
                config=None,
                imaging_method='tclean',
                in_tag='pbcorr_trimmed_k',
                out_tag='pbcorr_trimmed_k_fits',
                do_pb_too=True,
                in_pb_tag='trimmed_pb',
                out_pb_tag='trimmed_pb_fits',
                extra_ext_in='',
                extra_ext_out='',
                check_files=True,
        ):
            """
            For one target, config, product combination export to
            FITS. Optionally also export the primary beam files.
            """

            # Generate file names

            indir = self._kh.get_postprocess_dir_for_target(target)
            outdir = self._kh.get_postprocess_dir_for_target(target)
            fname_dict_in = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_in, imaging_method=imaging_method)
            fname_dict_out = self._fname_dict(
                target=target, config=config, product=product, extra_ext=extra_ext_out, imaging_method=imaging_method)

            infile = fname_dict_in[in_tag]
            outfile = fname_dict_out[out_tag]

            # Check input file existence

            if check_files:
                if not (os.path.isdir(indir + infile)):
                    logger.warning("Missing " + infile)
                    return ()

            # Export to FITS and clean up output

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Exporting data to FITS and cleaning up cubes for:")
            logger.info(str(target) + " , " + str(product) + " , " + str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            logger.info("Using ccr.export_and_cleanup.")
            logger.info("Export to " + outfile)
            logger.info("Writing from input cube " + infile)

            if not self._dry_run:
                ccr.export_and_cleanup(
                    infile=indir + infile,
                    outfile=outdir + outfile,
                    overwrite=True,
                    remove_cards=[],
                    add_cards={'OBJECT': target.upper()},
                    add_history=[],
                    zap_history=True,
                    round_beam=True,
                    roundbeam_tol=0.01,
                )

            if do_pb_too is False:
                return ()

            # Check input file existence

            infile_pb = fname_dict_in[in_pb_tag]
            outfile_pb = fname_dict_out[out_pb_tag]

            if check_files:
                if not (os.path.isdir(indir + infile_pb)):
                    logger.warning("Missing " + infile_pb)
                    return ()

            logger.info("Writing from primary beam " + infile_pb)
            logger.info("Writing output primary beam " + outfile_pb)

            if not self._dry_run:
                ccr.export_and_cleanup(
                    infile=indir + infile_pb,
                    outfile=outdir + outfile_pb,
                    overwrite=True,
                    remove_cards=[],
                    add_cards={'OBJECT': target.upper()},
                    add_history=[],
                    zap_history=True,
                    round_beam=False,
                    roundbeam_tol=0.01,
                )

            return ()

        def task_convolve_parts_for_mosaic(
                self,
                target=None,
                product=None,
                config=None,
                in_tag='pbcorr_round',
                out_tag='linmos_commonres',
                extra_ext_in='',
                extra_ext_out='',
                check_files=True,
        ):
            """
            For one target, config, product combination that is a linear
            mosaic, convolve all of the parts of the mosaic to share a
            common angular resolution, appropriate for gridding together
            into a single image.
            """

            # Generate file names

            indir = self._kh.get_postprocess_dir_for_target(target)
            outdir = self._kh.get_postprocess_dir_for_target(target)

            mosaic_parts = self._kh.get_parts_for_linmos(target)

            infile_list = []
            outfile_list = []

            for this_part in mosaic_parts:

                this_part_dict_in = self._fname_dict(
                    target=this_part, config=config, product=product,
                    extra_ext=extra_ext_in,
                )

                this_part_dict_out = self._fname_dict(
                    target=this_part, config=config, product=product,
                    extra_ext=extra_ext_out,
                )

                infile = indir + this_part_dict_in[in_tag]
                infile_exists = os.path.isdir(infile)

                outfile = outdir + this_part_dict_out[out_tag]

                if infile_exists:
                    infile_list.append(infile)
                    outfile_list.append(outfile)
                else:
                    if self.raise_exception_mosaic_part_missing:
                        raise FileNotFoundError("Missing file " + infile)
                    else:
                        logger.warning("Missing file " + infile)

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Convolving for mosaic for:")
            logger.info(str(target) + " , " + str(product) + " , " + str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            logger.info("Using cmr.common_res_for_mosaic.")
            logger.info("Convolving " + target)
            logger.info("Convolving original files " + str(infile_list))
            logger.info("Convolving to convolved output " + str(outfile_list))

            # Allow overrides for the pixel padding (the
            # number of pixels added to the greatest
            # common beam for calculating the target
            # resolution) and the target resolution.

            pixel_padding = 2.0
            target_res = None

            # TBD - check override dict for target
            # resolution and (maybe?) pixel padding.

            if not self._dry_run:
                cmr.common_res_for_mosaic(
                    infile_list=infile_list,
                    outfile_list=outfile_list,
                    do_convolve=True,
                    target_res=target_res,
                    pixel_padding=pixel_padding,
                    overwrite=True,
                )

            return ()

        def task_align_for_mosaic(
                self,
                target=None,
                product=None,
                config=None,
                in_tags=['linmos_commonres', 'weight', 'prepped_sd', 'sd_weight'],
                out_tags=['linmos_aligned', 'weight_aligned', 'sd_aligned', 'sd_weight_aligned'],
                extra_ext_in='',
                extra_ext_out='',
                check_files=True,
        ):
            """
            For one target, config, product combination that is a linear
            mosaic, align all parts of the mosaic to a common astrometric
            grid for combination into a single image.
            """

            # Map the input and output tags to one another in a dictionary

            if (type(in_tags) != type([])) or type(out_tags) != type([]):
                logger.error("Input and output tag lists must be lists.")
                return (None)

            if len(in_tags) != len(out_tags):
                logger.error("Mismatch in input and output list tag list.")
                return (None)

            out_tag_dict = {}
            for ii in range(len(in_tags)):
                out_tag_dict[in_tags[ii]] = out_tags[ii]

            # Generate file names

            indir = self._kh.get_postprocess_dir_for_target(target)
            outdir = self._kh.get_postprocess_dir_for_target(target)

            mosaic_parts = self._kh.get_parts_for_linmos(target)

            infile_list = []
            outfile_list = []

            for this_part in mosaic_parts:

                this_part_dict_in = self._fname_dict(
                    target=this_part, config=config, product=product,
                    extra_ext=extra_ext_in,
                )

                this_part_dict_out = self._fname_dict(
                    target=this_part, config=config, product=product,
                    extra_ext=extra_ext_out,
                )

                for this_tag_in in in_tags:

                    this_tag_out = out_tag_dict[this_tag_in]

                    infile = indir + this_part_dict_in[this_tag_in]
                    infile_exists = os.path.isdir(infile)

                    outfile = outdir + this_part_dict_out[this_tag_out]

                    if infile_exists:
                        infile_list.append(infile)
                        outfile_list.append(outfile)
                    else:
                        if self.raise_exception_mosaic_part_missing:
                            raise FileNotFoundError("Missing file " + infile)
                        else:
                            logger.warning("Missing file " + infile)

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Aligning for mosaic for:")
            logger.info(str(target) + " , " + str(product) + " , " + str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            logger.info("Using cmr.common_grid_for_mosaic.")
            logger.info("Aligning " + target)
            logger.info("Convolving original files " + str(infile_list))
            logger.info("Convolving to convolved output " + str(outfile_list))

            # TBD implement overrides

            ra_ctr = None
            dec_ctr = None
            delta_ra = None
            delta_dec = None

            if not self._dry_run:
                cmr.common_grid_for_mosaic(
                    infile_list=infile_list,
                    outfile_list=outfile_list,
                    ra_ctr=ra_ctr,
                    dec_ctr=dec_ctr,
                    delta_ra=delta_ra,
                    delta_dec=delta_dec,
                    allow_big_image=False,
                    too_big_pix=1e4,
                    asvelocity=True,
                    interpolation='cubic',
                    axes=[-1],
                    overwrite=True,
                )

            return ()

        def task_linear_mosaic(
                self,
                target=None,
                product=None,
                config=None,
                image_tag='linmos_aligned',  # 'sd_aligned'
                weight_tag='weight_aligned',  # 'sd_weight_aligned'
                out_tag='pbcorr_round',  # 'prepped_sd'
                extra_ext_in='',
                extra_ext_out='',
                check_files=True,
        ):
            """
            For one target, config, product combination that is a linear
            mosaic and has already been aligned and convolved, execute the
            linear mosaic. Needs to be run separately for single dish and
            interferometer data.
            """

            # Set input and output directories and define output file

            indir = self._kh.get_postprocess_dir_for_target(target)
            outdir = self._kh.get_postprocess_dir_for_target(target)

            fname_dict_out = self._fname_dict(
                target=target, config=config, product=product,
                extra_ext=extra_ext_out)

            outfile = fname_dict_out[out_tag]

            mosaic_parts = self._kh.get_parts_for_linmos(target)

            infile_list = []
            weightfile_list = []

            # Get the input and weight files for  individual parts.

            for this_part in mosaic_parts:

                this_part_dict_in = self._fname_dict(
                    target=this_part, config=config, product=product,
                    extra_ext=extra_ext_in,
                )

                # Only include these files if both imaging and weights exist
                infile = indir + this_part_dict_in[image_tag]
                weightfile = indir + this_part_dict_in[weight_tag]

                infile_exists = os.path.isdir(infile)
                weightfile_exists = os.path.isdir(weightfile)

                if infile_exists and weightfile_exists:
                    infile_list.append(infile)
                    weightfile_list.append(weightfile)
                else:
                    if not infile_exists:
                        if self.raise_exception_mosaic_part_missing:
                            raise FileNotFoundError("Missing file " + infile)
                        else:
                            logger.warning("Missing file " + infile)
                    if not weightfile_exists:
                        if self.raise_exception_mosaic_part_missing:
                            raise FileNotFoundError("Missing file " + weightfile)
                        else:
                            logger.warning("Missing file " + weightfile)

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Executing linear mosaic for:")
            logger.info(str(target) + " , " + str(product) + " , " + str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            logger.info("Using cmr.mosaic_aligned_data.")
            logger.info("Creating " + outfile)
            logger.info("Mosaicking original files " + str(infile_list))
            logger.info("Weighting by " + str(weightfile_list))

            if not self._dry_run:
                cmr.mosaic_aligned_data(
                    infile_list=infile_list,
                    weightfile_list=weightfile_list,
                    outfile=outdir + outfile,
                    overwrite=True)

            return ()

        # endregion

        # region Recipes execute a set of linked tasks for one data set.

        def recipe_prep_one_target(
                self,
                target=None,
                product=None,
                config=None,
                check_files=True,
                imaging_method='tclean',
                trim_coarse_beam_edge_channels=False,
        ):
            """
            Recipe that takes data from imaging through all steps that
            come before feathering and/or mosaicking. This means copying
            the data, primary beam correction, convolution to a round
            beam, importing and aligning the single dish data, and making
            weight files for targets that are part of linear mosaicks.

            The recipe assumes a lot of file name conventions so just
            needs the target/product/config defined.
            """

            # Work out file names and note whether the target is part of a
            # mosaic, has single dish data, etc.

            fname_dict = self._fname_dict(
                target=target, product=product, config=config, imaging_method=imaging_method)

            imaging_dir = self._kh.get_imaging_dir_for_target(target)
            has_imaging = os.path.isdir(imaging_dir + fname_dict['orig'])
            has_singledish = self._kh.has_singledish(target=target, product=product)
            is_part_of_mosaic = self._kh.is_target_in_mosaic(target)

            if not has_imaging:
                logger.warning("No imaging for " + fname_dict['orig'] + ". Returning.")
                return ()

            # Call tasks

            self.task_stage_interf_data(
                target=target, config=config, product=product,
                check_files=check_files,
                imaging_method=imaging_method,
                trim_coarse_beam_edge_channels=trim_coarse_beam_edge_channels,
            )

            self.task_pbcorr(
                target=target, config=config, product=product,
                check_files=check_files,
                imaging_method=imaging_method
            )

            self.task_round_beam(
                target=target, config=config, product=product,
                check_files=check_files,
                imaging_method=imaging_method
            )

            self.task_remove_degenerate_axes(
                target=target, config=config, product=product,
                check_files=check_files,
                imaging_method=imaging_method
            )

            if has_singledish and imaging_method not in ['sdintimaging']:
                self.task_stage_singledish(
                    target=target, config=config, product=product,
                    check_files=check_files
                )

            if is_part_of_mosaic:
                self.task_make_interf_weight(
                    target=target, config=config, product=product,
                    check_files=check_files, scale_by_noise=True,
                    imaging_method=imaging_method
                )

            if is_part_of_mosaic and has_singledish and imaging_method not in ['sdintimaging']:
                self.task_make_singledish_weight(
                    target=target, config=config, product=product,
                    check_files=check_files,
                )

            return ()

        def recipe_mosaic_one_target(
                self,
                target=None,
                product=None,
                config=None,
                imaging_method='tclean',
                check_files=True,
                extra_ext_in='',
                extra_ext_out='',
        ):
            """
            Linearly mosaic a single target, performing the convolution,
            alignment, and mosaicking steps.
            """

            # Check that the target is a mosaic

            is_mosaic = self._kh.is_target_linmos(target)

            if not is_mosaic:
                logger.warning("Not a mosaic, returning.")
                return ()

            mosaic_parts = self._kh.get_parts_for_linmos(target)

            if imaging_method == 'sdintimaging':
                has_imaging = False
                for mosaic_part in mosaic_parts:
                    fname_dict = self._fname_dict(
                        target=mosaic_part, product=product, config=config,
                        imaging_method=imaging_method)
                    imaging_dir = self._kh.get_imaging_dir_for_target(mosaic_part)
                    has_imaging = os.path.isdir(imaging_dir + fname_dict['orig'])
                    if has_imaging:
                        break
                if not has_imaging:
                    imaging_method = 'tclean'

            # Check if the individual parts have single dish data. If they
            # do, flip the single dish flag to true.

            parts_have_singledish = False

            for this_part in mosaic_parts:

                this_part_has_sd = self._kh.has_singledish(target=this_part, product=product)

                if this_part_has_sd:
                    parts_have_singledish = True

            # Check if this is a feather configuration. If so, then flip
            # the single dish flag to false. This overrides the presence
            # of data - we don't treat the singledish for feathered data.

            if config in self.get_feather_configs() or imaging_method in ['sdintimaging']:
                parts_have_singledish = False

            self.task_convolve_parts_for_mosaic(
                target=target,
                product=product,
                config=config,
                in_tag='pbcorr_round',
                out_tag='linmos_commonres',
                extra_ext_in=extra_ext_in,
                extra_ext_out=extra_ext_in,
                check_files=check_files,
            )

            in_tag_list = ['linmos_commonres', 'weight']
            out_tag_list = ['linmos_aligned', 'weight_aligned']

            if parts_have_singledish:
                in_tag_list.append('prepped_sd')
                in_tag_list.append('sd_weight')
                out_tag_list.append('sd_aligned')
                out_tag_list.append('sd_weight_aligned')

            self.task_align_for_mosaic(
                target=target,
                product=product,
                config=config,
                in_tags=in_tag_list,
                out_tags=out_tag_list,
                extra_ext_in=extra_ext_in,
                extra_ext_out=extra_ext_in,
                check_files=check_files,
            )

            self.task_linear_mosaic(
                target=target,
                product=product,
                config=config,
                image_tag='linmos_aligned',
                weight_tag='weight_aligned',
                out_tag='pbcorr_round',
                extra_ext_in=extra_ext_in,
                extra_ext_out=extra_ext_out,
                check_files=check_files,
            )

            if parts_have_singledish:
                self.task_linear_mosaic(
                    target=target,
                    product=product,
                    config=config,
                    image_tag='sd_aligned',
                    weight_tag='sd_weight_aligned',
                    out_tag='prepped_sd',
                    extra_ext_in=extra_ext_in,
                    extra_ext_out=extra_ext_out,
                    check_files=check_files,
                )

            return ()

        def recipe_cleanup_one_target(
                self,
                target=None,
                product=None,
                config=None,
                check_files=True,
                ext_ext='',
        ):
            """
            Recipe that cleans up the output for one target, converting to
            Kelvin, compressing and trimming the cube and then exporting
            as a FITS file.
            """

            self.task_compress(
                target=target, config=config, product=product,
                check_files=check_files, do_pb_too=True,
                do_trimrind=True,
                extra_ext_in=ext_ext, extra_ext_out=ext_ext
            )

            self.task_convert_units(
                target=target, config=config, product=product,
                check_files=check_files,
                extra_ext_in=ext_ext, extra_ext_out=ext_ext
            )

            self.task_export_to_fits(
                target=target, config=config, product=product,
                check_files=check_files, do_pb_too=True,
                extra_ext_in=ext_ext, extra_ext_out=ext_ext
            )

            return ()

        def recipe_convolve_to_scale(
                self,
                target=None,
                product=None,
                config=None,
                ext_ext='',
                export_to_fits=True,
                check_files=True,
        ):
            """
            Convolve a target, product, config combination to a succession
            of angulars scale using the task that convolves to a round
            beam.
            """

            res_list = self._kh.get_res_for_config(config)
            if res_list is None:
                logger.error("No target resolutions found for config " + config)
                return ()

            for this_res in res_list:
                check_target_is_part, target_name = self._kh.is_target_in_mosaic(target, return_target_name=True)

                # res_tag = self._kh.get_tag_for_res(this_res)
                res_tag = utilsResolutions.get_tag_for_res(this_res)
                res_arcsec = utilsResolutions.get_angular_resolution_for_res(this_res,
                                                                             distance=self._kh.get_distance_for_target(
                                                                                 target_name))

                # Check if the requested beam is smaller than the current one

                self.task_round_beam(
                    target=target, config=config, product=product,
                    in_tag='pbcorr_trimmed_k', out_tag='pbcorr_trimmed_k',
                    extra_ext_in=ext_ext, extra_ext_out=ext_ext + '_res' + res_tag,
                    force_beam_as=res_arcsec,
                    check_files=check_files
                )

                if export_to_fits:
                    self.task_export_to_fits(
                        target=target, config=config, product=product,
                        check_files=check_files, do_pb_too=True,
                        extra_ext_in=ext_ext + '_res' + res_tag, extra_ext_out=ext_ext + '_res' + res_tag,
                    )

        # endregion

        # region Loops

        def loop_postprocess(
                self,
                imaging_method='tclean',
                do_all=False,
                do_prep=False,
                do_feather=False,
                do_mosaic=False,
                do_cleanup=False,
                do_summarize=False,
                trim_coarse_beam_edge_channels=False,
                feather_apod=False,
                feather_noapod=False,
                feather_before_mosaic=False,
                make_directories=True,
        ):
            """
            Loops over the full set of targets, products, and
            configurations to run the postprocessing. Toggle the parts of
            the loop using the do_XXX booleans. Other choices affect the
            algorithms used.
            """

            if do_all:
                do_prep = True
                do_feather = True
                do_mosaic = True
                do_cleanup = True
                do_summarize = True

            if feather_apod is False and feather_noapod is False:
                logger.info("Defaulting to no apodization.")
                feather_noapod = True

            if len(self.get_targets()) == 0:
                logger.error("Need a target list.")
                return (None)

            if len(self.get_all_products()) == 0:
                logger.error("Need a products list.")
                return (None)

            if make_directories:
                self._kh.make_missing_directories(postprocess=True)

            # Prepare the interferometer data that has imaging for further
            # postprocessing. Includes staging the single dish data,
            # making weights, etc. These are in the recipe_prep_one_target

            if do_prep:

                for this_target, this_product, this_config in \
                        self.looper(do_targets=True, do_products=True, do_configs=True):

                    has_imaging = False

                    if imaging_method == 'sdintimaging':
                        fname_dict = self._fname_dict(
                            target=this_target, product=this_product, config=this_config,
                            imaging_method=imaging_method)
                        imaging_dir = self._kh.get_imaging_dir_for_target(this_target)
                        has_imaging = os.path.isdir(imaging_dir + fname_dict['orig'])
                        if has_imaging:
                            imaging_method_prep = 'sdintimaging'

                    if not has_imaging:
                        fname_dict = self._fname_dict(
                            target=this_target, product=this_product, config=this_config)
                        imaging_dir = self._kh.get_imaging_dir_for_target(this_target)
                        has_imaging = os.path.isdir(imaging_dir + fname_dict['orig'])
                        if has_imaging:
                            imaging_method_prep = 'tclean'

                    if not has_imaging:
                        logger.debug("Skipping " + this_target + " because it lacks imaging.")
                        logger.debug(imaging_dir + fname_dict['orig'])
                        continue

                    self.recipe_prep_one_target(
                        target=this_target, product=this_product, config=this_config,
                        check_files=True,
                        trim_coarse_beam_edge_channels=trim_coarse_beam_edge_channels,
                        imaging_method=imaging_method_prep)

            # Feather the interferometer configuration data that has
            # single dish imaging. We'll return to feather mosaicked
            # intereferometer and single dish data in the next steps.

            if do_feather:

                for this_target, this_product, this_config in \
                        self.looper(do_targets=True, do_products=True, do_configs=True, just_interf=True):

                    is_mosaic = self._kh.is_target_linmos(this_target)
                    if is_mosaic:
                        continue

                    if imaging_method == 'sdintimaging':
                        fname_dict = self._fname_dict(
                            target=this_target, product=this_product, config=this_config,
                            imaging_method=imaging_method)
                        imaging_dir = self._kh.get_imaging_dir_for_target(this_target)
                        using_sdint = os.path.isdir(imaging_dir + fname_dict['orig'])
                        if using_sdint:
                            logger.debug('Skipping feathering for %s, %s, %s because using sdintimaging' %
                                         (this_target, this_product, this_config))
                            continue

                    fname_dict = self._fname_dict(
                        target=this_target, product=this_product, config=this_config)

                    imaging_dir = self._kh.get_imaging_dir_for_target(this_target)
                    has_imaging = os.path.isdir(imaging_dir + fname_dict['orig'])
                    has_singledish = self._kh.has_singledish(target=this_target, product=this_product)

                    is_part_of_mosaic = self._kh.is_target_in_mosaic(this_target)
                    if is_part_of_mosaic and not feather_before_mosaic:
                        logger.debug("Skipping " + this_target + " because feather_before_mosaic is False.")
                        continue

                    if not has_imaging:
                        logger.debug("Skipping " + this_target + " because it lacks imaging.")
                        logger.debug(imaging_dir + fname_dict['orig'])
                        continue

                    if not has_singledish:
                        logger.debug("Skipping " + this_target + " because it lacks single dish.")
                        continue

                    if feather_apod:
                        self.task_feather(
                            target=this_target, product=this_product, config=this_config,
                            apodize=True, apod_ext='pb', extra_ext_out='_apod', check_files=True,
                            copy_weights=True,
                        )

                    if feather_noapod:
                        self.task_feather(
                            target=this_target, product=this_product, config=this_config,
                            apodize=False, extra_ext_out='', check_files=True,
                            copy_weights=True,
                        )

            # Mosaic the interferometer, single dish, and feathered data.

            if do_mosaic:

                # Loop over interferometer configurations

                for this_target, this_product, this_config in \
                        self.looper(do_targets=True, do_products=True, do_configs=True, just_interf=True):

                    is_mosaic = self._kh.is_target_linmos(this_target)
                    if not is_mosaic:
                        continue

                    if feather_before_mosaic:
                        logger.debug("Skipping " + this_target + " because feather_before_mosaic is True.")
                        continue

                    # Mosaic the interferometer data and the
                    # single dish data (need to verify if parts
                    # have single dish, enforce the same
                    # astrometric grid).

                    self.recipe_mosaic_one_target(
                        target=this_target, product=this_product, config=this_config,
                        check_files=True,
                        imaging_method=imaging_method,
                        extra_ext_in='',
                        extra_ext_out='',
                    )

                # Loop over feather configurations

                for this_target, this_product, this_config in \
                        self.looper(do_targets=True, do_products=True, do_configs=True, just_feather=True):

                    is_mosaic = self._kh.is_target_linmos(this_target)
                    if not is_mosaic:
                        continue

                    if feather_apod:
                        self.recipe_mosaic_one_target(
                            target=this_target, product=this_product, config=this_config,
                            check_files=True,
                            extra_ext_in='_apod',
                            extra_ext_out='',
                        )

                    if feather_noapod:
                        self.recipe_mosaic_one_target(
                            target=this_target, product=this_product, config=this_config,
                            check_files=True,
                            extra_ext_in='',
                            extra_ext_out='',
                        )

            # This round of feathering targets only mosaicked data. All
            # other data have been feathered above already.

            if do_feather and feather_before_mosaic:

                # N.B. if using sdintimaging this will just crash out since it hasn't staged any singledish. This is
                # intended!

                for this_target, this_product, this_config in \
                        self.looper(do_targets=True, do_products=True, do_configs=True, just_interf=True):

                    # Skip if we're not a mosaic
                    is_mosaic = self._kh.is_target_linmos(this_target)
                    if not is_mosaic:
                        continue

                    if feather_apod:
                        self.task_feather(
                            target=this_target, product=this_product, config=this_config,
                            apodize=True, apod_ext='pb', extra_ext_out='_apod', check_files=True,
                        )

                    if feather_noapod:
                        self.task_feather(
                            target=this_target, product=this_product, config=this_config,
                            apodize=False, extra_ext_out='', check_files=True,
                        )

            # Trim and downsample the data, convert to Kelvin, etc.

            if do_cleanup:

                for this_target, this_product, this_config in \
                        self.looper(do_targets=True,
                                    do_products=True,
                                    do_configs=True):

                    # At this point, if using sdintimaging rename all the interf config files to their associated feathered
                    # config files

                    if imaging_method == 'sdintimaging':
                        self.task_rename_sdintimaging(target=this_target, product=this_product, config=this_config,
                                                      imaging_method=imaging_method)

                    self.recipe_cleanup_one_target(
                        target=this_target,
                        product=this_product,
                        config=this_config,
                        check_files=True)

            # Build reports summarizing the properties of the final
            # postprocessed data.

            if do_summarize:
                pass

    # endregion
