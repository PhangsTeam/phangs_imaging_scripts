"""imagingHandler

This module makes images and cubes out of the uv data products created by uvdataHandler, imaging in
chunks of channels when cube sizes are large.

This code needs to be run inside CASA.

"""

import datetime
import glob
import logging
import os
import warnings
from copy import deepcopy, copy

import numpy as np

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

    import analysisUtils as au

    from . import casaImagingRoutines as imr
    from . import casaMaskingRoutines as msr
    from . import casaStuff
    from . import handlerTemplate
    from . import utilsFilenames
    from . import utilsLines as lines
    from .clean_call import CleanCall, CleanCallFunctionDecorator

    class ImagingChunkedHandler(handlerTemplate.HandlerTemplate):
        """
        Class to makes image cubes out of uv data for imaging a single spectral line.

        Parameters
        ----------
        target : str
            The target name.
        config : str
            The configuration name.
        product : str
            The product name.
        key_handler : dict
            The key handler dictionary.
        dry_run : bool, optional
            If True, do not actually make the images. Defaults to False.
        chunksize : int, optional
            The number of channels per image cube. Defaults to 10.
        imaging_method : str, optional
            The imaging method. Defaults to 'tclean'.
        recipe : str, optional
            The recipe. Defaults to 'phangsalma'.
        set_cell_imsize_on_init : bool, optional
            If True, set the cell size on initialization. Defaults to True.
        force_square : bool, optional
            If True, force the image size to be square. Defaults to False.
        oversamp : int, optional
            The oversampling factor of pixels per beam FWHM. Defaults to 5.
        make_temp_dir : bool, optional
            If True, make a temporary directory. Defaults to True.
        temp_key : str, optional
            The temporary key. Defaults to None.
        temp_path : str, optional
            The temporary path. Defaults to None. Set this to specify an independent path for the
            temporary directory that is unassociated with the default imaging directory.

        """

        ############
        # __init__ #
        ############

        def __init__(
            self,
            target,
            config,
            product,
            key_handler,
            dry_run = False,
            chunksize=10,
            imaging_method='tclean',
            recipe='phangsalma',
            set_cell_imsize_on_init=True,
            force_square=False,
            oversamp=5,
            make_temp_dir=True,
            temp_key=None,
            temp_path=None,
            copy_ms_to_temp=False,
            ):

            # inherit template class
            super().__init__(key_handler = key_handler, dry_run = dry_run)

            # Check existence in the keyhandler
            assert target in self._kh.get_targets()
            assert config in self._kh.get_interf_configs()
            assert product in self._kh.get_line_products()

            self.target = target
            self.config = config
            self.product = product

            # Enforce that chunked imaging be used for cubes only.
            # Otherwise, there's no need for this approach:
            is_line_product = self.product in self._kh.get_line_products()
            if not is_line_product:
                raise Exception("ImagingChunkedHandler is only implemented for creating data cubes."
                                " Use the normal imaging approach for imaging continuum data.")

            if chunksize == 1:
                raise Exception("chunksize must be greater than 1.")

            self._orig_imaging_dir = self._kh.get_imaging_dir_for_target(self.target, changeto=True)

            # Get the visibility name
            self.vis_file = utilsFilenames.get_vis_filename(target=target, product=product,
                                                            config=config)

            self.full_vis_file = os.path.join(self._orig_imaging_dir, self.vis_file)

            self.image_root = utilsFilenames.get_cube_filename(target=target, product=product, config=config,
                                                               casa=True, casaext='')

            if imaging_method not in ['tclean', 'sdintimaging']:
                logger.error('imaging_method should be either tclean or sdintimaging')
                raise Exception('imaging_method should be either tclean or sdintimaging')

            self.imaging_method = imaging_method

            known_recipes = ['phangsalma']
            if recipe not in known_recipes:
                logger.error("Recipe not known " + recipe)
                return (None)
            self.recipe = recipe

            self.recipe_list = self._kh.get_imaging_recipes(config=config, product=product)

            # Line setup:
            this_line_tag = self._kh.get_line_tag_for_line_product(product=product)
            if this_line_tag not in lines.line_list.keys():
                logger.error("Did not find line in line_list " + this_line_tag)
                raise Exception("Did not find line in line_list " + this_line_tag)
            self.line_tag = this_line_tag

            self.rest_freq_ghz = lines.line_list[this_line_tag]

            # Make needed directories
            self._kh.make_missing_directories(imaging=True)

            # Want to call this only once, then use throughout
            # TODO: this will fail if false as we need it for the base call.
            # Could just not give the option and force computing this on init
            if set_cell_imsize_on_init:
                self.set_cube_cell_and_imsize(force_square=force_square, oversamp=oversamp)

            # Find the number of channels in the MS to image
            # This defaults to using all (assuming it's a staged MS)
            #TODO: handle not imaging all channels and spw != 0.
            self.set_channel_num(spw=0)

            # Build the initial clean call to define our channel chunk ranges.
            self.base_clean_call = CleanCall(self.recipe_list, use_chunks=True, nchan=self.nchan)

            self.chunksize = chunksize

            self.chunk_channel_starts, self.chunk_channel_ends = \
                self.base_clean_call.return_chunked_channel_ranges(chunksize=chunksize)

            # If we're moving things around, take note of that here
            self.make_temp_dir = make_temp_dir
            self.copy_ms_to_temp = copy_ms_to_temp

            if self.make_temp_dir:
                if temp_path is None:
                    if temp_key is None:
                        self.temp_key = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
                    self._this_imaging_dir = f"{self._orig_imaging_dir}/temp_{self.image_root}_{self.temp_key}"
                else:
                    self._this_imaging_dir = temp_path

                if self.copy_ms_to_temp:
                    # The full visibility file will be in the imaging directory
                    self.full_vis_file = os.path.join(self._this_imaging_dir, self.vis_file)

            else:
                self._this_imaging_dir = self._orig_imaging_dir

            # Set a flag to check on whether we need to move clean-up the final products
            self._uses_tempdir = make_temp_dir

            self.configure_chunk_parameters()

        @property
        def tempdir_exists(self):
            """Return True if a temporary imaging directory was created and still exists on disk."""
            return self._uses_tempdir and os.path.isdir(self._this_imaging_dir)

        def set_channel_num(self, spw=0):
            '''
            Set number of channels from the linked MS.
            '''

            msmd = casaStuff.msmdtool()
            msmd.open(self.full_vis_file)
            self.nchan = msmd.nchan(spw)
            msmd.close()

            logger.info("Found {} channels in the vis file.".format(self.nchan))


        def configure_chunk_parameters(self):
            '''
            Create dictionaries with the required meta data for tracking the
            '''

            # Make a dictionary of the split MS names
            base_chunk_params = {'vis_name': None,
                                 'channel_range': [],
                                 'status': None}

            self.chunk_params = {}

            for chunk_num, (chan_start, chan_end) in enumerate(zip(self.chunk_channel_starts,
                                                                   self.chunk_channel_ends)):

                chan_end = int(chan_end)
                chan_start = int(chan_start)

                self.chunk_params[chunk_num] = deepcopy(base_chunk_params)

                # Channel range
                self.chunk_params[chunk_num]['channel_range'] = [chan_start, chan_end]

                # Vis file names
                chan_label = "{0}_{1}".format(chan_start, chan_end)

                # this_vis_name = "{0}_chan{1}".format(self.vis_file, chan_label)
                this_vis_name = self.full_vis_file

                self.chunk_params[chunk_num]['vis_name'] = this_vis_name

                self.chunk_params[chunk_num]['full_vis_name'] = self.full_vis_file

                # Image names

                this_image_name = "{0}_chan{1}".format(self.image_root, chan_label)
                self.chunk_params[chunk_num]['image_name'] = this_image_name

                full_imagename = "{0}/{1}".format(self._this_imaging_dir,
                                                  this_image_name)
                self.chunk_params[chunk_num]['full_imagename'] = full_imagename

            self.nchunks = len(self.chunk_params)

        ###############
        # _fname_dict #
        ###############

        def _fname_dict(
            self,
            imagename,
            imaging_method='tclean',
            ):
            """
            Handles file names used in the imaging processes.
            Cubes only for the chunked imaging process.
            """
            fname_dict = {}

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

            return fname_dict


        def run_imaging(
                self,
                chunk_num=None,
                do_all=False,
                do_dirty_image=False,
                do_revert_to_dirty=False,
                do_read_clean_mask=False,
                do_multiscale_clean=False,
                do_revert_to_multiscale=False,
                do_singlescale_mask=False,
                singlescale_mask_high_snr=None,
                singlescale_mask_low_snr=None,
                singlescale_mask_absolute=False,
                skip_singlescale_if_mask_empty=True,
                do_singlescale_clean=False,
                do_revert_to_singlescale=False,
                do_recombine_cubes=False,
                do_export_to_fits=True,
                do_cleanup=True,
                convergence_fracflux=0.01,
                convergence_noise_z_threshold=None,
                singlescale_threshold_value=1.0,
                extra_ext_in=None,
                suffix_in=None,
                extra_ext_out=None,
                dynamic_sizing=True,
                force_square=False,
                overwrite=False,
                ):
            """
            Run the imaging process.

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
                do_cleanup = True

            cwd = os.getcwd()

            # Create the imaging directory if it doesn't exist
            if not os.path.exists(self._this_imaging_dir):
                os.makedirs(self._this_imaging_dir)

            # Change to the relevant directory
            os.chdir(self._this_imaging_dir)

            # Copy the MS to the temp directory if it doesn't exist
            if self.make_temp_dir and self.copy_ms_to_temp:
                if not os.path.exists(self.full_vis_file):
                    os.system(f"cp -r {os.path.join(self._orig_imaging_dir, self.vis_file)} {self._this_imaging_dir}")

            # print starting message
            logger.info("")
            logger.info("--------------------------------------------------------")
            logger.info("START: Imaging target " + self.target + " config " + self.config + " product " + self.product)
            logger.info("--------------------------------------------------------")
            logger.info('Imaging recipe: ' + self.recipe)

            if self.recipe == 'phangsalma':
                self.recipe_phangsalma_imaging(
                    chunk_num=chunk_num,
                    extra_ext_in=extra_ext_in,
                    suffix_in=suffix_in,
                    extra_ext_out=extra_ext_out,
                    # imaging_method_override=None,
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
                    do_recombine_cubes=do_recombine_cubes,
                    do_export_to_fits=do_export_to_fits,
                    do_cleanup=do_cleanup,
                    convergence_fracflux=convergence_fracflux,
                    convergence_noise_z_threshold=convergence_noise_z_threshold,
                    singlescale_threshold_value=singlescale_threshold_value,
                    dynamic_sizing=dynamic_sizing,
                    force_square=force_square,
                    overwrite=overwrite,
                )

            os.chdir(cwd)

            # print ending message
            logger.info("--------------------------------------------------------")
            logger.info("END: Imaging target " + self.target + " config " + self.config + " product " + self.product)
            logger.info("--------------------------------------------------------")

        def return_valid_chunks(self, chunk_num=None):
            '''
            Allow specifying and validating running on a subset of chunk numbers.

            Parameters
            ----------
            chunk_num : int, list or None
                If None, will loop through all chunks. If a list or integer are given,
                this will loop through only the specified chunk numbers defined in
                `ImagingChunkedHandler.chunk_params`.

            Returns
            -------
            chunks_iter : list
                List of chunk numbers to be considered in a task.
            '''

            if chunk_num is None:
                chunks_iter = list(self.chunk_params.keys())
            else:
                if isinstance(chunk_num, list) or isinstance(chunk_num, np.ndarray):
                    chunks_iter = chunk_num
                elif isinstance(chunk_num, int):
                    chunks_iter = [chunk_num]
                else:
                    raise TypeError("Unable to parse chunk_num of given type {}. ".format(type(chunk_num)) +
                                    "chunk_num must be a list or integer.")

                # Ensure that all chunk numbers given are contained in self.chunk_params
                invalid_chunks = []
                for this_chunknum in chunks_iter:
                    if not this_chunknum in self.chunk_params:
                        invalid_chunks.append(this_chunknum)
                if len(invalid_chunks) > 0:
                    raise ValueError("Invalid chunk numbers given: {} ".format(invalid_chunks) +
                                     "Check self.chunk_params for valid chunk numbers.")

            return chunks_iter

        ##################################################################
        # Tasks - discrete steps on target, product, config combinations #
        ##################################################################

        # CASA Memo 13 shows that the MS splitting is likely not needed
        # https://drive.google.com/file/d/1_8JeN-MtDEqUYjRn7eIUbqcE0EyJeSqW/view

        # def task_split_chunked_vis(self, overwrite=False, chunk_num=None):
        #     '''
        #     Split out the specified chunk of channels.

        #     Parameters
        #     ----------
        #     overwrite : bool
        #         Overwrite an existing MS file for the chunk.
        #     chunk_num : int, list or None
        #         If None, will loop through all chunks. If a list or integer are given,
        #         this will loop through only the specified chunk numbers defined in
        #         `ImagingChunkedHandler.chunk_params`.
        #     '''

        #     chunks_iter = self.return_valid_chunks(chunk_num=chunk_num)

        #     # TODO: move to helper function.
        #     mytb = au.createCasaTool(casaStuff.tbtool)
        #     mytb.open(self.full_vis_file, nomodify = True)
        #     colnames = mytb.colnames()
        #     if 'CORRECTED_DATA' in colnames:
        #         logger.info("Data has a CORRECTED column. Will use that.")
        #         use_column = 'CORRECTED'
        #     else:
        #         logger.info("Data lacks a CORRECTED column. Will use DATA column.")
        #         use_column = 'DATA'
        #     mytb.close()

        #     for ii, key in enumerate(chunks_iter):

        #         this_vis_chunk = self.chunk_params[key]['full_vis_name']
        #         chan_start, chan_stop = self.chunk_params[key]['channel_range']

        #         logger.info("Splitting chunk number {0} for channels {1}~{2}".format(ii, chan_start,
        #                                                                             chan_stop))

        #         if os.path.exists(this_vis_chunk):
        #             if overwrite:
        #                 os.system("rm -rf {}".format(this_vis_chunk))
        #             else:
        #                 logger.info("Chunked vis for channels {} already exists. Skipping".format(key))
        #                 continue

        #         #TODO: how to handle multiple SPWs? This assumes the data has been staged so that the
        #         # channel numbers would all match in frequency.

        #         casaStuff.split(vis=self.full_vis_file, outputvis=this_vis_chunk,
        #                         spw='*:{0}~{1}'.format(chan_start, chan_stop),
        #                         datacolumn=use_column)


        def task_split_cube_to_chunks(self, imagename, overwrite=False, chunk_num=None,
                                      specaxis_name="Frequency"):
            '''
            Split a given cube into the channel chunks.

            Parameters
            ----------
            overwrite : bool
                Overwrite an existing MS file for the chunk.
            chunk_num : int, list or None
                If None, will loop through all chunks. If a list or integer are given,
                this will loop through only the specified chunk numbers defined in
                `ImagingChunkedHandler.chunk_params`.
            '''

            chunks_iter = self.return_valid_chunks(chunk_num=chunk_num)

            myia = au.createCasaTool(casaStuff.iatool)
            myrg = au.createCasaTool(casaStuff.rgtool)

            myia.open(imagename)

            # Find the spectral axis
            csys = myia.coordsys()
            try:
                spec_axis = np.where(np.asarray(csys.names()) == specaxis_name)[0][0]
            except IndexError:
                myia.close()

                raise IndexError("Cannot find spectral axis" + specaxis_name + " in " +
                                str(csys.names()))

            # Check given number of channels
            cube_shape = list(myia.shape())
            ndims = len(cube_shape)

            for ii, key in enumerate(chunks_iter):

                chan_start, chan_stop = self.chunk_params[key]['channel_range']
                chan_label = "{0}_{1}".format(chan_start, chan_stop)

                logger.info("Splitting chunk number {0} for channels {1}~{2}".format(ii, chan_start,
                                                                                    chan_stop))

                this_chunk_imagename = "{0}_{1}".format(imagename, chan_label)

                if os.path.exists(this_chunk_imagename):
                    if overwrite:
                        os.system("rm -rf {}".format(this_chunk_imagename))
                    else:
                        logger.info("Chunked vis for channels {} already exists. Skipping".format(key))
                        continue

                if verbose:
                    print("On channel "+str(chan+1)+" of "+str(start+nchan))

                lower_corner = [0] * ndims
                upper_corner = copy(cube_shape)

                # Set the channel
                lower_corner[spec_axis] = chan_start
                upper_corner[spec_axis] = chan_stop

                box = myrg.box(lower_corner, upper_corner)

                # Now make sliced image
                im_slice = myia.subimage(this_chunk_imagename,
                                         box)
                im_slice.done()

            myia.done()
            myia.close()


        def task_initialize_clean_call(
                self,
                chunk_num,
                stage='dirty',
        ):
            """
            Initialize a clean call object for a target, config, product
            combination and an imaging stage.
            """

            if not chunk_num in self.chunk_params:
                raise ValueError("chunk_num {} is not a valid chunk defined in chunk_params.".format(chunk_num))

            # Initialize from the base clean call mode on init
            clean_call = deepcopy(self.base_clean_call)

            vis_file = self.chunk_params[chunk_num]['vis_name']
            full_vis_file = self.chunk_params[chunk_num]['full_vis_name']

            if not os.path.isdir(full_vis_file):
                logger.error('Visibility file not found: ' + full_vis_file + " returning.")
                logger.error("vis_file for chunk does not exist. Run `task_split_chunked_vis` first.")
                return None

            if not hasattr(self, 'cell'):
                raise ValueError("Run self.set_cube_cell_and_imsize first.")
            clean_call.set_param('cell', self.cell, nowarning=True)
            clean_call.set_param('imsize', self.imsize, nowarning=True)

            # Set the visibility file (note that we assume we are in the working directory)

            clean_call.set_param('vis', self.full_vis_file, nowarning=True)

            # Set the output image file name (note no suffix for imaging root)
            image_root = self.chunk_params[chunk_num]['full_imagename']
            clean_call.set_param('imagename', image_root, nowarning=True)

            # Get the phase center associated with the target
            rastring, decstring = self._kh.get_phasecenter_for_target(target=self.target)
            phasecenter = 'J2000 {0} {1}'.format(rastring, decstring)
            clean_call.set_param('phasecenter', phasecenter)

            # Set the rest frequency or reference frequency
            if clean_call.get_param('specmode') != 'cube':
                logger.error('Line product detected but specmode is not cube.')
                raise Exception('Malformed clean call: Line product detected but specmode is not cube.')

            if stage == 'dirty':
                clean_call.set_param('deconvolver', 'hogbom')
            elif stage == 'multiscale':
                clean_call.set_param('deconvolver', 'multiscale')

                # Set the multiscale to use:
                scales_to_clean = self._kh.get_clean_scales_for_config(config=self.config)
                clean_call.set_multiscale_arcsec(scales=scales_to_clean)

            elif stage == 'singlescale':
                clean_call.set_param('deconvolver', 'hogbom')
            else:
                raise ValueError('stage must be one of [dirty, multiscale, singlescale].')

            clean_call.set_restfreq_ghz(self.rest_freq_ghz)

            # Set to the channel range given we split out max {chunksize} number of channels
            # Thus we can just specify -1 to image all channel in the split MS.
            chan_start, chan_end = self.chunk_params[chunk_num]['channel_range']
            nchan = (chan_end - chan_start) + 1
            clean_call.set_param('nchan', nchan)

            # Set start and width
            clean_call.set_param('start', chan_start)
            clean_call.set_param('width', 1)

            if self.imaging_method == 'sdintimaging':
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

        def set_cube_cell_and_imsize(self, force_square=False, oversamp=5):
            '''
            Set a single cell and imsize for all chunks.
            '''

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Calculating target cell size and image size for:")
            logger.info(self.vis_file)
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            # Call the estimating routine
            if not self._dry_run:
                self.cell, self.imsize = \
                    imr.estimate_cell_and_imsize(self.full_vis_file,
                                                 oversamp,
                                                 force_square=force_square)
            else:
                self.cell, self.imsize = '0.1arcsec', [1000, 1000]
                logger.info('DRY RUN skips calling imr.estimate_cell_and_imsize()')

            logger.info('cell={0}; imsize={1}'.format(self.cell, self.imsize))


        @CleanCallFunctionDecorator
        def task_setup_sdintimaging(
                self,
                clean_call=None,
                target=None,
                product=None,
                asvelocity=True,
                overwrite=False,
                chunk_num=None
        ):
            """
            Regrid existing TP image to the staged MS

            Parameters
            ----------
            clean_call : CleanCall object
            target : str
            product : str
            asvelocity : bool
            overwrite : bool
            chunk_num : int or None

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
                casaStuff.importfits(fitsimage=sd_fits_file,
                                     imagename=sd_image_file,
                                     overwrite=True,
                                     defaultaxes=True,
                                     defaultaxesvalues=["", "", "", "I"])

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

                spec_key = 'spectral2'
                if spec_key not in template_hdr['csys']:
                    spec_key = 'spectral1'

                spec_shap = template_hdr['shap'][-1]
                spec_crpix = template_hdr['csys'][spec_key]['wcs']['crpix']
                spec_cdelt = template_hdr['csys'][spec_key]['wcs']['cdelt']
                spec_crval = template_hdr['csys'][spec_key]['wcs']['crval']

                if not (spec_shap == n_chan and spec_crpix == 0.0 and spec_cdelt == d_freq and spec_crval == first_freq):

                    template_hdr['shap'][-1] = n_chan
                    template_hdr['csys'][spec_key]['wcs']['crpix'] = 0.0
                    template_hdr['csys'][spec_key]['wcs']['cdelt'] = d_freq
                    template_hdr['csys'][spec_key]['wcs']['crval'] = first_freq

                    casaStuff.imregrid(imagename=sd_image_file,
                                       output=sd_image_file + '_regrid',
                                       template=template_hdr,
                                       asvelocity=asvelocity,
                                       overwrite=True,
                                       )
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
                if not chunk_num is None:
                    start_chan = self.chunk_channel_starts[chunk_num]
                    end_chan = self.chunk_channel_ends[chunk_num]

                    casaStuff.imsubimage(imagename=sd_image_file,  outfile=sd_image_file+"_chunked",overwrite=True, chans=f"{start_chan}~{end_chan}")
                    sd_image_file= sd_image_file+"_chunked"


            return sd_image_file


        def task_gather_into_cube(self, root_name=None,
                                  remove_chunks=False,
                                  overwrite=False):
            '''
            Gather the chunked data into a final cube.
            '''

            if root_name is not None:
                root_name_label = "_{}".format(root_name)
            else:
                root_name_label = ""

            # These are the final cube names. We'll loop over cube products to concat together
            fname_dict = self._fname_dict(imagename="{0}{1}".format(self.image_root, root_name_label),
                                          imaging_method=self.imaging_method)

            del fname_dict['root']
            del fname_dict['suffix']

            # Make dictionary to track existing chunked image names
            chunk_fname_dict = dict.fromkeys(fname_dict)

            # We can't concat non cube products:
            skip_types = ['sumwt', 'weight']
            # If a temp dir was used but has since been removed, search the original imaging dir
            if not self.tempdir_exists:
                effective_imaging_dir = self._orig_imaging_dir
            else:
                effective_imaging_dir = self._this_imaging_dir

            # With the chunk names sorted, concat into cubes
            for img_type in fname_dict:

                chunk_fname_dict[img_type] = []

                missing_chunks = []

                logger.info("Concatenating chunks of type {} into a final cube".format(img_type))

                for chunk_num in self.chunk_params:

                    this_imagename = "{0}/{1}{2}.{3}".format(effective_imaging_dir,
                                                             self.chunk_params[chunk_num]['image_name'],
                                                             root_name_label, img_type)
                    logger.info("Image {} ".format(this_imagename))
                    if not os.path.exists(this_imagename):
                        #try sdintimaging:
                        this_imagename = "{0}/{1}{2}.joint.cube.{3}".format(
                            effective_imaging_dir,
                            self.chunk_params[chunk_num]['image_name'],
                            root_name_label, img_type)

                        if not os.path.exists(this_imagename):
                            missing_chunks.append(chunk_num)
                            continue

                    chunk_fname_dict[img_type].append(this_imagename)
                if img_type in skip_types:
                    continue

                if len(chunk_fname_dict[img_type]) != self.nchunks:
                    logger.error("Existing imaging products for type {} do not match the expected number of chunks.".format(img_type) +
                                "This cube will not be made.")
                    if len(missing_chunks) > 0:
                        logger.error("Missing the following chunks for the {0}: {1}".format(img_type, missing_chunks))
                        logger.error("The concatenated cube cannot be made for {}".format(img_type))
                    continue

                if len(missing_chunks) > 0:
                    logger.error("Missing the following chunks for the {0}: {1}".format(img_type, missing_chunks))
                    logger.error("The concatenated cube cannot be made for {}".format(img_type))
                    continue
                if os.path.exists(fname_dict[img_type]):
                    if overwrite:
                        os.system("rm -rf {}".format(fname_dict[img_type]))
                    else:
                        logger.warn("Cube name {} already exists. Skipping.".format(fname_dict[img_type]) +
                                    " Enable overwrite=True to force writing a new version.")
                        continue

                # If only one chunk exists, just copy it
                if len(chunk_fname_dict[img_type]) == 1:
                    logger.info("Only one chunk found for {}. Copying instead of concatenating.".format(img_type))
                    os.system("cp -r {0} {1}".format(this_imagename, self._orig_imaging_dir+fname_dict[img_type]))

                # Otherwise concatenate
                else:
                    ia = casaStuff.image()
                    im = ia.imageconcat(outfile=fname_dict[img_type],
                        infiles=chunk_fname_dict[img_type])
                    im.done()
                    ia.close()

            # (optional) clean up the per chunk imaging products
            if remove_chunks:
                logger.info("Removing chunked imaging products for image type {}".format(root_name))
                for img_type in list(chunk_fname_dict) + skip_types:
                    if chunk_fname_dict[img_type] is None or len(chunk_fname_dict[img_type]) == 0:
                        continue
                    for chunk_imagename in chunk_fname_dict[img_type]:
                        os.system("rm -rf {}".format(chunk_imagename))


        @CleanCallFunctionDecorator
        def task_make_dirty_image(
                self,
                chunk_num=None,
                backup=True,
                gather_chunks_into_cube=False,
                remove_chunks=False,
        ):
            """
            Execute the attached clean call with zero iterations to create
            a dirty image.

            Set backup to True will make a copy of the dirty image as
            {imagename}_dirty.image
            """

            cwd = os.getcwd()

            chunks_iter = self.return_valid_chunks(chunk_num=chunk_num)

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Making dirty images for {} chunks:".format(len(chunks_iter)))
            logger.info(str(chunks_iter))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            target = self.target
            product = self.product
            config = self.config
            overwrite = False
            if not self._dry_run:

                os.chdir(self._this_imaging_dir)

                for ii, this_chunk_num in enumerate(chunks_iter):

                    # Make the chunk clean call:
                    this_clean_call = self.task_initialize_clean_call(this_chunk_num, stage='dirty')

                    if self.imaging_method == 'sdintimaging':
                        sd_fits_file = self._kh.get_sd_filename(target=target, product=product)
                        feather_config = self._kh.get_feather_config_for_interf_config(interf_config=config)
                        if not sd_fits_file or not feather_config:
                            logger.warning('No singledish setup for %s, %s, %s, reverting to standard tclean' %
                                   (target, product, config))
                            imaging_method = 'tclean'
                        else:
                            sd_image_file = self.task_setup_sdintimaging(this_clean_call, target=target, product=product,
                                                                     overwrite=overwrite,
                                                                     chunk_num=this_chunk_num)
                            if not sd_image_file:
                                logger.error('Error in setting up singledish for sdintimaging')

                            # Set the clean call parameters as necessary.
                            this_clean_call.set_param('usedata', 'sdint')
                            this_clean_call.set_param('sdimage', sd_image_file)

                    logger.info("")
                    logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
                    logger.info("Making dirty image for chunk {}:".format(this_chunk_num))
                    logger.info(str(this_clean_call.get_param('imagename')))
                    logger.info("This is {0} out of {1} to be imaged".format(ii+1, len(chunks_iter)))
                    logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
                    logger.info("")

                    imr.make_dirty_image(this_clean_call, imaging_method=self.imaging_method)
                    if backup:
                        imr.copy_imaging(
                            input_root=this_clean_call.get_param('imagename'),
                            output_root=this_clean_call.get_param('imagename') + '_dirty',
                            imaging_method=self.imaging_method,
                            wipe_first=True)

            if gather_chunks_into_cube:
                self.task_gather_into_cube(root_name='dirty',
                                           remove_chunks=remove_chunks)

            os.chdir(cwd)


        @CleanCallFunctionDecorator
        def task_revert_to_imaging(
                self,
                clean_call=None,
                tag='dirty',
                chunk_num=None,
                revert_cube=False,
                verbose=False,
        ):
            """
            Reset the current imaging stack to an earlier imaging
            state. Requires a tag that will be used to define the old
            imaging moved into place. Wipes the current imaging.
            """

            chunks_iter = self.return_valid_chunks(chunk_num=chunk_num)

            if not self._dry_run:

                for ii, this_chunk_num in enumerate(chunks_iter):

                    # Make the chunk clean call:
                    this_clean_call = self.task_initialize_clean_call(this_chunk_num)

                    if verbose:
                        logger.info("")
                        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
                        logger.info("Resetting to " + tag + " imaging:")
                        logger.info(str(this_clean_call.get_param('imagename')))
                        logger.info("This is chunk {0} out of {1}.".format(ii+1, len(chunks_iter)))
                        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
                        logger.info("")

                    imr.copy_imaging(
                        input_root=this_clean_call.get_param('imagename') + '_' + tag,
                        output_root=this_clean_call.get_param('imagename'),
                        imaging_method=self.imaging_method,
                        wipe_first=True)


                # If it exists, try to revert the whole cube.
                if revert_cube:
                    raise NotImplementedError

                    if verbose:
                        logger.info("")
                        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
                        logger.info("Resetting to " + tag + " imaging:")
                        logger.info(str(this_clean_call.get_param('imagename')))
                        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
                        logger.info("")

                    imr.copy_imaging(
                        input_root=clean_call.get_param('imagename') + '_' + tag,
                        output_root=clean_call.get_param('imagename'),
                        imaging_method=imaging_method,
                        wipe_first=True)


        @CleanCallFunctionDecorator
        def task_read_clean_mask(
                self,
                clean_call=None,
                target=None,
                config=None,
                product=None,
        ):
            """
            Identify the clean mask associated with the target and
            product, read it from disk (in FITS form) and align it to the
            astrometry and grid of the current imaging.
            """

            # TODO: check the import and align can be done efficiently per chunk
            # Otherwise, we need the ability to only do this operation once, then split

            raise NotImplementedError

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
            if not casa_enabled:
                return ()

            # Get fname dict
            fname_dict = self._fname_dict(product=product, imagename=clean_call.get_param('imagename'),
                                          imaging_method=self.imaging_method)

            # import_and_align_mask
            msr.import_and_align_mask(in_file=this_cleanmask,
                                      out_file=fname_dict['mask'],
                                      template=fname_dict['image'],
                                      blank_to_match=True)
            # AKL - propose to deprecate
            # clean_call.set_param('usemask','user')

            # if imaging_method == 'sdintimaging':
            #     os.system('cp -r %s %s' % (out_file, out_file.replace('.joint.cube', '.int.cube')))


        @CleanCallFunctionDecorator
        def task_multiscale_clean(
                self,
                chunk_num=None,
                convergence_fracflux=0.01,
                convergence_noise_z_threshold=None,
                backup=True,
                gather_chunks_into_cube=False,
                remove_chunks=False,
        ):
            """
            Run a multiscale clean loop to convergence. This task
            currently hardcodes some of the PHANGS-ALMA best choice
            parameters.

            Set backup to True will make a copy of the multiscale cleaned
            image as {imagename}_multiscale.image
            """

            cwd = os.getcwd()

            if self._dry_run:
                return ()

            chunks_iter = self.return_valid_chunks(chunk_num=chunk_num)
            target = self.target
            product = self.product
            config = self.config
            overwrite = False
            for ii, this_chunk_num in enumerate(chunks_iter):

                os.chdir(self._this_imaging_dir)

                # Make the chunk clean call:
                this_clean_call = self.task_initialize_clean_call(this_chunk_num, stage='multiscale')

                if self.imaging_method == 'sdintimaging':
                    sd_fits_file = self._kh.get_sd_filename(target=target, product=product)
                    feather_config = self._kh.get_feather_config_for_interf_config(interf_config=config)
                    if not sd_fits_file or not feather_config:
                        logger.warning('No singledish setup for %s, %s, %s, reverting to standard tclean' %
                                   (target, product, config))
                        imaging_method = 'tclean'
                    else:
                        sd_image_file = self.task_setup_sdintimaging(this_clean_call, target=target, product=product,
                                                                     overwrite=overwrite,
                                                                     chunk_num=this_chunk_num)
                        if not sd_image_file:
                            logger.error('Error in setting up singledish for sdintimaging')

                        # Set the clean call parameters as necessary.
                        this_clean_call.set_param('usedata', 'sdint')
                        this_clean_call.set_param('sdimage', sd_image_file)

                if this_clean_call.get_param('deconvolver') not in ['multiscale', 'mtmfs']:
                    logger.warning("I expected a multiscale or mtmfs deconvolver but got " + str(
                        this_clean_call.get_param('deconvolver')) + ".")
                    raise Exception("Incorrect clean call! Should have a multiscale or mtmfs deconvolver.")

                logger.info("")
                logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
                logger.info("Running clean call to convergence for:")
                logger.info(this_clean_call.get_param('imagename'))
                logger.info("This is {0} out of {1} to be imaged".format(ii+1, len(chunks_iter)))
                logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
                logger.info("")

                imr.clean_loop(clean_call=this_clean_call,
                            imaging_method=self.imaging_method,
                            record_file=this_clean_call.get_param('imagename') + '_multiscale_record.txt',
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

                if backup:
                    imr.copy_imaging(
                        input_root=this_clean_call.get_param('imagename'),
                        output_root=this_clean_call.get_param('imagename') + '_multiscale',
                        imaging_method=self.imaging_method,
                        wipe_first=True)

            if gather_chunks_into_cube:
                self.task_gather_into_cube(root_name='multiscale',
                                           remove_chunks=remove_chunks)

            os.chdir(cwd)


        @CleanCallFunctionDecorator
        def task_singlescale_mask(
                self,
                chunk_num=None,
                high_snr=None,
                low_snr=None,
                absolute=False,
                force_mask_by_cube=False,
        ):
            """
            Create a signal-to-noise based mask within the existing clean
            mask for deep cleaning. Used before running a deep single
            scale clean.

            If high_snr is None, will default to 4.0 (line imaging), or 2.5 (continuum imaging)

            If low_snr is None, will default to 2.0 (line imaging), or 1.0 (continuum imaging)
            """

            # get imagename

            imagename = self._fname_dict(self.image_root,
                                         imaging_method=self.imaging_method)['image']

            # print message
            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Creating signal-to-noise based clean mask for:")
            logger.info(str(imagename))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            if self._dry_run:
                return()

            # NOTE: we've removed the non-line defaults here as this approach should only be used for
            # line imaging.
            if high_snr is None:
                high_snr = 4.0
            if low_snr is None:
                low_snr = 2.0

            # Split here to force gather and mask using the whole cube vs. mask by individual channels:
            if force_mask_by_cube:

                raise NotImplementedError("Needs a routine to split the mask cube back into channels.")

                if not os.path.isdir(imagename):
                    logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
                    logger.info("Creating the full line cube for signal masking.")
                    logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
                    self.task_gather_into_cube(root_name='',
                                               remove_chunks=False)

                    if not os.path.isdir(imagename):
                        logger.error("Image not found: " + imagename)
                        logger.error("Cube construction failed. Check that all chunks have been imaged")
                        return ()

                # signal_mask
                msr.signal_mask(imaging_method=self.imaging_method,
                                cube_root=fname_dict['root'],
                                out_file=fname_dict['mask'],
                                suffix_in=fname_dict['suffix'],
                                suffix_out='',
                                operation='AND',
                                high_snr=high_snr,
                                low_snr=low_snr,
                                absolute=absolute)

                # TODO: add a channel splitting routine given a cube name

            else:

                # This path masks per channel:

                logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
                logger.info("Applying signal masking on individual channels, not the whole cube.")
                logger.info("Enable `force_mask_by_cube` to enable signal masking on the whole cube.")
                logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")

                chunks_iter = self.return_valid_chunks(chunk_num=chunk_num)

                for ii, this_chunk_num in enumerate(chunks_iter):

                    logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
                    logger.info("Signal masking chunk {0} of {1}".format(ii, len(chunks_iter)))
                    logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")

                    image_root = self.chunk_params[this_chunk_num]['full_imagename']
                    mask_name = "{}.mask".format(self.chunk_params[this_chunk_num]['full_imagename'])

                    # signal_mask
                    msr.signal_mask(imaging_method=self.imaging_method,
                                    cube_root=image_root,
                                    out_file=mask_name,
                                    suffix_in='',
                                    suffix_out='',
                                    operation='AND',
                                    high_snr=high_snr,
                                    low_snr=low_snr,
                                    absolute=absolute,
                                    do_roll=False)


        @CleanCallFunctionDecorator
        def task_singlescale_clean(
                self,
                chunk_num=None,
                convergence_fracflux=0.01,
                convergence_noise_z_threshold=None,
                gather_chunks_into_cube=False,
                remove_chunks=False,
                threshold_value=1.0,
                skip_singlescale_if_mask_empty=True,
                backup=True,
        ):
            """
            Run a singlescale clean loop to convergence. This task
            currently hardcodes some of the PHANGS-ALMA best choice
            parameters.

            Set backup to True will make a copy of the singlescale cleaned
            image as {imagename}_singlescale.image
            """

            cwd = os.getcwd()

            if self._dry_run:
                return ()


            chunks_iter = self.return_valid_chunks(chunk_num=chunk_num)

            for ii, this_chunk_num in enumerate(chunks_iter):

                os.chdir(self._this_imaging_dir)

                # Make the chunk clean call:
                this_clean_call = self.task_initialize_clean_call(this_chunk_num, stage='singlescale')
                overwrite = False
                if self.imaging_method == 'sdintimaging':
                    sd_image_file = self.task_setup_sdintimaging(this_clean_call, target=self.target, product=self.product,
                                                                 overwrite=overwrite,
                                                                 chunk_num=this_chunk_num)
                    if not sd_image_file:
                        logger.error('Error in setting up singledish for sdintimaging')

                    # Set the clean call parameters as necessary.
                    this_clean_call.set_param('usedata', 'sdint')
                    this_clean_call.set_param('sdimage', sd_image_file)

                if this_clean_call.get_param('deconvolver') not in ['hogbom','mtmfs']:
                    logger.warning("I expected a singlescale or mtmfs deconvolver but got: " + this_clean_call.get_param('deconvolver'))
                    raise Exception("Incorrect clean call! Should have a hogbom or mtmfs deconvolver.")

                logger.info("")
                logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
                logger.info("Running clean call to convergence for:")
                logger.info(this_clean_call.get_param('imagename'))
                logger.info("This is {0} out of {1} to be imaged".format(ii+1, len(chunks_iter)))
                logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
                logger.info("")

                # Moved this from single scale masking.
                # Add in before the single scale clean loop:
                this_clean_call.set_param('usemask', 'user')

                skip_this_step = False
                if skip_singlescale_if_mask_empty:
                    image_root = self.chunk_params[this_chunk_num]['full_imagename']
                    if self.imaging_method == 'sdintimaging':
                        mask_name = "{}.joint.cube.mask".format(self.chunk_params[this_chunk_num]['full_imagename'])
                    else:
                        mask_name = "{}.mask".format(self.chunk_params[this_chunk_num]['full_imagename'])

                    mask_stats = msr.stat_cube(cube_file=mask_name)
                    if mask_stats['sum'] == 0:
                        skip_this_step = True
                        logger.info("")
                        logger.info("The clean mask is empty and SKIP_SINGLESCALE_IF_MASK_EMPTY is True. Skipping the singlescale clean step.")
                        logger.info("")

                if not skip_this_step:
                    imr.clean_loop(
                        clean_call=this_clean_call,
                        imaging_method=self.imaging_method,
                        record_file=this_clean_call.get_param('imagename') + '_singlescale_record.txt',
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

                if backup:
                    imr.copy_imaging(
                        input_root=this_clean_call.get_param('imagename'),
                        output_root=this_clean_call.get_param('imagename') + '_singlescale',
                        imaging_method=self.imaging_method,
                        wipe_first=True)

            if gather_chunks_into_cube:
                self.task_gather_into_cube(root_name='singlescale',
                                           remove_chunks=remove_chunks)

            os.chdir(cwd)

            return ()

        @CleanCallFunctionDecorator
        def task_complete_gather_into_cubes(self, root_name='all',
                                            remove_chunks=False, overwrite=False):
            '''
            Intended to create a final set of cubes.
            '''

            accepted_root_names = ['dirty', 'singlescale', 'multiscale', None]

            if root_name == 'all':
                root_names = accepted_root_names
            else:
                if root_name not in accepted_root_names:
                    raise Exception("Cannot find {0} in accepted types: {1}".format(root_name, accepted_root_names))
                root_names = [root_name]

            for root in root_names:

                self.task_gather_into_cube(root_name=root,
                                           remove_chunks=remove_chunks,
                                           overwrite=overwrite)

        @CleanCallFunctionDecorator
        def task_export_to_fits(
                self,
                tag=None,
        ):
            """
            Export the results of a clean_call to FITS files. Optionally
            append _tag to the file.
            """

            if tag is not None:
                root_name_label = "_{}".format(tag)
            else:
                root_name_label = ""

            image_root = "{0}{1}".format(self.image_root, root_name_label)

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Exporting to FITS:" + image_root)
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            if not self._dry_run:
                imr.export_imaging_to_fits(image_root)

            return ()

        def task_cleanup(self, tag=None, chunk_num=None):
            '''
            Cleanup imaging products per chunk.

            Parameters
            ----------
            tag : str or None
                A tag to append to the image root. Defaults to None.
            chunk_num : int or None, optional
                The chunk number to clean up.
            '''

            if tag is not None:
                root_name_label = "_{}".format(tag)
            else:
                root_name_label = ""

            chunks_iter = self.return_valid_chunks(chunk_num=chunk_num)

            cwd = os.getcwd()
            # If a temp dir was used but has since been removed, search the original imaging dir
            if not self.tempdir_exists:
                effective_imaging_dir = self._orig_imaging_dir
            else:
                effective_imaging_dir = self._this_imaging_dir

            os.chdir(effective_imaging_dir)

            for ii, this_chunk_num in enumerate(chunks_iter):

                chan_start, chan_end = self.chunk_params[this_chunk_num]['channel_range']
                chan_label = "{0}_{1}".format(chan_start, chan_end)

                image_root = f"{self.image_root}_chan{chan_label}{root_name_label}"

                if not self._dry_run:

                    files_to_remove = glob.glob(f"{image_root}*")
                    for f in files_to_remove:
                        if os.path.exists(f):
                            cmd = f"rm -rf {f}"
                            logger.info(cmd)
                            os.system(cmd)

            os.chdir(cwd)

        #############################
        # recipe_imaging_one_target #
        #############################

        def recipe_phangsalma_imaging(
                self,
                chunk_num=None,
                extra_ext_in=None,
                suffix_in=None,
                extra_ext_out=None,
                do_dirty_image=True,
                do_revert_to_dirty=True,
                do_read_clean_mask=True,
                do_multiscale_clean=True,
                do_revert_to_multiscale=True,
                do_singlescale_mask=True,
                singlescale_mask_high_snr=None,
                singlescale_mask_low_snr=None,
                singlescale_mask_absolute=False,
                skip_singlescale_if_mask_empty=True,
                do_singlescale_clean=True,
                do_revert_to_singlescale=True,
                do_recombine_cubes=False,
                do_export_to_fits=True,
                do_cleanup=True,
                convergence_fracflux=0.01,
                convergence_noise_z_threshold=None,
                singlescale_threshold_value=1.0,
                dynamic_sizing=True,
                force_square=False,
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

            # Get which chunks we'll loop over. If chunk_num is None, it's all of them
            if chunk_num is None:
                chunks_to_iter = range(self.nchunks)
            else:
                chunks_to_iter = [chunk_num]

            for chunk_to_iter in chunks_to_iter:

                # Make a dirty image (niter=0)
                if do_dirty_image:
                    self.task_make_dirty_image(chunk_num=chunk_to_iter,
                                               gather_chunks_into_cube=False,
                                               )

                # Reset the current imaging to the dirty image.
                if do_revert_to_dirty:
                    self.task_revert_to_imaging(chunk_num=chunk_to_iter,
                                                tag='dirty')

                # Read and align the clean mask to the astrometry of the image.
                do_read_clean_mask = False
                if do_read_clean_mask:
                    raise NotImplementedError
                #     self.task_read_clean_mask(
                #         # AKL - propose to deprecate interaction with the clean_call here
                #         clean_call=clean_call,
                #         target=target, config=config, product=product,
                #         imaging_method=imaging_method)

                # Run a multiscale clean until it converges.
                if do_multiscale_clean:
                    self.task_multiscale_clean(chunk_num=chunk_to_iter,
                                               convergence_fracflux=convergence_fracflux,
                                               convergence_noise_z_threshold=convergence_noise_z_threshold,
                                               gather_chunks_into_cube=False,
                                               )

                # Reset the current imaging to the results of the multiscale clean.
                if do_revert_to_multiscale:
                    self.task_revert_to_imaging(chunk_num=chunk_to_iter,
                                                tag='multiscale')

                # Make a signal-to-noise based mask for use in singlescale clean.
                if do_singlescale_mask:
                    self.task_singlescale_mask(chunk_num=chunk_to_iter,
                                               high_snr=singlescale_mask_high_snr,
                                               low_snr=singlescale_mask_low_snr,
                                               absolute=singlescale_mask_absolute)

                # Run a singlescale clean until it converges.
                if do_singlescale_clean:
                    self.task_singlescale_clean(chunk_num=chunk_to_iter,
                                                convergence_fracflux=convergence_fracflux,
                                                convergence_noise_z_threshold=convergence_noise_z_threshold,
                                                threshold_value=singlescale_threshold_value,
                                                skip_singlescale_if_mask_empty=skip_singlescale_if_mask_empty,
                                                gather_chunks_into_cube=False)

                # Reset the current imaging to the results of the singlescale clean.
                if do_revert_to_singlescale:
                    self.task_revert_to_imaging(chunk_num=chunk_to_iter,
                                                tag='singlescale')

            # Ensure products are re-combined into cubes. Do this in the current imaging directory
            if do_recombine_cubes:
                if chunk_num is None:
                    self.task_complete_gather_into_cubes(root_name='all')

                    # Export the products of the current clean to FITS files.
                    if do_export_to_fits:
                        self.task_export_to_fits(tag=None)

                    # Do cleanup.
                    if do_cleanup:
                        self.task_cleanup(tag=None)

                else:
                    warnings.warn(f"Recombination of cubes requires all chunks to be run. Given only chunk {chunk_num}.")

            # If using a temp dir, remove after everything is complete:
            if self._uses_tempdir:

                # Find files to move, but don't move the MS if we've copied it over
                files_to_move = glob.glob(os.path.join(self._this_imaging_dir, "*"))
                if self.full_vis_file in files_to_move:
                    files_to_move.remove(self.full_vis_file)

                for f in files_to_move:

                    # If we're not overwriting and the file exists, then skip
                    out_name = os.path.join(self._orig_imaging_dir, os.path.basename(f))
                    if os.path.exists(out_name) and not overwrite:
                        logger.debug(f"{out_name} already exists and overwrite is False, skipping")
                        continue

                    os.system(f"rm -rf {out_name}")
                    cmd = f"mv -f {f} {self._orig_imaging_dir}"
                    logger.info(cmd)
                    os.system(cmd)

                # Forcibly remove the temp imaging directory
                os.system(f'rm -rf {self._this_imaging_dir}')

                # Move back to the original directory
                os.chdir(self._orig_imaging_dir)
