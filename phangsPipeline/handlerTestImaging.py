"""handlerTestImaging

This module provides a test imaging handler for exploring different imaging
parameters (weights, tapers) without running full deconvolution. It creates
dirty images and PSFs for a single channel and summarizes the resulting
beam properties and image statistics.

This code needs to be run inside CASA.

Example:
    $ casa
    from phangsPipeline import handlerKeys as kh
    from phangsPipeline import handlerTestImaging as tih
    this_kh = kh.KeyHandler(master_key = 'config_keys/master_key.txt')
    this_tih = tih.TestImagingHandler(key_handler = this_kh)
    this_tih.set_targets(only = ['ngc3627'])

    # Define test parameters
    this_tih.set_test_params(
        weightings=['briggs', 'briggs', 'natural'],
        robustnums=[0.5, 2.0, None],
        tapers_arcsec=[0.0, 1.0, 0.0]
    )

    # Run the test imaging loop
    results = this_tih.loop_test_imaging()

    # Print summary
    this_tih.print_summary()

    # Generate all diagnostic plots and export CSV in one call
    this_tih.make_report(output_prefix='ngc3627_test')

    # Or call individual plot methods:
    this_tih.plot_psf_radial_profiles(output_file='psf_profiles.png')
    this_tih.plot_psf_gallery(output_file='psf_gallery.png',
                              zoom_radius_arcsec=5.0)
    this_tih.plot_image_gallery(output_file='image_gallery.png')
"""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Check casa environment by importing CASA-only packages
from .check_imports import is_casa_installed
casa_enabled = is_casa_installed()

if casa_enabled:
    logger.debug('casa_enabled = True')
    from . import casaImagingRoutines as imr
else:
    logger.debug('casa_enabled = False')

if casa_enabled:

    # Analysis utilities
    import analysisUtils as au

    from .clean_call import CleanCall

    from . import utilsLines as lines
    from . import handlerTemplate
    from . import utilsFilenames
    from . import casaStuff
    from . import utilsTestImagingPlots as tip


    class TestImagingHandler(handlerTemplate.HandlerTemplate):
        """
        Class to test different imaging parameters by creating dirty images
        and PSFs for a single channel. Useful for exploring the effects of
        different weighting schemes and uv-tapers on resolution and sensitivity.
        """

        ############
        # __init__ #
        ############

        def __init__(
            self,
            key_handler=None,
            dry_run=False,
        ):
            # Inherit template class
            handlerTemplate.HandlerTemplate.__init__(self, key_handler=key_handler, dry_run=dry_run)

            # Initialize test parameter lists
            self._test_params = []

            # Initialize results storage
            self._test_results = []

        ###################
        # set_test_params #
        ###################

        def set_test_params(
            self,
            weightings=None,
            robustnums=None,
            tapers_arcsec=None,
            test_names=None,
        ):
            """
            Set the imaging parameter combinations to test.

            Parameters
            ----------
            weightings : list
                List of weighting schemes ('briggs', 'natural', 'uniform').
            robustnums : list
                List of robust parameters (only used for briggs weighting).
                Use None for non-briggs weightings.
            tapers_arcsec : list
                List of outer UV taper values in arcseconds.
                Use 0.0 for no taper.
            test_names : list, optional
                Custom names for each test combination. If not provided,
                names are auto-generated from parameters.

            All lists must have the same length.
            """
            if weightings is None:
                weightings = ['briggs']
            if robustnums is None:
                robustnums = [0.5]
            if tapers_arcsec is None:
                # Default when no taper is given is to not taper.
                tapers_arcsec = [0.0] * len(weightings)

            # Ensure all lists have the same length
            n_tests = len(weightings)
            if len(robustnums) != n_tests or len(tapers_arcsec) != n_tests:
                logger.error("weightings, robustnums, and tapers_arcsec must have same length")
                raise ValueError("Parameter lists must have same length")

            self._test_params = []
            for i in range(n_tests):
                if test_names is not None and len(test_names) > i:
                    name = test_names[i]
                else:
                    # Auto-generate name
                    name = weightings[i]
                    if weightings[i] == 'briggs' and robustnums[i] is not None:
                        name += '_r' + str(robustnums[i])
                    if tapers_arcsec[i] > 0:
                        name += '_t' + str(tapers_arcsec[i])

                self._test_params.append({
                    'name': name,
                    'weighting': weightings[i],
                    'robust': robustnums[i],
                    'taper_arcsec': tapers_arcsec[i],
                })

            logger.info(f"Set {len(self._test_params)} test parameter combinations")
            return self._test_params

        def add_test_param(
            self,
            weighting='briggs',
            robust=0.5,
            taper_arcsec=0.0,
            name=None,
        ):
            """
            Add a single test parameter combination.

            Parameters
            ----------
            weighting : str
                Weighting scheme ('briggs', 'natural', 'uniform').
            robust : float or None
                Robust parameter for briggs weighting.
            taper_arcsec : float
                Outer UV taper in arcseconds (0.0 for no taper).
            name : str, optional
                Custom name for this test.
            """
            if name is None:
                name = weighting
                if weighting == 'briggs' and robust is not None:
                    name += '_r' + str(robust)
                if taper_arcsec > 0:
                    name += '_t' + str(taper_arcsec)

            self._test_params.append({
                'name': name,
                'weighting': weighting,
                'robust': robust,
                'taper_arcsec': taper_arcsec,
            })

            logger.info(f"Added test param: {name}")
            return self._test_params

        def get_test_params(self):
            """Return the list of test parameter combinations."""
            return self._test_params

        def clear_test_params(self):
            """Clear all test parameter combinations."""
            self._test_params = []
            return self._test_params

        ###############
        # _fname_dict #
        ###############

        def _fname_dict(
            self,
            product,
            imagename,
        ):
            """
            Handles file names used in the test imaging processes.
            For test imaging we always use tclean with cube specmode
            for a single channel.
            """
            fname_dict = {}
            fname_dict['root'] = imagename
            fname_dict['suffix'] = ''
            fname_dict['image'] = imagename + '.image'
            fname_dict['model'] = imagename + '.model'
            fname_dict['residual'] = imagename + '.residual'
            fname_dict['mask'] = imagename + '.mask'
            fname_dict['pb'] = imagename + '.pb'
            fname_dict['psf'] = imagename + '.psf'
            fname_dict['weight'] = imagename + '.weight'
            fname_dict['sumwt'] = imagename + '.sumwt'
            return fname_dict

        ########################
        # _get_beam_properties #
        ########################

        def _get_beam_properties(
            self,
            psf_file,
        ):
            """
            Extract beam properties from a PSF image.

            Parameters
            ----------
            psf_file : str
                Path to the PSF image.

            Returns
            -------
            dict
                Dictionary with beam major/minor axes (arcsec) and position angle (deg).
            """
            if not os.path.isdir(psf_file):
                logger.warning(f"PSF file not found: {psf_file}")
                return None

            myia = au.createCasaTool(casaStuff.iatool)
            myia.open(psf_file)

            beam_info = {}
            try:
                restoring_beam = myia.restoringbeam()
                if 'major' in restoring_beam:
                    beam_info['bmaj_arcsec'] = restoring_beam['major']['value']
                    if restoring_beam['major']['unit'] == 'deg':
                        beam_info['bmaj_arcsec'] *= 3600.0
                    beam_info['bmin_arcsec'] = restoring_beam['minor']['value']
                    if restoring_beam['minor']['unit'] == 'deg':
                        beam_info['bmin_arcsec'] *= 3600.0
                    beam_info['bpa_deg'] = restoring_beam['positionangle']['value']
                else:
                    # No beam info in restoring beam
                    beam_info['bmaj_arcsec'] = None
                    beam_info['bmin_arcsec'] = None
                    beam_info['bpa_deg'] = None
            except Exception as e:
                logger.warning(f"Could not extract beam from {psf_file}: {e}")
                beam_info['bmaj_arcsec'] = None
                beam_info['bmin_arcsec'] = None
                beam_info['bpa_deg'] = None

            myia.close()
            return beam_info

        ########################
        # _get_image_statistics #
        ########################

        def _get_image_statistics(
            self,
            image_file,
            pb_file=None,
            pb_limit=0.7,
        ):
            """
            Extract image statistics from a dirty image.

            Parameters
            ----------
            image_file : str
                Path to the dirty image.
            pb_file : str, optional
                Path to primary beam image for masking.
            pb_limit : float
                Primary beam level below which to exclude pixels.

            Returns
            -------
            dict
                Dictionary with image statistics (rms, max, min, etc.).
            """
            if not os.path.isdir(image_file):
                logger.warning(f"Image file not found: {image_file}")
                return None

            stats = {}

            # Get image statistics using imstat
            if pb_file is not None and os.path.isdir(pb_file):
                # Create a mask based on primary beam
                stat_result = casaStuff.imstat(imagename=image_file,
                                                mask=f'"{pb_file}" > {pb_limit}')
            else:
                stat_result = casaStuff.imstat(imagename=image_file)

            if stat_result is not None:
                stats['max'] = stat_result['max'][0] if 'max' in stat_result else None
                stats['min'] = stat_result['min'][0] if 'min' in stat_result else None
                stats['rms'] = stat_result['rms'][0] if 'rms' in stat_result else None
                stats['mean'] = stat_result['mean'][0] if 'mean' in stat_result else None
                stats['median'] = stat_result['median'][0] if 'median' in stat_result else None
                stats['npts'] = stat_result['npts'][0] if 'npts' in stat_result else None
            else:
                stats['max'] = None
                stats['min'] = None
                stats['rms'] = None
                stats['mean'] = None
                stats['median'] = None
                stats['npts'] = None

            return stats

        ###########################
        # task_initialize_test_call #
        ###########################

        def task_initialize_test_call(
            self,
            target=None,
            config=None,
            product=None,
            test_param=None,
            extra_ext_in=None,
            suffix_in=None,
            channel=None,
        ):
            """
            Initialize a clean call object for test imaging.

            Parameters
            ----------
            target : str
                Target name.
            config : str
                Configuration name.
            product : str
                Product name.
            test_param : dict
                Test parameter dictionary with weighting, robust, taper_arcsec.
            extra_ext_in : str, optional
                Extra extension for input files.
            suffix_in : str, optional
                Suffix for input files.
            channel : int, optional
                Channel to image. Defaults to 0 (first channel).
            """
            if target is None or config is None or product is None:
                logger.error('Please input target, config and product!')
                raise Exception('Please input target, config and product!')

            if test_param is None:
                logger.error('Please input test_param!')
                raise Exception('Please input test_param!')

            # Look up the recipes for this case
            recipe_list = self._kh.get_imaging_recipes(config=config, product=product)
            if recipe_list is None:
                logger.error(
                    f'Could not get imaging recipe for config {config} product {product}')
                raise Exception(
                    f'Could not get imaging recipe for config {config} product {product}')

            # Initialize the clean call
            clean_call = CleanCall(recipe_list, use_chunks=False)

            # Get the visibility name
            vis_file = utilsFilenames.get_vis_filename(
                target=target, product=product, config=config,
                ext=extra_ext_in, suffix=suffix_in)

            # Test existence
            full_vis_file = self._kh.get_imaging_dir_for_target(target=target) + vis_file
            if not os.path.isdir(full_vis_file):
                logger.error(f'Visibility file not found: {full_vis_file}')
                return None

            clean_call.set_param('vis', vis_file, nowarning=True)

            # Set the output image file name with test param name
            image_root = utilsFilenames.get_cube_filename(
                target=target, product=product, config=config,
                ext='test_' + test_param['name'], casa=True, casaext='')

            clean_call.set_param('imagename', image_root, nowarning=True)

            # Get the phase center
            rastring, decstring = self._kh.get_phasecenter_for_target(target=target)
            phasecenter = 'J2000 ' + rastring + ' ' + decstring
            clean_call.set_param('phasecenter', phasecenter)

            # Force cube mode for single channel
            clean_call.set_param('specmode', 'cube')
            clean_call.set_param('deconvolver', 'hogbom')

            # Set single channel
            if channel is None:
                channel = 0

            clean_call.set_param('nchan', 1)
            clean_call.set_param('start', channel)
            clean_call.set_param('width', 1)

            # Set rest frequency for line products
            is_line_product = product in self._kh.get_line_products()
            if is_line_product:
                this_line_tag = self._kh.get_line_tag_for_line_product(product=product)
                if this_line_tag in lines.line_list.keys():
                    rest_freq_ghz = lines.line_list[this_line_tag]
                    clean_call.set_restfreq_ghz(rest_freq_ghz)

            # Apply test parameters: weighting
            clean_call.set_param('weighting', test_param['weighting'], nowarning=True)

            if test_param['weighting'] == 'briggs' and test_param['robust'] is not None:
                clean_call.set_param('robust', test_param['robust'], nowarning=True)

            # Apply UV taper
            if test_param['taper_arcsec'] > 0:
                clean_call.set_round_uvtaper_arcsec(test_param['taper_arcsec'])
            else:
                clean_call.set_param('uvtaper', [], nowarning=True)

            return clean_call

        ############################
        # task_make_test_dirty_image #
        ############################

        def task_make_test_dirty_image(
            self,
            clean_call=None,
        ):
            """
            Execute the clean call with zero iterations to create a dirty
            image and PSF for testing purposes.
            """
            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Making test dirty image:")
            logger.info(str(clean_call.get_param('imagename')))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            if not self._dry_run and casa_enabled:
                imr.make_dirty_image(clean_call, imaging_method='tclean')

            return ()

        ######################
        # loop_test_imaging #
        ######################

        def loop_test_imaging(
            self,
            extra_ext_in=None,
            suffix_in=None,
            channel=None,
            make_directories=True,
            overwrite=False,
        ):
            """
            Loop over targets, products, configs, and test parameters to
            create test dirty images and PSFs, then extract beam properties
            and image statistics.

            Parameters
            ----------
            extra_ext_in : str, optional
                Extra extension for input files.
            suffix_in : str, optional
                Suffix for input files.
            channel : int, optional
                Channel to image. Defaults to 0 (first channel).
            make_directories : bool
                Create missing directories.
            overwrite : bool
                Overwrite existing test images.

            Returns
            -------
            list
                List of result dictionaries for each test.
            """
            if len(self._test_params) == 0:
                logger.error("No test parameters set. Use set_test_params() first.")
                return []

            if len(self.get_targets()) == 0:
                logger.error("Need a target list.")
                return []

            if len(self.get_all_products()) == 0:
                logger.error("Need a products list.")
                return []

            if make_directories:
                self._kh.make_missing_directories(imaging=True)

            # Clear previous results
            self._test_results = []

            # Save initial directory to restore when done
            initial_dir = os.getcwd()

            try:
                # Loop over targets, products, and configs
                for this_target, this_product, this_config in \
                        self.looper(do_targets=True, do_products=True, do_configs=True, just_interf=True):

                    # Change to the relevant directory
                    this_imaging_dir = self._kh.get_imaging_dir_for_target(this_target, changeto=True)

                    # Loop over test parameters
                    for test_param in self._test_params:
                        logger.info("")
                        logger.info("=" * 60)
                        logger.info(f"TEST IMAGING: {this_target} {this_config} {this_product}")
                        logger.info(f"Test params: {test_param['name']}")
                        logger.info("=" * 60)

                        # Initialize result dictionary
                        result = {
                            'target': this_target,
                            'config': this_config,
                            'product': this_product,
                            'test_name': test_param['name'],
                            'weighting': test_param['weighting'],
                            'robust': test_param['robust'],
                            'taper_arcsec': test_param['taper_arcsec'],
                        }

                        # Initialize clean call
                        clean_call = self.task_initialize_test_call(
                            target=this_target,
                            config=this_config,
                            product=this_product,
                            test_param=test_param,
                            extra_ext_in=extra_ext_in,
                            suffix_in=suffix_in,
                            channel=channel,
                        )

                        if clean_call is None:
                            logger.warning("Could not create clean call, skipping")
                            result['status'] = 'failed'
                            result['error'] = 'Could not create clean call'
                            self._test_results.append(result)
                            continue

                        # Get file names
                        fname_dict = self._fname_dict(
                            product=this_product,
                            imagename=clean_call.get_param('imagename')
                        )

                        # Check if we should skip
                        if not overwrite and os.path.isdir(fname_dict['psf']):
                            logger.info(f"Test image exists, skipping: {fname_dict['psf']}")
                            result['status'] = 'skipped'
                        else:
                            # Pick cell and imsize
                            if not self._dry_run:
                                cell, imsize = imr.estimate_cell_and_imsize(
                                    clean_call.get_param('vis'), oversamp=5, force_square=False)
                                clean_call.set_param('cell', cell, nowarning=True)
                                clean_call.set_param('imsize', imsize, nowarning=True)
                            else:
                                clean_call.set_param('cell', '0.1arcsec', nowarning=True)
                                clean_call.set_param('imsize', [1000, 1000], nowarning=True)

                            # Record pixel scale
                            cell_str = clean_call.get_param('cell')
                            if isinstance(cell_str, str):
                                result['cell_arcsec'] = float(
                                    cell_str.replace('arcsec', ''))
                            else:
                                result['cell_arcsec'] = float(cell_str)

                            # Make dirty image
                            self.task_make_test_dirty_image(clean_call=clean_call)
                            result['status'] = 'completed'

                        # Extract beam properties from PSF
                        if not self._dry_run:
                            beam_props = self._get_beam_properties(fname_dict['psf'])
                            if beam_props:
                                result.update(beam_props)

                            # Extract image statistics
                            img_stats = self._get_image_statistics(
                                fname_dict['image'],
                                pb_file=fname_dict['pb']
                            )
                            if img_stats:
                                result.update(img_stats)

                            # Compute PSF quality metrics via radial profile
                            if os.path.isdir(fname_dict['psf']):
                                bmaj = result.get('bmaj_arcsec')
                                bmin = result.get('bmin_arcsec')
                                if bmaj is not None and bmin is not None:
                                    data_2d, pix_scale = tip._read_image_data(
                                        fname_dict['psf'])
                                    if data_2d is not None:
                                        radii, psf_radial = tip._compute_radial_profile(
                                            data_2d, pix_scale,
                                            max_radius_arcsec=10.0 * bmaj)
                                        del data_2d
                                        result['kappa'] = tip.measure_kappa(
                                            radii, psf_radial, bmaj, bmin)
                                        result['skirt_level'] = tip.measure_skirt_level(
                                            radii, psf_radial, bmaj, bmin)
                                        result['epsilon'] = tip.measure_epsilon(
                                            radii, psf_radial, bmaj, bmin)
                                        # Cache radial profile for reuse in plotting
                                        result['_psf_radii'] = radii
                                        result['_psf_profile'] = psf_radial

                        self._test_results.append(result)

                        logger.info(
                            f"Result: cell={result.get('cell_arcsec', 'N/A')} arcsec, "
                            f"bmaj={result.get('bmaj_arcsec', 'N/A')} arcsec, "
                            f"bmin={result.get('bmin_arcsec', 'N/A')} arcsec, "
                            f"rms={result.get('rms', 'N/A')}, "
                            f"kappa={result.get('kappa', 'N/A')}, "
                            f"skirt={result.get('skirt_level', 'N/A')}, "
                            f"epsilon={result.get('epsilon', 'N/A')}")

            finally:
                os.chdir(initial_dir)
                logger.info(f"Restored working directory to {initial_dir}")

            return self._test_results

        #################
        # get_results #
        #################

        def get_results(self):
            """Return the list of test results."""
            return self._test_results

        ########################
        # _get_unique_targets #
        ########################

        def _get_unique_targets(self, results):
            """Return unique target names from results in order of first appearance."""
            seen = set()
            targets = []
            for r in results:
                t = r.get('target')
                if t is not None and t not in seen:
                    seen.add(t)
                    targets.append(t)
            return targets

        ##################
        # print_summary #
        ##################

        def print_summary(
            self,
            results=None,
        ):
            """
            Print a summary table of test imaging results, grouped
            by target.

            Parameters
            ----------
            results : list, optional
                List of result dictionaries. If None, uses stored results.
            """
            if results is None:
                results = self._test_results

            if len(results) == 0:
                logger.warning("No results to summarize")
                return

            targets = self._get_unique_targets(results)

            for this_target in targets:
                target_results = [r for r in results
                                  if r.get('target') == this_target]

                print("")
                print("=" * 160)
                print(f"TEST IMAGING SUMMARY: {this_target}")
                print("=" * 160)
                print(f"{'Config':<12} {'Product':<10} {'Test':<20} "
                      f"{'Cell':<8} {'Bmaj':<10} {'Bmin':<10} {'BPA':<8} {'RMS':<12} "
                      f"{'Kappa':<10} {'Skirt':<10} {'epsilon':<10} {'Status':<10}")
                print("-" * 160)

                for r in target_results:
                    cell = f"{r['cell_arcsec']:.4f}" if r.get('cell_arcsec') is not None else 'N/A'
                    bmaj = f"{r.get('bmaj_arcsec', 0):.3f}" if r.get('bmaj_arcsec') else 'N/A'
                    bmin = f"{r.get('bmin_arcsec', 0):.3f}" if r.get('bmin_arcsec') else 'N/A'
                    bpa = f"{r.get('bpa_deg', 0):.1f}" if r.get('bpa_deg') else 'N/A'
                    rms = f"{r.get('rms', 0):.2e}" if r.get('rms') else 'N/A'
                    kappa = f"{r['kappa']:.4f}" if r.get('kappa') is not None else 'N/A'
                    skirt = f"{r['skirt_level']:.4f}" if r.get('skirt_level') is not None else 'N/A'
                    epsilon = f"{r['epsilon']:.4f}" if r.get('epsilon') is not None else 'N/A'

                    print(f"{r.get('config', ''):<12} "
                          f"{r.get('product', ''):<10} {r.get('test_name', ''):<20} "
                          f"{cell:<8} {bmaj:<10} {bmin:<10} {bpa:<8} {rms:<12} "
                          f"{kappa:<10} {skirt:<10} {epsilon:<10} "
                          f"{r.get('status', ''):<10}")

                print("=" * 160)
                print("")

        ######################
        # export_summary_csv #
        ######################

        def export_summary_csv(
            self,
            filename='test_imaging_summary.csv',
            results=None,
        ):
            """
            Export test imaging results to per-target CSV files.

            When results contain multiple targets, a separate CSV is
            written for each target with the target name inserted into
            the filename (e.g. ``test_imaging_summary_ngc3627.csv``).

            Parameters
            ----------
            filename : str
                Output CSV filename (or template for multi-target).
            results : list, optional
                List of result dictionaries. If None, uses stored results.
            """
            if results is None:
                results = self._test_results

            if len(results) == 0:
                logger.warning("No results to export")
                return

            import csv

            columns = ['target', 'config', 'product', 'test_name', 'weighting',
                      'robust', 'taper_arcsec', 'cell_arcsec',
                      'bmaj_arcsec', 'bmin_arcsec',
                      'bpa_deg', 'max', 'min', 'rms', 'mean', 'median', 'npts',
                      'kappa', 'skirt_level', 'epsilon', 'status']

            targets = self._get_unique_targets(results)

            for this_target in targets:
                target_results = [r for r in results
                                  if r.get('target') == this_target]

                if len(targets) == 1:
                    out_file = filename
                else:
                    base, ext = os.path.splitext(filename)
                    out_file = f'{base}_{this_target}{ext}'

                out_file = self._ensure_output_in_dir(
                    out_file,
                    self._resolve_imaging_dir(results, this_target))

                with open(out_file, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=columns,
                                            extrasaction='ignore')
                    writer.writeheader()
                    for r in target_results:
                        writer.writerow(r)

                logger.info(f"Exported {len(target_results)} results for "
                            f"{this_target} to {out_file}")

        ####################
        # suggest_optimal #
        ####################

        def suggest_optimal(
            self,
            target_resolution_arcsec=None,
            results=None,
        ):
            """
            Suggest the optimal imaging parameters for each target.

            Parameters
            ----------
            target_resolution_arcsec : float, optional
                Target resolution in arcseconds. If provided, suggests
                parameters closest to this resolution.
            results : list, optional
                List of result dictionaries. If None, uses stored results.

            Returns
            -------
            dict
                Dictionary keyed by target name. Each value is the best
                matching result dictionary for that target, or None if
                no valid results exist.
            """
            if results is None:
                results = self._test_results

            if len(results) == 0:
                logger.warning("No results to analyze")
                return {}

            targets = self._get_unique_targets(results)
            recommendations = {}

            for this_target in targets:
                logger.info("")
                logger.info(f"--- Recommendation for {this_target} ---")

                valid_results = [r for r in results
                                 if r.get('target') == this_target
                                 and r.get('bmaj_arcsec') is not None]

                if len(valid_results) == 0:
                    logger.warning(f"  No valid results with beam measurements "
                                   f"for {this_target}")
                    recommendations[this_target] = None
                    continue

                if target_resolution_arcsec is not None:
                    best = min(valid_results,
                              key=lambda r: abs(r['bmaj_arcsec'] - target_resolution_arcsec))
                    logger.info(f"  Closest to target resolution "
                                f"{target_resolution_arcsec} arcsec:")
                else:
                    valid_with_rms = [r for r in valid_results
                                      if r.get('rms') is not None]
                    if len(valid_with_rms) > 0:
                        best = min(valid_with_rms, key=lambda r: r['rms'])
                        logger.info("  Best sensitivity (lowest RMS):")
                    else:
                        best = min(valid_results, key=lambda r: r['bmaj_arcsec'])
                        logger.info("  Highest resolution:")

                logger.info(f"  Test: {best['test_name']}")
                logger.info(f"  Weighting: {best['weighting']}, "
                            f"Robust: {best.get('robust')}")
                logger.info(f"  Taper: {best['taper_arcsec']} arcsec")
                logger.info(f"  Beam: {best['bmaj_arcsec']:.3f} x "
                            f"{best['bmin_arcsec']:.3f} arcsec")
                if best.get('rms'):
                    logger.info(f"  RMS: {best['rms']:.2e}")

                recommendations[this_target] = best

            return recommendations

        ##############################
        # plot_psf_radial_profiles #
        ##############################

        def plot_psf_radial_profiles(
            self,
            target=None,
            config=None,
            product=None,
            output_file='psf_radial_profiles.png',
            results=None,
            imaging_dir=None,
            max_radius_arcsec=None,
            bin_width_arcsec=None,
            log_scale=False,
            figsize=None,
            dpi=150,
            title=None,
        ):
            """
            Plot 1D radial PSF profiles, produced separately per target.

            When ``target`` is None, a separate figure is created for
            each target in the results, with the target name inserted
            into the output filename.

            Parameters
            ----------
            target : str, optional
                Restrict to this target. If None, loops over all targets.
            config : str, optional
                Filter results to this config.
            product : str, optional
                Filter results to this product.
            output_file : str, optional
                Output filename (template for multi-target).
            results : list, optional
                List of result dictionaries. If None, uses stored results.
            imaging_dir : str, optional
                Directory containing test images. If None, resolved from
                the key handler per target.
            max_radius_arcsec : float, optional
                Maximum radius for the profile.
            bin_width_arcsec : float, optional
                Radial bin width in arcseconds. Enforced to be at least
                one pixel wide. If None, defaults to one pixel.
            log_scale : bool
                Use symlog y-axis scale.
            figsize : tuple, optional
                Figure size in inches.
            dpi : int
                Figure resolution.
            title : str, optional
                Figure title.

            Returns
            -------
            dict
                Dictionary mapping target name to matplotlib Figure.
            """
            if results is None:
                results = self._test_results

            if len(results) == 0:
                logger.warning("No results to plot")
                return {}

            targets = [target] if target else self._get_unique_targets(results)
            all_figs = {}

            for this_target in targets:
                this_imaging_dir = imaging_dir
                if this_imaging_dir is None:
                    this_imaging_dir = self._resolve_imaging_dir(
                        results, this_target)

                this_output = self._per_target_filename(
                    output_file, this_target, len(targets))
                this_output = self._ensure_output_in_dir(
                    this_output, this_imaging_dir)

                fig = tip.make_psf_radial_profile_gallery(
                    results=results,
                    imaging_dir=this_imaging_dir,
                    output_file=this_output,
                    target=this_target,
                    config=config,
                    product=product,
                    max_radius_arcsec=max_radius_arcsec,
                    bin_width_arcsec=bin_width_arcsec,
                    log_scale=log_scale,
                    figsize=figsize,
                    dpi=dpi,
                    title=title,
                )
                all_figs[this_target] = fig

            return all_figs

        #####################
        # plot_psf_gallery #
        #####################

        def plot_psf_gallery(
            self,
            target=None,
            config=None,
            product=None,
            output_file='psf_image_gallery.png',
            results=None,
            imaging_dir=None,
            zoom_radius_arcsec=None,
            vmin=-0.1,
            vmax=1.0,
            cmap='RdBu_r',
            show_beam_ellipse=True,
            max_panels_per_page=9,
            figsize=None,
            dpi=150,
            title=None,
        ):
            """
            Plot a gallery of 2D PSF images, produced separately per
            target. Automatically splits into multiple pages per target
            if there are more results than max_panels_per_page.

            Parameters
            ----------
            target : str, optional
                Restrict to this target. If None, loops over all targets.
            config : str, optional
                Filter results to this config.
            product : str, optional
                Filter results to this product.
            output_file : str, optional
                Output filename (template for multi-target/multi-page).
            results : list, optional
                List of result dictionaries. If None, uses stored results.
            imaging_dir : str, optional
                Directory containing test images. If None, resolved from
                the key handler per target.
            zoom_radius_arcsec : float, optional
                Half-width of the zoomed region in arcseconds.
            vmin : float
                Minimum color scale value.
            vmax : float
                Maximum color scale value.
            cmap : str
                Matplotlib colormap name.
            show_beam_ellipse : bool
                Draw beam ellipses on each panel.
            max_panels_per_page : int
                Maximum panels per figure page (default 9).
            figsize : tuple, optional
                Figure size in inches.
            dpi : int
                Figure resolution.
            title : str, optional
                Figure supertitle.

            Returns
            -------
            dict
                Dictionary mapping target name to list of Figures.
            """
            if results is None:
                results = self._test_results

            if len(results) == 0:
                logger.warning("No results to plot")
                return {}

            targets = [target] if target else self._get_unique_targets(results)
            all_figs = {}

            for this_target in targets:
                this_imaging_dir = imaging_dir
                if this_imaging_dir is None:
                    this_imaging_dir = self._resolve_imaging_dir(
                        results, this_target)

                this_output = self._per_target_filename(
                    output_file, this_target, len(targets))
                this_output = self._ensure_output_in_dir(
                    this_output, this_imaging_dir)

                figs = tip.make_psf_image_gallery(
                    results=results,
                    imaging_dir=this_imaging_dir,
                    output_file=this_output,
                    target=this_target,
                    config=config,
                    product=product,
                    zoom_radius_arcsec=zoom_radius_arcsec,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    show_beam_ellipse=show_beam_ellipse,
                    max_panels_per_page=max_panels_per_page,
                    figsize=figsize,
                    dpi=dpi,
                    title=title,
                )
                all_figs[this_target] = figs

            return all_figs

        #######################
        # plot_image_gallery #
        #######################

        def plot_image_gallery(
            self,
            target=None,
            config=None,
            product=None,
            output_file='image_gallery.png',
            results=None,
            imaging_dir=None,
            zoom_radius_arcsec=None,
            vmin=None,
            vmax=None,
            cmap='inferno',
            show_beam_ellipse=True,
            symmetric_color=False,
            sigma_clip=3.0,
            max_panels_per_page=9,
            figsize=None,
            dpi=150,
            title=None,
        ):
            """
            Plot a gallery of 2D dirty images, produced separately per
            target. Automatically splits into multiple pages per target
            if there are more results than max_panels_per_page.

            Parameters
            ----------
            target : str, optional
                Restrict to this target. If None, loops over all targets.
            config : str, optional
                Filter results to this config.
            product : str, optional
                Filter results to this product.
            output_file : str, optional
                Output filename (template for multi-target/multi-page).
            results : list, optional
                List of result dictionaries. If None, uses stored results.
            imaging_dir : str, optional
                Directory containing test images. If None, resolved from
                the key handler per target.
            zoom_radius_arcsec : float, optional
                Half-width of the displayed region in arcseconds.
            vmin : float, optional
                Minimum color scale value.
            vmax : float, optional
                Maximum color scale value.
            cmap : str
                Matplotlib colormap name.
            show_beam_ellipse : bool
                Draw beam ellipses on each panel.
            symmetric_color : bool
                Use symmetric color range centered on zero.
            sigma_clip : float
                Sigma level for auto color range.
            max_panels_per_page : int
                Maximum panels per figure page (default 9).
            figsize : tuple, optional
                Figure size in inches.
            dpi : int
                Figure resolution.
            title : str, optional
                Figure supertitle.

            Returns
            -------
            dict
                Dictionary mapping target name to list of Figures.
            """
            if results is None:
                results = self._test_results

            if len(results) == 0:
                logger.warning("No results to plot")
                return {}

            targets = [target] if target else self._get_unique_targets(results)
            all_figs = {}

            for this_target in targets:
                this_imaging_dir = imaging_dir
                if this_imaging_dir is None:
                    this_imaging_dir = self._resolve_imaging_dir(
                        results, this_target)

                this_output = self._per_target_filename(
                    output_file, this_target, len(targets))
                this_output = self._ensure_output_in_dir(
                    this_output, this_imaging_dir)

                figs = tip.make_image_gallery(
                    results=results,
                    imaging_dir=this_imaging_dir,
                    output_file=this_output,
                    target=this_target,
                    config=config,
                    product=product,
                    zoom_radius_arcsec=zoom_radius_arcsec,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    show_beam_ellipse=show_beam_ellipse,
                    symmetric_color=symmetric_color,
                    sigma_clip=sigma_clip,
                    max_panels_per_page=max_panels_per_page,
                    figsize=figsize,
                    dpi=dpi,
                    title=title,
                )
                all_figs[this_target] = figs

            return all_figs

        ##############################
        # plot_metrics_vs_robust    #
        ##############################

        def plot_metrics_vs_robust(
            self,
            target=None,
            config=None,
            product=None,
            output_file='metrics_vs_robust.png',
            results=None,
            imaging_dir=None,
            metrics=None,
            figsize=None,
            dpi=150,
            title=None,
        ):
            """
            Plot imaging metrics as a function of the robust parameter,
            produced separately per target.

            Default panels: beam major axis, RMS, kappa, epsilon, and skirt
            level.

            Parameters
            ----------
            target : str, optional
                Restrict to this target. If None, loops over all targets.
            config : str, optional
                Filter results to this config.
            product : str, optional
                Filter results to this product.
            output_file : str, optional
                Output filename (template for multi-target).
            results : list, optional
                List of result dictionaries. If None, uses stored results.
            imaging_dir : str, optional
                Directory for output. If None, resolved from the key
                handler per target.
            metrics : list of tuple, optional
                List of ``(result_key, y_label)`` pairs. Defaults to
                bmaj, rms, kappa, epsilon, skirt_level.
            figsize : tuple, optional
                Figure size in inches.
            dpi : int
                Figure resolution.
            title : str, optional
                Figure super-title.

            Returns
            -------
            dict
                Dictionary mapping target name to matplotlib Figure.
            """
            if results is None:
                results = self._test_results

            if len(results) == 0:
                logger.warning("No results to plot")
                return {}

            targets = [target] if target else self._get_unique_targets(results)
            all_figs = {}

            for this_target in targets:
                this_imaging_dir = imaging_dir
                if this_imaging_dir is None:
                    this_imaging_dir = self._resolve_imaging_dir(
                        results, this_target)

                this_output = self._per_target_filename(
                    output_file, this_target, len(targets))
                this_output = self._ensure_output_in_dir(
                    this_output, this_imaging_dir)

                fig = tip.make_metrics_vs_robust_gallery(
                    results=results,
                    output_file=this_output,
                    target=this_target,
                    config=config,
                    product=product,
                    metrics=metrics,
                    figsize=figsize,
                    dpi=dpi,
                    title=title,
                )
                all_figs[this_target] = fig

            return all_figs

        ############################
        # plot_metrics_vs_beam    #
        ############################

        def plot_metrics_vs_beam(
            self,
            target=None,
            config=None,
            product=None,
            output_file='metrics_vs_beam.png',
            results=None,
            imaging_dir=None,
            metrics=None,
            figsize=None,
            dpi=150,
            title=None,
        ):
            """
            Plot imaging metrics as a function of beam size, produced
            separately per target.

            All weighting schemes and taper values are shown together
            on each panel, distinguished by colour and marker.
            Default panels: RMS, kappa, epsilon, and skirt level.

            Parameters
            ----------
            target : str, optional
                Restrict to this target. If None, loops over all targets.
            config : str, optional
                Filter results to this config.
            product : str, optional
                Filter results to this product.
            output_file : str, optional
                Output filename (template for multi-target).
            results : list, optional
                List of result dictionaries. If None, uses stored results.
            imaging_dir : str, optional
                Directory for output. If None, resolved from the key
                handler per target.
            metrics : list of tuple, optional
                List of ``(result_key, y_label)`` pairs.
            figsize : tuple, optional
                Figure size in inches.
            dpi : int
                Figure resolution.
            title : str, optional
                Figure super-title.

            Returns
            -------
            dict
                Dictionary mapping target name to matplotlib Figure.
            """
            if results is None:
                results = self._test_results

            if len(results) == 0:
                logger.warning("No results to plot")
                return {}

            targets = [target] if target else self._get_unique_targets(results)
            all_figs = {}

            for this_target in targets:
                this_imaging_dir = imaging_dir
                if this_imaging_dir is None:
                    this_imaging_dir = self._resolve_imaging_dir(
                        results, this_target)

                this_output = self._per_target_filename(
                    output_file, this_target, len(targets))
                this_output = self._ensure_output_in_dir(
                    this_output, this_imaging_dir)

                fig = tip.make_metrics_vs_beam_gallery(
                    results=results,
                    output_file=this_output,
                    target=this_target,
                    config=config,
                    product=product,
                    metrics=metrics,
                    figsize=figsize,
                    dpi=dpi,
                    title=title,
                )
                all_figs[this_target] = fig

            return all_figs

        #################
        # make_report #
        #################

        def make_report(
            self,
            output_prefix='test_imaging',
            output_dir=None,
            config=None,
            product=None,
            results=None,
            plots=None,
            plot_kwargs=None,
            dpi=150,
        ):
            """
            Generate a full diagnostic report per target.

            Loops over each target in the results and produces a
            separate set of outputs (CSV, plots, recommendation) for
            each one. The set of outputs is controlled by the ``plots``
            parameter and can be extended by adding new entries to the
            internal registry.

            Parameters
            ----------
            output_prefix : str
                Prefix for all output filenames. When there are multiple
                targets the target name is appended automatically
                (e.g. ``'test_imaging'`` -> ``'test_imaging_ngc3627_psf_profiles.png'``).
                With a single target, the prefix is used as-is
                (e.g. ``'test_imaging'`` -> ``'test_imaging_psf_profiles.png'``).
            output_dir : str, optional
                Directory for output files. If None, uses the imaging
                directory for each target.
            config : str, optional
                Filter results to this config.
            product : str, optional
                Filter results to this product.
            results : list, optional
                List of result dictionaries. If None, uses stored results.
            plots : list of str, optional
                Which outputs to produce. Supported keys:

                - ``'csv'`` : summary CSV table
                - ``'psf_profiles'`` : 1D radial PSF profiles
                - ``'psf_gallery'`` : 2D PSF image grid
                - ``'image_gallery'`` : 2D dirty image grid
                - ``'metrics_vs_robust'`` : metrics vs. robust parameter
                - ``'metrics_vs_beam'`` : metrics vs. beam size
                - ``'html'`` : self-contained HTML summary report

                If None, all of the above are produced.
            plot_kwargs : dict, optional
                Per-plot keyword overrides. Keys are the plot names from
                the ``plots`` list; values are dicts of keyword arguments
                passed to the corresponding method. For example::

                    plot_kwargs={
                        'psf_gallery': {'zoom_radius_arcsec': 5.0},
                        'psf_profiles': {'log_scale': True},
                    }
            dpi : int
                Default figure resolution for all plots.

            Returns
            -------
            dict
                Dictionary keyed by target name. Each value is a dict
                mapping output keys (e.g. ``'csv'``, ``'psf_gallery'``)
                to their results.
            """
            if results is None:
                results = self._test_results

            if len(results) == 0:
                logger.warning("No results for report")
                return {}

            if plots is None:
                plots = ['csv', 'psf_profiles', 'psf_gallery',
                         'image_gallery', 'metrics_vs_robust',
                         'metrics_vs_beam', 'html']

            if plot_kwargs is None:
                plot_kwargs = {}

            targets = self._get_unique_targets(results)
            all_reports = {}

            for this_target in targets:
                logger.info("")
                logger.info("=" * 60)
                logger.info(f"REPORT: {this_target}")
                logger.info("=" * 60)

                target_results = [r for r in results
                                  if r.get('target') == this_target]

                imaging_dir = self._resolve_imaging_dir(
                    target_results, this_target)

                this_output_dir = output_dir if output_dir is not None else imaging_dir
                if len(targets) > 1:
                    prefix = output_prefix + '_' + this_target
                else:
                    prefix = output_prefix

                def _out(suffix):
                    return os.path.join(this_output_dir, prefix + '_' + suffix)

                # Registry of report outputs.
                # Extend this dict to add new report outputs.
                registry = {
                    'csv': {
                        'method': self.export_summary_csv,
                        'output_file': _out('summary.csv'),
                        'default_kwargs': {},
                        'output_key': 'filename',
                    },
                    'psf_profiles': {
                        'method': self.plot_psf_radial_profiles,
                        'output_file': _out('psf_profiles.png'),
                        'default_kwargs': {},
                        'output_key': 'output_file',
                    },
                    'psf_gallery': {
                        'method': self.plot_psf_gallery,
                        'output_file': _out('psf_gallery.png'),
                        'default_kwargs': {},
                        'output_key': 'output_file',
                    },
                    'image_gallery': {
                        'method': self.plot_image_gallery,
                        'output_file': _out('image_gallery.png'),
                        'default_kwargs': {},
                        'output_key': 'output_file',
                    },
                    'metrics_vs_robust': {
                        'method': self.plot_metrics_vs_robust,
                        'output_file': _out('metrics_vs_robust.png'),
                        'default_kwargs': {},
                        'output_key': 'output_file',
                    },
                    'metrics_vs_beam': {
                        'method': self.plot_metrics_vs_beam,
                        'output_file': _out('metrics_vs_beam.png'),
                        'default_kwargs': {},
                        'output_key': 'output_file',
                    },
                }

                report = {}

                for key in plots:
                    # HTML report is generated last, after all plots
                    if key == 'html':
                        continue

                    if key not in registry:
                        logger.warning(f"Unknown report output: '{key}', "
                                       f"skipping")
                        continue

                    entry = registry[key]
                    kwargs = dict(entry['default_kwargs'])
                    kwargs.update(plot_kwargs.get(key, {}))

                    kwargs[entry['output_key']] = kwargs.get(
                        entry['output_key'], entry['output_file'])

                    if key == 'csv':
                        kwargs.setdefault('results', target_results)
                    else:
                        kwargs.setdefault('results', target_results)
                        kwargs.setdefault('imaging_dir', imaging_dir)
                        kwargs.setdefault('target', this_target)
                        kwargs.setdefault('config', config)
                        kwargs.setdefault('product', product)
                        kwargs.setdefault('dpi', dpi)

                    try:
                        result = entry['method'](**kwargs)
                        report[key] = result
                        logger.info(f"  {this_target}: produced '{key}' -> "
                                    f"{kwargs[entry['output_key']]}")
                        # Close figures immediately after saving to free memory
                        self._close_figures(result)
                    except Exception as e:
                        logger.error(f"  {this_target}: failed '{key}': {e}")
                        report[key] = None

                # Generate HTML report last so it can embed all plots
                if 'html' in plots:
                    html_file = _out('report.html')
                    plot_files = self._collect_plot_files(
                        report, registry)
                    try:
                        tip.make_html_report(
                            results=target_results,
                            plot_files=plot_files,
                            output_file=html_file,
                            target=this_target,
                        )
                        report['html'] = html_file
                        logger.info(f"  {this_target}: produced 'html' -> "
                                    f"{html_file}")
                    except Exception as e:
                        logger.error(f"  {this_target}: failed 'html': {e}")
                        report['html'] = None

                all_reports[this_target] = report

            return all_reports

        ###########################
        # _collect_plot_files    #
        ###########################

        def _collect_plot_files(self, report, registry):
            """Collect output plot file paths from a report dict.

            Walks the report dict produced by ``make_report`` and
            extracts file paths for each generated plot, suitable for
            passing to ``make_html_report``.

            All plot methods return ``{target: fig}`` or
            ``{target: [fig, ...]}``; this method unwraps those dicts
            and uses the figure-list length to determine how many
            paginated files (``_1.png``, ``_2.png``, ...) were written.

            Parameters
            ----------
            report : dict
                Per-target report dict mapping keys to results.
            registry : dict
                The registry dict from ``make_report``.

            Returns
            -------
            dict
                Mapping of plot key to file path (str) or list of paths.
            """
            plot_files = {}
            for key, value in report.items():
                if key in ('csv', 'html') or value is None:
                    continue
                entry = registry.get(key, {})
                out_file = entry.get('output_file')
                if out_file is None:
                    continue

                # All plot methods return {target: fig} or {target: [figs]}.
                # Unwrap the per-target dict to get the actual figure(s).
                inner = value
                if isinstance(inner, dict):
                    vals = list(inner.values())
                    inner = vals[0] if len(vals) == 1 else None

                if inner is None:
                    continue

                # Use the figure list length to determine how many pages
                # were produced (works even after figures are closed).
                n_pages = len(inner) if isinstance(inner, list) else 1

                if n_pages > 1:
                    # Multi-page gallery: look for _1.png, _2.png, ...
                    base, ext = os.path.splitext(out_file)
                    files = []
                    for i in range(n_pages):
                        candidate = f'{base}_{i + 1}{ext}'
                        if os.path.isfile(candidate):
                            files.append(candidate)
                    if files:
                        plot_files[key] = files
                elif os.path.isfile(out_file):
                    plot_files[key] = out_file

            return plot_files

        ####################
        # _close_figures #
        ####################

        def _close_figures(self, obj):
            """Recursively close matplotlib Figure objects to free memory.

            Handles the nested dict/list structures returned by the plot
            methods so that ``make_report`` can free each figure as soon
            as it has been saved to disk.
            """
            if obj is None:
                return
            if isinstance(obj, dict):
                for v in obj.values():
                    self._close_figures(v)
            elif isinstance(obj, list):
                for v in obj:
                    self._close_figures(v)
            elif hasattr(obj, 'savefig'):
                try:
                    import matplotlib.pyplot as plt
                    plt.close(obj)
                except Exception:
                    pass

        ###########################
        # _per_target_filename #
        ###########################

        def _per_target_filename(self, output_file, target, n_targets):
            """
            Insert the target name into an output filename when there
            are multiple targets.

            Parameters
            ----------
            output_file : str or None
                Base output filename.
            target : str
                Target name.
            n_targets : int
                Total number of targets being processed.

            Returns
            -------
            str or None
            """
            if output_file is None:
                return None
            if n_targets <= 1:
                return output_file
            base, ext = os.path.splitext(output_file)
            return f'{base}_{target}{ext}'

        ############################
        # _ensure_output_in_dir #
        ############################

        def _ensure_output_in_dir(self, output_file, directory):
            """
            If *output_file* is a bare filename (no directory component),
            join it with *directory* so products are saved in the imaging
            directory by default.

            Parameters
            ----------
            output_file : str or None
                Output filename. If None, returned as-is.
            directory : str or None
                Directory to prepend. If None, no change is made.

            Returns
            -------
            str or None
            """
            if output_file is None or directory is None:
                return output_file
            if os.path.dirname(output_file):
                # Already has a directory component — respect it
                return output_file
            return os.path.join(directory, output_file)

        ##########################
        # _resolve_imaging_dir #
        ##########################

        def _resolve_imaging_dir(
            self,
            results,
            target=None,
        ):
            """
            Resolve the imaging directory from the key handler.

            Uses the provided target name, or falls back to the target
            from the first result entry.

            Parameters
            ----------
            results : list of dict
                Results list.
            target : str, optional
                Target name to look up.

            Returns
            -------
            str
                Path to the imaging directory.
            """
            if target is None and len(results) > 0:
                target = results[0].get('target')

            if target is not None and self._kh is not None:
                try:
                    return self._kh.get_imaging_dir_for_target(target=target)
                except Exception as e:
                    logger.warning(f"Could not resolve imaging dir for {target}: {e}")

            return '.'
