"""releaseHandler

Assemble the cubes and maps made by the handlerProduct program
into a release for the team. Mostly just copying between directories.

Example:
    $ ipython
    from phangsPipeline import handlerKeys as hk
    from phangsPipeline import utilsFilenames as uf
    from phangsPipeline import handlerRelease as hr
    this_hk = hk.KeyHandler(master_key = '../phangsalma_keys/master_key.txt')
    this_hr = hr.ReleaseHandler(key_handler = this_hk)
    this_hr.set_interf_configs(only=['7m'])
    this_hr.set_feather_configs(only=['7m+tp'])
    this_hr.set_line_products(only=['co21'])
    this_hr.set_no_cont_products(True)
    this_hr.set_targets(only=['ngc4321'])
"""

import logging
import os
import shutil

from . import handlerTemplate
from . import utilsFilenames
from . import utilsResolutions

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ReleaseHandler(handlerTemplate.HandlerTemplate):
    """
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



    ###########################################
    # Defined file names for various products #
    ###########################################

    def _fname_dict(
        self,
        target=None,
        config=None,
        product=None,
        ):

        if target is None:
            logger.error("Need a target.")
            return()

        if product is None:
            logger.error("Need a product.")
            return()

        if config is None:
            logger.error("Need a config.")
            return()

        fname_dict = {} # here we will create a subdict for each cube resolution (res_tag)

        indir_postprocess = self._kh.get_postprocess_dir_for_target(target=target, changeto=False)
        indir_postprocess = os.path.abspath(indir_postprocess)
        logger.debug('indir_postprocess: '+indir_postprocess)

        indir_product = self._kh.get_product_dir_for_target(target=target, changeto=False)
        indir_product = os.path.abspath(indir_product)
        logger.debug('indir_product: '+indir_product)

        outdir = self._kh.get_release_dir_for_target(target=target, changeto=False)
        outdir = os.path.abspath(outdir)
        logger.debug('outdir: '+outdir)

        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        res_list = self._kh.get_res_for_config(config)
        if res_list is None:
            logger.error('No target resolutions found for target '+target+' and config'+config)
            raise Exception('No target resolutions found for target '+target+' and config'+config)

        cube_filename = utilsFilenames.get_cube_filename(target = target,
                                                         config = config,
                                                         product = product,
                                                         ext = 'pbcorr_trimmed_k',
                                                         casa = False)

            #
        fname_dict['native'] = {}
        if os.path.isfile(os.path.join(indir_postprocess, cube_filename)):
            fname_dict['native']['pbcorr_trimmed_k'] = os.path.join(indir_postprocess, cube_filename)
            hybridmask_filename = utilsFilenames.get_cube_filename(target = target,
                                                                       config = config,
                                                                       product = product,
                                                                       ext = 'hybridmask',
                                                                       casa = False)
            if os.path.isfile(os.path.join(indir_product, hybridmask_filename)):
                fname_dict['native']['hybridmask'] = os.path.join(indir_product, hybridmask_filename)

            signalmask_filename = utilsFilenames.get_cube_filename(target = target,
                                                                       config = config,
                                                                       product = product,
                                                                       ext = 'signalmask',
                                                                       casa = False)
            if os.path.isfile(os.path.join(indir_product, signalmask_filename)):
                fname_dict['native']['signalmask'] = os.path.join(indir_product, signalmask_filename)

                for this_mask in ['broad', 'strict']:
                    fname_dict['native'][this_mask] = {}
                    for this_mom in ["mom0", "mom1", "mom2", "ew", "tmax", "vmax", "vquad"]:
                        for this_prefix in ["","e"]:
                            mom_filename = utilsFilenames.get_cube_filename(target = target,
                                                                            config = config,
                                                                            product = product,
                                                                        ext = this_mask+"_"+this_prefix+this_mom,
                                                                        casa = False)
                            if os.path.isfile(os.path.join(indir_product, mom_filename)):
                                fname_dict['native'][this_mask][this_prefix+this_mom] = os.path.join(indir_product, mom_filename)

            for this_res in res_list:
                #res_tag = self._kh.get_tag_for_res(this_res)
                res_tag = utilsResolutions.get_tag_for_res(this_res)
            cube_filename = utilsFilenames.get_cube_filename(target = target,
                                                             config = config,
                                                             product = product,
                                                             ext = 'pbcorr_trimmed_k'+'_res'+res_tag,
                                                             casa = False)
            if os.path.isfile(os.path.join(indir_postprocess, cube_filename)):
                    fname_dict[res_tag] = {}
                    fname_dict[res_tag]['pbcorr_trimmed_k'] = os.path.join(indir_postprocess, cube_filename)
                    hybridmask_filename = utilsFilenames.get_cube_filename(target = target,
                                                                            config = config,
                                                                            product = product,
                                                                            ext = 'hybridmask'+'_res'+res_tag,
                                                                            casa = False)
                    if os.path.isfile(os.path.join(indir_product, hybridmask_filename)):
                        fname_dict[res_tag]['hybridmask'] = os.path.join(indir_product, hybridmask_filename)

                    signalmask_filename = utilsFilenames.get_cube_filename(target = target,
                                                                           config = config,
                                                                           product = product,
                                                                           ext = 'signalmask'+'_res'+res_tag,
                                                                           casa = False)
                    if os.path.isfile(os.path.join(indir_product, signalmask_filename)):
                        fname_dict[res_tag]['signalmask'] = os.path.join(indir_product, signalmask_filename)

                    for this_mask in ['broad', 'strict']:
                        fname_dict[res_tag][this_mask] = {}
                        for this_mom in ["mom0", "mom1", "mom2", "ew", "tmax", "vmax", "vquad"]:
                            for this_prefix in ["","e"]:
                                mom_filename = utilsFilenames.get_cube_filename(target = target,
                                                                                config = config,
                                                                                product = product,
                                                                            ext = this_mask+"_"+this_prefix+this_mom+"_res"+res_tag,
                                                                            casa = False)
                                if os.path.isfile(os.path.join(indir_product, mom_filename)):
                                    fname_dict[res_tag][this_mask][this_prefix+this_mom] = os.path.join(indir_product, mom_filename)

            return fname_dict, outdir


    ########################
    # loop_products_making #
    ########################

    def loop_build_release(
        self,
        make_directories=True,
        ):
        """
        """

        # Replacing build_release.pro

        if len(self.get_targets()) == 0:
            logger.error("Need a target list.")
            return(None)

        if len(self.get_all_products()) == 0:
            logger.error("Need a products list.")
            return(None)

        if make_directories:
            self._kh.make_missing_directories(release = True)

        for this_target, this_product, this_config in \
            self.looper(do_targets=True,do_products=True,do_configs=True):
            fname_dict, outdir = self._fname_dict(target = this_target,
                                                  config = this_config,
                                                  product = this_product)

            self.task_building_release(fname_dict = fname_dict,
                                        outdir = outdir)
            #
            # end of loop


    ##########################
    # task_writting_products #
    ##########################
        def task_building_release(
            self,
            fname_dict,
            outdir,
            do_cube = True,
            do_hybridmask = True,
            do_signalmask = True,
            do_broad = True,
            do_strict = True,
            ):
            """
            """
            for res_tag in fname_dict:
                if do_cube==True:
                    inputname = fname_dict[res_tag]['pbcorr_trimmed_k']
                    outputname = inputname.split("/")[-1]
                    if os.path.isfile(os.path.join(outdir,outputname)):
                        os.remove(os.path.join(outdir,outputname))
                        logger.debug('Deleting old file "'+os.path.join(outdir,outputname)+'"')
                    #
                    shutil.copy(inputname,outdir)
                    logger.debug('building release "'+outputname+'"')

                if do_hybridmask==True:
                    inputname = fname_dict[res_tag]['hybridmask']
                    outputname = inputname.split("/")[-1]
                    if os.path.isfile(os.path.join(outdir,outputname)):
                        os.remove(os.path.join(outdir,outputname))
                        logger.debug('Deleting old file "'+os.path.join(outdir,outputname)+'"')
                    #
                    shutil.copy(inputname,outdir)
                    logger.debug('building release "'+outputname+'"')

                if do_signalmask==True:
                    inputname = fname_dict[res_tag]['signalmask']
                    outputname = inputname.split("/")[-1]
                    if os.path.isfile(os.path.join(outdir,outputname)):
                        os.remove(os.path.join(outdir,outputname))
                        logger.debug('Deleting old file "'+os.path.join(outdir,outputname)+'"')
                    #
                    shutil.copy(inputname,outdir)
                    logger.debug('building release "'+outputname+'"')

                if do_broad==True:
                    inputkeys = fname_dict[res_tag]['broad']
                    for inputkey in inputkeys:
                        inputname = fname_dict[res_tag]['broad'][inputkey]
                        outputname = inputname.split("/")[-1]
                        if os.path.isfile(os.path.join(outdir,outputname)):
                            os.remove(os.path.join(outdir,outputname))
                            logger.debug('Deleting old file "'+os.path.join(outdir,outputname)+'"')
                        #
                        shutil.copy(inputname,outdir)
                        logger.debug('building release "'+outputname+'"')

                if do_strict==True:
                    inputkeys = fname_dict[res_tag]['strict']
                    for inputkey in inputkeys:
                        inputname = fname_dict[res_tag]['strict'][inputkey]
                        outputname = inputname.split("/")[-1]
                        if os.path.isfile(os.path.join(outdir,outputname)):
                            os.remove(os.path.join(outdir,outputname))
                            logger.debug('Deleting old file "'+os.path.join(outdir,outputname)+'"')
                        #
                        shutil.copy(inputname,outdir)
                        logger.debug('building release "'+outputname+'"')

