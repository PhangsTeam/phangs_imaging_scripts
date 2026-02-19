#!/usr/bin/env python
# 
"""
This "statsHandler" is a PHANGS-ALMA internal-use object and not
intended as a general part of the pipeline. It uses the general
handler infrastructure to loop over measurement sets and compile
statistics, which are then dumped to JSON files for further
analysis. Other users should feel free to adapt this to their uses,
but it's not essential or generally integrated.
"""

import json
import logging
import os

from . import casaImagingRoutines as cir
from . import handlerKeys
from . import handlerTemplate
from . import utilsFilenames

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class StatsHandler(handlerTemplate.HandlerTemplate):
    """
    Make handler for statistics.
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
        handlerTemplate.HandlerTemplate.__init__(self, 
                                                 key_handler = key_handler, 
                                                 dry_run = dry_run)
    
    def go(
        self, 
        outfile_singlescale='stats_singlesale.json',
        outfile_multiscale='stats_multiscale.json',
        ):
        """main function to execute.
        """
        ms_dict = {}
        ss_dict = {}
        for this_target, this_product, this_config in \
            self.looper(do_targets=True, do_products=True, do_configs=True, just_interf=True):
            # 
            this_ms_dict = self.recipe_residual_regression(
                target=this_target, 
                product=this_product, 
                config=this_config,
                casaext='_multiscale', 
                )
            if this_ms_dict is not None:
                ms_dict[this_ms_dict['cubename']] = this_ms_dict

            # 
            this_ss_dict = self.recipe_residual_regression(
                target=this_target, 
                product=this_product, 
                config=this_config,
                casaext='_singlescale', 
                )
            if this_ss_dict is not None:
                ss_dict[this_ss_dict['cubename']] = this_ss_dict
    
        # Convert to a table and write to disk
        with open('stats_multiscale.json', 'w') as fout:
            json.dump(ms_dict , fout)
        with open(r"stats_multiscale.json", "r") as read_file:
            test = json.load(read_file)
            print(test)

        # Convert to a table and write to disk
        with open('stats_singlescale.json', 'w') as fout:
            json.dump(ss_dict , fout)
        with open(r"stats_singlescale.json", "r") as read_file:
            test = json.load(read_file)
            print(test)

    def _fname_dict(
        self,
        product, 
        imagename, 
        ):
        """define file name dict for analysis in this code.
        """
        fname_dict = {}
        is_line_product = product in self._kh.get_line_products()
        if is_line_product:
            fname_dict['root'] = imagename
            fname_dict['suffix'] = ''
            fname_dict['image'] = imagename+'.image'
            fname_dict['model'] = imagename+'.model'
            fname_dict['residual'] = imagename+'.residual'
            fname_dict['mask'] = imagename+'.mask'
            fname_dict['pb'] = imagename+'.pb'
            fname_dict['psf'] = imagename+'.psf'
            fname_dict['weight'] = imagename+'.weight'
            fname_dict['sumwt'] = imagename+'.sumwt'
        else:
            fname_dict['root'] = imagename
            fname_dict['suffix'] = '.tt0'
            fname_dict['image'] = imagename+'.image.tt0'
            fname_dict['model'] = imagename+'.model.tt0'
            fname_dict['residual'] = imagename+'.residual.tt0'
            fname_dict['mask'] = imagename+'.mask.tt0'
            fname_dict['pb'] = imagename+'.pb.tt0'
            fname_dict['psf'] = imagename+'.psf.tt0'
            fname_dict['weight'] = imagename+'.weight.tt0'
            fname_dict['sumwt'] = imagename+'.sumwt.tt0'
            fname_dict['alpha'] = imagename+'.alpha'
            fname_dict['beta'] = imagename+'.beta'
        return fname_dict
    
    
    def recipe_residual_regression(
        self,
        target=None, 
        product=None, 
        config=None, 
        casaext='', 
        ):
        """recipe to make clean residual regression for one target.
        """
        if target is None:
            raise Exception('Please input a target!')
        if product is None:
            raise Exception('Please input a product!')
        if config is None:
            raise Exception('Please input a config!')
        # 
        logger.info('------------------------------------------------------------')
        logger.info('Processing target %s product %s config %s'%(target, product, config))
        logger.info('------------------------------------------------------------')
        # 
        image_root = utilsFilenames.get_cube_filename(\
            target=target, product=product, config=config,
            ext=None, casa=True, casaext=casaext)
        # 
        fname_dict = self._fname_dict(\
            product = product, 
            imagename = image_root)
        # 
        image_file = fname_dict['image']
        residual_file = fname_dict['residual']
        mask_file = fname_dict['mask']
        # 
        target_whole_name = self._kh.get_mosaic_target_for_parts(target)
        if target_whole_name is None:
            target_whole_name = target
        # 
        imaging_dir = self._kh.get_imaging_dir_for_target(target_whole_name, changeto=False)
        # 
        image_file = os.path.join(imaging_dir, image_file)
        residual_file = os.path.join(imaging_dir, residual_file)
        mask_file = os.path.join(imaging_dir, mask_file)
        # 

        stat_dict = self.task_residual_regression(
            image_file=image_file, 
            residual_file=residual_file, 
            mask_file=mask_file)
        if stat_dict is None:
            return(stat_dict)

        stat_dict['target'] = target
        stat_dict['product'] = product
        stat_dict['config'] = config

        return(stat_dict)

    def task_residual_regression(
        self, 
        image_file, 
        residual_file, 
        mask_file, 
        ):
        """calculates the clean residual statistics
        """
        logger.info('image    : %r'%(image_file))
        logger.info('residual : %r'%(residual_file))
        logger.info('mask     : %r'%(mask_file))
        
        stat_dict = cir.calc_residual_statistics(
            resid_name=residual_file,mask_name=mask_file)

        return(stat_dict)
        


########
# main #
########
if __name__ == '__main__':
    
    master_key = 'keys/master_key.txt'
    #master_key = '/Users/dzliu/Work/AlmaPhangs/Works/20200630_PHANGS_ALMA_clean_records/test_phangs_working_dir/keys/master_key.txt'
    if not os.path.isfile(master_key):
        master_key = input("Please input your master key file path: ")
    
    if master_key.find("'") >= 0:
        master_key = master_key.replace("'", "")
    if master_key.find('"') >= 0:
        master_key = master_key.replace('"', '')
    
    if not os.path.isfile(master_key):
        raise Exception('Error! Input master key file does not exist: %r'%(master_key))
    
    this_key_handler = handlerKeys.KeyHandler(master_key = master_key)
    this_handler = StatsHandler(key_handler = this_key_handler)
    this_handler.set_line_products(only=['co21'])
    this_handler.set_no_cont_products(True)
    this_handler.go()








