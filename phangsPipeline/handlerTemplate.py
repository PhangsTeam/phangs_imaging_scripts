"""
handlerTemplate

This is a template handler object. It acts as parent to our other
handlers and includes basic list and shared functionality.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class HandlerTemplate:
    """
    Template handler class inherited by specific handler objects.
    """

    def __init__(
        self,
        key_handler = None,
        dry_run = False,
        ):

        # Initialize lists

        self._targets_first = None
        self._targets_last = None
        self._targets_skip = None
        self._targets_only = None
        self._cont_skip = None
        self._cont_only = None
        self._lines_skip = None
        self._lines_only = None
        self._interf_configs_skip = None
        self._interf_configs_only = None
        self._feather_configs_skip = None
        self._feather_configs_only = None
        self._singledish_configs_skip = None
        self._singledish_configs_only = None

        self._targets_list = []
        self._line_products_list = []
        self._cont_products_list = []
        self._interf_configs_list = []
        self._feather_configs_list = []
        self._singledish_configs_list = []

        # Initialize switches on config types
        self._no_interf = False
        self._no_feather = False
        self._no_singledish = False

        # Initialize switches on product types
        self._no_cont = False
        self._no_line = False

        if key_handler is not None:
            self.set_key_handler(key_handler, nobuild=True)

        # Initialize the list variables
        self.set_targets(nobuild=True)

        self.set_line_products(nobuild=True)
        self.set_cont_products(nobuild=True)

        self.set_interf_configs(nobuild=True)
        self.set_feather_configs(nobuild=True)
        self.set_singledish_configs(nobuild=True)

        # Build the lists
        self._build_lists()

        # Toggle whether tasks are executed
        self.set_dry_run(dry_run)

#region Parameter toggles

    ##########################################
    # Toggle control flow related parameters #
    ##########################################

    def set_key_handler(
        self,
        key_handler = None,
        nobuild = False):
        """
        Set the keyhandler object being used by the pipeline. The
        handlerKeys object interaces with configuration files, target
        lists, etc.
        """
        self._kh = key_handler
        if not nobuild:
            self._build_lists()
        return(None)

    def set_dry_run(
        self,
        dry_run = False):
        """
        Toggle the program to execute a 'dry run.' In this case it
        will not actually execute calls but will run through loops,
        print messages, etc..
        """
        self._dry_run = dry_run
        return(None)

#endregion

#region List building routines

    ##############################################################
    # Routines to set targets, configs, products and build lists #
    ##############################################################

    def set_targets(
        self, 
        first=None, 
        last=None, 
        skip=[], 
        only=[],
        nobuild=False):
        """
        Set conditions on the list of targets to be considered when a
        loop is run. By default, consider all targets.
        """
        self._targets_first = first
        self._targets_last = last

        if np.isscalar(skip):
            self._targets_skip = [skip]
        else:
            self._targets_skip = skip

        if np.isscalar(only):
            self._targets_only = [only]
        else:
            self._targets_only = only

        if not nobuild:
            self._build_lists()
        return(None)

    def set_line_products(
        self, 
        skip=[], 
        only=[], 
        nobuild=False,
        ):
        """
        Set conditions on the list of line products to be considered
        when a loop is run. By default, consider all products.
        """
        if np.isscalar(skip):
            self._lines_skip = [skip]
        else:
            self._lines_skip = skip

        if np.isscalar(only):
            self._lines_only = [only]
        else:
            self._lines_only = only

        if not nobuild:
            self._build_lists()
        return(None)

    def set_cont_products(
        self, 
        skip=[], 
        only=[], 
        nobuild=False,
        ):
        """
        Set conditions on the list of continuum products to be
        considered when a loop is run. By default, consider all
        products.
        """
        if np.isscalar(skip):
            self._cont_skip = [skip]
        else:
            self._cont_skip = skip

        if np.isscalar(only):
            self._cont_only = [only]
        else:
            self._cont_only = only

        if not nobuild:
            self._build_lists()
        return(None)

    def set_interf_configs(
        self, 
        skip=[], 
        only=[], 
        nobuild=False,
        ):
        """
        Set conditions on the list of interferometric array
        configurations to be considered when a loop is run. By
        default, consider all configurations.
        """
        if np.isscalar(skip):
            self._interf_configs_skip = [skip]
        else:
            self._interf_configs_skip = skip

        if np.isscalar(only):
            self._interf_configs_only = [only]
        else:
            #self._inerf_configs_only = only #BUG#
            self._interf_configs_only = only

        if not nobuild:
            self._build_lists()
        return(None)

    def set_feather_configs(
        self, 
        skip=[], 
        only=[],
        nobuild=False,
        ):
        """
        Set conditions on the list of feathered array configurations
        to be considered when a loop is run. By default, consider
        all configurations.
        """
        if np.isscalar(skip):
            self._feather_configs_skip = [skip]
        else:
            self._feather_configs_skip = skip

        if np.isscalar(only):
            self._feather_configs_only = [only]
        else:
            self._feather_configs_only = only

        if not nobuild:
            self._build_lists()
        return(None)

    def set_singledish_configs(
        self, 
        skip=[], 
        only=[],
        nobuild=False,
        ):
        """
        Set conditions on the list of singledish configurations
        to be considered when a loop is run. By default, consider
        all configurations.
        """
        if np.isscalar(skip):
            self._singledish_configs_skip = [skip]
        else:
            self._singledish_configs_skip = skip

        if np.isscalar(only):
            self._singledish_configs_only = [only]
        else:
            self._singledish_configs_only = only

        if not nobuild:
            self._build_lists()
        return(None)

    def set_no_line_products(
        self,
        no_line = False):
        """
        Toggle the program to skip all line products when a loop or
        task is run.
        """
        self._no_line = no_line
        self._build_lists()

    def set_no_cont_products(
        self,
        no_cont = False):
        """
        Toggle the program to skip all continuum products when a
        loop is run.
        """
        self._no_cont = no_cont
        self._build_lists()

    def set_no_interf_configs(
        self,
        no_interf = False):
        """
        Toggle the program to skip all interferometric configurations
        when a loop is run.
        """
        self._no_interf = no_interf
        self._build_lists()

    def set_no_feather_configs(
        self,
        no_feather = False):
        """
        Toggle the program to skip all feathered configurations when a
        loop is run.
        """
        self._no_feather = no_feather
        self._build_lists()

    def set_no_singledish_configs(
        self,
        no_singledish = False):
        """
        Toggle the program to skip all singledish configurations when a
        loop is run.
        """
        self._no_singledish = no_singledish
        self._build_lists()

    def _build_lists(
        self
        ):
        """
        Build the lists of targets, mosaics, products, and
        configurations to loop over when a loop is run.
        """

        # Make sure there is an attached handlerKeys object.
        
        if self._kh is None:
            logger.error("Cannot build lists without a handlerKeys.")
            raise Exception("Cannot build lists without a handlerKeys.")
            return(None)

        self._targets_list = self._kh.get_targets(
            only = self._targets_only,
            skip = self._targets_skip,
            first = self._targets_first,
            last = self._targets_last,
            )

        if self._no_line:
            self._line_products_list = []
        else:
            self._line_products_list = self._kh.get_line_products(
                only = self._lines_only,
                skip = self._lines_skip,
                )

        if self._no_cont:
            self._cont_products_list = []
        else:
            self._cont_products_list = self._kh.get_continuum_products(
                only = self._cont_only,
                skip = self._cont_skip,
                )

        if self._no_interf:
            self._interf_configs_list = []
        else:
            self._interf_configs_list = self._kh.get_interf_configs(
                only = self._interf_configs_only,
                skip = self._interf_configs_skip,
                )
        
        if self._no_feather:
            self._feather_configs_list = []
        else:
            self._feather_configs_list = self._kh.get_feather_configs(
                only = self._feather_configs_only,
                skip = self._feather_configs_skip,
                )
        
        if self._no_singledish:
            self._singledish_configs_list = []
        else:
            self._singledish_configs_list = self._kh.get_singledish_configs(
                only = self._singledish_configs_only,
                skip = self._singledish_configs_skip,
                )

        return()

#endregion

#region Access

    ############################################################
    # Internal access to products and configurations for lists #
    ############################################################

    def get_targets(
        self
        ):
        """
        Return the list of targets to consider.
        """
        if self._targets_list is None:
            return([])
        else:
            return(self._targets_list)
        
    def get_line_products(
        self
        ):
        """
        Return the list of line products to consider.
        """
        if self._line_products_list is None:
            return([])
        else:
            return(self._line_products_list)

    def get_cont_products(
        self
        ):
        """
        Return the list of continuum products to consider.
        """
        if self._cont_products_list is None:
            return([])
        else:
            return(self._cont_products_list)

    def get_all_products(
        self
        ):
        """
        Get a combined list of line and continuum products to be
        considered.
        """

        if len(self.get_cont_products()) == 0:
            return(self.get_line_products())

        if len(self.get_line_products()) == 0:
            return(self.get_cont_products())
        
        return(self.get_line_products() + self.get_cont_products())

    def get_interf_configs(
        self
        ):
        """
        Return the list of interferometric configs to consider.
        """
        if self._interf_configs_list is None:
            return([])
        else:
            return(self._interf_configs_list)

    def get_feather_configs(
        self
        ):
        """
        Return the list of feather configs to consider.
        """
        if self._feather_configs_list is None:
            return([])
        else:
            return(self._feather_configs_list)

    def get_singledish_configs(
        self
        ):
        """
        Return the list of singledish configs to consider.
        """
        if self._singledish_configs_list is None:
            return([])
        else:
            return(self._singledish_configs_list)

    def get_all_configs(
        self
        ):    
        """
        Get a combined list of feather and interferometric configs to
        consider.
        """
        all_configs = []
        if self.get_interf_configs() is not None and not self._no_interf:
            if len(self.get_interf_configs()) > 0:
                all_configs.extend(self.get_interf_configs())

        if self.get_feather_configs() is not None and not self._no_feather:
            if len(self.get_feather_configs()) > 0:
                all_configs.extend(self.get_feather_configs())
        
        if len(all_configs) == 0:
            all_configs = None
        
        return all_configs

#endregion

#region Loop execution

    ###################################################
    # Loop over targets, products, and configurations #
    ###################################################

    def looper(
        self,
        do_targets=True,
        do_products=True,
        just_line=False,
        just_cont=False,
        do_configs=True,
        just_interf=False,
        just_feather=False,
        just_singledish=False,
        ):
        """
        Return (target, product, config) tuples for all selected
        combinations. Boolean switches toggle what gets included in
        the loop.
        """

        target_list = self.get_targets()

        product_list = self.get_all_products()
        if just_line and just_cont:
            logger.error("Both just_line and just_cont set. Defaulting to all products.")
        if just_line and not just_cont:
            product_list = self.get_line_products()
        if just_cont and not just_line:
            product_list = self.get_cont_products()
            
        config_list = self.get_all_configs()
        if just_interf and just_feather:
            logger.error("Both just_interf and just_feather set. Defaulting to all configs.")
        if just_interf and not just_feather:
            config_list = self.get_interf_configs()
        if just_feather and not just_interf:
            config_list = self.get_feather_configs()
        if just_singledish:
            config_list = self.get_singledish_configs()      

        # All three quantities

        if do_targets and do_products and do_configs:
            logger.info("Looping over target, product, and config.")
            for this_target in target_list:
                for this_product in product_list:
                    for this_config in config_list:
                        yield this_target, this_product, this_config
                
        # Two quantity loop

        if do_targets and do_products and not do_configs:
            logger.info("Looping over target and product.")
            for this_target in target_list:
                for this_product in product_list:
                    yield this_target, this_product

        if do_targets and do_configs and not do_products:
            logger.info("Looping over target and config.")
            for this_target in target_list:
                for this_config in config_list:
                    yield this_target, this_config

        if do_products and do_configs and not do_targets:
            logger.info("Looping over product and config.")
            for this_product in product_list:
                for this_config in config_list:
                    yield this_product, this_config

        # Single quantity loop

        if do_targets and not do_configs and not do_products:
            logger.info("Looping over target.")
            for this_target in target_list:
                yield this_target

        if do_configs and not do_targets and not do_products:
            logger.info("Looping over config.")
            for this_config in config_list:
                yield this_config

        if do_products and not do_targets and not do_configs:
            logger.info("Looping over product.")
            for this_product in product_list:
                yield this_product

#endregion
