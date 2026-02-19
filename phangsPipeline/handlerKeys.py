"""
Parts of the PHANGS pipeline that handle the targets, data files,
etc. This is the program that navigates the galaxy list, directory
structure, etc. This part is pure python.
"""

import ast
import logging
import os

from . import utilsKeyReaders as key_readers
from . import utilsLists as list_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

VALID_IMAGING_STAGES = ['dirty', 'multiscale', 'singlescale']


class KeyHandler:
    """
    Class to handle data files that indicate the names and data sets
    associated with reducing a large ALMA imaging project.
    """

    def __init__(self,
                 master_key='key_templates/master_key.txt',
                 quiet=False,
                 dochecks=True,
                 ):

        self._dochecks = dochecks

        self._master_key = None

        self._target_dict = None
        self._config_dict = None
        self._imaging_dict = None

        self._casaversion_dict = None
        self._alma_download_dict = None
        self._ms_dict = None
        self._sd_dict = None
        self._cleanmask_dict = None
        self._linmos_dict = None
        self._distance_dict = None
        self._window_dict = None
        self._moment_dict = None

        self._dir_for_target = None
        self._override_dict = None

        self.build_key_handler(master_key)

    ##############################################################
    # FILE READING AND INITIALIZATION
    ##############################################################

    def build_key_handler(self, master_key=None):
        """
        Construct the key handler object.
        """

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&&%&%&%&%&%&%&%&%&%&%")
        logger.info("Initializing the PHANGS-ALMA pipeline KeyHandler.")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&&%&%&%&%&%&%&%&%&%&%")
        logger.info("")

        if os.path.isfile(master_key) is False:
            logger.error("Master key " + master_key + " not found. Aborting.")
            raise Exception("Master key " + master_key + " not found. Aborting.")
            return False

        self._master_key = os.path.abspath(master_key)
        self._read_master_key()

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Reading individual key files.")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        self._read_all_keys()

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Running checks and cross-links.")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        if self._dochecks:
            self.check_ms_existence()

        if self._dochecks:
            self.check_sd_existence()

        self._target_list = []
        self._missing_targets = []
        self._build_target_list()
        if self._dochecks:
            self.print_missing_targets()

        self._expand_dir_key()
        if self._dochecks:
            self.check_dir_existence()

        self._build_whole_target_list()
        self._map_targets_to_mosaics()
        self._map_configs()

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("Printing configurations.")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("")

        self.print_configs()

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("Printing spectral products.")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("")

        self.print_products()

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("Printing derived data products.")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("")

        self.print_derived()

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("Printing missing distances.")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("")

        self.print_missing_distances()

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("Printing missing velocity windows.")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("")

        self.print_missing_window()

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Master key reading and checks complete.")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

    def _parse_path(self, input_path):
        """
        Parse relative path.
        """
        if input_path.startswith('/') or input_path.startswith(os.sep):
            output_path = input_path
        elif input_path.startswith('~') or input_path.startswith('$'):
            output_path = os.path.expanduser(input_path)
        else:
            output_path = os.path.abspath(input_path)
        if not (output_path.endswith('/') or output_path.endswith(os.sep)):
            output_path += '/'
        return output_path

    ##############################################################
    # READ THE MASTER KEY
    ##############################################################

    # This steers the whole pipeline process and infrastructure and links
    # from fields into attributes of the object.

    def _read_master_key(self):
        """
        Read the master key.
        """
        logger.info("Reading the master key.")

        logger.info("Master key file: " + self._master_key)
        fname = self._master_key
        infile = open(fname, 'r')

        # Initialize

        self._key_dir = os.path.dirname(
            self._master_key) + os.sep  # None #<20200305><DL># We do not have to set a 'key_dir' in "master_key.txt", just use its directory. If set, then we still use the one in the 'key_dir' in "master_key.txt".
        self._imaging_root = ''  # os.getcwd()+'/../imaging/'
        self._postprocess_root = ''  # os.getcwd()+'/../postprocess/'
        self._derived_root = ''  # os.getcwd()+'/../derived/'
        self._release_root = ''  # os.getcwd()+'/../release/'
        self._cleanmask_root = ''  # os.getcwd() + '/../cleanmask/'
        self._vfield_root = ''  # os.getcwd() + '/../vfield/'

        self._ms_roots = []
        self._sd_roots = []
        self._cleanmask_roots = []
        self._vfield_roots = []
        self._casaversion_keys = []
        self._alma_download_keys = []
        self._ms_keys = []
        self._sd_keys = []
        self._cleanmask_keys = []

        self._config_keys = []
        self._derived_keys = []
        self._moment_keys = []
        self._target_keys = []
        self._linmos_keys = []
        self._dir_keys = []
        self._imaging_keys = []
        self._override_keys = []
        self._distance_keys = []
        self._window_keys = []

        first_key_dir = True
        first_imaging_root = True
        first_postprocess_root = True
        first_derived_root = True
        first_release_root = True

        lines_read = 0
        while True:
            line = infile.readline()
            if len(line) == 0:
                break
            if line[0] == '#' or line == '\n':
                continue

            words = line.split()
            # All key entries go key-value
            if len(words) != 2:
                continue

            this_key = words[0]
            this_value = words[1]

            if this_key == 'imaging_root':
                self._imaging_root = self._parse_path(this_value)
                if first_imaging_root:
                    first_imaging_root = False
                else:
                    logger.warning("Multiple imaging_root definitions. Using the last one.")
                lines_read += 1

            if this_key == 'postprocess_root':
                self._postprocess_root = self._parse_path(this_value)
                if first_postprocess_root:
                    first_postprocess_root = False
                else:
                    logger.warning("Multiple postprocess_root definitions. Using the last one.")
                lines_read += 1

            if this_key == 'derived_root':
                self._derived_root = self._parse_path(this_value)
                if first_derived_root:
                    first_derived_root = False
                else:
                    logger.warning("Multiple derived_root definitions. Using the last one.")
                lines_read += 1

            if this_key == 'release_root':
                self._release_root = self._parse_path(this_value)
                if first_release_root:
                    first_release_root = False
                else:
                    logger.warning("Multiple release_root definitions. Using the last one.")
                lines_read += 1

            if this_key == 'key_dir':
                self._key_dir = self._parse_path(this_value)
                if first_key_dir:
                    first_key_dir = False
                else:
                    logger.warning("Multiple key directory definitions. Using the last one.")
                lines_read += 1

            if this_key == 'ms_root':
                self._ms_roots.append(self._parse_path(this_value))
                lines_read += 1

            if this_key == 'singledish_root':
                self._sd_roots.append(self._parse_path(this_value))
                lines_read += 1

            if this_key == 'cleanmask_root':
                self._cleanmask_root = this_value
                self._cleanmask_roots.append(self._parse_path(this_value))
                lines_read += 1

            if this_key == 'vfield_root':
                self._vfield_root = this_value
                self._vfield_roots.append(self._parse_path(this_value))
                lines_read += 1

            if this_key == 'casaversion_key':
                self._casaversion_keys.append(this_value)
                lines_read += 1

            if this_key == 'alma_download_key':
                self._alma_download_keys.append(this_value)
                lines_read += 1

            if this_key == 'ms_key':
                self._ms_keys.append(this_value)
                lines_read += 1

            if this_key == 'config_key':
                self._config_keys.append(this_value)
                lines_read += 1

            if this_key == 'derived_key':
                self._derived_keys.append(this_value)
                lines_read += 1

            if this_key == 'moment_key':
                self._moment_keys.append(this_value)
                lines_read += 1

            if this_key == 'cleanmask_key':
                self._cleanmask_keys.append(this_value)
                lines_read += 1

            if this_key == 'dir_key':
                self._dir_keys.append(this_value)
                lines_read += 1

            if this_key == 'target_key':
                self._target_keys.append(this_value)
                lines_read += 1

            if this_key == 'imaging_key':
                self._imaging_keys.append(this_value)
                lines_read += 1

            if this_key == 'override_key':
                self._override_keys.append(this_value)
                lines_read += 1

            if this_key == 'linmos_key':
                self._linmos_keys.append(this_value)
                lines_read += 1

            if this_key == 'singledish_key':
                self._sd_keys.append(this_value)
                lines_read += 1

            if this_key == 'distance_key':
                self._distance_keys.append(this_value)
                lines_read += 1

            if this_key == 'window_key':
                self._window_keys.append(this_value)
                lines_read += 1

        logger.info("Successfully imported " + str(lines_read) + " key/value pairs.")

        infile.close()

        if self._dochecks:
            self.check_key_existence()

        return True

    def check_key_existence(self):
        """
        Check file existence for the input keys defined in the master file.
        """

        logger.info("------------------------------------")
        logger.info("Checking the existence of key files.")
        logger.info("------------------------------------")

        all_valid = True
        errors = 0

        self._key_dir_exists = os.path.isdir(self._key_dir)
        if not self._key_dir_exists:
            logger.error("Missing the key directory. Currently set to " + self._key_dir)
            logger.error("I need the key directory to proceed. Set key_dir in your master_key file.")
            all_valid = False
            errors += 1
            # return(all_valid)

        for keyname in ['imaging', 'postprocess', 'derived', 'release']:
            keypath = getattr(self, '_%s_root' % keyname)
            if keypath is None or keypath == '':
                logger.error("The %s root directory has not been set." % keyname)
                logger.error("Please set a valid %s_root in your master_key file." % keyname)
                all_valid = False
                errors += 1

        if len(self._ms_roots) == 0:
            logger.error("The ms root directory has not been set.")
            logger.error("Please set one or more ms_root in your master_key file.")
            all_valid = False
            errors += 1

        for keypath in self._ms_roots:
            if not os.path.isdir(keypath):
                logger.error("The ms root directory does not exist: %r. "
                             "This will cause an error if not using downloader handler." % keypath)
                errors += 1
        if len(self._cleanmask_roots) > 0:
            for keypath in self._cleanmask_roots:
                if not os.path.isdir(keypath):
                    logger.error("The cleanmask root directory does not exist: %r" % keypath)
                    logger.error("Please set the correct cleanmask_root in your master_key file.")
                    all_valid = False
                    errors += 1

        if len(self._vfield_roots) == 0:
            logger.warning("The vfield root directory has not been set.")
            logger.warning("Please set vfield_root in your master_key file if you want to use an input vfield for shuffling.")

        all_key_lists = [self._casaversion_keys,
                         self._alma_download_keys,
                         self._ms_keys,
                         self._dir_keys,
                         self._target_keys,
                         self._override_keys,
                         self._imaging_keys,
                         self._linmos_keys,
                         self._sd_keys,
                         self._config_keys,
                         self._cleanmask_keys,
                         self._distance_keys,
                         self._window_keys,
                         self._derived_keys,
                         self._moment_keys,
                         ]
        for this_list in all_key_lists:
            for this_key in this_list:
                this_key_exists = os.path.isfile(self._key_dir + this_key)
                if not this_key_exists:
                    errors += 1
                    if this_list not in [self._ms_keys, self._dir_keys, self._linmos_keys]:
                        all_valid = False
                        logger.error("key " + this_key + " is defined but does not exist in " + self._key_dir)
                    else:
                        logger.warning('key %s is defined but does not exist. This will cause issues if not using the '
                                       'download handler' % this_key)

        if all_valid:
            logger.info("Checked key file existence with %d errors. "
                        "This may cause issues if not using download handlers." % errors)
        else:
            logger.error("Checked key file existence. Found " + str(errors) + " errors.")
            raise Exception("Checked key file existence. Found " + str(errors) + " errors.")

        return all_valid

    ##############################################################
    # READ THE KEYS INTO DICTIONARIES
    ##############################################################

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Define routines to read the imaging dictionary
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # The imaging dictionary requires outside knowledge, e.g., because it
    # uses wild cards and requires entries for each case. Initialize and
    # read it here.

    def _initialize_imaging_dict(self):
        """
        Initialize the imaging dictionary. Requires the config dict to
        exist already to work successfully.
        """

        logger.info("Initializing the imaging dictionary using known configs and stages.")

        full_dict = {}

        # Loop over configs
        for this_config in self.get_interf_configs():
            full_dict[this_config] = {}

            # Loop over line products
            for this_product in self.get_line_products():
                full_dict[this_config][this_product] = {}

                # Loop over stages
                for this_stage in VALID_IMAGING_STAGES:
                    full_dict[this_config][this_product][this_stage] = None

            # Loop over continuum products
            for this_product in self.get_continuum_products():
                full_dict[this_config][this_product] = {}
                # Loop over stages
                for this_stage in VALID_IMAGING_STAGES:
                    full_dict[this_config][this_product][this_stage] = None

        self._imaging_dict = full_dict

        return ()

    def _read_imaging_key(self, fname=None, existing_dict=None, delim=None):
        """
        Read an imaging recipe key. Needs to be in the keyHandler
        object because the order of entries matters AND wild cards
        matter and these interact.
        """

        # Check file existence

        if os.path.isfile(fname) is False:
            logger.error("I tried to read key " + fname + " but it does not exist.")
            return existing_dict

        logger.info("Reading: " + fname)

        # Expected Format

        expected_words = 4
        expected_format = "config product stage recipe"

        # Open File

        infile = open(fname, 'r')

        # Initialize the dictionary

        if existing_dict is None:
            if self._imaging_dict is None:
                self._initialize_imaging_dict()
            out_dict = self._imaging_dict
        else:
            out_dict = existing_dict

        # Loop over the lines
        lines_read = 0
        while True:
            line = infile.readline()
            if len(line) == 0:
                break
            if key_readers.skip_line(line, expected_words=expected_words,
                                     delim=delim, expected_format=expected_format):
                continue

            this_config, this_product, this_stage, this_recipe = \
                key_readers.parse_one_line(line, delim=delim)

            if this_config.lower() != 'all':
                if this_config not in self.get_interf_configs():
                    logger.warning("Config not recognized. Line is:")
                    logger.warning(line)
                    continue

            if this_product.lower() != 'all_line' and this_product.lower() != 'all_cont':
                if (this_product not in self.get_line_products()) and \
                        (this_product not in self.get_continuum_products()):
                    logger.warning("Product not recognized. Line is:")
                    logger.warning(line)
                    continue

            if this_stage.lower() != 'all' and this_stage.lower() not in VALID_IMAGING_STAGES:
                logger.warning("Imaging stage not recognized. Line is:")
                logger.warning(line)
                continue

            # Just to make the last bit clean, force everything into a list

            if this_config.lower() == 'all':
                config_list = self.get_interf_configs()
            else:
                config_list = [this_config]

            if this_product.lower() == 'all_line':
                product_list = self.get_line_products()
            elif this_product.lower() == 'all_cont':
                product_list = self.get_continuum_products()
            else:
                product_list = [this_product]

            if this_stage == 'all':
                stage_list = VALID_IMAGING_STAGES
            else:
                stage_list = [this_stage]

            for each_config in config_list:
                for each_product in product_list:
                    for each_stage in stage_list:
                        out_dict[each_config][each_product][each_stage] = this_recipe
                        # out_dict[each_config][each_product][each_stage] = this_recipe
                        # logger.debug('self._imaging_dict[%r][%r][%r] = %r'%(each_config, each_product, each_stage, this_recipe))

            lines_read += 1

        self._imaging_dict = out_dict

        infile.close()

        logger.info("Read " + str(lines_read) + " lines into the imaging recipe dictionary.")

        return out_dict

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Define routines to read the derived product dictionary
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # The derived product dictionary requires outside knowledge, e.g.,
    # because it uses wild cards and involved cross-indexing.

    def _initialize_derived_dict(self):
        """
        Initialize the derive dictionary. Requires the config and
        products dict to exist already to work successfully.
        """

        logger.info("Initializing the derived dictionary using known configs and products.")

        full_dict = {}

        # Loop over configs
        for this_config in self.get_all_configs():
            full_dict[this_config] = {}

            # Loop over line products
            for this_product in self.get_line_products():
                full_dict[this_config][this_product] = {
                    'phys_res': {},
                    'ang_res': {},
                    'mask_configs': [],
                    'moments': [],
                }

            # Loop over continuum products
            for this_product in self.get_continuum_products():
                full_dict[this_config][this_product] = {}
                full_dict[this_config][this_product] = {
                    'phys_res': {},
                    'ang_res': {},
                    'mask_configs': [],
                    'moments': [],
                }

        self._derived_dict = full_dict

        return ()

    def _read_derived_key(self, fname='', existing_dict=None, delim=None):
        """
        Read a file that defines the calculation of derived products.
        """

        # Check file existence

        if os.path.isfile(fname) is False:
            logger.error("I tried to read key " + fname + " but it does not exist.")
            return existing_dict

        logger.info("Reading: " + fname)

        # Expected Format

        expected_words = 4
        expected_format = "config product param value"

        # Known parameters

        known_param_list = ['mask_configs', 'ang_res', 'phys_res',
                            'noise_kw', 'strictmask_kw', 'broadmask_kw',
                            'flatstrictmask_kw', 'flatbroadmask_kw',
                            'convolve_kw', 'moments', 'shuffle_kw',
                            ]

        # Open File

        infile = open(fname, 'r')

        # Initialize the dictionary

        if self._derived_dict is None:
            self._initialize_derived_dict()
        out_dict = self._derived_dict

        # Loop over the lines
        lines_read = 0
        while True:
            line = infile.readline()
            if len(line) == 0:
                break

            if key_readers.skip_line(line, expected_words=expected_words,
                                     delim=delim, expected_format=expected_format):
                continue

            this_config, this_product, this_param, this_value = \
                key_readers.parse_one_line(line, delim=delim)

            if this_param.lower() not in known_param_list:
                logger.warning("Parameter not in known parameter list. Skipping. Line is:")
                logger.warning(line)
                continue

            if this_config.lower() != 'all':
                if this_config not in self.get_all_configs():
                    logger.warning("Config not recognized. Line is:")
                    logger.warning(line)
                    continue

            if this_product.lower() != 'all_line' and \
                    this_product.lower() != 'all_cont' and \
                    this_product.lower() != 'all':
                if (this_product not in self.get_line_products()) and \
                        (this_product not in self.get_continuum_products()):
                    logger.warning("Spectral product not recognized. Line is:")
                    logger.warning(line)
                    continue

            # Force configs and products into a list format

            if this_config.lower() == 'all':
                config_list = self.get_all_configs()
            else:
                config_list = [this_config]

            if this_product.lower() == 'all_line':
                product_list = self.get_line_products()
            elif this_product.lower() == 'all_cont':
                product_list = self.get_continuum_products()
            elif this_product.lower() == 'all':
                product_list = self.get_line_products() + self.get_continuum_products()
            else:
                product_list = [this_product]

            # Read in the read data

            for each_config in config_list:
                for each_product in product_list:

                    if this_param.lower() == 'phys_res':
                        this_res_dict = ast.literal_eval(this_value)
                        if not isinstance(this_res_dict, dict):
                            logger.warning("Format of res string not a dictionary. Line is:")
                            logger.warning(line)
                            continue
                        for res_tag in this_res_dict.keys():
                            # EWR : error checking / sanity here comparing string to numerical value.

                            out_dict[each_config][each_product]['phys_res'][res_tag] = this_res_dict[res_tag]

                    if this_param.lower() == 'ang_res':
                        this_res_dict = ast.literal_eval(this_value)
                        if not isinstance(this_res_dict, dict):
                            logger.warning("Format of res string not a dictionary. Line is:")
                            logger.warning(line)
                        for res_tag in this_res_dict.keys():
                            # EWR : error checking / sanity here comparing string to numerical value.

                            out_dict[each_config][each_product]['ang_res'][res_tag] = this_res_dict[res_tag]

                    if this_param.lower() == 'mask_configs':
                        this_mask_list = ast.literal_eval(this_value)
                        if type(this_mask_list) != type([]):
                            logger.warning("Format of mask configs not a list. Line is:")
                            logger.warning(line)
                        for cross_config in this_mask_list:
                            current_list = out_dict[each_config][each_product]['mask_configs']
                            if cross_config not in current_list:
                                current_list.append(cross_config)
                                out_dict[each_config][each_product]['mask_configs'] = current_list

                    if this_param.lower() == 'moments':
                        this_moment_list = ast.literal_eval(this_value)
                        if type(this_moment_list) != type([]):
                            logger.warning("Format of moments not a list. Line is:")
                            logger.warning(line)
                        for this_moment in this_moment_list:
                            current_list = out_dict[each_config][each_product]['moments']
                            if this_moment not in current_list:
                                current_list.append(this_moment)
                                out_dict[each_config][each_product]['moments'] = current_list

                    # Keywords for masking, noise

                    for valid_dict in ['strictmask_kw', 'broadmask_kw', 'noise_kw', 'convolve_kw', \
                                       'shuffle_kw', 'flatstrictmask_kw', 'flatbroadmask_kw']:
                        if this_param.lower().strip() != valid_dict:
                            continue
                        this_kw_dict = ast.literal_eval(this_value)
                        if not isinstance(this_kw_dict, dict):
                            logger.warning("Format of mask keywords is a dict. Line is:")
                            logger.warning(line)
                            continue

                        if valid_dict not in out_dict[each_config][each_product].keys():
                            out_dict[each_config][each_product][valid_dict] = {}

                        for this_tag in this_kw_dict.keys():
                            out_dict[each_config][each_product][valid_dict][this_tag] = \
                                this_kw_dict[this_tag]

            lines_read += 1

        # Close and return

        infile.close()

        logger.info("Read " + str(lines_read) + " lines into a derived product definition dictionary.")

        return out_dict

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Batch read the other keys.
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # Mostly these wrap around the programs in utilsKeyReaders , which
    # parse individual key files into dictionaries.

    def _read_all_keys(self):
        """
        Read the other keys into dictionary attributes.
        """

        self._casaversion_dict = key_readers.batch_read(
            key_list=self._casaversion_keys, reader_function=key_readers.read_casaversion_key,
            key_dir=self._key_dir)

        self._alma_download_dict = key_readers.batch_read(
            key_list=self._alma_download_keys, reader_function=key_readers.read_alma_download_key,
            key_dir=self._key_dir)

        self._ms_dict = key_readers.batch_read(
            key_list=self._ms_keys, reader_function=key_readers.read_ms_key,
            key_dir=self._key_dir)

        self._cleanmask_dict = key_readers.batch_read(
            key_list=self._cleanmask_keys, reader_function=key_readers.read_cleanmask_key,
            key_dir=self._key_dir)

        self._sd_dict = key_readers.batch_read(
            key_list=self._sd_keys, reader_function=key_readers.read_singledish_key,
            key_dir=self._key_dir)

        self._config_dict = key_readers.batch_read(
            key_list=self._config_keys, reader_function=key_readers.read_config_key,
            key_dir=self._key_dir)

        self._target_dict = key_readers.batch_read(
            key_list=self._target_keys, reader_function=key_readers.read_target_key,
            key_dir=self._key_dir)

        self._linmos_dict = key_readers.batch_read(
            key_list=self._linmos_keys, reader_function=key_readers.read_linmos_key,
            key_dir=self._key_dir)

        self._dir_for_target = key_readers.batch_read(
            key_list=self._dir_keys, reader_function=key_readers.read_dir_key,
            key_dir=self._key_dir)

        self._initialize_imaging_dict()
        self._imaging_dict = key_readers.batch_read(
            key_list=self._imaging_keys, reader_function=self._read_imaging_key,
            key_dir=self._key_dir, existing_dict=self._imaging_dict)

        self._initialize_derived_dict()
        self._derived_dict = key_readers.batch_read(
            key_list=self._derived_keys, reader_function=self._read_derived_key,
            key_dir=self._key_dir, existing_dict=self._derived_dict)

        self._override_dict = key_readers.batch_read(
            key_list=self._override_keys, reader_function=key_readers.read_override_key,
            key_dir=self._key_dir)

        self._distance_dict = key_readers.batch_read(
            key_list=self._distance_keys, reader_function=key_readers.read_distance_key,
            key_dir=self._key_dir)
        
        self._window_dict = key_readers.batch_read(
            key_list=self._window_keys, reader_function=key_readers.read_window_key,
            key_dir=self._key_dir)

        self._moment_dict = key_readers.batch_read(
            key_list=self._moment_keys, reader_function=key_readers.read_moment_key,
            key_dir=self._key_dir)

    ##############################################################
    # LINKING AND KEY BUILDING
    ##############################################################

    def _build_target_list(self, check=True):
        """
        After the keys have been read in, create a complete target
        list and check for inconsistencies or missing entries.
        """

        logger.info("Building the target list.")

        if self._target_dict is None:
            logger.error("I don't have a target dictionary. Can't proceed.")
            return

        self._target_list = list(self._target_dict.keys())
        self._target_list = sorted(self._target_list)

        self._missing_targets = []

        missing_targets = []

        if self._ms_dict is not None:
            ms_targets = self._ms_dict.keys()
            for target in ms_targets:
                if target not in self._target_list:
                    # logger.error(target+ " is in the measurement set key but not the target list.")
                    if target not in missing_targets:
                        missing_targets.append(target)

        if self._dir_for_target is not None:
            dir_targets = self._dir_for_target.keys()
            for target in dir_targets:
                if target not in self._target_list:
                    # logger.error(target+ " is in the directory key but not the target list.")
                    if target not in missing_targets:
                        missing_targets.append(target)

        if self._sd_dict is not None:
            sd_targets = self._sd_dict.keys()
            for target in sd_targets:
                if target not in self._target_list:
                    # logger.error(target+ " is in the single dish key but not the target list.")
                    if target not in missing_targets:
                        missing_targets.append(target)

        if self._linmos_dict is not None:
            linmos_targets = self._linmos_dict.keys()
            for target in linmos_targets:
                if target not in self._target_list:
                    # logger.error(target+ " is in the linear mosaic key but not the target list.")
                    if target not in missing_targets:
                        missing_targets.append(target)

        if self._distance_dict is not None:
            distance_targets = self._distance_dict.keys()
            for target in distance_targets:
                if target not in self._target_list:
                    # logger.error(target+ " is in the distance key but not the target list.")
                    if target not in missing_targets:
                        missing_targets.append(target)

        if self._window_dict is not None:
            window_targets = self._window_dict.keys()
            for target in window_targets:
                if target not in self._target_list:
                    logger.error(target+ " is in the velocity window key but not the target list.")
                    if target not in missing_targets:
                        missing_targets.append(target)

        self._missing_targets = missing_targets

        logger.info("Total of " + str(len(self._target_list)) + " targets.")
        n_missing = len(self._missing_targets)
        if n_missing == 0:
            logger.info("No cases found where I expect a target but lack a definition.")
        else:
            logger.warning(str(n_missing) + " cases where I expected a target definition but didn't find one.")

        return ()

    def _build_whole_target_list(self):
        """
        Create a list where the parts that will go into a linear
        mosaic are gone and only whole targets remain.
        """

        logger.info("Building the list of whole targets.")

        if self._target_dict is None:
            logger.error("I don't have a target dictionary. Can't proceed.")
            return ()

        if self._target_list is None:
            self._build_target_list()

        self._whole_target_list = list(self._target_list)

        if self._linmos_dict is None:
            return ()

        linmos_targets = self._linmos_dict.keys()
        for this_mosaic in linmos_targets:
            if this_mosaic not in self._whole_target_list:
                self._whole_target_list.append(this_mosaic)
            for this_part in self._linmos_dict[this_mosaic]:
                if this_part in self._whole_target_list:
                    self._whole_target_list.remove(this_part)

        self._whole_target_list = sorted(self._whole_target_list)

        logger.info("Total of " + str(len(self._whole_target_list)) + " 'whole' targets.")

        return ()

    def _map_targets_to_mosaics(self):
        """
        Create a dictionary that maps from individual targets to mosaics.
        """

        logger.info("Mapping targets to linear mosaics.")

        if self._target_dict is None:
            logger.error("I don't have a target dictionary. Can't proceed.")
            return ()

        if self._target_list is None:
            self._build_target_list()

        self._mosaic_assign_dict = {}

        if self._linmos_dict is None:
            return ()

        linmos_targets = self._linmos_dict.keys()
        for this_mosaic in linmos_targets:
            part_list = self._linmos_dict[this_mosaic]
            for this_part in part_list:
                self._mosaic_assign_dict[this_part] = this_mosaic

        logger.info("Total of " + str(len(self._mosaic_assign_dict.keys())) + \
                    " targets assigned to mosaics.")

        return ()

    def _map_configs(self):
        """
        Map interferferometric and feather configurations to one another.
        """

        logger.info("Cross-matching interferometric and feather configs")

        if 'interf_config' not in self._config_dict.keys():
            return ()

        # Initialize
        for interf_config in self._config_dict['interf_config'].keys():

            self._config_dict['interf_config'][interf_config]['feather_config'] = None

            if 'feather_config' not in self._config_dict.keys():
                continue

            for feather_config in self._config_dict['feather_config'].keys():
                if self._config_dict['feather_config'][feather_config]['interf_config'] != interf_config:
                    continue
                self._config_dict['interf_config'][interf_config]['feather_config'] = feather_config

        return ()

    ##############################################################
    # Programs to run checks on the keyHandler and the data
    ##############################################################

    def set_dochecks(self, dochecks=True):
        """

        """
        self._dochecks = dochecks
        return ()

    def print_missing_targets(self):
        """
        Print the targets missing a definition in the target list key.
        """

        if len(self._target_list) == 0:
            self._build_target_list()

        logger.info("-------------------------")
        logger.info("Printing missing targets.")
        logger.info("-------------------------")

        if len(self._missing_targets) == 0:
            logger.info("No missing targets.")
            return None

        logger.warning("Missing a total of " + str(
            len(self._missing_targets)) + " target definitions.")  # missing sources in the target_definitions.txt
        for target in self._missing_targets:
            logger.error("missing: " + target)

        return ()

    def check_ms_existence(self):
        """
        Check that the measurement sets in the ms key all exist in the
        specified directories.
        """

        logger.info("-------------------------------------------")
        logger.info("Checking the existence of measurement sets.")
        logger.info("-------------------------------------------")

        if self._ms_dict is None:
            return ()

        found_count = 0
        missing_count = 0
        for target in self._ms_dict.keys():
            for project_tag in self._ms_dict[target].keys():
                for array_tag in self._ms_dict[target][project_tag].keys():
                    for obs_tag in self._ms_dict[target][project_tag][array_tag].keys():
                        found = False
                        local_found_count = 0
                        for ms_root in self._ms_roots:
                            this_ms = ms_root + self._ms_dict[target][project_tag][array_tag][obs_tag]['file']
                            if os.path.isdir(this_ms):
                                found = True
                                found_count += 1
                                local_found_count += 1
                            if local_found_count > 1:
                                logger.error(
                                    "Found multiple copies of ms for " + target + " " + project_tag + " " + array_tag)
                        if found:
                            continue
                        missing_count += 1
                        logger.error("Missing ms for " + target + " " + project_tag + " " + array_tag)

        logger.info("Verified the existence of " + str(found_count) + " measurement sets.")
        if missing_count == 0:
            logger.info("No measurement sets found to be missing.")
        else:
            logger.error("Missing " + str(missing_count) + " measurement set key entries.")

        return ()

    def check_sd_existence(self):
        """
        Check that the FITS files in the singledish key all exist in
        the specified directories.
        """

        logger.info("-------------------------------------------")
        logger.info("Checking the existence of single dish data.")
        logger.info("-------------------------------------------")

        if self._sd_dict is None:
            return ()

        found_count = 0
        missing_count = 0
        for target in self._sd_dict.keys():
            for product in self._sd_dict[target].keys():
                found = False
                local_found_count = 0
                for this_root in self._sd_roots:
                    this_fname = this_root + self._sd_dict[target][product]
                    if os.path.isfile(this_fname):
                        found = True
                        found_count += 1
                        local_found_count += 1
                if local_found_count > 1:
                    logger.error("Found multiple copies of singledish data for " + target + " " + product)
                if found:
                    continue
                missing_count += 1
                logger.warning("Missing singledish data for " + target + " " + product)

        logger.info("Verified the existence of " + str(found_count) + " single dish data sets.")
        if missing_count == 0:
            logger.info("No single dish data found to be missing.")
        else:
            logger.warning("Missing " + str(missing_count) + " single dish key entries.")

        return ()

    def _expand_dir_key(self):
        """
        After the keys have been read in, work out the
        directory/target mapping in more detail.
        """

        if self._target_list is None:
            self._build_target_list()

        for target in self._target_list:
            if target in self._dir_for_target.keys():
                continue
            self._dir_for_target[target] = target

        self._targets_for_dir = {}
        for target in self._target_list:
            this_dir = self._dir_for_target[target]
            if this_dir in self._targets_for_dir.keys():
                current_targets = self._targets_for_dir[this_dir]
                if target in current_targets:
                    continue
                current_targets.append(target)
                current_targets = sorted(current_targets)
                self._targets_for_dir[this_dir] = current_targets
            else:
                self._targets_for_dir[this_dir] = [target]

        return ()

    def check_dir_existence(self, imaging=True, postprocess=True, derived=True, release=True):
        """
        Check the existence of the directories for imaging and post-processing.
        """

        logger.info("--------------------------------------")
        logger.info("Checking the existence of directories.")
        logger.info("--------------------------------------")

        dir_list = self._targets_for_dir.keys()

        found_dirs = 0
        missing_dirs = []
        for this_dir in dir_list:

            if imaging:
                if os.path.isdir(self._imaging_root + this_dir):
                    found_dirs += 1
                else:
                    logger.warning("Missing imaging directory: " + self._imaging_root + this_dir)
                    missing_dirs.append(self._imaging_root + this_dir)

            if postprocess:
                if os.path.isdir(self._postprocess_root + this_dir):
                    found_dirs += 1
                else:
                    logging.warning("Missing post-processing directory: " + self._postprocess_root + this_dir)
                    missing_dirs.append(self._postprocess_root + this_dir)

            if derived:
                if os.path.isdir(self._derived_root + this_dir):
                    found_dirs += 1
                else:
                    logging.warning("Missing derived directory: " + self._derived_root + this_dir)
                    missing_dirs.append(self._derived_root + this_dir)

            if release:
                if os.path.isdir(self._release_root + this_dir):
                    found_dirs += 1
                else:
                    logging.warning("Missing release directory: " + self._release_root + this_dir)
                    missing_dirs.append(self._release_root + this_dir)

        logging.info("Found " + str(found_dirs) + " directories.")

        missing_count = (len(missing_dirs))
        if missing_count == 0:
            logger.info("No directories appear to be missing.")
        else:
            logger.warning("Missing " + str(missing_count) + " directories. Returning that list.")

        return missing_dirs

    ##############################################################
    # Access properties and key information.
    ##############################################################

    def _get_dir_for_target(self, target=None, changeto=False,
                            imaging=False, postprocess=False,
                            derived=False, release=False,
                            cleanmask=False, singledish=False,
                            vfield=False):
        """
        Return the imaging working directory given a target name. If
        changeto is true, then change directory to that location.
        """

        if target == None:
            logging.error("Please specify a target.")
            return None

        if target not in self._dir_for_target.keys():
            logging.error("Target " + target + " is not in the list of targets.")
            return None

        if (imaging and postprocess) or (imaging and derived) or \
                (derived and postprocess):
            logging.error("Multiple flags set, pick only one type of directory.")
            return None

        if (release and imaging) or (release and derived) or \
                (release and postprocess):
            logging.error("Multiple flags set, pick only one type of directory.")
            return None

        if imaging:
            this_dir = self._imaging_root + self._dir_for_target[target] + '/'
        elif postprocess:
            this_dir = self._postprocess_root + self._dir_for_target[target] + '/'
        elif derived:
            this_dir = self._derived_root + self._dir_for_target[target] + '/'
        elif release:
            this_dir = self._release_root + self._dir_for_target[target] + '/'
        elif cleanmask:
            this_dir = self._cleanmask_roots[-1] + self._dir_for_target[target] + '/'
        elif singledish:
            this_dir = self._sd_roots[-1] + self._dir_for_target[target] + '/'
        elif vfield:
            this_dir = self._vfield_root + '/'
        else:
            logging.error("Pick a type of directory. None found.")
            return None

        if changeto:
            if not os.path.isdir(this_dir):
                logging.error("You requested to change to " + this_dir + " but it does not exist.")
                return this_dir
            os.chdir(this_dir)
            return this_dir
        return this_dir

    def get_imaging_dir_for_target(self, target=None, changeto=False):
        """
        Return the imaging working directory given a target name. If
        changeto is true, then change directory to that location.
        """
        return self._get_dir_for_target(target=target, changeto=changeto, imaging=True)

    def get_postprocess_dir_for_target(self, target=None, changeto=False):
        """
        Return the postprocessing working directory given a target name. If
        changeto is true, then change directory to that location.
        """
        return self._get_dir_for_target(target=target, changeto=changeto, postprocess=True)

    def get_derived_dir_for_target(self, target=None, changeto=False):
        """
        Return the derived working directory given a target name. If
        changeto is true, then change directory to that location.
        """
        return self._get_dir_for_target(target=target, changeto=changeto, derived=True)

    def get_release_dir_for_target(self, target=None, changeto=False):
        """
        Return the release working directory given a target name. If
        changeto is true, then change directory to that location.
        """
        return self._get_dir_for_target(target=target, changeto=changeto, release=True)

    def get_vfield_dir_for_target(self, target=None, changeto=False):
        """
        Return the vfield working directory given a target name. If
        changeto is true, then change directory to that location.
        """
        return self._get_dir_for_target(target=target, changeto=changeto, vfield=True)
    
    def get_cleanmask_dir_for_target(self, target=None, changeto=False):
        """
        Return the release working directory given a target name. If
        changeto is true, then change directory to that location.
        """
        return self._get_dir_for_target(target=target, changeto=changeto, cleanmask=True)

    def get_singledish_dir_for_target(self, target=None, changeto=False):
        """
        Return the release working directory given a target name. If
        changeto is true, then change directory to that location.
        """
        return self._get_dir_for_target(target=target, changeto=changeto, singledish=True)

    def get_path_for_casaversion(self, casa_version=None):
        """
        Get the CASA path for a given version of CASA.
        """
        if casa_version is None:
            return None
        try:
            # Preferentially search an exact match
            casa_path = self._casaversion_dict[casa_version]
            return casa_path
        except KeyError:

            # Loop over, trying to find CASA version without the -XX version (i.e. 5.4.0 instead of 5.4.0-70)
            for key in self._casaversion_dict.keys():
                if key.split('-')[0] == casa_version.split('-')[0]:
                    return self._casaversion_dict[key]

            # Give up if we don't find anything
            return None

    def get_targets(self, only=None, skip=None, first=None, last=None):
        """
        List the full set of targets.

        Modified by keywords only, skip, first, last. Will only return
        targets in only, skip targets in skip, and return targets
        alphabetically after first and before last.
        """
        this_target_list = \
            list_utils.select_from_list(self._target_list, first=first, last=last,
                                        skip=skip, only=only, loose=True)
        return this_target_list

    def get_targets_in_ms_key(self, only=None, skip=None, first=None, last=None):
        """
        List all targets that have uv data.

        Modified by keywords only, skip, first, last. Will only return
        targets in only, skip targets in skip, and return targets
        alphabetically after first and before last.
        """
        targets = self._target_list
        targets_with_ms = []
        for target in targets:
            if target in self._ms_dict.keys():
                targets_with_ms.append(target)

        this_target_list = \
            list_utils.select_from_list(targets_with_ms, first=first, last=last,
                                        skip=skip, only=only, loose=True)
        return this_target_list

    def get_linmos_targets(self, only=None, skip=None, first=None, last=None):
        """
        List all linear mosaics.

        Modified by keywords only, skip, first, last. Will only return
        targets in only, skip targets in skip, and return targets
        alphabetically after first and before last.
        """
        if self._linmos_dict is None:
            return None
        mosaic_targets = list(self._linmos_dict.keys())

        this_target_list = \
            list_utils.select_from_list(mosaic_targets, first=first, last=last,
                                        skip=skip, only=only, loose=True)
        return this_target_list

    def get_whole_targets(self, only=None, skip=None, first=None, last=None):
        """
        List only full galaxy names (no parts, e.g., _1 or _2). Very
        similar to the directory list.

        Modified by keywords only, skip, first, last. Will only return
        targets in only, skip targets in skip, and return targets
        alphabetically after first and before last.
        """
        this_whole_target_list = \
            list_utils.select_from_list(self._whole_target_list, first=first, last=last,
                                        skip=skip, only=only, loose=True)
        return this_whole_target_list

    def get_alma_download_restrictions(self, target=None, product=None, config=None):
        """
        Get any download restrictions for ALMA data
        """
        if target is None:
            logging.error("Please specify a target.")
            raise Exception("Please specify a target.")
            return None
        if product is None:
            logging.error("Please specify a product.")
            raise Exception("Please specify a product.")
            return None
        if config is None:
            logging.error("Please specify a config.")
            raise Exception("Please specify a config.")
            return None

        restrictions_dict = None

        # Drill down through the dictionary to get at any restrictions. Preferentially match, but fall back to 'all'
        # options if not explicitly found

        all_targets = [target, 'all']  # TODO: MAYBE THIS WILL WORK WHO KNOWS. SEE HOW PRIORITY WORKS
        all_products = [product, 'all']
        all_configs = [config, 'all']

        dict_found = False

        for target in all_targets:
            for product in all_products:
                for config in all_configs:
                    try:
                        restrictions_dict = self._alma_download_dict[target][product][config]
                        dict_found = True
                        break
                    except KeyError:
                        pass

                if dict_found:
                    break
            if dict_found:
                break

        return restrictions_dict

    def get_interf_configs(self, only=None, skip=None):
        """
        Get a list of interferometer array configurations

        Modified by keywords only, skip. Will only return targets in
        only, skip targets in skip.
        """
        if not isinstance(self._config_dict, dict):
            return None
        if 'interf_config' not in self._config_dict.keys():
            return None
        interf_configs = self._config_dict['interf_config'].keys()
        this_list = \
            list_utils.select_from_list(interf_configs, skip=skip, only=only, loose=True)
        return this_list

    def get_feather_configs(self, only=None, skip=None):
        """
        Get a list of feathered single dish + interferometer array
        configruations.

        Modified by keywords only, skip. Will only return targets in
        only, skip targets in skip.
        """
        if not isinstance(self._config_dict, dict):
            return None
        if 'feather_config' not in self._config_dict.keys():
            return None
        feather_configs = self._config_dict['feather_config'].keys()
        this_list = \
            list_utils.select_from_list(feather_configs, skip=skip, only=only, loose=True)
        return this_list

    def get_singledish_configs(self, only=None, skip=None):
        """
        Get a list of singledish configruations.

        Modified by keywords only, skip. Will only return targets in
        only, skip targets in skip.
        """
        if not isinstance(self._config_dict, dict):
            return None
        if 'singledish_config' not in self._config_dict.keys():
            return None
        singledish_configs = self._config_dict['singledish_config'].keys()
        this_list = \
            list_utils.select_from_list(singledish_configs, skip=skip, only=only, loose=True)
        return this_list

    def get_all_configs(self):
        """
        Get a list of all interf and feather configs.
        """
        interf_config = self.get_interf_configs()
        feather_config = self.get_feather_configs()
        all_configs = []
        if interf_config is not None:
            all_configs.extend(interf_config)
        if feather_config is not None:
            all_configs.extend(feather_config)
        if len(all_configs) == 0:
            all_configs = None
        return all_configs

    def get_line_products(self, only=None, skip=None):
        """
        Get a list of line 'products,' i.e., line plus velocity
        resolution combinations.

        Modified by keywords only, skip. Will only return targets in
        only, skip targets in skip.
        """
        if not isinstance(self._config_dict, dict):
            return []
        if 'line_product' not in self._config_dict.keys():
            return []
        line_products = self._config_dict['line_product'].keys()
        this_list = \
            list_utils.select_from_list(line_products, skip=skip, only=only, loose=True)
        return this_list

    def get_continuum_products(self, only=None, skip=None):
        """
        Get a list of continuum 'products'.

        Modified by keywords only, skip. Will only return targets in
        only, skip targets in skip.
        """
        if not isinstance(self._config_dict, dict):
            return []
        if 'cont_product' not in self._config_dict.keys():
            return []
        cont_products = self._config_dict['cont_product'].keys()
        this_list = \
            list_utils.select_from_list(cont_products, skip=skip, only=only, loose=True)
        return this_list

    def get_all_mosaic_targets(self):
        """Get all mosaic targets defined in "linearmosaic_definitions.txt" which have parts.
        """
        if self._linmos_dict is not None:
            if len(self._linmos_dict) > 0:
                return sorted(list(self._linmos_dict.keys()))
        return None

    def get_all_targets(self):
        """Get all targets defined in "target_definitions.txt", including mosaic targets and their parts and non mosaic targets.
        """
        if self._target_list is not None:
            if len(self._target_list) > 0:
                return sorted(list(set(self._target_list)))
        return None

    def get_all_non_mosaic_targets(self):
        """Get all targets which are not mosaic targets.
        """
        all_mosaic_targets = self.get_all_mosaic_targets()
        all_targets = self.get_all_targets()
        if all_targets is not None:
            if all_mosaic_targets is not None:
                return [t for t in all_targets if (t not in all_mosaic_targets)]
            else:
                return all_targets
        return None

    def get_mosaic_target_for_parts(self, target_part_name=None):
        """Get mosaic target name "ngc4321" given a part name like "ngc4321_1". This is the inverted operation of `get_parts_for_linmos`.

        This is also the same as `is_target_in_mosaic(target_part_name, return_target_name=True)`.
        """
        if target_part_name is None:
            return None
        if self._linmos_dict is not None:
            if len(self._linmos_dict) > 0:
                if self._mosaic_assign_dict is None or len(self._mosaic_assign_dict) == 0:
                    self._map_targets_to_mosaics()
                if len(self._mosaic_assign_dict) > 0:
                    if target_part_name in self._mosaic_assign_dict.keys():
                        return self._mosaic_assign_dict[target_part_name]
        return None

    def is_target_linmos(self, target=None):
        """
        Return True or False based on whether the target is a linear
        mosaic. True means that this target is the OUTPUT of a linear
        mosaic operation.
        """
        if target == None:
            return False

        if self._linmos_dict is None:
            logging.error("No linear mosaic dictionary defined.")
            return False

        if target in self._linmos_dict.keys():
            return True

        return False

    def get_parts_for_linmos(self, target=None):
        """
        Return the parts for a linear mosaic.
        """
        if target == None:
            logging.error("Please specify a target.")
            return None

        if self._linmos_dict is None:
            logging.error("No linear mosaic dictionary defined.")
            return None

        if target not in self._linmos_dict.keys():
            logging.error("No linear mosaic defined for " + target)
            return None

        return self._linmos_dict[target]

    def is_target_in_mosaic(self, target, return_target_name=False):
        """
        Return true or false depending on whether the target is in a linear mosaic.
        """
        if target in self._mosaic_assign_dict.keys():
            if return_target_name:
                return True, self._mosaic_assign_dict[target]
            else:
                return True
        else:
            if return_target_name:
                return False, target
            else:
                return False

    def get_imaging_recipes(self, config=None, product=None, stage=None):
        """
        Return the imaging recipe for the input config and product.
        """
        if config is None or product is None:
            return None
        if config in self._imaging_dict:
            if product in self._imaging_dict[config]:
                imaging_recipe_dir = self._key_dir  # <TODO><DL># imaging_recipe_dir
                if not imaging_recipe_dir.endswith(os.sep):
                    imaging_recipe_dir += os.sep
                if stage is not None:
                    if stage in self._imaging_dict[config][product]:
                        imaging_recipe_file = self._imaging_dict[config][product][stage]
                        if imaging_recipe_file is None:
                            logger.error(
                                'No imaging recipe is defined for config ' + config + ' product ' + product + ' and imaging stage ' + stage + '. Please check your "imaging_recipes.txt"!')
                            raise Exception(
                                'No imaging recipe is defined for config ' + config + ' product ' + product + ' and imaging stage ' + stage + '. Please check your "imaging_recipes.txt"!')
                        return imaging_recipe_dir + self._imaging_dict[config][product][
                            stage]  # <TODO><DL># multiple recipes for one stage?
                else:
                    for t in VALID_IMAGING_STAGES:
                        imaging_recipe_file = self._imaging_dict[config][product][t]
                        if imaging_recipe_file is None:
                            logger.error(
                                'No imaging recipe is defined for config ' + config + ' product ' + product + ' and imaging stage ' + t + '. Please check your "imaging_recipes.txt"!')
                            raise Exception(
                                'No imaging recipe is defined for config ' + config + ' product ' + product + ' and imaging stage ' + t + '. Please check your "imaging_recipes.txt"!')
                    return [imaging_recipe_dir + self._imaging_dict[config][product][t] for t in VALID_IMAGING_STAGES]
        return None

    def get_distance_for_target(self, target=None):
        """
        Get the distance (in Mpc) associated with a target. If the
        target is part of a mosaic, return the distance to the whole
        galaxy.
        """
        if target is None:
            logging.error("Please specify a target.")
            raise Exception("Please specify a target.")

        _, target_name = self.is_target_in_mosaic(target, return_target_name=True)

        distance = None

        if self._distance_dict is not None:

            if target_name in self._distance_dict:
                if 'distance' in self._distance_dict[target_name]:
                    distance = self._distance_dict[target_name]['distance']

        return distance
    
    def get_window_for_target(self, target=None):
        """
        Get the velocity window (in km/s) associated with a target. If the
        target is part of a mosaic, return the velocity window to the whole
        galaxy.
        """
        if target is None:
            logging.error("Please specify a target.")
            raise Exception("Please specify a target.")

        _, target_name = self.is_target_in_mosaic(target, return_target_name=True)

        window = None

        if self._window_dict is not None:

            if target_name in self._window_dict:
                if 'window' in self._window_dict[target_name]:
                    window = self._window_dict[target_name]['window']

        return window

    def get_vfield_file_for_target(self, target=None):
        """
        Get the velocity field file associated with a target. If the
        target is part of a mosaic, return the velocity field file for the whole
        galaxy.
        """
        if target is None:
            logging.error("Please specify a target.")
            raise Exception("Please specify a target.")

        _, target_name = self.is_target_in_mosaic(target, return_target_name=True)

        vfield_file = None

        if self._window_dict is not None:

            if target_name in self._window_dict:
                if 'vfield_file' in self._window_dict[target_name]:
                    vfield_file = self._window_dict[target_name]['vfield_file']

        return vfield_file

    def get_system_velocity_and_velocity_width_for_target(
            self,
            target=None,
            check_parent=False,
    ):
        """

        Inputs
        ------
        check_parent: bool, optional
            If part of a linear mosaic, return the vsys, vwidth of the parent
            target.
        """
        if target is None:
            logging.error("Please specify a target.")
            raise Exception("Please specify a target.")

        target_vsys = None
        target_vwidth = None
        if target in self._target_dict:
            if 'vsys' in self._target_dict[target] and 'vwidth' in self._target_dict[target]:
                target_vsys = self._target_dict[target]['vsys']
                target_vwidth = self._target_dict[target]['vwidth']

                # If this is part of a linear mosaic, check that vwidth is
                # equivalent so the full line extent is used for all chunks.
                if check_parent and target in self._mosaic_assign_dict:
                    parent = self._mosaic_assign_dict[target]
                    parent_vsys = self._target_dict[parent]['vsys']
                    parent_vwidth = self._target_dict[parent]['vwidth']

                    target_vsys = parent_vsys
                    target_vwidth = parent_vwidth

                    logger.warning("... Found parent target {}. Using parent ".format(parent) +
                                   "vsys, vwidth for {}".format(target))

        if target_vsys is None or target_vwidth is None:
            logging.error(
                'No vsys or vwidth values set for the target ' + target + '. Please check your "target_definitions.txt".')
            raise Exception('No vsys or vwidth values set for the target ' + target)

        return target_vsys, target_vwidth

    def get_phasecenter_for_target(
            self,
            target=None,
    ):
        """
        Return the strings (ra, dec) that define the phase center for
        a target in the target dictionary.
        """
        if target is None:
            logging.error("Please specify a target.")
            raise Exception("Please specify a target.")

        rastring = None
        decstring = None
        if target in self._target_dict:
            if 'rastring' in self._target_dict[target]:
                rastring = self._target_dict[target]['rastring']
            if 'decstring' in self._target_dict[target]:
                decstring = self._target_dict[target]['decstring']

        if rastring is None or decstring is None:
            logging.error('Missing phase center for target ' + target)
            raise Exception('Missing phase center for target ' + target)

        return rastring, decstring

    def get_channel_width_for_line_product(self, product=None):
        """
        Get the channel width (in km/s) associated with a line
        product.
        """
        if product is None:
            logging.error("Please specify a product.")
            raise Exception("Please specify a product.")

        channel_kms = None
        if 'line_product' in self._config_dict:
            if product in self._config_dict['line_product']:
                if 'channel_kms' in self._config_dict['line_product'][product]:
                    channel_kms = self._config_dict['line_product'][product]['channel_kms']

        if channel_kms is None:
            logging.error('No channel_kms value set for line product ' + product)
            raise Exception('No channel_kms value set for line product ' + product)

        return channel_kms

    def get_line_tag_for_line_product(self, product=None):
        """
        Get the line tag (in utilsLines) associated with a line data
        product.
        """
        if product is None:
            logging.error("Please specify a product.")
            raise Exception("Please specify a product.")
            return None

        line_tag = None
        if 'line_product' in self._config_dict:
            if product in self._config_dict['line_product']:
                if 'line_tag' in self._config_dict['line_product'][product]:
                    line_tag = self._config_dict['line_product'][product]['line_tag']

        if line_tag is None:
            logging.error('No line_tag value set for the input line product ' + product)
            raise Exception('No line_tag value set for the input line product ' + product)

        return line_tag

    def get_statwt_edge_for_line_product(self, product=None):
        """
        Get the velocity width of the edge region to use when running
        statwt on a line product.
        """
        if product is None:
            logging.error("Please specify a product.")
            raise Exception("Please specify a product.")
            return None

        statwt_edge = None
        if 'line_product' in self._config_dict:
            if product in self._config_dict['line_product']:
                if 'statwt_edge_kms' in self._config_dict['line_product'][product]:
                    statwt_edge = self._config_dict['line_product'][product]['statwt_edge_kms']

        if statwt_edge is None:
            logging.info('No statwt_edge found for ' + product)

        return statwt_edge

    def get_contsub_excludefreqrange(self, product=None):
        """
        Get the frequency range to force excluding for continuum subtraction for a line product.
        """

        if product is None:
            logging.error("Please specify a product.")
            raise Exception("Please specify a product.")
            return None

        exclude_freq_ranges_ghz = None
        if 'line_product' in self._config_dict:
            if product in self._config_dict['line_product']:
                if 'exclude_freq_ranges_ghz' in self._config_dict['line_product'][product]:
                    exclude_freq_ranges_ghz = self._config_dict['line_product'][product]['exclude_freq_ranges_ghz']

        if exclude_freq_ranges_ghz is None:
            logging.info('No exclude_freq_ranges_ghz found for ' + product + ' .')

        return exclude_freq_ranges_ghz

    def get_contsub_flagedgefraction(self, product=None):
        """
        Get the frequency range to force excluding for continuum subtraction for a line product.
        """

        if product is None:
            logging.error("Please specify a product.")
            raise Exception("Please specify a product.")
            return None

        flag_edge_fraction = None
        if 'line_product' in self._config_dict:
            if product in self._config_dict['line_product']:
                if 'flag_edge_fraction' in self._config_dict['line_product'][product]:
                    flag_edge_fraction = self._config_dict['line_product'][product]['flag_edge_fraction']

        if flag_edge_fraction is None:
            logging.info('No flag_edge_fraction found for ' + product + ' Defaults to 0.0.')
            flag_edge_fraction = 0.0

        return flag_edge_fraction

    def get_contsub_fitorder(self, product=None):
        """
        Get the fitorder to be used for continuum subtraction for a line product.
        """

        if product is None:
            logging.error("Please specify a product.")
            raise Exception("Please specify a product.")
            return None

        fitorder = None
        if 'line_product' in self._config_dict:
            if product in self._config_dict['line_product']:
                if 'fitorder' in self._config_dict['line_product'][product]:
                    fitorder = self._config_dict['line_product'][product]['fitorder']

        if fitorder is None:
            logging.info('No fitorder found for ' + product + ' . Defaulting to order zero.')
            fitorder = 0

        return fitorder

    def get_contsub_combinespw(self, product=None):
        """
        Query whether the continuum subtraction should combine all SPWs.
        """

        if product is None:
            logging.error("Please specify a product.")
            raise Exception("Please specify a product.")
            return None

        combinespw = None
        if 'line_product' in self._config_dict:
            if product in self._config_dict['line_product']:
                if 'combinespw' in self._config_dict['line_product'][product]:
                    combinespw = self._config_dict['line_product'][product]['combinespw']

        if combinespw is None:
            logging.info('No combinespw flag found for ' + product + ' . Defaulting to False.')
            combinespw = False

        return combinespw

    def get_lines_to_flag(self, product=None):
        """
        Get the list of lines to flag when either constructing a
        continuum product or carrying out continuum subtraction on a
        line product.
        """
        if product is None:
            logging.error("Please specify a product.")
            raise Exception("Please specify a product.")
            return None

        lines_to_flag = []
        if 'cont_product' in self._config_dict.keys():
            if product in self._config_dict['cont_product']:
                if 'lines_to_flag' in self._config_dict['cont_product'][product]:
                    lines_to_flag = self._config_dict['cont_product'][product]['lines_to_flag']

        if 'line_product' in self._config_dict.keys():
            if product in self._config_dict['line_product']:
                if 'lines_to_flag' in self._config_dict['line_product'][product]:
                    lines_to_flag = self._config_dict['line_product'][product]['lines_to_flag']

        if len(lines_to_flag) == 0:
            logging.warning('No lines to flag for the input product ' + product)
            # raise Exception('No lines to flag for the input continuum product '+product)

        return lines_to_flag

    def get_channel_width_for_cont_product(self, product=None):
        """
        Get the channel width (in km/s) associated with a continuum product.
        """
        if product is None:
            logging.error("Please specify a product.")
            raise Exception("Please specify a product.")

        channel_ghz = None
        if 'cont_product' in self._config_dict:
            if product in self._config_dict['cont_product']:
                if 'channel_ghz' in self._config_dict['cont_product'][product]:
                    channel_ghz = \
                        self._config_dict['cont_product'][product]['channel_ghz']

        if channel_ghz is None:
            logging.info('No channel_ghz found for ' + product)

        return channel_ghz

    def get_freq_ranges_for_cont_product(self, product=None):
        """
        Get the frequency ranges (in GHz) associated with a continuum product.
        """
        if product is None:
            logging.error("Please specify a product.")
            raise Exception("Please specify a product.")

        freq_ranges_ghz = None
        if 'cont_product' in self._config_dict:
            if product in self._config_dict['cont_product']:
                if 'freq_ranges_ghz' in self._config_dict['cont_product'][product]:
                    freq_ranges_ghz = \
                        self._config_dict['cont_product'][product]['freq_ranges_ghz']

        if freq_ranges_ghz is None:
            logging.error('No freq_ranges_ghz value set for cont product ' + product)

        return freq_ranges_ghz

    def get_array_tags_for_config(self, config=None):
        """
        Get the list of array tags associated with an interferometric configuration.
        """
        if config is None:
            logging.error("Please specify a config.")
            return None

        if 'interf_config' in self._config_dict:
            if config in self._config_dict['interf_config']:
                if 'array_tags' in self._config_dict['interf_config'][config]:
                    return self._config_dict['interf_config'][config]['array_tags']

        if 'singledish_config' in self._config_dict:
            if config in self._config_dict['singledish_config']:
                if 'array_tags' in self._config_dict['singledish_config'][config]:
                    return self._config_dict['singledish_config'][config]['array_tags']
                else:
                    return [config]

        return None

    def get_timebin_for_array_tag(self, array_tag=None):
        """
        Get the timebin for an array tag. Returns 0s by default.
        """
        if 'array_tag' not in self._config_dict.keys():
            # logger.debug("No array_tag defined in config definitions.")
            return '0s'
        if array_tag.strip() not in self._config_dict['array_tag'].keys():
            logger.debug("Tag " + str(array_tag) + " not in config definitions.")
            return '0s'
        if 'timebin' not in self._config_dict['array_tag'][array_tag].keys():
            logger.debug("Tag " + str(array_tag) + " has no timebin.")
            return '0s'
        return self._config_dict['array_tag'][array_tag]['timebin']

    def get_joint_imaging_dirs_for_singledish_config(self, config='tp'):
        """
        Get joint_imaging_dirs.
        """
        if 'singledish_config' in self._config_dict:
            if config in self._config_dict['singledish_config']:
                if 'joint_imaging_dirs' in self._config_dict['singledish_config'][config]:
                    joint_imaging_dirs = self._config_dict['singledish_config'][config]['joint_imaging_dirs']
                    if not isinstance(joint_imaging_dirs, (list, tuple)):
                        joint_imaging_dirs = joint_imaging_dirs.split(' ')
                    return joint_imaging_dirs
                else:
                    return None

        return None

    def get_joint_imaging_suffix_for_singledish_config(self, config='tp'):
        """
        Get joint_imaging_suffix.
        """
        if 'singledish_config' in self._config_dict:
            if config in self._config_dict['singledish_config']:
                if 'joint_imaging_suffix' in self._config_dict['singledish_config'][config]:
                    return self._config_dict['singledish_config'][config]['joint_imaging_suffix']
                else:
                    return ''

        return ''

    def loop_over_input_ms(
            self,
            target=None,
            config=None,
            project=None,
            check_linmos=False,
            strict_config=True,
    ):
        """
        Loop over the the target name, project tag, array tag, and
        obsnum for each input visibility file. If a target is supplied
        then restrict to that target, trying to match to a linear
        mosaic if the target is not represented in the dictionary. If
        a configuration is supplied, then only include array_tags that
        contribute to that configuration.

        Note that the interaction with configs that contain multiple
        arrays is tricky. By default, if "strict_config" is TRUE, it
        will only loop over targets that have data from ALL arrays in
        the configuration. For example to be included in a "12m+7m"
        configuration you need both "12m" AND "7m" data. Set
        strict_config to FALSE to adjust this behaviour.
        """

        # if user has input a target or target list, match to it
        just_targets = []
        if target is not None:
            if type(target) == type(''):
                input_targets = [target]
            elif type(target) == type([]):
                input_targets = target
            else:
                logger.error("Expected list or string.")
                raise Exception("Expected list or string.")

            for this_target in input_targets:
                if this_target not in just_targets:
                    just_targets.append(this_target)

                # Optionally, also include targets that are linear
                # mosaics but don't have their own assigned
                # measurement sets. This is set to False by default.

                if check_linmos:
                    if self.is_target_linmos(target=this_target):
                        parts = self.get_parts_for_linmos(target=this_target)
                        if parts is None:
                            continue
                        for this_part in parts:
                            if this_part not in just_targets:
                                just_targets.append(this_part)

        # if the user has input a project list, match to that
        just_projects = []
        if project is not None:
            if type(project) == type(''):
                just_projects = [project]
            elif type(target) == type([]):
                just_projects = project
            else:
                logger.error("Expected list or string.")
                raise Exception("Expected list or string.")

        # if user has input a config or config list, only loop over
        # array tags included in those configurations.
        just_arraytags = []
        if config is not None:
            if type(config) == type(''):
                input_configs = [config]
            elif type(config) == type([]):
                input_configs = config
            else:
                logger.error("Expected list or string.")
                raise Exception("Expected list or string.")

            for this_config in input_configs:
                this_array_tags = self.get_array_tags_for_config(this_config)
                if this_array_tags is None:
                    continue
                for this_tag in this_array_tags:
                    if this_tag not in just_arraytags:
                        just_arraytags.append(this_tag)

        # Loop over targets
        target_list = list(self._ms_dict.keys())
        target_list.sort()
        for this_target in target_list:

            # if user has input targets, match it
            if len(just_targets) > 0:
                if not (this_target in just_targets):
                    continue

            # If we're being strict, only consider targets that have
            # data associated with the user-supplied configs.

            if strict_config:

                # This list holds the valid array tags for this target.

                valid_arraytags = []

                # This mode only works with a user-supplied list of
                # configs. Else we loop over all measurement sets.

                if config is not None:
                    if type(config) == type(''):
                        input_configs = [config]
                    elif type(config) == type([]):
                        input_configs = config
                    else:
                        logger.error("Expected list or string.")
                        raise Exception("Expected list or string.")

                    has_data_for_any_config = False

                    # Check if the target has data for that configuration
                    for this_config in input_configs:

                        if self.has_data_for_config(target=this_target, config=this_config, strict=True):
                            has_data_for_any_config = True

                            # Note the array tags in this, known to be valid, configuration
                            for this_arraytag in self.get_array_tags_for_config(this_config):
                                if valid_arraytags.count(this_arraytag) == 0:
                                    valid_arraytags.append(this_arraytag)

                    # If there are no valid configurations skip.
                    if not has_data_for_any_config:
                        continue

            # loop over projects
            project_list = list(self._ms_dict[this_target].keys())
            project_list.sort()
            for this_project in project_list:

                if len(just_projects) > 0:
                    if not (this_project in just_projects):
                        continue

                # loop over array tags
                arraytag_list = list(self._ms_dict[this_target][this_project].keys())
                arraytag_list.sort()
                for this_arraytag in arraytag_list:

                    if len(just_arraytags) > 0:
                        if not (this_arraytag in just_arraytags):
                            continue

                    if strict_config and config is not None:
                        if valid_arraytags.count(this_arraytag) == 0:
                            continue

                    # loop over obs nums

                    obsnum_list = list(self._ms_dict[this_target][this_project][this_arraytag].keys())
                    obsnum_list.sort()
                    for this_obsnum in obsnum_list:
                        yield this_target, this_project, this_arraytag, this_obsnum

    def get_file_for_input_ms(
            self,
            target=None,
            project=None,
            array_tag=None,
            obsnum=None,
    ):
        """
        Return the full file path to of an input measurement set given
        a target, project, array_tag, obsnum combination.
        """

        # Verify input

        if target is None:
            logging.error("Please specify a target.")
            return None
        if project is None:
            logging.error("Please specify a project.")
            return None
        if array_tag is None:
            logging.error("Please specify an array_tag.")
            return None
        if obsnum is None:
            logging.error("Please specify an obsnum.")
            return None

        # Check that the provided input is in the ms_dict

        check_valid = False
        if target in self._ms_dict.keys():
            if project in self._ms_dict[target].keys():
                if array_tag in self._ms_dict[target][project].keys():
                    if obsnum in self._ms_dict[target][project][array_tag].keys():
                        check_valid = True

        if not check_valid:
            logger.error('No target ' + target + ' project ' + project + ' array_tag ' + \
                         array_tag + ' obsnum ' + str(obsnum) + ' defined in ms dictionary.')
            raise Exception('No target ' + target + ' project ' + project + ' array_tag ' + \
                            array_tag + ' obsnum ' + str(obsnum) + ' defined in ms dictionary.')

        # Else locate the full path to the file

        ms_file_path = ''
        file_paths = []
        for ms_root in self._ms_roots:
            file_path = ms_root + self._ms_dict[target][project][array_tag][obsnum]['file']
            if os.path.isdir(file_path):
                file_paths.append(file_path)

        # Error check the results (redundant with read-in checks but good to have)

        if len(file_paths) == 0:
            logging.warning('Could not find measurement set for target ' + \
                            target + ' project ' + project + ' array_tag ' + \
                            array_tag + ' obsnum ' + str(obsnum) + '!')
            return None
        else:
            if len(file_paths) > 1:
                logging.warning('Found multiple measurement sets for target ' + \
                                target + ' project ' + project + ' array_tag ' + \
                                array_tag + ' obsnum ' + str(obsnum) + \
                                ':\n' + '\n'.join(file_paths) + '\n' + 'Returning the first one.')

        ms_file_path = file_paths[0]
        return ms_file_path

    def has_data_for_config(
            self,
            target=None,
            config=None,
            strict=True,
    ):
        """
        Test whether a target has data for a configuration in the ms
        key. If "strict" is TRUE then require that a target has data
        from ALL arrays that make up the configuration.
        """

        if target is None:
            logging.error("Please specify a target.")
            return None
        if config is None:
            logging.error("Please specify a config.")
            return None

        config_array_tags = self.get_array_tags_for_config(config)

        arraytags_for_target = []

        for this_target in self._ms_dict.keys():

            if this_target != target:
                continue

            for this_project in self._ms_dict[this_target].keys():

                for this_arraytag in self._ms_dict[this_target][this_project].keys():
                    arraytags_for_target.append(this_arraytag)

        has_any = False
        missing_any = False

        for this_config_arraytag in config_array_tags:

            missing_this_one = True

            for this_target_arraytag in arraytags_for_target:

                if this_config_arraytag == this_target_arraytag:
                    missing_this_one = False
                    has_any = True

            if missing_this_one:
                missing_any = True

        if strict:
            if missing_any:
                return False
            else:
                return True
        else:
            if has_any:
                return True
            else:
                return False

        return False

    def get_field_for_input_ms(
            self,
            target=None,
            project=None,
            array_tag=None,
            obsnum=None,
    ):
        """
        Return the science field given a target, project, array_tag,
        obsnum combination.
        """

        # Verify input

        if target is None:
            logging.error("Please specify a target.")
            return None
        if project is None:
            logging.error("Please specify a project.")
            return None
        if array_tag is None:
            logging.error("Please specify an array_tag.")
            return None
        if obsnum is None:
            logging.error("Please specify an obsnum.")
            return None

        # Check that the provided input is in the ms_dict

        check_valid = False
        if target in self._ms_dict.keys():
            if project in self._ms_dict[target].keys():
                if array_tag in self._ms_dict[target][project].keys():
                    if obsnum in self._ms_dict[target][project][array_tag].keys():
                        check_valid = True

        if not check_valid:
            logger.error('No target ' + target + ' project ' + project + ' array_tag ' + \
                         array_tag + ' obsnum ' + str(obsnum) + ' defined in ms dictionary.')
            raise Exception('No target ' + target + ' project ' + project + ' array_tag ' + \
                            array_tag + ' obsnum ' + str(obsnum) + ' defined in ms dictionary.')

        return self._ms_dict[target][project][array_tag][obsnum]['field']

    def has_singledish(
            self,
            target=None,
            product=None,
    ):
        """
        Return true or false indicating if this target and product
        combination has associated single dish data.
        """

        if self._sd_dict is None:
            logging.error("No single dish key defined.")
            return None

        if target == None:
            logging.error("Please specify a target.")
            return None

        if product == None:
            logging.error("Please specify a product.")
            return None

        if target not in self._sd_dict.keys():
            return False

        this_dict = self._sd_dict[target]

        if product not in this_dict.keys():
            return False

        return True

    def get_sd_filename(
            self,
            target=None,
            product=None,
            nocheck=False,
    ):
        """
        Return the single dish filename for a target and product combination.
        """

        if self._sd_dict is None:
            logging.error("No single dish key defined.")
            return None

        if target == None:
            logging.error("Please specify a target.")
            return None

        if product == None:
            logging.error("Please specify a product.")
            return None

        if target not in self._sd_dict.keys():
            logging.warning("Not in single dish keys: " + target)
            return None

        this_dict = self._sd_dict[target]

        if product not in this_dict.keys():
            logging.warning("Product not found for " + target + " : " + product)
            return None

        found = False
        found_count = 0
        last_found_file = None
        for this_root in self._sd_roots:
            this_fname = this_root + this_dict[product]
            if os.path.isfile(this_fname) or os.path.isdir(this_fname) or nocheck:
                found = True
                found_count += 1
                last_found_file = this_fname

        if found_count > 1:
            logging.error("Found multiple copies of single dish data for " + target + " " + product)
            logging.error("Returning last one, but this is likely an error.")
            return last_found_file

        if found_count == 0:
            logging.error("Did not find single dish data for " + target + " " + product)
            return None

        return last_found_file

    def get_cleanmask_filename(
            self,
            target=None,
            product=None,
    ):
        """
        Get the file name of the clean mask associated with a target and product.
        """

        if self._cleanmask_dict is None:
            logging.error("No cleanmask dictionary defined.")
            return None

        if target == None:
            logging.error("Please specify a target.")
            return None

        if target not in self._cleanmask_dict.keys():
            logging.warning("Not in cleanmask keys: " + target)
            return None

        this_dict = self._cleanmask_dict[target]

        if 'all' in this_dict.keys():
            this_product = 'all'
        else:
            this_product = product

        if this_product not in this_dict.keys():
            logging.warning("Cleanmask not found for " + target + " and product " + str(this_product))
            return None

        found = False
        found_count = 0
        last_found_file = None
        for this_root in self._cleanmask_roots:
            this_fname = this_root + this_dict[this_product]
            if os.path.isfile(this_fname) or os.path.isdir(this_fname):
                found = True
                found_count += 1
                last_found_file = this_fname

        if found_count > 1:
            logging.error("Found multiple copies of cleanmask for " + target + " " + str(this_product))
            logging.error("Returning last one, but this is likely an error.")
            return last_found_file

        if found_count == 0:
            logging.error("Did not find a cleanmask for " + target + " " + str(this_product))
            return None

        logger.debug('Using clean mask file "' + os.path.basename(
            last_found_file) + '" for target "' + target + '" and product "' + product + '"')

        return last_found_file

        return ()

    def get_feather_config_for_interf_config(
            self,
            interf_config=None
    ):
        """
        Get the interferometric configuration to go with a feather configuration.
        """

        if interf_config is None:
            return None

        if 'interf_config' not in self._config_dict.keys():
            return None

        interf_config_dict = self._config_dict['interf_config']

        if interf_config not in interf_config_dict.keys():
            return None

        if 'feather_config' not in interf_config_dict[interf_config].keys():
            return None

        return interf_config_dict[interf_config]['feather_config']

    def get_interf_config_for_feather_config(
            self,
            feather_config=None
    ):
        """
        Get the feather configuration to go with an interferometric configuration.
        """

        if feather_config is None:
            return None

        if 'interf_config' not in self._config_dict.keys():
            return None

        feather_config_dict = self._config_dict['feather_config']

        if feather_config not in feather_config_dict.keys():
            return None

        if 'interf_config' not in feather_config_dict[feather_config].keys():
            return None

        return feather_config_dict[feather_config]['interf_config']

    def get_clean_scales_for_config(
            self,
            config=None,
    ):
        """
        Return the angular scales used for multiscale clean for an
        interferometric configuration.
        """

        if config is None:
            return None

        if config in self._config_dict['interf_config'].keys():
            this_dict = self._config_dict['interf_config'][config]
        else:
            return None

        return this_dict['clean_scales_arcsec']

    def get_ang_res_dict(self, config=None, product=None,
                         ):
        """
        Return the angular resolutions for derived product creation
        for a combination of resolution and spectral product.
        """

        if config is None:
            logger.warning("Need a config.")
            return None

        if product is None:
            logger.warning("Need a product.")
            return None

        if config not in self._derived_dict.keys():
            return {}

        if product not in self._derived_dict[config].keys():
            return {}

        return self._derived_dict[config][product]['ang_res']

    def get_phys_res_dict(
            self,
            config=None,
            product=None,
    ):
        """
        Return the physical resolutions for derived product creation
        for a combination of resolution and spectral product.
        """

        if config is None:
            logger.warning("Need a config.")
            return None

        if product is None:
            logger.warning("Need a product.")
            return None

        if config not in self._derived_dict.keys():
            return {}

        if product not in self._derived_dict[config].keys():
            return {}

        return self._derived_dict[config][product]['phys_res']

    def get_derived_kwargs(self, config=None, product=None,
                           kwarg_type='strictmask_kw'):
        """
        Get the dictionary of keyword arguments from the derived key
        for masking or noise estimation. Valid kwarg_types are
        'strictmask_kw', 'broadmask_kw', 'noise_kw'
        'mask_configs', 'ang_res', 'phys_res',
        'noise_kw', 'strictmask_kw', 'broadmask_kw',
        'convolve_kw', 'moments', 'shuffle_kw',
        'flatstrictmask_kw', 'flatbroadmask_kw'
        """

        if config is None:
            logger.warning("Need a config.")
            return None

        if product is None:
            logger.warning("Need a product.")
            return None

        if config not in self._derived_dict.keys():
            return {}

        if product not in self._derived_dict[config].keys():
            return {}
        
        if kwarg_type not in self._derived_dict[config][product].keys():
            return {}

        return self._derived_dict[config][product][kwarg_type].copy()

    def get_linked_mask_configs(
            self,
            config=None,
            product=None,
    ):
        """
        Return the list of linked configurations used in making hybrid
        masks.
        """

        if config is None:
            return None

        if product is None:
            return None

        if config not in self._derived_dict.keys():
            return []

        if product not in self._derived_dict[config].keys():
            return []

        return self._derived_dict[config][product]['mask_configs']

    def get_moment_list(
            self,
            config=None,
            product=None,
    ):
        """
        Return the list of moments to build for a config + product.
        masks.
        """

        if config is None:
            return None

        if product is None:
            return None

        if config not in self._derived_dict.keys():
            return []

        if product not in self._derived_dict[config].keys():
            return []

        return self._derived_dict[config][product]['moments']

    def get_params_for_moment(self, moment=None):
        """
        Return parameter dictionary for a moment.
        """

        if moment is None:
            return None

        if moment not in self._moment_dict.keys():
            logger.error("Moment not found in keys: " + str(moment))
            return None

        return self._moment_dict[moment]

    def get_params_for_singledish(
            self,
            singledish_config=None
    ):
        """
        Return parameter dictionary for a singledish config.
        """

        if singledish_config is None:
            return None

        if 'singledish_config' not in self._config_dict.keys():
            return None

        singledish_config_dict = self._config_dict['singledish_config']

        if singledish_config not in singledish_config_dict.keys():
            return None

        return singledish_config_dict[singledish_config]

    def print_configs(
            self
    ):
        """
        Print out the configurations for inspection.
        """

        if self._config_dict is None:
            return ()

        logger.info("Interferometric Configurations")
        for this_config in self._config_dict['interf_config'].keys():
            logger.info("... " + this_config)
            this_arrays = self._config_dict['interf_config'][this_config]['array_tags']
            this_other_config = self._config_dict['interf_config'][this_config]['feather_config']
            scales_for_clean = self._config_dict['interf_config'][this_config]['clean_scales_arcsec']
            logger.info("... ... includes arrays " + str(this_arrays))
            logger.info("... ... maps to feather config " + str(this_other_config))
            logger.info("... ... clean these scales in arcsec " + str(scales_for_clean))

        if 'feather_config' in self._config_dict:
            logger.info("Feather Configurations")
            for this_config in self._config_dict['feather_config'].keys():
                logger.info("... " + this_config)
                this_other_config = self._config_dict['feather_config'][this_config]['interf_config']
                logger.info("... ... maps to interferometer config " + str(this_other_config))

        return ()

    def print_products(
            self
    ):
        """
        Print out the products for inspection.
        """

        if self._config_dict is None:
            return ()

        logger.info("Continuum data products")
        for this_product in self.get_continuum_products():
            logger.info("... " + this_product)
            lines_to_flag = self._config_dict['cont_product'][this_product]['lines_to_flag']
            logger.info("... ... lines to flag " + str(lines_to_flag))

        logger.info("Line data products")
        for this_product in self.get_line_products():
            logger.info("... " + this_product)
            if 'channel_kms' in self._config_dict['line_product'][this_product].keys():
                channel_width = self._config_dict['line_product'][this_product]['channel_kms']
                logger.info("... ... channel width [km/s] " + str(channel_width))
            if 'line_tag' in self._config_dict['line_product'][this_product].keys():
                line_name = self._config_dict['line_product'][this_product]['line_tag']
                logger.info("... ... line name code " + str(line_name))

        return ()

    def print_missing_distances(
            self
    ):
        """
        Print out the missing distances
        """

        if self._distance_dict is None:
            logger.info("... there is no distance dictionary defined.")
            return ()

        missing_targets = []
        for this_target in self.get_all_targets():
            if self.get_distance_for_target(target=this_target) is None:
                missing_targets.append(this_target)

        if len(missing_targets) == 0:
            logger.info("... no targets are missing distances!")
            return ()

        logger.info("... targets missing distances: " + str(missing_targets))
        return ()

    def print_missing_window(
            self
        ):
        """
        Print out the missing velocity window
        """

        if self._window_dict is None:
            logger.info("... there is no velocity window dictionary defined.")
            return ()

        missing_targets = []
        for this_target in self.get_all_targets():
            if self.get_window_for_target(target=this_target) is None:
                missing_targets.append(this_target)

        if len(missing_targets) == 0:
            logger.info("... no targets are missing velocity window!")
            return ()

        logger.info("... targets missing velocity window: " + str(missing_targets))
        return ()

    def print_derived(
            self
    ):
        """
        Print out the information for derived products.
        """

        if self._derived_dict is None:
            return ()

        for this_config in self.get_all_configs():
            all_products = self.get_continuum_products() + self.get_line_products()
            for this_product in all_products:
                logger.info("... derived products for " + this_config + " " + this_product)

                ang_res = self._derived_dict[this_config][this_product]['ang_res']
                phys_res = self._derived_dict[this_config][this_product]['phys_res']
                linked_configs = self._derived_dict[this_config][this_product]['mask_configs']
                moments = self._derived_dict[this_config][this_product]['moments']

                logger.info("... ... angular resolutions [arcsec]: " + str(ang_res))
                logger.info("... ... physical resolutions [pc]: " + str(phys_res))
                logger.info("... ... linked configs for masking: " + str(linked_configs))
                logger.info("... ... moments to produce: " + str(moments))

        return ()

    def has_overrides_for_key(self, key=None):
        """Check whether the override dictionary contains the input key or not.

        The key is usually a file name or a galaxy name.
        """

        if self._override_dict is None:
            return False

        # check key
        if key is None:
            return False

        # check key
        if key in self._override_dict:
            return True
        else:
            return False

    def get_overrides(self, key=None, param=None, default=None):
        """
        Check the override dictionary for entries given some key,
        parameter pair.
        """

        if self._override_dict is None:
            return default

        # check key and param
        if key is None and param is None:
            logger.error('Please input key and param to get overrides!')
            raise Exception('Please input key and param to get overrides!')
            return default

        # check key in override dict or not
        if not (key in self._override_dict):
            # logger.debug('Warning! The input key "'+str(key)+'" is not in the overrides file!')
            return default

        # if user has input key only, return the dict
        if param is None:
            # logger.debug('Warning! Only has key input when getting overrides! Will return the override dictionary for this key "'+str(key)+'".')
            return self._override_dict[key]  # <TODO># dzliu: what if user has just input a key?

        if not (param in self._override_dict[key]):
            # logger.debug('Warning! The input key "'+str(key)+'" does not have a param "'+str(param)+'" in the overrides file! Return default value '+str(default))
            return default

        logger.debug('Overriding key "' + str(key) + '" param "' + str(param) + '" value ' + str(
            self._override_dict[key][param]) + ', default ' + str(default))
        return self._override_dict[key][param]

    # endregion

    # region Manipulate files and file structure

    def make_missing_directories(self, ms_root=False, imaging=False, postprocess=False, derived=False, release=False):
        """
        Make any missing imaging or postprocessing directories.
        """

        if not ms_root and not imaging and not postprocess and not derived and not release:
            logging.error("Set either ms_root or imaging or postprocess or product or release to True. Returning.")
            return False

        if ms_root:
            # Just create the first directory
            if not os.path.isdir(self._ms_roots[0]):
                logging.info("Creating MS root directory: " + self._ms_roots[0])
                os.makedirs(self._ms_roots[0])

        if imaging:
            if not os.path.isdir(self._imaging_root):
                # logging.info("Missing imaging root directory.")
                logging.info("Creating imaging root directory: " + self._imaging_root)
                # return(False)
                os.makedirs(self._imaging_root)

        if postprocess:
            if not os.path.isdir(self._postprocess_root):
                # logging.info("Missing postprocess root directory.")
                logging.info("Creating postprocess root directory: " + self._postprocess_root)
                # return(False)
                os.makedirs(self._postprocess_root)

        if derived:
            if not os.path.isdir(self._derived_root):
                # logging.info("Missing derived root directory.")
                logging.info("Creating derived root directory: " + self._derived_root)
                # return(False)
                os.makedirs(self._derived_root)

        if release:
            if not os.path.isdir(self._release_root):
                # logging.info("Missing release root directory.")
                logging.info("Creating release root directory: " + self._release_root)
                # return(False)
                os.makedirs(self._release_root)

        missing_dirs = self.check_dir_existence(imaging=imaging, postprocess=postprocess, derived=derived,
                                                release=release)
        made_directories = 0
        for this_missing_dir in missing_dirs:
            made_directories += 1
            os.makedirs(this_missing_dir)

        missing_dirs = self.check_dir_existence(imaging=imaging, postprocess=postprocess, derived=derived,
                                                release=release)

        logging.info("Made " + str(made_directories) + " directories. Now " + str(len(missing_dirs)) + " missing.")

        if len(missing_dirs) == 0:
            return True

        return False

# endregion
