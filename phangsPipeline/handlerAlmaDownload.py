"""AlmaDownloaderHandler

This module will query the ALMA archive, download and calibrate data as appropriate.
Since it uses astroquery, should be used outside monolithic CASA versions.

Example:
    $ ipython
    from phangsPipeline import handlerKeys as kh
    from phangsPipeline import handlerAlmaDownload as adl
    this_kh = kh.KeyHandler(master_key = 'phangsalma_keys/master_key.txt')
    this_adl = adl.DerivedHandler(key_handler=this_kh)
    this_adl.loop_alma_download()

Note that as this module will rewrite keys, you should re-initialise the key handler
after running through this module.

"""
import copy
import fnmatch
import glob
import logging
import multiprocessing as mp
import os
import re
import tarfile
from functools import partial

import astropy
import astropy.units as u
import numpy as np
import requests
from astropy import table
from astropy.coordinates import SkyCoord
from astroquery.alma import Alma, Conf
from astroquery.alma.utils import parse_frequency_support
from bs4 import BeautifulSoup

from . import handlerKeys
from . import handlerTemplate
from . import utilsLines

# Tags to identify antenna arrays
ANTENNA_ARRAY_SETUP = {'12m': ['DV', 'DA'],
                       '7m': ['CM'],
                       'tp': ['PM']}

ALLOWED_MS_GROUPINGS = ['mosaic', 'join', 'separate']


def query_target(target, max_query_failures=10):
    """Light wrapper around astroquery object query to allow for HTTP errors"""
    query_success = False
    query_failures = 0
    while not query_success and query_failures <= max_query_failures:
        try:
            try:
                observations = Alma.query_object(target)
                return observations
            except astropy.coordinates.name_resolve.NameResolveError:
                return None
        except requests.exceptions.HTTPError:
            query_failures += 1

    # If we can't reach the server, fail out
    if not query_success:
        logger.warning('Maximum ALMA archive queries reached. Server is probably down.')
        raise Exception('Maximum ALMA archive queries reached. Server is probably down.')


def query_region(coords, radius=None, max_query_failures=10):
    """Light wrapper around astroquery region query to allow for HTTP errors"""

    if radius is None:
        raise ValueError('Search radius should be defined!')

    query_success = False
    query_failures = 0
    while not query_success and query_failures <= max_query_failures:
        try:
            observations = Alma.query_region(coords, radius=radius)
            return observations
        except requests.exceptions.HTTPError:
            query_failures += 1

    # If we can't reach the server, fail out
    if not query_success:
        logger.warning('Maximum ALMA archive queries reached. Server is probably down.')
        raise Exception('Maximum ALMA archive queries reached. Server is probably down.')


def astroquery_download(row, cache_location=None, username=None):
    """Download ALMA archive files"""
    myAlma = Alma()

    if cache_location is not None:
        myAlma.cache_location = cache_location
    if username is not None:
        myAlma.login(username)

    myAlma.download_files([row['access_url']], cache=True)
    return True


def get_casa_version_from_qa_report(qa_report):
    """Pulls CASA pipeline version from HTML QA report.

    Taken from Daizhong's A3COSMOS stuff, thanks Daizhong!

    """

    casa_pipeline_version = None

    with open(qa_report, 'r') as html:
        html_content = html.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        for soup_th in soup.findAll('th', text='CASA Version'):
            for soup_td in soup_th.find_next_siblings():
                if len(soup_td.text) > 0:
                    if soup_td.text.find('.') > 0:
                        if soup_td.text.find(' ') > 0:
                            casa_pipeline_version = soup_td.text.split()[0]
                        else:
                            casa_pipeline_version = soup_td.text
                if casa_pipeline_version is not None:
                    break

            if casa_pipeline_version is not None:
                break

        # Try again to find span that includes 'CASA version:'
        if casa_pipeline_version is None:
            for soup_span in soup.findAll('span'):
                if soup_span.text.startswith('CASA version:'):
                    casa_pipeline_version = re.sub(r'CASA version: ([0-9.]+).*', r'\1', soup_span.text)
                    if casa_pipeline_version is not None:
                        break

        # Try again to find span that include 'CASA version' and next is 'Report date'
        if casa_pipeline_version is None:
            for soup_span in soup.findAll('span'):
                if soup_span.text == 'CASA version':
                    soup_tr = soup_span.find_parent('tr')
                    soup_tr_td = soup_tr.find_all('td')
                    soup_tr_td_text = [t.text.strip() for t in soup_tr_td]
                    soup_tr_td_index = soup_tr_td_text.index('CASA version')
                    soup_tr_next = soup_tr.find_next_sibling('tr')
                    soup_tr_next_td = soup_tr_next.find_all('td')
                    soup_tr_next_td_text = [t.text.strip() for t in soup_tr_next_td]
                    found_text = soup_tr_next_td_text[soup_tr_td_index]
                    if re.match(r'^[0-9.-]+$', found_text):
                        casa_pipeline_version = found_text
                    if casa_pipeline_version is not None:
                        break

    return casa_pipeline_version


def get_casa_version_from_weblog(weblog):
    """Pulls CASA pipeline version from weblog

    Again, taken from Daizhong's A3COSMOS stuff, thanks Daizhong!

    """

    casa_pipeline_version = None

    weblog_obj = tarfile.open(weblog)

    for weblog_item in weblog_obj.getmembers():
        if os.path.basename(weblog_item.name) == 'index.html':
            weblog_index_html = weblog_obj.extractfile(weblog_item)
            weblog_index_html_content = weblog_index_html.read()

            soup = BeautifulSoup(weblog_index_html_content, 'html.parser')

            for soup_th in soup.findAll('th', text='CASA Version'):
                for soup_td in soup_th.find_next_siblings():
                    if len(soup_td.text) > 0:
                        if soup_td.text.find('.') > 0:
                            if soup_td.text.find(' ') > 0:
                                casa_pipeline_version = soup_td.text.split()[0]
                            else:
                                casa_pipeline_version = soup_td.text
                    if casa_pipeline_version is not None:
                        break

                if casa_pipeline_version is not None:
                    break

            if casa_pipeline_version is not None:
                break

    return casa_pipeline_version


def get_casa_version_from_calibration_script(script):
    """Pulls CASA pipeline version from calibration script

    Again, taken from Daizhong's A3COSMOS stuff, thanks Daizhong!

    """

    casa_pipeline_version = None

    with open(script, 'r') as fp:

        for line in fp:
            if line.find('PLEASE USE THE SAME VERSION OF CASA') >= 0:
                if re.match(r'^.*PLEASE USE THE SAME VERSION OF CASA.*:[ \t]*([0-9.]+)\b.*$', line):
                    casa_pipeline_version = re.sub(r'^.*PLEASE USE THE SAME VERSION OF CASA.*:[ \t]*([0-9.]+)\b.*$',
                                                   r'\1', line)
                    # Trim any whitespace
                    casa_pipeline_version = casa_pipeline_version.strip()
                    break

    return casa_pipeline_version


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class AlmaDownloadHandler(handlerTemplate.HandlerTemplate):
    """
    Class to automate downloading and calibrating ALMA data.

    N.B. Since clean masks are generated externally, this code will not modify that key file. It will, however,
    modify ms_key, dir_key, linmos_key, singledish_key, and target_key.

    This script may also produce weirdness if you have a multiple files for each key type.

    """

    ############
    # __init__ #
    ############

    def __init__(
            self,
            key_handler=None,
            master_key=None,
            restore_previous_target_keys=True,
            dry_run=False,
    ):
        # inherit template class
        handlerTemplate.HandlerTemplate.__init__(self,
                                                 key_handler=key_handler,
                                                 dry_run=dry_run)

        # If we've run this before, restore the original key file and reinitialise
        key_root = self._kh._key_dir
        target_key_file = self._kh._target_keys[0]
        target_key_path = os.path.join(key_root, target_key_file)
        pre_dl_target_key_path = target_key_path + '_pre_AlmaDownload'

        if os.path.exists(pre_dl_target_key_path) and restore_previous_target_keys:
            logger.info('This is a rerun, restoring old target keys')

            os.system('rm -rf %s' % target_key_path)
            os.system('mv %s %s' % (pre_dl_target_key_path, target_key_path))

            key_handler = handlerKeys.KeyHandler(master_key=master_key)
            handlerTemplate.HandlerTemplate.__init__(self,
                                                     key_handler=key_handler,
                                                     dry_run=dry_run)

    def loop_alma_download(
            self,
            do_all=False,
            make_directories=True,
            do_download=False,
            do_calibrate=False,
            do_build_key_files=False,
            do_tp=False,
            allow_proprietary=False,
            username=None,
            query_radius=10 * u.arcmin,
            suppress_casa_output=True,
            split_ms='mosaic',
            overwrite_download=False,
            overwrite_calibrate=False,
            overwrite_build_key_files=True,
            overwrite_all=False,
    ):
        """Download and calibrate ALMA data.

        Major steps:

        (1) Query archive for target/line/antenna config and download
        (2) Figure out which version of CASA to run the scriptForPI in, and run that
        (3) Build the file keys from the downloaded, calibrated files

        N.B. After running this you will need to reinitialise the key handler, since this makes edits to various
        key files

        Args:
            do_tp (bool, optional): If True, will also download TP data and sort out for the TP pipeline. Defaults
                to False.
            allow_proprietary (bool, optional): If True, will log in to the ALMA servers using username, to allow
                for download of proprietary data. This requires a working keychain package! Defaults to False.
            username (str, optional): ALMA username to login. On first time running this, it will ask for a
                password.
            query_radius (astropy.units, optional): Search radius for fallback coordinate query. Defaults to 10
                arcmin, same as the ALMA archive search.
            split_ms (str, optional): Can be one of 'mosaic', 'join', 'separate'. Determines how the MS file key is
                set up for later staging and imaging. If mosaic, will attempt to be smart and split observations by
                project ID/science goal. If 'join', all observations will be joined together into a potentially
                huge map. If 'separate', then each calibrated MS will be assigned to a separate target. Defaults to
                'mosaic', which mimics the way the PHANGS-ALMA targets are set up.

        """

        if do_all:
            make_directories = True
            do_download = True
            do_calibrate = True
            do_build_key_files = True
            do_tp = True
        if overwrite_all:
            overwrite_download = True
            overwrite_calibrate = True
            overwrite_build_key_files = True

        # Error checking

        if len(self.get_targets()) == 0:
            logger.error("Need a target list.")
            return None

        if len(self.get_all_products()) == 0:
            logger.error("Need a products list.")
            return None

        # If requested, make the directories
        if make_directories:
            self._kh.make_missing_directories(ms_root=True)

        # If allowing proprietary data, login here and save password for later
        if allow_proprietary:
            if username is None:
                logger.warning('username should be set!')
                raise Exception('username should be set!')
            Conf.username = username
            alma = Alma()
            alma.login(username, store_password=True)

        # If requested, query/download/extract data
        if do_download:

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("Beginning download/extraction of ALMA data")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("")

            for this_target, this_product, this_config in \
                    self.looper(do_targets=True, do_products=True, do_configs=True, just_interf=True):
                uids = self.task_query(target=this_target,
                                       product=this_product,
                                       config=this_config,
                                       query_radius=query_radius,
                                       overwrite=overwrite_download)
                self.task_download(target=this_target,
                                   product=this_product,
                                   config=this_config,
                                   uids=uids,
                                   username=username,
                                   overwrite=overwrite_download)

            # Also potentially include TP
            if do_tp:

                for this_target, this_product in \
                        self.looper(do_targets=True, do_products=True, do_configs=False):
                    uids = self.task_query(target=this_target,
                                           product=this_product,
                                           config='tp')
                    self.task_download(target=this_target,
                                       product=this_product,
                                       config='tp',
                                       uids=uids,
                                       username=username,
                                       overwrite=overwrite_download)

        # If requested, run scriptForPI
        if do_calibrate:

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("Beginning calibration of ALMA data")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("")

            for this_target, this_config in \
                    self.looper(do_targets=True, do_products=False, do_configs=True, just_interf=True):
                self.task_run_scriptforpi(target=this_target,
                                          config=this_config,
                                          suppress_casa_output=suppress_casa_output,
                                          overwrite=overwrite_calibrate)

        # If requested, build key files
        if do_build_key_files:
            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("Building key files")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("")

            self.task_build_key_files(do_tp=do_tp,
                                      split_ms=split_ms,
                                      overwrite=overwrite_build_key_files)

    def task_query(self,
                   target=None,
                   product=None,
                   config=None,
                   query_radius=10 * u.arcmin,
                   max_query_failures=10,
                   overwrite=False,
                   ):
        """Query ALMA archive.

        Uses astroquery to search the archive for data which matches (a) the target, (b) the product, (c) the
        antenna config (12m or 7m), and then velocity resolution.

        Args:
            max_query_failures (int, optional): Number of times to try querying the database before failing out.
                Defaults to 10
        Returns:
            list of UIDs to download

        """

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Querying ALMA database for:")
        logger.info('{0}, {1}, {2}'.format(target, product, config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        if target is None:
            logger.warning('Require a target')
            return None
        if product is None:
            logger.warning('Require a product')
            return None
        if config is None or config not in ANTENNA_ARRAY_SETUP.keys():
            logger.warning('Require a valid config (%s)' % list(ANTENNA_ARRAY_SETUP.keys()))
            return None

        ms_root = self._kh._ms_roots[0]

        # Pull out line, channel width, and central frequency for this combination of target, product, config
        line = self._kh.get_line_tag_for_line_product(product)
        channel_kms = self._kh._config_dict['line_product'][product]['channel_kms']
        vsys, vwidth = self._kh.get_system_velocity_and_velocity_width_for_target(target, check_parent=False)
        line_low_ghz, line_high_ghz = utilsLines.get_ghz_range_for_line(line=line,
                                                                        vsys_kms=vsys,
                                                                        vwidth_kms=vwidth,
                                                                        )
        line_ghz = ((line_high_ghz + line_low_ghz) / 2) * u.GHz

        # Perform a target query.
        observations = query_target(target, max_query_failures=max_query_failures)

        # If the target doesn't resolve, fall back to RA/Dec search.key file building
        if observations is None:
            ra, dec = self._kh.get_phasecenter_for_target(target=target)
            coords = SkyCoord('%s %s' % (ra, dec))
            observations = query_region(coords=coords, radius=query_radius, max_query_failures=max_query_failures)

        # Include custom switches
        download_restrictions = self._kh.get_alma_download_restrictions(target=target,
                                                                        product=product,
                                                                        config=config)

        parsed_obs = table.Table()

        for observation in observations:

            # Check if the file actually already exists

            proposal_id = observation['proposal_id']
            file_uid = observation['asdm_uid'].replace(':', '_').replace('/', '_')
            tar_file_name = '%s_%s.asdm.sdm.tar' % (proposal_id, file_uid)

            if config != 'tp':
                file_name = os.path.join(ms_root, target, config, tar_file_name)
            else:
                file_name = os.path.join(ms_root, target, config, product, tar_file_name)

            if os.path.exists(file_name) and not overwrite:
                continue

            # Do checks on velocity resolution
            velocity_res = observation['velocity_resolution'] / 1000
            if velocity_res > channel_kms:
                continue

            # Checks on arrays we want
            antenna_arrays = observation['antenna_arrays']

            array_wanted = False

            array_setup_tags = ANTENNA_ARRAY_SETUP[config]
            for array_setup_tag in array_setup_tags:
                if array_setup_tag in antenna_arrays:
                    array_wanted = True

                    # Sometimes it seems like TP can use other antenna (maybe this is only early science?) so double
                    # check here. TP only has 4 antenna so
                    if config != 'tp':
                        # for tp_array_tag in ANTENNA_ARRAY_SETUP['tp']:
                        #     if tp_array_tag in antenna_arrays:
                        if antenna_arrays.count(':') <= 4:
                            array_wanted = False

            if not array_wanted:
                continue

            # Check the line we want is in the frequency range

            freqs = parse_frequency_support(observation['frequency_support'])

            freq_wanted = False

            for freq in freqs:
                if freq[0] <= line_ghz <= freq[1]:
                    freq_wanted = True

            if not freq_wanted:
                continue

            # Check for any additional restrictions
            restriction_found = False
            if download_restrictions is not None:
                for key in download_restrictions.keys():
                    download_restriction = download_restrictions[key]
                    if not isinstance(download_restriction, list):
                        download_restriction = [download_restriction]

                    if observation[key] not in download_restriction:
                        restriction_found = True
                        break

            if restriction_found:
                continue

            # After we've parsed everything down, append that observation row to a new table
            parsed_obs = table.vstack([parsed_obs, observation])

        if len(parsed_obs) == 0:
            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info('No suitable UIDs found')
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("")
            return None

        uids = np.unique(parsed_obs['member_ous_uid'])
        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info('Found %d suitable UIDs:' % len(uids))
        for uid in uids:
            logger.info('-> %s' % uid)
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        return uids

    def task_download(self,
                      target=None,
                      config=None,
                      product=None,
                      uids=None,
                      username=None,
                      n_simultaneous=5,
                      overwrite=False,
                      ):
        """Downloads queried UIDs.

        Hooks the queried UIDs to download raw/ancillary/readme data. Will go into a folder structure as
            ms_root/target/config/(product for TP), to make navigation a little easier

        Args:
            username (str, optional): ALMA archive username. Defaults to None.
            n_simultaneous (int, optional): Number of download processes to spawn, to speed up download time.
                Defaults to 5.

        """

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Downloading/extracting data for:")
        logger.info('{0}, {1}, {2}'.format(target, product, config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        if target is None:
            logger.warning('Require a target')
            return None
        if config is None:
            logger.warning('Require a config')
            return None
        if product is None and config == 'tp':
            logger.warning('Require a product for TP')
            return None
        if uids is None:
            logger.warning('Require UIDs to download')
            return None

        # Setup download location, this will be ms_root/{target}/{config}
        ms_root = self._kh._ms_roots[0]

        if config == 'tp':
            dl_dir = os.path.join(ms_root, target, config, product)
        else:
            dl_dir = os.path.join(ms_root, target, config)

        # Delete download folder if we're overwriting
        if overwrite:
            os.system('rm -rf %s' % dl_dir)

        if not os.path.exists(dl_dir):
            os.makedirs(dl_dir)

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info('Downloading ALMA data for %d UIDs:' % len(uids))
        for uid in uids:
            logger.info('-> %s' % uid)
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        link_list = Alma.get_data_info(uids, expand_tarfiles=False)

        # Trim out any pointless files
        rows_to_remove = []
        for i, link in enumerate(link_list):
            if 'asdm.sdm' not in link['access_url'] and \
                    'README' not in link['access_url'] and \
                    'auxiliary' not in link['access_url']:
                rows_to_remove.append(i)
        link_list.remove_rows(rows_to_remove)

        # Download files. Allow multiple downloads via pool
        with mp.Pool(n_simultaneous) as p:
            map_result = p.map_async(partial(astroquery_download, cache_location=dl_dir, username=username),
                                     link_list)
            map_result.wait()

        # Extract tar files
        original_dir = os.getcwd()
        os.chdir(dl_dir)

        original_tar_files = sorted(glob.glob('*.tar'))

        # Check if we've already untarred
        tar_files = []
        for tar_file in original_tar_files:
            if not os.path.exists('%s_touch' % tar_file):
                tar_files.append(tar_file)

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info('Untarring %d files:' % len(tar_files))
        for tar_file in tar_files:
            logger.info('-> %s' % tar_file)
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        # Loop over and once they've been untarred make a file so we know to skip next time

        for tar_file in tar_files:
            os.system('tar --skip-old-files -xf %s' % tar_file)
            os.system('touch %s_touch' % tar_file)

        os.chdir(original_dir)

    def task_run_scriptforpi(self,
                             target=None,
                             config=None,
                             suppress_casa_output=True,
                             overwrite=False):
        """Runs scriptForPI on downloaded datasets.

        Figures out the pipeline version of CASA to run (currently, just from the QA report), and then will execute
        that script to produce a calibrated measurement set

        Args:
            suppress_casa_output (bool, optional): If True, will suppress most CASA output while running
                scriptForPI. Mostly useful for debugging, and will still put the whole log into the /script
                directory. Defaults to True.

        """

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Calibrating data for:")
        logger.info('{0}, {1}'.format(target, config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")

        if target is None:
            logger.warning('Require a target')
            return None
        if config is None or config not in ANTENNA_ARRAY_SETUP.keys():
            logger.warning('Require a valid config (%s)' % list(ANTENNA_ARRAY_SETUP.keys()))
            return None

        ms_root = self._kh._ms_roots[0]
        dl_dir = os.path.join(ms_root, target, config)

        if not os.path.exists(dl_dir):
            logger.warning('Directory for %s, %s does not exist. Returning' % (target, config))
            return None

        original_dir = os.getcwd()
        os.chdir(dl_dir)

        casa_version_files = {}

        # Search for calibration scripts, QA reports, and weblogs. Prefer the calibration script highest, since that
        # will crash out if the wrong version is chosen
        file_types = {'calibration_script': '*.scriptForCalibration.py',
                      'qa_report': '*.qa2_report.html',
                      'weblog': '*.weblog.*',
                      }

        for file_type in file_types.keys():
            file_ext = file_types[file_type]

            for root, dirnames, filenames in os.walk(os.getcwd()):
                for filename in fnmatch.filter(filenames, file_ext):
                    par_dir = os.path.dirname(root)
                    if par_dir not in casa_version_files.keys():
                        casa_version_files[par_dir] = {'filename': os.path.join(root, filename),
                                                       'type': file_type}

        for root_dir in sorted(casa_version_files.keys()):

            casa_version_file_type = casa_version_files[root_dir]['type']
            casa_version_file = casa_version_files[root_dir]['filename']

            member_dir = os.path.sep + os.path.join(*casa_version_file.split(os.path.sep)[:-2])
            calibrated_dir = os.path.join(member_dir, 'calibrated')
            script_dir = os.path.join(member_dir, 'script')

            if os.path.exists(calibrated_dir) and not overwrite:
                logger.info("")
                logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
                logger.info('Data already calibrated, will not rerun for %s' % member_dir.replace(ms_root, ''))
                logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
                logger.info("")
                continue

            # Clear out the calibrated directory
            os.system('rm -rf %s' % calibrated_dir)

            if casa_version_file_type == 'qa_report':
                casa_pipeline_version = get_casa_version_from_qa_report(casa_version_file)
            elif casa_version_file_type == 'weblog':
                casa_pipeline_version = get_casa_version_from_weblog(casa_version_file)
            elif casa_version_file_type == 'calibration_script':
                casa_pipeline_version = get_casa_version_from_calibration_script(casa_version_file)
            else:
                logger.warning('Version file type %s not understood' % casa_version_file_type)
                raise Exception('Version file type %s not understood' % casa_version_file_type)

            if casa_pipeline_version is None:
                logger.warning('Could not find a CASA version to run scriptForPI')
                raise Exception('Could not find a CASA version to run scriptForPI')

            # Get the CASA path
            casa_path = self._kh.get_path_for_casaversion(casa_pipeline_version)

            if casa_path is None:
                logger.warning('No CASA path defined for %s' % casa_pipeline_version)
                raise Exception('No CASA path defined for %s' % casa_pipeline_version)

            # Make sure we're properly pointing at CASA
            if not casa_path.endswith('casa'):
                casa_path = os.path.join(casa_path, 'casa')

            os.chdir(script_dir)
            if os.path.exists('scriptForPI.py'):
                script_names = ['scriptForPI.py']
            else:
                script_names = glob.glob('*scriptForPI.py')

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info('Calibrating %s' % member_dir.replace(ms_root, ''))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
            logger.info("")

            for script_name in script_names:
                logger.info("")
                logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
                logger.info('Running %s in CASA %s' % (script_name, casa_pipeline_version))
                logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
                logger.info("")

                # If the CASA version is 4.6, there is no pipeline version so don't use that switch
                if '4.6.' in casa_pipeline_version:
                    pipeline_cmd = ''
                else:
                    pipeline_cmd = '--pipeline'

                cmd = '%s %s --nologger -c %s' % (casa_path, pipeline_cmd, script_name)
                if suppress_casa_output:
                    cmd += ' >/dev/null 2>&1'
                exit_code = os.system(cmd)

                # Sometimes this actually fails gracefully, so check we've got some files
                output_mses = glob.glob('../calibrated/*.ms.split.cal') + glob.glob('../calibrated/*.ms')

                # If we don't execute properly, this might be a case that it's an early cycle that's been re-imaged
                # and the numbers are wrong in the report. Loop round some early CASA versions and try again
                if exit_code != 0 or len(output_mses) == 0:

                    # TODO: Loop over some early CASA versions. Looking at the list of pipeline versions
                    #  (https://almascience.nrao.edu/processing/science-pipeline), these should cover them all, but
                    #  leaving a note here in case we run into some weird cases
                    fallback_casa_versions = ['4.2.2', '4.3.1', '4.5.3', '4.7.2']

                    for fallback_casa_version in fallback_casa_versions:

                        os.system('rm -rf ../calibrated')

                        fallback_casa_path = self._kh.get_path_for_casaversion(fallback_casa_version)

                        if fallback_casa_path is None:
                            logger.warning('No CASA path defined for %s. Skipping' % fallback_casa_version)
                            continue

                        # Make sure we're properly pointing at CASA
                        if not fallback_casa_path.endswith('casa'):
                            fallback_casa_path = os.path.join(fallback_casa_path, 'casa')

                        logger.info("")
                        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
                        logger.info('CASA %s failed, falling back to %s' %
                                    (casa_pipeline_version, fallback_casa_version))
                        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
                        logger.info("")

                        # If the CASA version is 4.6, there is no pipeline version so don't use that switch
                        if '4.6.' in fallback_casa_version:
                            pipeline_cmd = ''
                        else:
                            pipeline_cmd = '--pipeline'

                        cmd = '%s %s --nologger -c %s' % (fallback_casa_path, pipeline_cmd, script_name)
                        if suppress_casa_output:
                            cmd += ' >/dev/null 2>&1'
                        fallback_exit_code = os.system(cmd)

                        # Sometimes this actually fails gracefully, so check we've got some files
                        output_mses = glob.glob('../calibrated/*.ms.split.cal') + glob.glob('../calibrated/*.ms')

                        if fallback_exit_code != 0 or len(output_mses) == 0:
                            os.system('rm -rf ../calibrated')
                            logger.warning("")
                            logger.warning("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
                            logger.warning('Calibration still failing for %s' % fallback_casa_version)
                            logger.warning("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
                            logger.warning("")
                        else:
                            break

            os.chdir(dl_dir)

        os.chdir(dl_dir)

        # Check everything now has a calibrated dir, which indicates the script has run
        calibrated_dirs = []
        for path, dirs, files in os.walk(os.getcwd()):
            if 'calibration' in dirs:
                calibrated_dirs.append(os.path.join(path, 'calibrated'))

        missing_dirs = False
        for calibrated_dir in calibrated_dirs:
            if not os.path.exists(calibrated_dir):
                # TODO: This will find any missing 'calibrated' directories, so keeping as a TODO for corner cases
                logger.warning('Unexpected missing calibrated directory! %s' % calibrated_dir)
                missing_dirs = True

        if missing_dirs:
            raise Exception('Missing some calibrated directories!')

        # Move back to the original directory
        os.chdir(original_dir)

    def task_build_key_files(self,
                             do_tp=False,
                             split_ms='mosaic',
                             overwrite=False,
                             max_query_failures=10):
        """Builds MS file key from calibrated measurement sets.

        Recursively search through calibrated measurement sets and build up into a key file to be read in for later
        imaging.

        """

        if split_ms not in ALLOWED_MS_GROUPINGS:
            logger.warning('split_ms=%s not allowed. Should be one of %s' % (split_ms, ALLOWED_MS_GROUPINGS))
            return None

        key_root = self._kh._key_dir
        ms_root = self._kh._ms_roots[0]

        # Pull out file names
        dir_key_file = self._kh._dir_keys[0]
        linmos_key_file = self._kh._linmos_keys[0]
        ms_key_file = self._kh._ms_keys[0]
        target_key_file = self._kh._target_keys[0]

        dir_key_file_name = os.path.join(key_root, dir_key_file)
        linmos_key_file_name = os.path.join(key_root, linmos_key_file)
        ms_file_name = os.path.join(key_root, ms_key_file)
        target_file_name = os.path.join(key_root, target_key_file)

        if do_tp:
            sd_key_file = self._kh._sd_keys[0]
            sd_key_file_name = os.path.join(key_root, sd_key_file)

        file_names_stacked = [dir_key_file_name,
                              linmos_key_file_name,
                              ms_file_name]

        if any([os.path.exists(file_name) for file_name in file_names_stacked]) and not overwrite:
            logger.info('Files already exists and overwrite is False. Will not overwrite')
            return None

        os.system('rm -rf %s' % ms_file_name)

        ms_file = open(ms_file_name, 'w+')

        mosaic_info = {}
        singledish_info = {}

        targets = self._kh.get_targets()

        for target in targets:

            observations = query_target(target, max_query_failures=max_query_failures)

            # If the target doesn't resolve, fall back to RA/Dec search.key file building
            if observations is None:
                ra, dec = self._kh.get_phasecenter_for_target(target=target)
                coords = SkyCoord('%s %s' % (ra, dec))
                observations = query_region(coords=coords, radius=query_radius,
                                            max_query_failures=max_query_failures)

            dl_dir = os.path.join(ms_root, target)

            original_dir = os.getcwd()
            os.chdir(dl_dir)

            target_ms_dict = {}
            all_configs = []
            all_project_ids = []
            all_science_goals = []

            # Search recursively for files
            for root, dirnames, filenames in os.walk(os.getcwd()):

                if 'tp' in root.split(os.path.sep) and do_tp:
                    # Find the member.* directory for TP data
                    search_term = 'member.*'
                    search_type = 'tp'
                else:
                    # TODO: This will miss non '.ms.split.cal', so keeping this as a TODO for corner cases.
                    search_term = '*.ms.split.cal'
                    search_type = 'int'

                for dirname in fnmatch.filter(dirnames, search_term):

                    full_dir = os.path.join(root, dirname).split(ms_root)[1]
                    full_dir_split = full_dir.split(os.path.sep)

                    # Pull out config, product, project ID, science goal, member UID
                    if search_type == 'int':
                        config = full_dir_split[1]
                        product = None
                        project_id = full_dir_split[2]
                        science_goal = full_dir_split[3]
                        member_uid = full_dir_split[5]
                    elif search_type == 'tp':
                        config = full_dir_split[1]
                        product = full_dir_split[2]
                        project_id = full_dir_split[3]
                        science_goal = full_dir_split[4]
                        member_uid = full_dir_split[6]
                    else:
                        raise Exception('search_type %s not known!' % search_type)

                    # Use member uid to get at coordinates of the observation

                    member_uid = member_uid.replace('___', '://').replace('_', '/').replace('member.', '')
                    observations_ms = observations[member_uid == observations['member_ous_uid']][0]
                    ra, dec = observations_ms['s_ra'], observations_ms['s_dec']

                    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
                    ra_str, dec_str = coord.to_string('hmsdms').split()

                    if config not in all_configs:
                        all_configs.append(config)
                    if project_id not in all_project_ids:
                        all_project_ids.append(project_id)
                    if science_goal not in all_science_goals:
                        all_science_goals.append(science_goal)

                    target_ms_dict[full_dir] = {'config': config,
                                                'product': product,
                                                'project_id': project_id,
                                                'science_goal': science_goal,
                                                'ra': ra_str,
                                                'dec': dec_str
                                                }

            # Start writing these things out
            mosaic_no = 1
            if split_ms == 'mosaic':

                # For mosaics, group up by project ID/science goal. This captures mosaics at different times and
                # multiple mosaics within a single project

                using_mosaic = False

                if len(all_project_ids) == 1 and len(all_science_goals) == 1:
                    target_key = copy.deepcopy(target)
                else:
                    target_key = '%s_%d' % (target, mosaic_no)
                    using_mosaic = True

                for project_id in all_project_ids:
                    for science_goal in all_science_goals:
                        observation_number = {config: 1 for config in all_configs}
                        match_found = False
                        match_key = None
                        for key in target_ms_dict.keys():
                            if target_ms_dict[key]['project_id'] == project_id and \
                                    target_ms_dict[key]['science_goal'] == science_goal:
                                match_found = True
                                match_key = copy.deepcopy(key)
                                config = target_ms_dict[key]['config']
                                product = target_ms_dict[key]['product']
                                ms_file.write('%s\t%s\tall\t%s\t%d\t%s\n' %
                                              (target_key, project_id, config, observation_number[config], key))
                                observation_number[config] += 1

                                # Save any singledish info
                                if config == 'tp':
                                    singledish_info[target_key] = {'original_target': target,
                                                                   'product': product,
                                                                   'ms_filename': key}

                        if match_found:

                            ra, dec = target_ms_dict[match_key]['ra'], target_ms_dict[match_key]['dec']

                            if using_mosaic:
                                mosaic_info[target_key] = {}
                                mosaic_info[target_key]['original_target'] = target
                                mosaic_info[target_key]['ra'] = ra
                                mosaic_info[target_key]['dec'] = dec

                            mosaic_no += 1
                            target_key = '%s_%d' % (target, mosaic_no)
                            ms_file.write('\n')

            elif split_ms == 'join':

                # For join, collapse all observations for a single target down so imaging will clean everything at
                # once

                target_key = copy.deepcopy(target)
                observation_number = {config: 1 for config in all_configs}
                for key in target_ms_dict.keys():
                    config = target_ms_dict[key]['config']
                    product = target_ms_dict[key]['product']
                    project_id = target_ms_dict[key]['project_id']
                    ms_file.write('%s\t%s\tall\t%s\t%d\t%s\n' %
                                  (target_key, project_id, config, observation_number[config], key))
                    observation_number[config] += 1

                    # Save any singledish info
                    if config == 'tp':
                        singledish_info[target_key] = {'original_target': target,
                                                       'product': product,
                                                       'ms_filename': key}

                ms_file.write('\n')

            elif split_ms == 'separate':

                # For separate, every observation will be imaged separately

                target_key = '%s_%d' % (target, mosaic_no)
                observation_number = 1
                for key in target_ms_dict.keys():
                    config = target_ms_dict[key]['config']
                    product = target_ms_dict[key]['product']
                    project_id = target_ms_dict[key]['project_id']

                    ms_file.write('%s\t%s\tall\t%s\t%d\t%s\n' %
                                  (target_key, project_id, config, observation_number, key))
                    ms_file.write('\n')

                    ra, dec = target_ms_dict[key]['ra'], target_ms_dict[key]['dec']

                    mosaic_info[target_key] = {}
                    mosaic_info[target_key]['original_target'] = target
                    mosaic_info[target_key]['product'] = product
                    mosaic_info[target_key]['ra'] = ra
                    mosaic_info[target_key]['dec'] = dec

                    # Save any singledish info
                    if config == 'tp':
                        singledish_info[target_key] = {'original_target': target,
                                                       'product': product,
                                                       'ms_filename': key}

                    mosaic_no += 1
                    target_key = '%s_%d' % (target, mosaic_no)

            os.chdir(original_dir)

        ms_file.close()

        # Write out dir key

        os.system('rm -rf %s' % dir_key_file_name)
        dir_file = open(dir_key_file_name, 'w+')

        for key in mosaic_info.keys():
            target_name = mosaic_info[key]['original_target']
            dir_file.write('%s\t%s\n' % (key, target_name))

        dir_file.write('\n')
        dir_file.close()

        # Write out linmos keys

        os.system('rm -rf %s' % linmos_key_file_name)
        linmos_file = open(linmos_key_file_name, 'w+')

        for key in mosaic_info.keys():
            target_name = mosaic_info[key]['original_target']
            linmos_file.write('%s\t%s\n' % (target_name, key))

        linmos_file.write('\n')
        linmos_file.close()

        # Write out singledish keys.

        if do_tp:

            os.system('rm -rf %s' % sd_key_file_name)
            sd_file = open(sd_key_file_name, 'w+')

            for key in singledish_info.keys():
                product = singledish_info[key]['product']
                tp_filename = '%s_%s.fits' % (key, product)
                sd_file.write('%s\t%s\t%s\n' % (key, product, tp_filename))

            sd_file.close()

        # Write out target definitions. Start by renaming the original, so we can pull it back in if this gets rerun

        os.system('rm -rf %s_pre_AlmaDownload' % target_file_name)
        os.system('mv %s %s_pre_AlmaDownload' % (target_file_name, target_file_name))

        target_file = open(target_file_name, 'w+')

        for target in targets:

            ra, dec = self._kh.get_phasecenter_for_target(target=target)

            # Pull out velocity and width
            vel, vel_width = self._kh.get_system_velocity_and_velocity_width_for_target(target=target)

            target_file.write('%s\t%s\t%s\t%s\t%s\n' % (target, ra, dec, vel, vel_width))

            for target_mosaic in mosaic_info.keys():
                if mosaic_info[target_mosaic]['original_target'] == target:
                    ra = mosaic_info[target_mosaic]['ra']
                    dec = mosaic_info[target_mosaic]['dec']

                    target_file.write('%s\t%s\t%s\t%s\t%s\n' % (target_mosaic, ra, dec, vel, vel_width))

        target_file.close()
