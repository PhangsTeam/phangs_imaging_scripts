# This is the line list.

import logging
import re

import numpy as np

from . import utilsLists as lists

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Drawn from Splatalogue at http://www.cv.nrao.edu/php/splat/

# Lists that combine multiple transitions for a single line

line_families = {
    'co': [
        'co10', 'co21', 'co32', 'co43', 'co54', 'co65'],
    '13co': [
        '13co10', '13co21', '13co32', '13co43', '13co54', '13co65'],
    'c18o': [
        'c18o10', 'c18o21', 'c18o32', 'c18o43', 'c18o54', 'c18o65'],
    'c17o': [
        'c17o10', 'c17o21', 'c17o32', 'c17o43', 'c17o54', 'c17o65'],
    'hcn': [
        'hcn10', 'hcn21', 'hcn32', 'hcn43', 'hcn54', 'hcn65', 'hcn76'],
    'h13cn': [
        'h13cn10', 'h13cn21', 'h13cn32', 'h13cn43', 'h13cn54', 'h13cn65',
        'h13cn76'],
    'hnc': [
        'hnc10', 'hnc21', 'hnc32', 'hnc43', 'hnc54', 'hnc65', 'hnc76'],
    'hnco': [  # Only ALMA band 3 (for now)
        'hnco43',  # 4( 0, 4)- 3( 0, 3)
        'hnco54',  # 5( 0, 5)- 4( 0, 4)
    ],
    'hn13c': [
        'hn13c10', 'hn13c21', 'hn13c32', 'hn13c43', 'hn13c54', 'hn13c65',
        'hn13c76'],
    'hcop': [
        'hcop10', 'hcop21', 'hcop32', 'hcop43', 'hcop54', 'hcop65',
        'hcop76'],
    'h13cop': [
        'h13cop10', 'h13cop21', 'h13cop32', 'h13cop43', 'h13cop54',
        'h13cop65', 'h13cop76'],
    'cs': [
        'cs10', 'cs21', 'cs32', 'cs43', 'cs54', 'cs65', 'cs76', 'cs87',
        'cs98', 'cs109', 'cs1110', 'cs1211', 'cs1312', 'cs1413'],
    '13cs': [
        '13cs10', '13cs21', '13cs32', '13cs43', '13cs54', '13cs65', '13cs76',
        '13cs87', '13cs98', '13cs109', '13cs1110', '13cs1211', '13cs1312',
        '13cs1413'],
    'cn': [  # Only ALMA band 3 (for now)
        'cn10lo',  # J=1/2-1/2
        'cn10hi',  # J=3/2-1/2
    ],
    'sio': [
        'sio10', 'sio21', 'sio32', 'sio43', 'sio54', 'sio65', 'sio76', 'sio87',
        'sio98', 'sio109', 'sio1110', 'sio1211', 'sio1312', 'sio1413',
        'sio1514', 'sio1615'],
    'hi': ['hi21cm'],
    'ci': ['ci10', 'ci21'],
    'nh3': ['nh311', 'nh322', 'nh333', 'nh344'],
    'n2hp': ['n2hp10', 'n2hp21', 'n2hp32', 'n2hp43'],
    'halpha': [  # only those in ALMA bands (for now)
        'h19alpha',
        'h21alpha',
        'h24alpha', 'h25alpha',
        'h26alpha', 'h27alpha', 'h28alpha',
        'h29alpha', 'h30alpha',
        'h31alpha', 'h32alpha', 'h33alpha',
        'h34alpha', 'h35alpha', 'h36alpha',
        'h38alpha', 'h39alpha', 'h40alpha', 'h41alpha',
        'h42alpha', 'h43alpha', 'h44alpha', 'h45alpha',
        'h53alpha', 'h54alpha', 'h55alpha', 'h56alpha', 'h57alpha', 'h58alpha',
    ],
    'oh': ['oh1612', 'oh1665', 'oh1667', 'oh1720'],
    'cch': ['cch10_21'],
    }

# The line list dictionary

line_list = {
    'co65': 691.47308,
    'co54': 576.26793,
    'co43': 461.04077,
    'co32': 345.79599,
    'co21': 230.53800,
    'co10': 115.27120,
    '13co65': 661.06728,
    '13co54': 550.92629,
    '13co43': 440.76517,
    '13co32': 330.58797,
    '13co21': 220.39868,
    '13co10': 110.20135,
    'c18o65': 658.55328,
    'c18o54': 548.83101,
    'c18o43': 439.08877,
    'c18o32': 329.33055,
    'c18o21': 219.56035,
    'c18o10': 109.78217,
    'c17o10': 112.358982,
    'c17o21': 224.714187,
    'c17o32': 337.060988,
    'c17o43': 449.395191,
    'c17o54': 561.712735,
    'c17o65': 674.009184,
    'hcn10': 88.63185,  # J=1-0, F=2-1
    'hcn21': 177.26111,  # J=2-1, F=2-1
    'hcn32': 265.88618,
    'hcn43': 354.50548,
    'hcn54': 443.11616,
    'hcn65': 531.71639,
    'hcn76': 620.30410,
    'h13cn10': 86.33992140,
    'h13cn21': 172.67785120,
    'h13cn32': 259.01179760,
    'h13cn43': 345.33976930,
    'h13cn54': 431.65977480,
    'h13cn65': 517.96982100,
    'h13cn76': 604.26791400,
    'cs10': 48.99095,
    'cs21': 97.98095,
    'cs32': 146.96903,
    'cs43': 195.95421,
    'cs54': 244.93556,
    'cs65': 293.91209,
    'cs76': 342.88285,
    'cs87': 391.84689,
    'cs98': 440.80323,
    'cs109': 489.75092,
    'cs1110': 538.68900,
    'cs1211': 587.61649,
    'cs1312': 636.53246,
    'cs1413': 685.43592,
    '13cs10': 46.24756320,
    '13cs21': 92.49430800,
    '13cs32': 138.73933500,
    '13cs43': 184.98177200,
    '13cs54': 231.22068520,
    '13cs65': 277.45540500,
    '13cs76': 323.68497300,
    '13cs87': 369.90855050,
    '13cs98': 416.12527510,
    '13cs109': 462.33429010,
    '13cs1110': 508.53473910,
    '13cs1211': 554.72576570,
    '13cs1312': 600.90648000,
    '13cs1413': 647.07615000,
    'cn10lo': 113.1912787,  # J=1/2-1/2, F=3/2-3/2
    'cn10hi': 113.4909702,  # J=3/2-1/2, F=5/2-3/2
    'hcop10': 89.18852,
    'hcop21': 178.37506,
    'hcop32': 267.55763,
    'hcop43': 356.73422,
    'hcop54': 445.90287,
    'hcop65': 535.06158,
    'hcop76': 624.20836,
    'h13cop10': 86.75428840,
    'h13cop21': 173.50670030,
    'h13cop32': 260.25533900,
    'h13cop43': 346.99834400,
    'h13cop54': 433.73383270,
    'h13cop65': 520.45988430,
    'h13cop76': 607.17464560,
    'hnc10': 90.66357,
    'hnc21': 181.32476,
    'hnc32': 271.98114,
    'hnc43': 362.63030,
    'hnc54': 453.26992,
    'hnc65': 543.89755,
    'hnc76': 634.51083,
    'hnco43': 87.925237,  # 4( 0, 4)- 3( 0, 3)
    'hnco54': 109.905749,  # 5( 0, 5)- 4( 0, 4)
    'hn13c10': 87.09085000,
    'hn13c21': 174.17940800,
    'hn13c32': 261.26331010,
    'hn13c43': 348.34026950,
    'hn13c54': 435.40796260,
    'hn13c65': 522.46407300,
    'hn13c76': 609.50628400,
    'ci10': 492.16065,  # 3P1-3P0
    'ci21': 809.34197,  # 3P2-3P1
    'sio10': 43.42376,
    'sio21': 86.84696,
    'sio32': 130.26861,
    'sio43': 173.68831,
    'sio54': 217.10498,
    'sio65': 260.51802,
    'sio76': 303.92696,
    'sio87': 347.33063,
    'sio98': 390.72845,
    'sio109': 434.11955,
    'sio1110': 477.50310,
    'sio1211': 520.87820,
    'sio1312': 564.24396,
    'sio1413': 607.59942,
    'sio1514': 650.94359,
    'sio1615': 694.27543,
    'hi21cm': 1.420405751,
    'nh311': 23.6944955,
    'nh322': 23.72263333,
    'nh333': 23.8701296,
    'nh344': 24.1394169,
    'n2hp10': 93.1733977,
    'n2hp21': 186.3446844,
    'n2hp32': 279.5117491,
    'n2hp43': 372.6724808,
    'h19alpha': 888.047022,
    'h21alpha': 662.404162,
    'h24alpha': 447.540278,
    'h25alpha': 396.900834,
    'h26alpha': 353.622747,
    'h27alpha': 316.415425,
    'h28alpha': 284.250571,
    'h29alpha': 256.302035,
    'h30alpha': 231.900928,
    'h31alpha': 210.501771,
    'h32alpha': 191.656728,
    'h33alpha': 174.995805,
    'h34alpha': 160.211511,
    'h35alpha': 147.046878,
    'h36alpha': 135.286032,
    'h38alpha': 115.274399,
    'h39alpha': 106.737357,
    'h40alpha': 99.022952,
    'h41alpha': 92.034434,
    'h42alpha': 85.688390,
    'h43alpha': 79.912651,
    'h44alpha': 74.644562,
    'h45alpha': 69.829551,
    'h53alpha': 42.951968,
    'h54alpha': 40.630498,
    'h55alpha': 38.473358,
    'h56alpha': 36.466260,
    'h57alpha': 34.596383,
    'h58alpha': 32.852196,
    'oh1612': 1.61223090,
    'oh1665': 1.66540180,
    'oh1667': 1.66735900,
    'oh1720': 1.72052990,
    'cch10_21': 87.31692500, # N= 1-0, J=3/2-1/2, F= 2-1
    }


# Run some consistency checks
def run_checks():
    """
    Some internal consistency checks.
    """
    all_okay = True

    for family in line_families:
        this_list = line_families[family]
        for this_line in this_list:
            if this_line not in line_list.keys():
                print(
                    "Line missing from line list but "
                    "in line families: " + this_line)
                all_okay = False

    if all_okay:
        print("All lines in line families present in line list.")

    no_repeats = True

    for this_line in line_list:
        for other_line in line_list:
            if this_line == other_line:
                continue
            if line_list[this_line] == line_list[other_line]:
                print(
                    "Duplicate frequencies for: " + this_line +
                    " and " + other_line + " . Check for typos.")
                no_repeats = False

    if no_repeats:
        print("No repeat frequencies in list.")


# Find line in line list
def get_line_name_and_frequency(line, exit_on_error=True):
    """
    Access the line_dictionary and return the name and frequency
    matched to some input line name.
    """

    matched_line_name = None
    matched_line_freq = None

    # try to find by input line name
    if matched_line_name is None:
        if line in line_list:
            matched_line_name = line
            matched_line_freq = line_list[matched_line_name]

    # if not found, try to find by input line name in lower case
    if matched_line_name is None:
        if line.lower() in line_list:
            matched_line_name = line.lower()
            matched_line_freq = line_list[matched_line_name]

    # if not found, try to find by input line name in lower case
    # and removed non-letters
    if matched_line_name is None:
        line_name_cleaned = re.sub(r'[^0-9a-zA-Z]', r'', line.lower())
        if line_name_cleaned in line_list:
            matched_line_name = line_name_cleaned
            matched_line_freq = line_list[matched_line_name]

    # report error
    if matched_line_name is None:
        if exit_on_error:
            logger.error(
                'Error! Could not find the input line "' + line +
                '" in our line_list module. Candiate line names are: ' +
                str(line_list.keys()))
            raise Exception(
                'Error! Could not find the input line "' + line +
                '" in our line_list module.')
        else:
            logger.warning(
                'Could not find the input line "' + line +
                '" in our line_list module. ')

    # return
    return matched_line_name, matched_line_freq


# Find line in line families
def get_line_names_in_line_family(line, exit_on_error=True):
    """
    Return the list of line names in a line family.
    """

    matched_line_family_name = None
    matched_line_names = []

    line_name_cleaned = re.sub(r'[^0-9a-zA-Z]', r'', line.lower())
    # try
    if matched_line_family_name is None:
        if line_name_cleaned in line_families:
            matched_line_family_name = line_name_cleaned
            matched_line_names.extend(line_families[line_name_cleaned])
    # report error
    if matched_line_family_name is None:
        if exit_on_error:
            logger.error(
                'Error! Could not find the input line family "' + line +
                '" in our line_list module. Candiate line families are: ' +
                str(line_families.keys()))
            raise Exception(
                'Error! Could not find the input line family "' + line +
                '" in our line_list module.')
        else:
            logger.warning(
                'Could not find the input line family "' + line +
                '" in our line_list module. ')
    # return
    return matched_line_names


def is_line_family(line_tag=''):
    line_tag_cleaned = re.sub(r'[^0-9a-zA-Z]', r'', line_tag.lower())
    return (line_tag_cleaned in line_families.keys())


def get_ghz_range_for_line(
        line=None, restfreq_ghz=None, vsys_kms=None, vwidth_kms=None,
        vlow_kms=None, vhigh_kms=None):
    """
    Return a low, high frequency range for a line code and either vsys,
    vwidth or vlow, vhigh.
    """

    # Physical constants
    sol_kms = 2.9979246e5

    vsys_method = (vsys_kms is not None) and (vwidth_kms is not None)
    vlow_method = (vlow_kms is not None) and (vhigh_kms is not None)

    if not vsys_method and not vlow_method:
        logger.warning(
            "Neither vsys+vwidth and vlow+vhigh specified. Returning.")
        return None
    elif vlow_method:
        use_vsys = False
        if vsys_method:
            logger.warning(
                "Both vsys+vwidth and vlow+vhigh specified. "
                "Using vlow method.")
    else:
        use_vsys = True

    if restfreq_ghz is None:
        if line is None:
            logging.error(
                "Specify a line name or provide a rest frequency in GHz.")
            raise Exception("No rest frequency specified.")
        restfreq_ghz = (
            get_line_name_and_frequency(line, exit_on_error=True))[1]

    if use_vsys:
        vlow_kms = vsys_kms-vwidth_kms/2.0
        vhigh_kms = vsys_kms+vwidth_kms/2.0

    line_edge_ghz = [restfreq_ghz*(1.-(vlow_kms)/sol_kms),
                     restfreq_ghz*(1.-(vhigh_kms)/sol_kms)]
    line_high_ghz = np.max(line_edge_ghz)
    line_low_ghz = np.min(line_edge_ghz)

    return line_low_ghz, line_high_ghz


def get_ghz_range_for_list(
        line_list=[], vsys_kms=None, vwidth_kms=None,
        vlow_kms=None, vhigh_kms=None):
    """
    Return a low, high frequency range for a list of line or line
    family codes and either vsys, vwidth or vlow, vhigh.
    """

    if np.isscalar(line_list):
        line_list = [line_list]

    full_line_list = []
    for this_line in line_list:
        if is_line_family(this_line):
            full_line_list.extend(get_line_names_in_line_family(this_line))
        else:
            full_line_list.append(this_line)

    initial_list = []
    for this_line in full_line_list:
        this_low, this_high = get_ghz_range_for_line(
            line=this_line, vsys_kms=vsys_kms, vwidth_kms=vwidth_kms,
            vlow_kms=vlow_kms, vhigh_kms=vhigh_kms)
        initial_list.append((this_low, this_high))

    final_list = lists.merge_pairs(initial_list)
    return final_list
