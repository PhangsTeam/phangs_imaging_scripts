"""
utilsFieldSelection.py
"""
import os
import re
import shutil

import analysisUtils as aU
import numpy as np

from . import casaStuff

tb = aU.createCasaTool(casaStuff.tbtool)
split = casaStuff.split

# 
# Function to convert ra dec degrees to hmsdms
# 
def deg2hms(ra_deg, dec_deg):
    ra_hours = ra_deg/360*24
    ra_minutes = (ra_hours-int(ra_hours))*60
    ra_seconds = (ra_minutes-int(ra_minutes))*60
    dec_sign = np.sign(dec_deg)
    dec_minutes = (dec_deg/dec_sign-int(dec_deg/dec_sign))*60
    dec_seconds = (dec_minutes-int(dec_minutes))*60
    return 'J2000 {:02d}h{:02d}m{:.4f}s {:+02d}d{:02d}m{:.4f}s'.format(
            int(ra_hours), int(ra_minutes), ra_seconds, 
            int(dec_sign)*int(dec_deg/dec_sign), int(dec_minutes), dec_seconds)

def hms2deg(ra_dec_str):
    if ra_dec_str.startswith('J2000'):
        ra_dec_str = ra_dec_str.replace('J2000', '')
    ra_str, dec_str = ra_dec_str.strip().split()
    ra_match = re.match(r'^([0-9]+)[h:]([0-9]+)[m:]([0-9.]+)[s]*$', ra_str)
    dec_match = re.match(r'^([+-]*)([0-9]+)[d:\.]([0-9]+)[m:\.]([0-9.]+)[s]*$', dec_str)
    if not ra_match or not dec_match:
        raise ValueError('Error! Could not convert ra dec hms to deg {}'.format(ra_dec_str))
    ra_deg = (float(ra_match.group(1)) + float(ra_match.group(2))/60.0 + float(ra_match.group(3))/3600.0) / 24 * 360
    dec_deg = float(dec_match.group(1)+'1') * (
                float(dec_match.group(2)) + float(dec_match.group(3))/60.0 + float(dec_match.group(4))/3600.0)
    return ra_deg, dec_deg

# 
# Find fields that contains a given coordinate within the primary beam
# 
def extract_field_selections(
        vis, 
        lower_left_ra_dec, 
        upper_right_ra_dec, 
        field_name = None, 
        pb_factor = 1.6, # how many factor of FWHM to consider as the matching circle diameter
        pb_fwhm = None, # arcsec
        ant_diam = 12.0, # meters
        return_field_mosaic_dict = False, 
        return_string = True, 
        verbose = False, 
    ):
    # using global tb
    field_mosaic = {}
    field_mosaic['id'] = []
    field_mosaic['name'] = []
    field_mosaic['ra'] = []
    field_mosaic['dec'] = []
    field_mosaic['reference_ra'] = []
    field_mosaic['reference_dec'] = []
    field_mosaic['phase_ra'] = []
    field_mosaic['phase_dec'] = []
    field_mosaic['delay_ra'] = []
    field_mosaic['delay_dec'] = []
    field_mosaic['time'] = []
    tb.open(os.path.join(vis, 'FIELD'), nomodify=True)
    for i in range(tb.nrows()):
        name = tb.getcell('NAME', i)
        if field_name is not None:
            if name != field_name:
                continue
        reference_dir = tb.getcell('REFERENCE_DIR', i)
        phase_dir = tb.getcell('PHASE_DIR', i)
        delay_dir = tb.getcell('DELAY_DIR', i)
        reference_dec = np.rad2deg(reference_dir[1][0])
        reference_ra = np.rad2deg(reference_dir[0][0])
        if reference_ra < 0.0:
            reference_ra += 360.0
        phase_dec = np.rad2deg(phase_dir[1][0])
        phase_ra = np.rad2deg(phase_dir[0][0])
        if phase_ra < 0.0:
            phase_ra += 360.0
        delay_dec = np.rad2deg(delay_dir[1][0])
        delay_ra = np.rad2deg(delay_dir[0][0])
        if delay_ra < 0.0:
            delay_ra += 360.0
        ra, dec = delay_ra, delay_dec
        field_mosaic['id'].append(i)
        field_mosaic['name'].append(name)
        field_mosaic['ra'].append(ra)
        field_mosaic['dec'].append(dec)
        field_mosaic['reference_ra'].append(reference_ra)
        field_mosaic['reference_dec'].append(reference_dec)
        field_mosaic['phase_ra'].append(phase_ra)
        field_mosaic['phase_dec'].append(phase_dec)
        field_mosaic['delay_ra'].append(delay_ra)
        field_mosaic['delay_dec'].append(delay_dec)
        field_mosaic['time'].append(tb.getcell('TIME', i))
    tb.close()
    if verbose:
        #print('field_mosaic: \n{}'.format(pprint.pformat(field_mosaic, indent=4)))
        print('field_mosaic: {}'.format(field_mosaic))
    # 
    # compute pb
    if pb_fwhm is None:
        spw_dict = aU.getScienceSpws(vis, intent='OBSERVE_TARGET#ON_SOURCE', returnFreqRanges=True)
        #print('spw_dict.values()', spw_dict.values())
        min_freq = np.min(np.array(list(spw_dict.values())).ravel())
        pb_fwhm = 1.14*1.22*(3e8/min_freq)*3600*180/(ant_diam*3.1415926)
    match_radius = pb_factor * pb_fwhm / 2.0 # arcsec
    if verbose:
        print('match_radius: {} arcsec'.format(match_radius))
    # 
    # parse target ra dec
    if str(lower_left_ra_dec).find('m')>0 or str(lower_left_ra_dec).find(':')>0:
        lower_left_ra, lower_left_dec = hms2deg(lower_left_ra_dec)
    else:
        lower_left_ra, lower_left_dec = lower_left_ra_dec
    # 
    if str(upper_right_ra_dec).find('m')>0 or str(upper_right_ra_dec).find(':')>0:
        upper_right_ra, upper_right_dec = hms2deg(upper_right_ra_dec)
    else:
        upper_right_ra, upper_right_dec = upper_right_ra_dec
    # 
    if lower_left_ra > upper_right_ra:
        lower_left_ra, upper_right_ra = upper_right_ra, lower_left_ra
    if lower_left_dec > upper_right_dec:
        lower_left_dec, upper_right_dec = upper_right_dec, lower_left_dec
    lower_left_ra -= match_radius/3600./np.cos(np.deg2rad(lower_left_dec))
    lower_left_dec -= match_radius/3600.
    upper_right_ra += match_radius/3600./np.cos(np.deg2rad(upper_right_dec))
    upper_right_dec += match_radius/3600.
    # 
    # find valid fields that contains the given target
    valid_fields = []
    for i in range(len(field_mosaic['id'])):
        field_ra, field_dec = field_mosaic['ra'][i], field_mosaic['dec'][i]
        is_matched = False
        if ((field_ra-lower_left_ra)*(field_ra-upper_right_ra) <= 0.0) and \
           ((field_dec-lower_left_dec)*(field_dec-upper_right_dec) <= 0.0):
            is_matched = True
        if verbose:
            print('Checking field {} ra dec {:.7f} {:.7f} vs target rect {:.7f} {:.7f} {:.7f} {:.7f} (with matching buffer {:.1f} arcsec) is_matched {}'.format(
                   i, field_ra, field_dec, lower_left_ra, lower_left_dec, upper_right_ra, upper_right_dec, match_radius, is_matched))
        if is_matched:
            valid_fields.append(str(field_mosaic['id'][i]))
    if verbose:
        print('Valid fields: {}'.format(valid_fields))
    if return_string:
        valid_fields = ','.join(valid_fields)
    if return_field_mosaic_dict:
        return valid_fields, field_mosaic
    return valid_fields

# 
# Examine ant diameter
# 
def get_ant_diam(vis):
    tb.open(vis+'/ANTENNA')
    ant_diam = tb.getcol('DISH_DIAMETER').mean()
    tb.close()
    return ant_diam

# 
# Examine datacolumn
# 
def get_datacolumn(vis):
    tb.open(vis)
    if 'CORRECTED_DATA' in tb.colnames():
        datacolumn = 'CORRECTED'
    else:
        datacolumn = 'DATA'
    tb.close()
    return datacolumn


# 
# Loop input vis
# 
"""
Example 1: 
    process_ms_list(
        ms_list = ['1.ms', '2.ms', '3.ms'], 
        lower_left_ra_dec = (150.00, 1.9), 
        upper_right_ra_dec = (151.00, 2.0),
    )

Example 2: 
    process_ms_list(
        ms_list = ['1.ms', '2.ms', '3.ms'], 
        lower_left_ra_dec = 'J2000 ', 
        upper_right_ra_dec = [151.00, 2.0],
    )
"""
def process_ms_list(
        ms_list,                    # a list of measurement sets
        lower_left_ra_dec,          # a tuple or a string of RA and Dec
        upper_right_ra_dec,         # a tuple or a string of RA and Dec
        suffix = '.split.fields',   # output suffix
        pb_factor = 1.6,            # matching buffer is 160% primary beam FWHM/2. 
        overwrite = False, 
        verbose = False, 
    ):
    if suffix == '':
        raise ValueError('suffix cannot be empty!')
    for i, vis in enumerate(ms_list):
        outputvis = vis + suffix
        if os.path.exists(outputvis): 
            if not overwrite:
                print('Found existing {} and overwrite is False. Skipping it.'.format(outputvis))
                continue
            else:
                if os.path.exists(outputvis+'.backup'):
                    shutil.rmtree(outputvis+'.backup')
                shutil.move(outputvis, outputvis+'.backup')
        ant_diam = get_ant_diam(vis)
        valid_fields = extract_field_selections(vis, lower_left_ra_dec, upper_right_ra_dec, ant_diam=ant_diam, verbose=verbose)
        if valid_fields != '':
            datacolumn = get_datacolumn(vis)
            if verbose:
                print('Splitting {!r} field={!r} -> {!r} ({}/{})'.format(vis, valid_fields, outputvis, i+1, len(ms_list)))
            split(vis, outputvis, field=valid_fields, datacolumn=datacolumn)
            if verbose:
                print('Processed {!r} -> {!r} ({}/{})'.format(vis, outputvis, i+1, len(ms_list)))
