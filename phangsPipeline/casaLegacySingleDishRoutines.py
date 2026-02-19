"""
Standalone routines to analyze and manipulate single dish data.

This is based on "TP_ALMA_data_reduction/ALMA-TP-tools.py".

Last modifications:
  - Initial developments by C. Herrera.
  - 31.01.2017: read_source_coordinates
  - 01.02.2017: More than 1 line can be defined to be excluded for baseline corrections (bug fixed 21/03/2017)
  - 02.02.2017: Handle TOPO ALMA frame vs the given LSRK velocity for extraction of cube and baseline
  - 27.03.2017: extract_jyperk. It was not working for Cycle 1 data.
  - 26.07.2017: add flag of 7m antennas (CM#)
  - 26.07.2017: correct spw Tsys value associated with the averaged spw science value
                (tsysmap[spws_scie[i]+1] = spws_tsys[i]-> tsysmap[spws_scie[i]+1] = spws_tsys[ddif.argmin()])
  - 26.07.2017: modified convert_vel2chan_line, because some asap files had mixed the IFs,
                having IFNO and IFID different.
  - 10.10.2017: handle imaging of 2 SGs of the same galaxy.
  - 28.11.2017: change directory to conciliate 12m+7m data reduction with TP data reduction directory trees.
  - 01.06.2017: Add tarfile because in some projects the jyperk file is in a tar file (auxproduct.tgz).
  - 21.09.2020: Add call to GET_SOURCENAME to handle mismatched source names between the galaxy specific script and #ON_SOURCE target from the table
  - 05.09.2020: There is a version modified by C. Faesi 2020/09/05 to be run in casa 5.1 or later for Cycle 7 data.
                The modification includes setting `bdfflags=True` when calling importadsm, and the calling of
                `/bin/bdflags2MS` and `es.fixForCSV2555` and later commands in `import_and_split_ant`
                are not needed for Cycle 7 data.
  - 01.07.2021: Adapted to phangs alma pipeline, renamed the code as casaSingleDishRoutines, by D. Liu.
  - 02.07.2021: Trying to adapt for CASA 5, renamed the code as casaSingleDishNewRoutines, by D. Liu.
  - 14.02.2025: Some updates to reduce multiple lines in EBs.

Still need to do (probably outdated):
  - Work on errors when files are not found, where asdm import did not work fine, etc.
  - Add timer (suggestion by CF)
  - Add GET_SOURCENAME in main script to call the right source name. DONE CMF 21.09.2020.
  - 2021-07-05 can not split with ant='0&0' ? if no split, can not obtain a reasonable final fits image cube?!

"""
import ast
import copy
import datetime
import glob
import json
import logging
import os
import re
import shutil
import tarfile
import time

import analysisUtils as au
import numpy as np

from . import casaStuff
from .utilsSingleDish import getTPSampling, get_first_arr_val

es = au.stuffForScienceDataReduction()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#region Routines for basic characterization

#endregion

#region Routines to analyze and extract lines in measurement sets

# Physical constants
sol_kms = 2.99792458e5
c_light = sol_kms   # Speed of light in km/s
pi      = np.pi


# path constants
path_script = '../script/'       # Path to the script folder.
path_raw    = '../raw/'          # Path to the raw folder.
path_dataproduct = '../data/'    # Path to data products.

fsuffix = '.ms'

# Check if data was calibrated with the pipeline
def checkpipeline():

    if len(glob.glob(path_script+'*.xml')) > 0:
        logger.info("> Data was reduced by ALMA/JAO using an automatized pipeline ")
        logger.info("> Setting the variable 'pipeline' to True")
        return True
    else:
        logger.info("> Data was reduced by ALMA/JAO using scripts ")
        logger.info("> Setting the variable 'pipeline' to False")
        return False

# Creating CASA tools
#def createCasaTool(mytool):
#
#    if (type(casac.Quantity) != type):  # casa 4.x
#        myt = mytool()
#    else:  # casa 3.x
#        myt = mytool.create()
#    return(myt)

# Retrieve name of the column
def getDataColumnName(inputms):

    mytb = au.createCasaTool(casaStuff.tbtool)
    mytb.open(inputms)
    colnames = mytb.colnames()
    if 'FLOAT_DATA' in colnames:
        data_query= 'FLOAT_DATA'
    else:
        data_query = 'DATA'
    mytb.close()
    return(data_query)

def getDataColumnForSDBaseline(vis):
    """
    Returns the names of the corrected data columns (corrected) in a measurement set.
    """
    mytb = au.createCasaTool(casaStuff.tbtool)
    mytb.open(vis)
    names = copy.copy(mytb.colnames())
    mytb.close()
    columns = []
    for i in ['DATA','FLOAT_DATA','CORRECTED_DATA']:
        if i in names:
            columns.append(i)
    #logger.debug('getDataColumnForSDBaseline: vis = %r'%(vis))
    #logger.debug('getDataColumnForSDBaseline: colnames = %s'%(names))
    #logger.debug('getDataColumnForSDBaseline: columns = %s'%(columns))
    if 'CORRECTED_DATA' in columns:
        return 'corrected'
    elif 'FLOAT_DATA' in columns:
        return 'float_data'
    else:
        return 'data'

def getDataColumnForPlotMS(vis):
    return getDataColumnForSDBaseline(vis)

def getDataColumnForSplit(vis):
    return getDataColumnForSDBaseline(vis)

def check_data_dir_being_touched(filename, clear_failed_run = False):
    if os.path.exists(filename+'.touch'):
        if clear_failed_run:
            rm_data_dir(filename, check_being_touched = False)
            rm_data_dir(filename+'.touch', check_being_touched = False)
        else:
            logger.error("Found "+filename+'.touch! Seems something is still running or failed? Please delete the *.touch dir to start over:\n'+os.path.abspath(filename+'.touch'))
            raise Exception("Found "+filename+'.touch! Seems something is still running or failed? Please delete the *.touch dir to start over:\n'+os.path.abspath(filename+'.touch'))

def rm_data_dir(filename, check_being_touched = True):
    if check_being_touched:
        check_data_dir_being_touched(filename)
    if os.path.exists(filename):
        logger.info('Deleting '+filename)
        shutil.rmtree(filename)
    if os.path.exists(filename+'.flagversions'):
        logger.info('Deleting '+filename+'.flagversions')
        shutil.rmtree(filename+'.flagversions')

def cp_data_dir(filename_in, filename_out, check_being_touched = True, log_copied_from = False):
    if not os.path.exists(filename_in):
        logger.error("Data dir not found! Please check: "+os.path.abspath(filename_in))
        raise Exception("Data dir not found! Please check: "+os.path.abspath(filename_in))
    rm_data_dir(filename_out, check_being_touched = check_being_touched)
    logger.info('Copying '+filename_in+' to '+filename_out)
    shutil.copytree(filename_in, filename_out)
    if os.path.exists(filename_in+'.flagversions'):
        shutil.copytree(filename_in+'.flagversions', filename_out+'.flagversions')
    if log_copied_from:
        with open(filename_out+'.copied.from.txt', 'w') as fp:
            fp.write(filename_in+'\n')

# by ALMA
def scaleAutocorr(vis, scale=1., antenna='', spw='', field='', scan=''):

    if os.path.exists(vis) == False:
        logger.warning("Could not find MS.")
        return
    if os.path.exists(vis+'/table.dat') == False:
        logger.warning("No table.dat. This does not appear to be an MS.")
        return

    mymsmd = au.createCasaTool(casaStuff.msmdtool)
    mytb = au.createCasaTool(casaStuff.tbtool)

    conditions = ["ANTENNA1==ANTENNA2"]

    mymsmd.open(vis)

    if antenna != '':
        if not isinstance(antenna, (list, tuple)):
            antenna = [antenna]
        antennaids = []
        for i in antenna:
            if re.match("^[0-9]+$", str(i)): # digits only: antenna ID
                antennaids.append(int(i))
            else: # otherwise: antenna name
                antennaids.append(int(mymsmd.antennaids(i)[0]))
        conditions.append("ANTENNA1 in %s" % str(antennaids))
    if spw != '':
        if not isinstance(spw, (list, tuple)):
            spw = [spw]
        datadescids = []
        for i in spw:
            datadescids.append(int(mymsmd.datadescids(spw=int(i))[0]))
        conditions.append("DATA_DESC_ID in %s" % str(datadescids))
    if field != '':
        if not isinstance(field, (list, tuple)):
            field = [field]
        fieldids = []
        for i in field:
            if re.match("^[0-9]+$", str(i)): # digits only: field ID
                fieldids.append(int(i))
            else: # otherwise: field name
                fieldids.append(int(mymsmd.fieldsforname(i)[0]))
        conditions.append("FIELD_ID in %s" % str(fieldids))
    if scan != '':
        if not isinstance(scan, (list, tuple)):
            scan = [scan]
        scannumbers = [int(i) for i in scan]
        conditions.append("SCAN_NUMBER in %s" % str(scannumbers))

    mymsmd.close()

    logger.info("Opening the table "+vis)
    mytb.open(vis, nomodify=False)
    subtb = mytb.query(" && ".join(conditions))
    datacolumns = []
    for datacolumn in subtb.colnames():
        if datacolumn in ['DATA','FLOAT_DATA','MODEL_DATA','CORRECTED_DATA']:
            datacolumns.append(datacolumn)
    for datacolumn in datacolumns:
        try:
            data = subtb.getcol(datacolumn)
            logger.info("Dimension of the selected data: %s" % str(data.shape))
            subtb.putcol(datacolumn, data*scale)
        except:
            logger.info("An error occurred upon reading/writing the data column "+datacolumn+"! The scaleAutocorr function may have failed!")
    logger.info("Closing the table.")
    mytb.flush()
    subtb.close()
    mytb.close()

# Create vector with antenna names
def read_ants_names(filename):

    mytb = au.createCasaTool(casaStuff.tbtool)
    mytb.open(filename + '/ANTENNA')
    vec_ants = mytb.getcol('NAME')
    mytb.close()

    return vec_ants

# Correct the Tsysmap (useful for old data)
def get_tsysmap(tsysmap,spws_scie,spws_tsys,freq_rep_scie,freq_rep_tsys):

    for i in range(len(freq_rep_scie)):
        diff = [abs(freq_rep_tsys[j] - freq_rep_scie[i]) for j in range(len(freq_rep_tsys))]
        ddif = np.array(diff)
        tsysmap[spws_scie[i]]   = spws_tsys[ddif.argmin()]
        tsysmap[spws_scie[i]+1] = spws_tsys[ddif.argmin()]
    logger.info("Final map used for the observations: (they should have the same frequency)")
    for i in range(len(spws_scie)):
        logger.info('  %s, %s'%(spws_scie[i],tsysmap[spws_scie[i]]))
    return tsysmap

# Read spw information (source and Tsys)
def read_spw(filename,source):

    # Tsys spws (index)
    mytb = au.createCasaTool(casaStuff.tbtool)
    mytb.open(filename + '/SYSCAL')

    spwstsys  = mytb.getcol('SPECTRAL_WINDOW_ID')
    spws_tsys = np.unique(spwstsys).tolist()
    mytb.close()

    # Science spws (index)
    mytb.open(filename + '/SOURCE')
    names  = mytb.getcol('NAME')
    numli  = mytb.getcol('NUM_LINES')
    ss     = np.where((names == source) & (numli ==  1))
    spws_scie      = [int(get_first_arr_val(mytb.getcol('SPECTRAL_WINDOW_ID',startrow=i,nrow=1))) for i in ss[0]]
    rest_freq_scie = [float(get_first_arr_val(mytb.getcol('REST_FREQUENCY', startrow=i, nrow=1))) for i in ss[0]]
    mytb.close()
    mytb.open(filename + '/SPECTRAL_WINDOW')
    names          = mytb.getcol('NAME')
    rest_freq_scie = [rest_freq_scie[i] for i in range(len(spws_scie)) if "FULL_RES" in names[spws_scie[i]]]
    spws_scie      = [spw for spw in spws_scie if "FULL_RES" in names[spw]]
    spws_scie      = au.getScienceSpws(filename)
    spws_scie      = spws_scie.split(",")
    spws_scie = [int(i) for i in spws_scie]

    # Read number of channels, frequency at channel zero and compute representative frequency
    freq_zero_scie  = list(range(len(spws_scie)))
    chan_width_scie = list(range(len(spws_scie)))
    num_chan_scie   = list(range(len(spws_scie)))
    freq_rep_scie   = list(range(len(spws_scie)))
    for i in range(len(spws_scie)):
        freq_zero_scie[i]  = float(get_first_arr_val(mytb.getcol('REF_FREQUENCY',startrow=spws_scie[i],nrow=1)))
        chan_width_scie[i] = float(get_first_arr_val(mytb.getcol('CHAN_WIDTH',startrow=spws_scie[i],nrow=1)[0]))
        num_chan_scie[i]   = float(get_first_arr_val(mytb.getcol('NUM_CHAN',startrow=spws_scie[i],nrow=1)))
        freq_rep_scie[i]   = (num_chan_scie[i]/2*chan_width_scie[i]+freq_zero_scie[i])/1e6
    freq_zero_tsys  = list(range(len(spws_tsys)))
    chan_width_tsys = list(range(len(spws_tsys)))
    num_chan_tsys   = list(range(len(spws_tsys)))
    freq_rep_tsys   = list(range(len(spws_tsys)))
    for i in range(len(spws_tsys)):
        freq_zero_tsys[i]  = float(get_first_arr_val(mytb.getcol('REF_FREQUENCY',startrow=spws_tsys[i],nrow=1)))
        chan_width_tsys[i] = float(get_first_arr_val(mytb.getcol('CHAN_WIDTH',startrow=spws_tsys[i],nrow=1)[0]))
        num_chan_tsys[i]   = float(get_first_arr_val(mytb.getcol('NUM_CHAN',startrow=spws_tsys[i],nrow=1)))
        freq_rep_tsys[i]   =  (num_chan_tsys[i]/2*chan_width_tsys[i]+freq_zero_tsys[i])/1e6
    mytb.close()

    return spws_scie,spws_tsys,freq_rep_scie,freq_rep_tsys,chan_width_scie,num_chan_scie

# Get information of the source velocity
def read_vel_source(filename,source):

    mytb  = au.createCasaTool(casaStuff.tbtool)
    mytb.open(filename + '/SOURCE')
    names = mytb.getcol('NAME')
    numli = mytb.getcol('NUM_LINES')
    ss    = np.where((names == source) & (numli ==  1))[0]
    vel_source = float(get_first_arr_val(mytb.getcol('SYSVEL',startrow=ss[0],nrow=1)))/1e3
    vel_frame = mytb.getcolkeywords('SYSVEL')['MEASINFO']['Ref']
    logger.info("Frame of source velocity  is: "+vel_frame)
    mytb.close()

    return vel_source

# SPW where the requested line is located
def get_spw_line(vel_source,freq_rest,spws_info):

    #science spws
    spws_scie,spws_tsys,freq_rep_scie,freq_rep_tsys,chan_width_scie,num_chan_scie  = spws_info

    found = False
    for i in range(len(spws_scie)):
        freq_ini = (freq_rep_scie[i]-num_chan_scie[i]/2*chan_width_scie[i]*1e-6)/(1-vel_source/c_light)  # initial frequency in spw -> still to be check since observations are in TOPO
        freq_fin = (freq_rep_scie[i]+num_chan_scie[i]/2*chan_width_scie[i]*1e-6)/(1-vel_source/c_light)  # final frequency in spw -> still to be check since observations are in TOPO
        if freq_rest > min(freq_ini,freq_fin) and freq_rest < max(freq_ini,freq_fin):
            found = True
            return spws_scie[i]
    if found == False:
        logger.info("** Requested line with rest frequency "+str(freq_rest/1e3)+" GHz is not on the data **")
        return False

# Extract flagging from original data reduction file.
def extract_flagging(filename, pipeline, flag_dir='', flag_file=''):

    if os.path.exists(path_script+'file_flags.py'):
        shutil.move(path_script+'file_flags.py', path_script+'file_flags.py.backup')
    file_flag = open(path_script+'file_flags.py', 'w')
    #fileflagread = ori_path+'/galaxy-specific-scripts/flags-folder/'+flag_file
    if flag_dir == '' or flag_file == '':
        fileflagread = 'FILEDOESNOTEXIST'
    else:
        fileflagread = os.path.join(flag_dir, flag_file)

    if pipeline == True:
        if os.path.exists(fileflagread) == False:
            logger.info("No flagging will be done. If you want to flag something, please create a file ")
            logger.info("with the specific flags using the task sdflag."  )
            logger.info("Example: ")
            logger.info("sdflag(infile = 'uid___A002_X9998b8_X5d5.ms.PM04.asap',")
            logger.info("  mode = 'manual',")
            logger.info("  spw = '19:0~119;3960~4079,21:0~500;3960~4079',")
            logger.info("  overwrite = True)")
            logger.info(" Save it as GalName-flagfile.py in galaxy-specific-scripts/flags-folder")
        else:
            logger.info("Reading file "+fileflagread+" for flagging")
            with open(fileflagread) as f: lines_f = f.readlines()
            for i in range(len(lines_f)): file_flag.write(lines_f[i])
            logger.info("Flags saved in "+path_script+'file_flags.py')
    else:
        file_script = path_script+filename+'.scriptForSDCalibration.py'
        with open(file_script) as f: lines_f = f.readlines()
        with open(file_script) as f:
            for i, line in enumerate(f):
                ll = i
                if "sdflag(infile" in line:
                    ss = line.index("sdflag(i")
                    while len(lines_f[ll].split()) != 0:
                        file_flag.write((lines_f[ll])[ss:len(lines_f[ll])])
                        ll = ll+1
        if os.path.exists(fileflagread) == True:
            logger.info("Reading file "+fileflagread+" for flagging")
            with open(fileflagread) as f: lines_f = f.readlines()
            for i in range(len(lines_f)): file_flag.write(lines_f[i])

        logger.info("Flags saved in "+path_script+'file_flags.py')
    file_flag.close()

# Convert the given velocity to channels (using MS file)
def convert_vel2chan(filename,freq_rest,vel_cube,spw_line,vel_source,spws_info,coords):

    spws_scie,freq_rep_scie,chan_width_scie,num_chan_scie = spws_info[0],spws_info[2],spws_info[4],spws_info[5]

    spw_select = np.array(spws_scie) == spw_line
    if spw_select.sum() == 0:
        raise ValueError("Unable to find {0} in {1}".format(spw_line, spws_scie))

    freq_rep_line = np.array(freq_rep_scie)[spw_select]
    chan_width_line = np.array(chan_width_scie)[spw_select] / 1.e6
    num_chan_line = np.array(num_chan_scie, dtype=int)[spw_select]

    if freq_rep_line.size == 1:
        freq_rep_line = freq_rep_line[0]
    if chan_width_line.size == 1:
        chan_width_line = chan_width_line[0]
    if num_chan_line.size == 1:
        num_chan_line = num_chan_line[0]

    vel1 = float((vel_cube.split('~'))[0])
    vel2 = float((vel_cube.split('~'))[1])
    freq1 = (1-vel1/c_light)*freq_rest
    freq2 = (1-vel2/c_light)*freq_rest

    ra  = coords.split()[1]
    ra  = ra.replace("h",":")
    ra  = ra.replace("m",":")
    dec = coords.split()[2]
    dec = dec.replace("d",":")
    dec = dec.replace("m",":")

    date = au.getObservationStartDate(filename)
    date = (date.split()[0]).replace('-','/')+'/'+date.split()[1]

    freq1_topo = au.lsrkToTopo(freq1,date,ra,dec)
    freq2_topo = au.lsrkToTopo(freq2,date,ra,dec)

    freq_chan0  = freq_rep_line-(num_chan_line/2-0.5)*chan_width_line

    chan1 = int(round((freq1_topo-freq_chan0)/chan_width_line))
    chan2 = int(round((freq2_topo-freq_chan0)/chan_width_line))

    return min(chan1,chan2),max(chan1,chan2)

# Convert the given velocity to channels (using ASAP file with unique spw)
def convert_vel2chan_line(filename_in,freq_rest,vel_line,spw_line,coords,date):
    # freq_rest must be in units of MHz

    vel1 = float((vel_line.split('~'))[0])
    vel2 = float((vel_line.split('~'))[1])

    freq1 = (1-vel1/c_light)*freq_rest*1e6 # in units of Hz
    freq2 = (1-vel2/c_light)*freq_rest*1e6 # in units of Hz

    ra  = coords.split()[1]
    ra  = ra.replace("h",":")
    ra  = ra.replace("m",":")
    dec = coords.split()[2]
    dec = dec.replace("d",":")
    dec = dec.replace("m",":")

    freq1_topo = au.lsrkToTopo(freq1, date, ra, dec)
    freq2_topo = au.lsrkToTopo(freq2, date, ra, dec)

    if fsuffix == '.asap':
        mytb  = au.createCasaTool(casaStuff.tbtool)
        mytb.open(filename_in)
        nchan = mytb.getkeyword('nChan')
        if_eq = mytb.getcol('FREQ_ID',startrow=1,nrow=1)
        bandw = mytb.getkeyword('Bandwidth')
        mytb.close()

        mytb.open(filename_in+'/FREQUENCIES')
        freq_chanref = mytb.getcol('REFVAL',startrow=if_eq,nrow=1) # /1e6 # keep in units of Hz
        chanref      = mytb.getcol('REFPIX',startrow=if_eq,nrow=1)
        chan_width   = mytb.getcol('INCREMENT',startrow=if_eq,nrow=1) # /1e6 # keep in units of Hz
        mytb.close()

    else:
        mytb  = au.createCasaTool(casaStuff.tbtool)
        mytb.open(filename_in+os.sep+'SPECTRAL_WINDOW')
        spwids = np.arange(mytb.nrows())
        spw_line = -1
        for ispw in spwids:
            nchan = mytb.getcell('NUM_CHAN', ispw)
            chan_freqs = mytb.getcell('CHAN_FREQ', ispw)
            chan_widths = mytb.getcell('CHAN_WIDTH', ispw)
            # check if line fully in this spw
            if (np.min(chan_freqs)-freq1_topo)*(np.max(chan_freqs)-freq1_topo) < 0 and \
               (np.min(chan_freqs)-freq2_topo)*(np.max(chan_freqs)-freq2_topo) < 0:
                spw_line = ispw
                break
        if spw_line < 0:
            logger.error('Error! Could not find a spectral window that fully contains the line from frequency %s to %s Hz.'%(freq1_topo, freq2_topo))
            raise Exception('Error! Could not find a spectral window that fully contains the line from frequency %s to %s Hz.'%(freq1_topo, freq2_topo))
        nchan = mytb.getcell('NUM_CHAN', spw_line)
        chan_freqs = mytb.getcell('CHAN_FREQ', spw_line)
        chan_widths = mytb.getcell('CHAN_WIDTH', spw_line)
        chanref = 0
        freq_chanref = chan_freqs[chanref]
        chan_width = chan_widths[chanref]
        mytb.close()
        # note that the returned chan indices below start from 0
        # also note that the spw_line may change

    freq_chan0 = freq_chanref-chanref*chan_width
    chan1 = int(round((freq1_topo-freq_chan0)/chan_width))
    chan2 = int(round((freq2_topo-freq_chan0)/chan_width))

    return min(chan1,chan2),max(chan1,chan2),nchan,spw_line

# Create string with spw and channel for baseline correction
def str_spw4baseline(filename_in,freq_rest,vel_line,spw_line,coords):

    #filename = re.search('(.+?).ms',filename_in).group(0) # this is the asap data? <TODO><20210705> dzliu commented out this, not sure..

    date = au.getObservationStartDate(filename_in)
    date = (date.split()[0]).replace('-','/')+'/'+date.split()[1]
    vel_line_s = vel_line.split(';')
    nlines = len(vel_line_s)
    channels_v = list(range(nlines*2))
    for i in range(nlines):
        vel_str = vel_line_s[i]
        chan1_line,chan2_line,nchan_line,spw_line = convert_vel2chan_line(filename_in,freq_rest,vel_str,spw_line,coords,date)
        channels_v[2*i+1] = chan2_line
        channels_v[2*i]   = chan1_line
    channels_v = sorted(channels_v)
    # String to define spws for baseline correction
    spw_extr = str(spw_line)+":0~"+str(channels_v[0])+";"
    if nlines > 1:
        for i in range(nlines-1):
            spw_extr = spw_extr + str(channels_v[2*i+1])+"~"+ str(channels_v[2*i+2])+";"
    spw_extr = spw_extr + str(channels_v[-1])+"~"+str(max(channels_v[-1],nchan_line))

    return spw_extr

# Extract variable jyperk, used to convert from K to Jy.
def extract_jyperk(filename, spw_line, pipeline):
    logger.info("Extracting Jy per K conversion factor")
    if pipeline == True:
        file_script = 'jyperk.csv'
        ant_arr = []
        spw_arr = []
        val_arr = []
        if os.path.isfile(file_script) == False:
            filetgz = glob.glob("*auxproducts.tgz")
            if len(filetgz) > 0:
                tar = tarfile.open(filetgz[0])
                tar.extractall()
                tar.close()

            # Check if it's still not there or search for jyperk_query.csv
            if not os.path.exists(file_script):
                alt_file_script = 'jyperk_query.csv'
                if not os.path.exists(alt_file_script):
                    raise ValueError("Unable to find the jyperk.csv or jyperk_query.csv files.")
                else:
                    file_script = alt_file_script

        with open(file_script) as f:
            for line in f:
                if filename in line:
                    line_arr = line.split(',')
                    ant_arr.append(line_arr[1])
                    spw_arr.append(int(line_arr[2]))
                    val_arr.append(line_arr[4][0:line_arr[4].index('\n')])
        jyperk = {k: {e:{'mean':{}} for e in np.unique(spw_arr)} for k in np.unique(ant_arr)}
        for i in range(len(ant_arr)): jyperk[ant_arr[i]][spw_arr[i]]['mean']= float(val_arr[i])
        return jyperk
    else:
        file_script = path_script+filename+'.scriptForSDCalibration.py'
        vec_jyperk = ''
        with open(file_script) as f: lines_f = f.readlines()
        with open(file_script) as f:
            for i, line in enumerate(f):
                ll = i
                if "jyperk = " in line:
                    ss = line.index("jyperk")
                    while len(lines_f[ll].split()) != 0:
                        if ll == i+1: ss2 = lines_f[ll].index("{")
                        if ll == i:
                            vec_jyperk = vec_jyperk+(lines_f[ll])[ss:len(lines_f[ll])]
                        else:
                            vec_jyperk = vec_jyperk+(lines_f[ll])[ss2:len(lines_f[ll])]
                        ll = ll+1
        kw = {}
        exec(vec_jyperk) in kw
        jyperk = kw['jyperk']
        return jyperk

# Read source coordinates
def read_source_coordinates(filename,source):

    coord_source = au.getRADecForSource(filename,source)
    RA_h  = (coord_source.split(' ')[0]).split(':')[0]
    RA_m  = (coord_source.split(' ')[0]).split(':')[1]
    RA_s  = (coord_source.split(' ')[0]).split(':')[2]
    DEC_d = (coord_source.split(' ')[1]).split(':')[0]
    DEC_m = (coord_source.split(' ')[1]).split(':')[1]
    DEC_s = (coord_source.split(' ')[1]).split(':')[2]
    coord = "J2000  "+str(RA_h)+"h"+str(RA_m)+"m"+str(RA_s[0:6])+" "+str(DEC_d)+"d"+str(DEC_m)+"m"+str(DEC_s)
    return coord

# Get source name
def get_sourcename(filename,
                   source='all',
                   ):

    mytb   = au.createCasaTool(casaStuff.msmdtool)
    mytb.open(filename)

    # Get source field names
    fieldnames = mytb.fieldnames()

    # If we're looking for all intents, then just take the (first) science target
    if source == 'all':
        field = mytb.fieldsforintent('OBSERVE_TARGET#ON_SOURCE')[0]

    # Else, key in on the source name. Do this in a regex-style
    else:

        # For field selection, this can be done with forward slashes,
        # so make sure to strip them
        if source.startswith("/"):
            source = source.strip("/")

        # Get science fieldnames as names
        fieldnames_asname = mytb.fieldsforintent('OBSERVE_TARGET#ON_SOURCE', asnames=True)

        regex_matches = [re.fullmatch(source, x) for x in fieldnames_asname]
        regex_idxs = [x is not None for x in regex_matches]
        found_field = any(regex_idxs)

        # If we've found a match, update the source name with the first
        # fieldname
        if found_field:
            source = fieldnames_asname[regex_idxs][0]

        if not found_field:
            logger.warning(f"source {source} not in field names: {', '.join(fieldnames)}")
            return None

        field = mytb.fieldsforname(source)

        # If we don't find anything, get the science targets
        if len(field) == 0:
            field = mytb.fieldsforintent('OBSERVE_TARGET#ON_SOURCE')

        # Take the first result
        field = field[0]

    source_name = fieldnames[field]

    mytb.close()

    return source_name

# Create string of spws to apply the Tsys
def str_spw_apply_tsys(spws_info):
    #science spws
    spws_scie,spws_tsys,freq_rep_scie,freq_rep_tsys = spws_info[0:4]

    spws_all = spws_tsys+spws_scie
    #spws_all = sorted(spws_all)
    spws_all = sorted(list(set(spws_all)))
    #spws_tsys_str = (str(spws_tsys))[1:len(str(spws_tsys))-1]
    #spws_scie_str = (str(spws_scie))[1:len(str(spws_scie))-1]
    #spws_all_str  = (str(spws_all))[1:len(str(spws_all))-1]
    spws_tsys_str = ','.join([str(t) for t in spws_tsys])
    spws_scie_str = ','.join([str(t) for t in spws_scie])
    spws_all_str = ','.join([str(t) for t in spws_all])

    return spws_scie_str,spws_tsys_str,spws_all_str

# Check date of observations to decide if the non-linearity correction should be applied or not.
def check_date_nonlinearity(filename):

    date_obs    = au.getObservationStart(filename)/24/60/60.
    date_change = au.dateStringToMJD('2015/10/01 00:00:00')

    if abs(date_obs-date_change) <= 1:
        logger.info("Data obtained within 1 day of the change, be careful!"    )
    if date_obs >= date_change:
        logger.info("Data obtained after 2015/10/01, non-linearity not applied")
        return False
    if date_obs < date_change:
        logger.info("Data obtained before 2015/10/01, non-linearity applied")
        return True

# Check if we are in the correct directory
def checkdir(currentdir,path_galaxy):
    if path_galaxy in currentdir:
        return True
    else:
        return False

def checktmp():
    if os.path.isdir('../'+path_galaxy) == False:
        logger.info("Temporal folder does not exists. Creating it and copying raw data")
        os.system('mkdir -p ../'+path_galaxy)
        os.system('cp -rf  ../../../'+path_galaxy[4:-1]+'/calibration ../'+path_galaxy)
        os.system('cp -rf  ../../../'+path_galaxy[4:-1]+'/raw         ../'+path_galaxy)
        os.system('cp -rf  ../../../'+path_galaxy[4:-1]+'/script      ../'+path_galaxy)

#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# Data reduction steps
#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
#
# Step 0
#*-*-*-*-*-*
def check_exists(filename):
    filename_asdm = filename[0:filename.find('.ms')]+'.asdm.sdm'
    logger.info("> Checking ALMA raw data existence: ")
    logger.info("  "+os.path.abspath(os.path.join(path_raw, filename_asdm))+" "+str(os.path.exists(path_raw+filename_asdm)))
    if os.path.exists(path_raw+filename_asdm) == True:
        return True
    else:
        logger.info("** Original ALMA data "+filename_asdm +" does NOT exist: **")
        logger.info("   Skipping file ")
        return False

#*-*-*-*-*-*_*-*-*-*-*-*
# Step 1  Import data
#*-*-*-*-*-*-*-*-*-*-*
def import_and_split_ant(filename, precycle7=True, doallants=True, dosplitants=True, doplots=True):
    """Import and split antenna for single dish raw data.

    We will copy the raw "*.asdm.sdm" data from original place to working place.

    Args:
        filename (str): The data name for output with suffix ".ms". Does not include the file path, which should be defined in the global variable `path_raw`.
        precycle7 (bool): Whether the data is taken pre-Cycle7, i.e., Cycle 0-6.
        doallants (bool): Whether making an MS data with all antennae in it.
    """
    # <TODO> Can we be more smart on defining the precycle7 variable?

    logger.info("==================================================")
    logger.info("= Step 1 - Import ASDM data and split by antenna =")
    logger.info("==================================================")
    logger.info("Current directory: "+os.getcwd())

    if os.path.exists('done_step_1_for_'+filename[0:-3]):
        logger.info('Found file: done_step_1_for_'+filename[0:-3]+'. Will not re-do step 1.')
        return

    if not os.path.isdir('plots'):
        os.makedirs('plots') # folder containing all plots
    if not os.path.isdir('obs_lists'):
        os.makedirs('obs_lists') # folder containing all observation lists (i.e., listobs, sdlist)


    # 1.1 Import of the ASDM
    logger.info("1.1 Importing from ASDM to MS")

    # clear up previous failed runs if *.touch exists
    check_data_dir_being_touched(filename, clear_failed_run=True)

    if not os.path.exists(filename):

        # mark the current running with a *.touch directory
        os.mkdir(filename+'.touch')

        # copy raw data to filename0, then run importasdm
        filename0 = filename[0:filename.find('.ms')] # remove the suffix ".ms"
        if not (os.path.exists(filename0) and os.path.isfile(filename0+'.copied.from.txt')):
            cp_data_dir(path_raw+filename0+'.asdm.sdm', filename0, log_copied_from = True)

        if precycle7:
            bdfflags=False
        else:
            bdfflags=True

        logger.info('Running CASA importasdm: '+filename0+' -> '+filename)
        casaStuff.importasdm(filename0,
            asis='Antenna Station Receiver Source CalAtmosphere CalWVR CorrelatorMode SBSummary',
            bdfflags=bdfflags,
            process_caldevice=False,
            with_pointing_correction=True)

        if precycle7:

            # Transfer specific flags (BDF flags) from the ADSM to the MS file
            logger.info(os.environ['CASAPATH'].split()[0]+'/bin/bdflags2MS -f "COR DELA INT MIS SIG SYN TFB WVR ZER" '+filename0+' '+filename)
            os.system(os.environ['CASAPATH'].split()[0]+'/bin/bdflags2MS -f "COR DELA INT MIS SIG SYN TFB WVR ZER" '+filename0+' '+filename)

            # Check for known issue, CSV-2555: Inconsistency in FIELD_ID, SOURCE_ID and Spw_ID in single dish data
            es.fixForCSV2555(filename)

        # 1.2 Listobs
        logger.info("1.2 Creating listobs for MS file")
        outname = filename+'.listobs.txt'
        if os.path.exists('obs_lists/'+outname):
            os.system('rm -rf obs_lists/'+outname)
        casaStuff.listobs(vis = filename,
            listfile = 'obs_lists/'+outname)

        if doplots == True:
            logger.info("Running getTPSampling, saving plots to "+'plots/'+filename+'.sampling.png')
            getTPSampling(vis = filename,
            showplot = True,
            plotfile = 'plots/'+filename+'.sampling.png')

        # 1.3 A priori flagging: e.g., mount is off source, calibration device is not in correct position, power levels are not optimized, WCA not loaded...
        logger.info("1.3 Applying a priori flagging, check plots/"+filename+".flagcmd.png plot to see these flags.")
        if doplots:
            casaStuff.flagcmd(vis = filename,
                inpmode = 'table',
                useapplied = True,
                action = 'plot',
                plotfile = 'plots/'+filename+'.flagcmd.png')

        casaStuff.flagcmd(vis = filename,
            inpmode = 'table',
            useapplied = True,
            action = 'apply')

        # mark the current running as finished by deleting the *.touch directory
        os.rmdir(filename+'.touch')

    else:

        logger.info('Found imported data: '+filename+' - Steps 1.2 and 1.3 are skipped.')

    # If there are, flag 7m antennas
    vec_ants = read_ants_names(filename)
    ants_7m = [s for s in vec_ants if "CM" in s]
    if len(ants_7m) > 0:
        logger.info('Found 7m antennae, flagging those.')
        str_ants = ', '.join(ants_7m)
        casaStuff.flagdata(vis = filename,
                 mode = 'manual',
                 antenna = str_ants,
                 action = 'apply')

    # if doallants, make an MS with all antennae in it
    if doallants:
        cp_data_dir(filename, filename+'.allant'+fsuffix)

    # if dosplitants, make an MS for each antenna, with a file name like filename+'.'+ant+fsuffix
    if dosplitants:
        # 1.4 Split by antenna
        logger.info("1.4 Splitting the file by antennas")

        vec_ants_t = read_ants_names(filename)
        vec_ants = [s for s in vec_ants_t if any(xs in s for xs in ['PM','DV'])]
        for ant in vec_ants:
            rm_data_dir(filename+'.'+ant+fsuffix)

        for ant in vec_ants:
            use_casa_split_antenna = True
            if use_casa_split_antenna:
                logger.info('Running split to make '+filename+'.'+ant+fsuffix+', datacolumn is '+getDataColumnForSplit(filename))
                casaStuff.split(vis = filename,
                    outputvis = filename+'.'+ant+fsuffix,
                    antenna = '%s&&&'%(ant),
                    datacolumn = getDataColumnForSplit(filename))
                    #<Note># CASA split with antenna = '0&0' does not work, should use '0&&&' to get only autocorrelations,
                    #        see https://casa.nrao.edu/docs/taskref/split-task.html
            else:
                #<TODO># these are not well tested
                # this is an alternative way to split single antenna autocorr data
                filename_in = filename
                filename_out = filename+'.'+ant+fsuffix+'.tmp'
                cp_data_dir(filename_in, filename_out)
                #
                other_ants = copy.copy(vec_ants)
                other_ants.remove(ant)
                str_other_ants = ';'.join(other_ants)
                logger.info('Running flagdata to flag '+str_other_ants+' in '+filename_out)
                casaStuff.flagdata(vis = filename_out,
                                   mode = 'manual',
                                   antenna = str_other_ants,
                                   action = 'apply')
                #
                filename_in = filename+'.'+ant+fsuffix+'.tmp'
                filename_out = filename+'.'+ant+fsuffix
                rm_data_dir(filename_out)
                logger.info('Running split to make '+filename_out+', datacolumn is '+getDataColumnForSplit(filename_in))
                casaStuff.split(vis = filename_in,
                                outputvis = filename_out,
                                keepflags = False,
                                datacolumn = getDataColumnForSplit(filename_in))

        #1.5 sdlist
        logger.info("1.5 Create listobs for each splitted file.")
        for ant in vec_ants:
            if os.path.exists('obs_lists/'+filename+'.'+ant+fsuffix+'.listobs.txt'):
                os.remove('obs_lists/'+filename+'.'+ant+fsuffix+'.listobs.txt')
            casaStuff.listobs(vis = filename+'.'+ant+fsuffix+'',
                listfile = 'obs_lists/'+filename+'.'+ant+fsuffix+'.listobs.txt')


    with open('done_step_1_for_'+filename[0:-3], 'w') as outlogfile:
        outlogfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + time.strftime('%Z'))


#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# Step 2  Generate Tsys and apply flagging
#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

def gen_tsys_and_flag(filename, spws_info, pipeline, flag_dir='', flag_file='', doplots=False):

    logger.info("========================================================")
    logger.info(" Step 2  Generate Tsys and apply flagging")
    logger.info("========================================================")
    logger.info("Current directory: "+os.getcwd())

    if os.path.exists('done_step_2_for_'+filename[0:-3]):
        logger.info('Found file: done_step_2_for_'+filename[0:-3]+'. Will not re-do step 2.')
        with open(filename+'.spwmap.json', 'r') as fp:
            spwmap = json.load(fp)
            spwmap = ast.literal_eval(json.dumps(spwmap)) # Removing uni-code chars
        return spwmap

    #if checkdir(os.getcwd(),path_galaxy) == False:
    #    os.chdir('../'+path_galaxy+'calibration')
    if not os.path.isdir('plots'):
        os.makedirs('plots') # folder containing all plots

    # 2.1 Generation of the Tsys cal table
    logger.info(" 2.1 Generating Tsys calibration table")

    rm_data_dir(filename+'.tsys')

    logger.info('Running gencal to make '+filename+'.tsys')
    casaStuff.gencal(vis = filename,
                     caltable = filename+'.tsys',
                     caltype = 'tsys')

    # 2.2 Create png plots of CASA Tsys and bandpass solution
    logger.info(" 2.2 Create plots of Tsys and bandpass solution")

    if doplots:
        if os.path.exists('plots/'+filename+'.tsys.plots.overlayTime/'+filename+'.tsys'):
            os.system('rm -Rf plots/'+filename+'.tsys.plots.overlayTime/'+filename+'.tsys')

        casaStuff.plotbandpass(caltable=filename+'.tsys',
                               overlay='time',
                               xaxis='freq', yaxis='amp',
                               subplot=22,
                               buildpdf=False,
                               interactive=False,
                               showatm=True,
                               pwv='auto',
                               chanrange='92.1875%',
                               showfdm=True,
                               field='',
                               figfile='plots/'+filename+'.tsys.plots.overlayTime/'+filename+'.tsys')

        # Create png plots for Tsys per source with antennas
        es.checkCalTable(filename+'.tsys', msName=filename, interactive=False)
        if os.path.exists('plots/'+filename+'.tsys.plots'):
            os.system('rm -rf plots/'+filename+'.tsys.plots')
        os.system('mv '+filename+'.tsys.plots'+' '+'plots/')

    # 2.3 Do initial flagging
    logger.info("2.3 Initial flagging, reading flags in file file_flags.py. You can modify this file to add more flags")
    extract_flagging(filename, pipeline, flag_dir=flag_dir, flag_file=flag_file)    # Extract flags from original ALMA calibration script (sdflag entries)
    if os.path.exists(path_script+'file_flags.py'):
        #execfile(path_script+'file_flags.py')    #<TODO><DZLIU>#
        exec(open(path_script+'file_flags.py').read())

    # 2.4 Create Tsys map
    logger.info("2.4 Creating Tsysmaps" )
    # Read spws and frquencies for science and tsys
    spws_scie,spws_tsys,freq_rep_scie,freq_rep_tsys = spws_info[0:4]

    #from recipes.almahelpers import tsysspwmap
    tsysmap = casaStuff.tsysspwmap(vis = filename, tsystable = filename+'.tsys', trim = False)

    logger.info("Spectral windows for science are: %s, %s"%(spws_scie, freq_rep_scie))
    logger.info("Spectral windows for tsys are   : %s, %s"%(spws_tsys, freq_rep_tsys))
    logger.info("Original map between science and tsys spws: (they should have the same frequency)")
    for i in range(len(spws_scie)):
        logger.info('%s, %s'%(spws_scie[i],tsysmap[spws_scie[i]]))

    #tsysmap = get_tsysmap(tsysmap,spws_scie,spws_tsys,freq_rep_scie,freq_rep_tsys)

    spwmap = {}
    for i in spws_scie:
        if not tsysmap[i] in spwmap.keys():
            spwmap[tsysmap[i]] = []
        spwmap[tsysmap[i]].append(i)

    with open(filename+'.spwmap.json', 'w') as fp:
        json.dump(spwmap, fp, sort_keys=True, indent=4) # write spwmap to json file

    with open('done_step_2_for_'+filename[0:-3], 'w') as outlogfile:
        outlogfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + time.strftime('%Z'))

    return spwmap

#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# Step 3  From counts to Kelvin
#-*-*-*-*-*-*-*-*-*-*
def counts2kelvin(filename, ant_list=None, spws_info=None, spwmap=None, doplots=False):

    logger.info("==================================")
    logger.info("= Step 3 - From counts to Kelvin =")
    logger.info("==================================")
    logger.info("Current directory: "+os.getcwd())

    if os.path.exists('done_step_3_for_'+filename[0:-3]):
        logger.info('Found file: done_step_3_for_'+filename[0:-3]+'. Will not re-do step 3.')
        return

    if ant_list is None:
        ant_list = []

    if spwmap is None:
        logger.error('Error! spwmap is not defined when calling counts2kelvin()!')
        raise Exception('Error! spwmap is not defined when calling counts2kelvin()!')

    if spws_info is None:
        logger.error('Error! spws_info is not defined when calling counts2kelvin()!')
        raise Exception('Error! spws_info is not defined when calling counts2kelvin()!')

    if not os.path.isdir('plots'):
        os.makedirs('plots') # folder containing all plots

    logger.info("3.1 Converting data into Kelvin Ta* = Tsys * (ON-OFF)/OFF")

    # Get string with needed spws to apply Tsys
    spws_scie_str, spws_tsys_str, spws_all_str = str_spw_apply_tsys(spws_info)
    print('filename: '+str(filename))
    print('ant_list: '+str(ant_list))
    print('spws_scie_str: '+str(spws_scie_str))
    print('spws_tsys_str: '+str(spws_tsys_str))
    print('spws_all_str: '+str(spws_all_str))
    print('spwmap: '+str(spwmap))
    print('doplots: '+str(doplots))

    fin    = fsuffix
    finout = fsuffix+'.2'

    if len(ant_list) == 0:
        ant_list = [None]

    for ant in ant_list:

        if ant is not None:
            filename_in  = filename+'.'+ant+fin
            filename_out = filename+'.'+ant+finout
        else:
            filename_in  = filename+'.allant'+fin
            filename_out = filename+'.allant'+finout

        rm_data_dir(filename_out)

        cp_data_dir(filename_in, filename_out)

        logger.info('Running sdcal to make '+filename_out)
        casaStuff.sdcal(infile = filename_out,
                calmode = 'ps,tsys,apply',
                spw = spws_all_str,
                spwmap = spwmap,
                outfile = filename_out,
                overwrite = True,
                )
        # -- https://casa.nrao.edu/casadocs/casa-5.4.1/single-dish-calibration/single-dish-data-calibration-and-reduction
        # Note that we didn't specify the Tsys spectral windows in the call to sdcal.
        # For ALMA single-dish data from Cycle 3 onward, this is okay since the Tsys
        # and science data share the same spectral window.
        # Alternatively, the mapping between the Tsys
        # and science spectral windows can be explicitly set with spwmap and spw.
        # In this case, we would use:
        # sdcal(infile=vis, calmode='ps,tsys,apply', spwmap={17:[17], 19:[19], 21:[21],23:[23]}, spw='17,19,21,23')

        if doplots == True:
            es.SDcheckSpectra(filename_out, msName=filename_out, spwIds=spws_scie_str, interactive=False)
            # must use new analysisUtils.py with getCasaVersion()
            # this will create plot files in directory filename_out+'.plots'
            # note that these plots are uncalibrated


        apply_nl = check_date_nonlinearity(filename)
        if apply_nl == True:
            logger.info("3.2 Applying non-linearity correction factor if data were obtained before the 2015-10-01")
            logger.warning("This only works in pre-CASA 5!")

        # end for ant loop

    with open('done_step_3_for_'+filename[0:-3], 'w') as outlogfile:
        outlogfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + time.strftime('%Z'))


#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# Step 4  Extract the cube including the line
#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
def extract_cube(filename, source, name_line, ant_list=None, freq_rest=None, spws_info=None, vel_source=None, vel_cube=None, doplots=False, overwrite=False):

    logger.info("=========================================================")
    logger.info("= Step 4 - Extracting cube including the requested line =")
    logger.info("=========================================================")
    logger.info("Current directory: "+os.getcwd())

    if os.path.exists('done_step_4_for_'+filename[0:-3]+'_'+name_line):
        logger.info('Found file: done_step_4_for_'+filename[0:-3]+'_'+name_line+'. Will not re-do step 4.')
        return

    if ant_list is None:
        ant_list = []

    if freq_rest is None:
        logger.error('Error! freq_rest is not defined when calling extract_cube()!')
        raise Exception('Error! freq_rest is not defined when calling extract_cube()!')

    if spws_info is None:
        logger.error('Error! spws_info is not defined when calling extract_cube()!')
        raise Exception('Error! spws_info is not defined when calling extract_cube()!')

    if vel_source is None:
        logger.error('Error! vel_source is not defined when calling extract_cube()!')
        raise Exception('Error! vel_source is not defined when calling extract_cube()!')

    if vel_cube is None:
        logger.error('Error! vel_cube is not defined when calling extract_cube()!')
        raise Exception('Error! vel_cube is not defined when calling extract_cube()!')

    if not os.path.isdir('plots'):
        os.makedirs('plots') # folder containing all plots
    if not os.path.isdir('obs_lists'):
        os.mkdirs('obs_lists') # folder containing all observation lists (i.e., listobs, sdlist)

    # Defining extensions
    fin    = fsuffix+'.2'
    finout = fsuffix+'.3'+'.'+name_line

    if len(ant_list) == 0:
        ant_list = [None]

    for ant in ant_list:

        if ant is not None:
            filename_in  = filename+'.'+ant+fin
            filename_out = filename+'.'+ant+finout
        else:
            filename_in  = filename+'.allant'+fin
            filename_out = filename+'.allant'+finout

        # Get the spw where the requested line is located
        spw_line = get_spw_line(vel_source,freq_rest,spws_info)

        logger.info("source: "+str(source))
        logger.info("vel_source: "+str(vel_source))
        logger.info("freq_rest: "+str(freq_rest))
        logger.info("spw_line: "+str(spw_line))

        # Plotting the line
        if doplots:

            plotfile = 'plots/'+filename_in+'.spw'+str(spw_line)+'.spec.png'
            if os.path.exists(plotfile) and overwrite:
                os.remove(plotfile)
            if not os.path.exists(plotfile):
                logger.info("4.1 Plotting each spw")
                logger.info('Running plotms to make '+plotfile)
                casaStuff.plotms(vis=filename_in,
                    ydatacolumn=getDataColumnForPlotMS(filename_in),
                    intent='OBSERVE_TARGET#ON_SOURCE',
                    field=source, spw=str(spw_line),
                    averagedata=True, avgtime='86400', avgscan=True,
                    xaxis='vel', yaxis='amp', coloraxis='ant1', showlegend=True,
                    iteraxis='corr', xselfscale=True, xsharedaxis=True, gridrows=2,
                    highres=True, dpi=300, showmajorgrid=True, majorstyle='dot',
                    plotfile=plotfile, overwrite=True,
                    )

        # Get the string of the channels to be extracted from the original cube
        coords = read_source_coordinates(filename,source)
        chan1_cube,chan2_cube = convert_vel2chan(filename,freq_rest,vel_cube,spw_line,vel_source,spws_info,coords)
        spw_extr = str(spw_line)+":"+str(chan1_cube)+"~"+str(chan2_cube)

        logger.info("4.2 Extracting a cube with the line")

        rm_data_dir(filename_out)

        logger.info('Running split to make '+filename_out+', datacolumn is '+getDataColumnForSplit(filename_in))
        casaStuff.split(vis=filename_in,
               field=source,
               spw=spw_extr,
               outputvis=filename_out,
               datacolumn=getDataColumnForSplit(filename_in))

        listfile = 'obs_list/'+filename_out+'.listobs.txt'
        if os.path.exists(listfile):
            logger.info('Deleting '+listfile)
            os.remove(listfile)
        logger.info('Running listobs to make '+listfile)
        casaStuff.listobs(vis=filename_out,
               listfile=listfile)

        if doplots == True:
            logger.info("4.3 Plotting the line spectrum averaged in time")
            if name_line != '':
                name_line2 = re.sub(r'_([0-9]+kmsres)', r'_originalres', name_line)
            else:
                name_line2 = 'unknown'
            plotfile = 'plots/'+filename_out+'.line.'+name_line2+'.spec.png'
            if os.path.exists(plotfile):
                os.remove(plotfile)
            logger.info('Running plotms to make '+plotfile)
            casaStuff.plotms(vis=filename_out,
                ydatacolumn=getDataColumnForPlotMS(filename_out),
                intent='OBSERVE_TARGET#ON_SOURCE',
                restfreq=str(freq_rest)+'MHz',
                averagedata=True, avgtime='86400', avgscan=True,
                xaxis='vel', yaxis='amp', coloraxis='ant1', showlegend=True,
                iteraxis='corr', xselfscale=True, xsharedaxis=True, gridrows=2,
                highres=True, dpi=300, showmajorgrid=True, majorstyle='dot',
                plotfile=plotfile, overwrite=True,
                )

        # end for ant loop

    with open('done_step_4_for_'+filename[0:-3]+'_'+name_line, 'w') as outlogfile:
        outlogfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + time.strftime('%Z'))

#-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# Step 5 Baseline correction
#-*-*-*-*-*-*-*-*-*-*

def baseline(filename, source, name_line, ant_list=None, freq_rest=None, spws_info=None, vel_source=None, vel_line=None, bl_order=1, doplots=True):
    logger.info("================================")
    logger.info("= Step 5 - Baseline correction =")
    logger.info("================================")
    logger.info("Current directory: "+os.getcwd())

    if os.path.exists('done_step_5_for_'+filename[0:-3]+'_'+name_line):
        logger.info('Found file: done_step_5_for_'+filename[0:-3]+'_'+name_line+'. Will not re-do step 5.')
        return

    if ant_list is None:
        ant_list = []

    if freq_rest is None:
        logger.error('Error! freq_rest is not defined when calling baseline()!')
        raise Exception('Error! freq_rest is not defined when calling baseline()!')

    if spws_info is None:
        logger.error('Error! spws_info is not defined when calling baseline()!')
        raise Exception('Error! spws_info is not defined when calling baseline()!')

    if vel_source is None:
        logger.error('Error! vel_source is not defined when calling baseline()!')
        raise Exception('Error! vel_source is not defined when calling baseline()!')

    if vel_line is None:
        logger.error('Error! vel_line is not defined when calling baseline()!')
        raise Exception('Error! vel_line is not defined when calling baseline()!')

    if not os.path.isdir('plots'):
        os.makedirs('plots') # folder containing all plots

    # Definition of extension
    fin    = fsuffix+'.3'+'.'+name_line
    finout = fsuffix+'.4'+'.'+name_line

    if len(ant_list) == 0:
        ant_list = [None]

    for ant in ant_list:

        if ant is not None:
            filename_in  = filename+'.'+ant+fin
            filename_out = filename+'.'+ant+finout
        else:
            filename_in  = filename+'.allant'+fin
            filename_out = filename+'.allant'+finout

        # Extract the ID of the spw where the line is
        spw_line = get_spw_line(vel_source,freq_rest,spws_info)

        # Convert the velocity range in channels and get spw string for baseline fitting
        coords   = read_source_coordinates(filename,source)
        spw_extr = str_spw4baseline(filename_in,freq_rest,vel_line,spw_line,coords)

        # Subtracting the baseline
        rm_data_dir(filename_out)

        logger.info('Running sdbaseline to make '+filename_out+', spw = '+str(spw_extr)+', order = '+str(bl_order))
        try:
            casaStuff.sdbaseline(infile = filename_in,
                             datacolumn = getDataColumnForSDBaseline(filename_in),
                             spw = spw_extr,
                             maskmode = 'list',
                             blfunc = 'poly',
                             order = bl_order,
                             outfile = filename_out,
                             overwrite = True)
        except: # CASA 4.7.X sdbaseline has no datacolumn arg
            casaStuff.sdbaseline(infile = filename_in,
                             #datacolumn = getDataColumnForSDBaseline(filename_in),
                             spw = spw_extr,
                             maskmode = 'list',
                             blfunc = 'poly',
                             order = bl_order,
                             outfile = filename_out,
                             overwrite = True)

        if doplots:
            # PLotting the result from the baseline correction. Spectra avergarfed in time
            plotfile = 'plots/'+filename_out+'_baseline_corrected.png'
            if os.path.exists(plotfile):
                os.remove(plotfile)
            logger.info('Running plotms to make '+plotfile)
            casaStuff.plotms(vis=filename_out,
                ydatacolumn=getDataColumnForPlotMS(filename_out),
                intent='OBSERVE_TARGET#ON_SOURCE',
                restfreq=str(freq_rest)+'MHz',
                averagedata=True, avgtime='86400', avgscan=True,
                xaxis='vel', yaxis='amp', coloraxis='ant1', showlegend=True,
                iteraxis='corr', xselfscale=True, xsharedaxis=True, gridrows=2,
                highres=True, dpi=300, showmajorgrid=True, majorstyle='dot',
                plotfile=plotfile, overwrite=True,
                )

        os.system('mv  *blparam.txt  obs_lists/')

        # end for ant loop

    with open('done_step_5_for_'+filename[0:-3]+'_'+name_line, 'w') as outlogfile:
        outlogfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + time.strftime('%Z'))

#-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# Step 6 Concatenate antennas
#-*-*-*-*-*-*-*-*-*-*
def concat_ants(filename, name_line, ant_list=None, freq_rest=None, spws_info=None, vel_source=None, pipeline=True):

    logger.info("========================================================")
    logger.info("= Step 6 - Concatenate antennas and K to Jy conversion =")
    logger.info("========================================================")
    logger.info("Current directory: "+os.getcwd())

    if os.path.exists('done_step_6_for_'+filename[0:-3]+'_'+name_line):
        logger.info('Found file: done_step_6_for_'+filename[0:-3]+'_'+name_line+'. Will not re-do step 6.')
        return

    if ant_list is None:
        ant_list = []

    if freq_rest is None:
        logger.error('Error! freq_rest is not defined when calling extract_cube()!')
        raise Exception('Error! freq_rest is not defined when calling extract_cube()!')

    if spws_info is None:
        logger.error('Error! spws_info is not defined when calling extract_cube()!')
        raise Exception('Error! spws_info is not defined when calling extract_cube()!')

    if vel_source is None:
        logger.error('Error! vel_source is not defined when calling extract_cube()!')
        raise Exception('Error! vel_source is not defined when calling extract_cube()!')

    # Defining extensions
    fin    = fsuffix+'.4'+'.'+name_line
    finout = '.ms.5'+'.'+name_line

    # check antenna list
    #if len(ant_list) == 0:
    #    ant_list = [None]

    # prepare antenna list to concate
    #lis_fils = [f for f in os.listdir(".") if (f.endswith(fin) and f.startswith(filename))]
    #vec_As   = [f[f.find(filename)+len(filename)+1:f.rfind(fin)] for f in lis_fils]
    if len(ant_list) > 0:
        lis_fils = []
        for ant in ant_list:
            filename_in = filename+'.'+ant+fin
            filename_out = filename+'.'+ant+finout
            rm_data_dir(filename_out)
            cp_data_dir(filename_in, filename_out) # they are all *.ms, just copy it over
            lis_fils.append(filename_out)
        # Concatenation
        logger.info("6.2 Concatenating antennas")
        #lis_fils = [f for f in os.listdir(".") if f.endswith('.ms.5') and f.startswith(filename)]
        rm_data_dir(filename+'.cal'+'.'+name_line)
        logger.info('Running concat to make '+filename+'.cal'+'.'+name_line)
        casaStuff.concat(vis = lis_fils, concatvis = filename+'.cal'+'.'+name_line)
    else:
        filename_in  = filename+'.allant'+fin
        filename_out = filename+'.allant'+finout
        cp_data_dir(filename_in, filename_out)
        cp_data_dir(filename_out, filename+'.cal')


    # Convert the Science Target Units from Kelvin to Jansky
    logger.info("6.3 Convert the Science Target Units from Kelvin to Jansky")
    spw_line = get_spw_line(vel_source, freq_rest, spws_info) # get the original spw ID
    jyperk = extract_jyperk(filename, spw_line, pipeline)

    cp_data_dir(filename+'.cal'+'.'+name_line, filename+'.cal.jy'+'.'+name_line)

    logger.info('Running scaleAutocorr on '+filename+'.cal.jy'+'.'+name_line)
    for ant in jyperk.keys():
        logger.info('ant: %s, spw_line: %s, jyperk[ant][spw_line][\'mean\']: %s'%(ant, spw_line, jyperk[ant][spw_line]['mean']))
        scaleAutocorr(vis=filename+'.cal.jy'+'.'+name_line, scale=jyperk[ant][spw_line]['mean'], antenna=ant, spw=0) # spw is always 0


    # Rename line spw to spw=0
    logger.info("6.4 Renaming spw of line "+str(spw_line)+" to 0")
    fin = '.cal.jy'+'.'+name_line
    finout = '.cal.jy.tmp'+'.'+name_line
    cp_data_dir(filename+fin, filename+finout)

    fin = '.cal.jy.tmp'+'.'+name_line
    finout = '.cal.jy'+'.'+name_line
    rm_data_dir(filename+finout)
    logger.info('Running split to make '+filename+finout+', datacolumn is '+getDataColumnForSplit(filename+fin))
    casaStuff.split(vis=filename+fin,
         outputvis=filename+finout,
         datacolumn=getDataColumnForSplit(filename+fin))


    # listobs
    if os.path.exists(filename+finout+'.listobs.txt'):
        os.remove(filename+finout+'.listobs.txt')
    casaStuff.listobs(vis=filename+finout, listfile=filename+finout+'.listobs.txt')


    with open('done_step_6_for_'+filename[0:-3]+'_'+name_line, 'w') as outlogfile:
        outlogfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + time.strftime('%Z'))

#-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# Step 7 - Imaging
#-*-*-*-*-*-*-*-*-*-*
def imaging(source, name_line, phcenter, vel_source, source_vel_kms, vwidth_kms, chan_dv_kms, freq_rest_im,
        joint_imaging_dirs=None, joint_imaging_suffix='', doplots=False):

    logger.info("====================")
    logger.info("= Step 7 - Imaging =")
    logger.info("====================")
    logger.info("Current directory: "+os.getcwd())

    if joint_imaging_suffix == '':
        suffix = ''
    else:
        suffix = '.' + joint_imaging_suffix.lstrip('.')

    if os.path.exists('done_step_7'+'_'+name_line+suffix):
        logger.info('Found file: done_step_7'+'_'+name_line+suffix+'. Will not re-do step 7.')
        return

    fwhmfactor = 1.13                               # Factor to estimate the ALMA theoretical beam
    diameter   = 12                                 # Diameter of ALMA antennas in meters

    # Search for files already calibrated
    Msnames = glob.glob(os.path.join("*",
                                     "calibration",
                                     f"*.cal.jy.{name_line}"
                                     )
                        )

    if doplots:
        plotfile = True
    else:
        plotfile = ''

    # If 2 SGs have to be imaged together, look for *cal.jy files for the second part of the galaxy
    if joint_imaging_dirs is None:
        joint_imaging_dirs = []
    if len(joint_imaging_dirs) > 0:
        logger.info("More than one Science goals are considerated to create the final image of the galaxy "+source)
        for path2 in joint_imaging_dirs:
            #path2 = joint_imaging_dir # ori_path+'/../'+path_galaxy2+'calibration/' # <TODO> for NGC4254b NGC4321b NGC3627b
            logger.info('PATH for join imaging '+path2)
            Msnames2 = [path2+f for f in os.listdir(path2) if f.endswith('.cal.jy'+'.'+name_line)]
            for Msname in Msnames:
                if Msname in Msnames2:
                    Msnames2.remove(Msname)
            Msnames = Msnames+Msnames2
    logger.info('Msnames: %s N=%d'%(Msnames, len(Msnames)))

    if len(Msnames) == 0:
        raise Exception('Error! No Msnames to do imaging!')
    elif len(Msnames) == 1:
        try:
            xSampling, ySampling, maxsize = getTPSampling(Msnames[0], showplot=False, plotfile=plotfile)
        except ValueError:
            xSampling, ySampling, maxsize = getTPSampling(Msnames[0], showplot=False, pickFirstRaster=False, plotfile=plotfile)

        infile = Msnames[0]

    else:
        #casaStuff.concat(vis = Msnames, concatvis = 'Msnames.cal'+'.'+name_line)
        #xSampling, ySampling, maxsize = getTPSampling('Msnames.cal'+'.'+name_line, showplot=False, plotfile=plotfile)

        # Try concatenating MSs to handle the too many files issue in CASA 5
        ms_concat_filename = 'ALMA_TP.'+source+'.'+name_line+suffix+'.ms_concat'
        if os.path.exists(ms_concat_filename):
            shutil.rmtree(ms_concat_filename)
        casaStuff.concat(vis=Msnames, concatvis=ms_concat_filename, respectname=True)
        logger.info('Msnames: %s, N=%d, concatenated: %s'%(Msnames, len(Msnames), ms_concat_filename))

        xSampling, ySampling, maxsize = getTPSampling(ms_concat_filename, showplot=False, plotfile=plotfile)
        logger.info('xSampling: %s, ySampling: %s, maxsize: %s'%(str(xSampling), str(ySampling), str(maxsize)))

        infile = ms_concat_filename

    # Read frequency
    mymsmd = au.createCasaTool(casaStuff.msmdtool)
    mymsmd.open(infile)
    freq = mymsmd.meanfreq(0)
    mymsmd.close()
    logger.info("Reading frequency in image: "+str(freq))

    # Coordinate of phasecenter read from the data or used as input
    if phcenter == False:
        coord_phase = read_source_coordinates(infile, source)
        logger.info("Coordinate of phasecenter, read from the data: ")
        logger.info(str(coord_phase))
    else:
        logger.info("Coordinate of phasecenter entered by the user: ")
        coord_phase = phcenter
        logger.info(str(coord_phase))

    # Source velocity for imaging, read from the data or used as input
    if source_vel_kms == False:
        source_vel_kms = vel_source
        logger.info("Velocity of source used for imaging read from the data: ")
        logger.info(str(source_vel_kms))
    else:
        logger.info("Velocity of source used for imaging entered by the user: ")
        source_vel_kms = source_vel_kms
        logger.info(str(source_vel_kms))

    theorybeam = fwhmfactor*c_light*1e3/freq/diameter*180/pi*3600
    cell       = theorybeam/9.0
    if 'factorim' in globals():
        imsize  = int(round(maxsize/cell)*factorim)
    else:
        imsize     = int(round(maxsize/cell)*1.5)

    start_vel      = source_vel_kms-vwidth_kms/2
    nchans_vel     = int(round(vwidth_kms/chan_dv_kms))

    if os.path.exists('ALMA_TP.'+source+'.'+name_line+suffix+'.image'):
        shutil.rmtree('ALMA_TP.'+source+'.'+name_line+suffix+'.image')
    if os.path.exists('ALMA_TP.'+source+'.'+name_line+suffix+'.image.compleximage'):
        shutil.rmtree('ALMA_TP.'+source+'.'+name_line+suffix+'.image.compleximage')

    # CASA 6? has a better tsdimaging function, use it if possible
    if hasattr(casaStuff, 'tsdimaging'):
        func_sdimaging = casaStuff.tsdimaging
        outfile = 'ALMA_TP.'+source+'.'+name_line+suffix # tsdimaging will output '.image' and '.weight'
        outimage = outfile+'.image'
    else:
        func_sdimaging = casaStuff.sdimaging
        outfile = 'ALMA_TP.'+source+'.'+name_line+suffix+'.image'
        outimage = outfile

    # Start imaging
    logger.info("Start imaging")
    logger.info("Imaging from velocity "+str(start_vel)+", using "+str(nchans_vel)+" channels.")
    logger.info("Rest frequency is "+str(freq_rest_im)+" GHz.")
    logger.info("Cell and image sizes are: "+str(cell)+"arcsec and "+str(imsize))
    func_sdimaging(
        infiles = infile,
        mode = 'velocity',
        nchan = nchans_vel,
        width = str(chan_dv_kms)+'km/s',
        start = str(start_vel)+'km/s',
        veltype  = "radio",
        outframe = 'LSRK',
        restfreq = str(freq_rest_im)+'GHz',
        gridfunction = 'SF',
        convsupport = 6,
        phasecenter = coord_phase,
        imsize = imsize,
        cell = str(cell)+'arcsec',
        overwrite = True,
        outfile = outfile)

    # Correct the brightness unit in the image header
    casaStuff.imhead(
        imagename = outimage, # 'ALMA_TP.'+source+'.'+name_line+suffix+'.image',
        mode = 'put',
        hdkey = 'bunit',
        hdvalue = 'Jy/beam')

    # Add Restoring Beam Header Information to the Science Image
    minor, major, fwhmsfBeam, sfbeam = au.sfBeam(frequency=freq*1e-9,
        pixelsize=cell,
        convsupport=6,
        img=None, #to use Gaussian theorybeam
        stokes='both',
        xSamplingArcsec=xSampling,
        ySamplingArcsec=ySampling,
        fwhmfactor=fwhmfactor,
        diameter=diameter)

    #ia.open('ALMA_TP.'+source+'.'+name_line+suffix+'.image')
    #ia.setrestoringbeam(major = str(sfbeam)+'arcsec', minor = str(sfbeam)+'arcsec', pa = '0deg')
    #ia.done()
    myia = au.createCasaTool(casaStuff.iatool)
    myia.open(outimage) # 'ALMA_TP.'+source+'.'+name_line+suffix+'.image'
    myia.setrestoringbeam(major = str(sfbeam)+'arcsec', minor = str(sfbeam)+'arcsec', pa = '0deg')
    mask = myia.getchunk(getmask=True)
    myia.close()

    # Finally, trim the image and weight cube down
    mask_spec_x = np.any(mask, axis=tuple([i for i, x in enumerate(list(mask.shape)) if i != 0]))

    # Get bounding box in RA/Dec. Use padding of 1 pixel
    pad = 1
    xmin = np.max([0,np.min(np.where(mask_spec_x))-pad])
    xmax = np.min([np.max(np.where(mask_spec_x))+pad,mask.shape[0]-1])

    #mask_spec_y = np.sum(np.sum(mask*1.0,axis=2),axis=0) > 0
    mask_spec_y = np.any(mask, axis=tuple([i for i, x in enumerate(list(mask.shape)) if i != 1]))
    ymin = np.max([0,np.min(np.where(mask_spec_y))-pad])
    ymax = np.min([np.max(np.where(mask_spec_y))+pad,mask.shape[1]-1])

    box_string = '' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax)

    casaStuff.imsubimage(
        imagename=outimage,
        outfile=outimage+'_trimmed',
        box=box_string,
    )
    casaStuff.imsubimage(
        imagename=outimage.replace('.image', '.weight'),
        outfile=outimage.replace('.image', '.weight')+'_trimmed',
        box=box_string,
    )

    # Move things around
    os.system('mv ' + outimage + ' ' + outimage + '.orig')
    os.system('mv ' + outimage + '_trimmed ' + outimage)

    os.system('mv ' + outimage.replace('.image', '.weight') + ' ' + outimage.replace('.image', '.weight') + '_orig')
    os.system('mv ' + outimage.replace('.image', '.weight') + '_trimmed ' + outimage.replace('.image', '.weight'))

    if doplots == True:
        casaStuff.viewer(outimage) # 'ALMA_TP.'+source+'.'+name_line+suffix+'.image'

    with open('done_step_7'+'_'+name_line+suffix, 'w') as outlogfile:
        outlogfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + time.strftime('%Z'))


#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# Step 8 - Export fits file
#-*-*-*-*-*-*-*-*-*-*-*-*-*

def export_fits(name_line, source, output_file, joint_imaging_suffix=''):

    logger.info("========================")
    logger.info("= Step 8 - Export FITS =")
    logger.info("========================")
    logger.info("Current directory: "+os.getcwd())

    if joint_imaging_suffix == '':
        suffix = ''
    else:
        suffix = '.' + joint_imaging_suffix.lstrip('.')

    if os.path.exists('done_step_8'+'_'+name_line+suffix):
        logger.info('Found file: done_step_8'+'_'+name_line+suffix+'. Will not re-do step 8.')
        return

    #if os.path.isdir(ori_path+'/'+path_dataproduct) == False:
    #    os.system('mkdir '+ori_path+'/'+path_dataproduct)       # folder containing all plots

    #if checkdir(os.getcwd(),path_galaxy) == False:
    #    os.chdir('../'+path_galaxy+'calibration')

    # Prepare to output image and weight data
    imagename = 'ALMA_TP.'+source+'.'+name_line+suffix+'.image'
    if os.path.exists(imagename+'.weight'):
        weightname = imagename+'.weight'
    else:
        weightname = 'ALMA_TP.'+source+'.'+name_line+suffix+'.weight'
    imagefile = imagename + '.fits'
    weightfile = weightname + '.fits'

    # Export to fits file
    if os.path.exists(imagefile):
        os.system('rm -Rf  '+imagefile)
    if os.path.exists(weightfile):
        os.system('rm -Rf  '+weightfile)
    casaStuff.exportfits(imagename = imagename,
               fitsimage = imagefile)
    casaStuff.exportfits(imagename = weightname,
               fitsimage = weightfile)

    logger.info('> Exported FITS to "%s"'%(imagefile))
    logger.info('> Exported FITS to "%s"'%(weightfile))

    shutil.copy2(imagefile, output_file)

    logger.info('> Copied FITS to "%s"'%(output_file))

    with open('done_step_8'+'_'+name_line+suffix, 'w') as outlogfile:
        outlogfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + time.strftime('%Z'))



#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# Main body TP ALMA data reduction.
#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
def run_ALMA_TP_tools(
        path_galaxy = '',
        ms_dirs = '',
        flag_file = '',
        doplots = True,
        dosplitants = True,
        bl_order = 1,
        max_flag_frac = 0.9,
        in_sources = '',
        freq_rest = np.nan,
        vel_cube = '',
        vel_line = '',
        phase_center = '',
        source_vel_kms = np.nan,
        vwidth_kms = np.nan,
        chan_dv_kms = np.nan,
        freq_rest_im = np.nan,
        name_line = '',
        output_file = '',
        joint_imaging_dirs = None,
        joint_imaging_suffix = '',
        do_step = [],
        EBexclude = None,
    ):

    if (path_galaxy == ''
            or ms_dirs == ''
            or in_sources == ''
            or np.isnan(freq_rest)
            or np.isnan(source_vel_kms)
            or np.isnan(vwidth_kms)
            or np.isnan(chan_dv_kms)
            or np.isnan(freq_rest_im)
            or name_line == ''
            or output_file == ''
    ):
        logger.info('Error! Invalid input arguments when calling run_ALMA_TP_tools.')
        return

    # Get current directory
    ori_path = os.getcwd()

    logger.info("==================================")
    logger.info(" Starting TP ALMA data reduction  ")
    logger.info("==================================")

    logger.info("> You are executing the ALMA-TP-pipeline script from the directory: ")
    logger.info("  "+ori_path)

    # Keep a check around that we've found anything
    found_any_ebs = False

    filenames = []
    sourcenames = []

    logger.info("Found the following MS directories:")
    for ms_dir in ms_dirs:
        logger.info(f"> {ms_dir}")

    # Loop over MS dirs
    for ms_idx, ms_dir in enumerate(ms_dirs):

        # Move to calibration directory
        calib_dir = os.path.join(path_galaxy, ms_dir, "calibration")

        logger.info("> Changing directory to " + calib_dir)
        os.chdir(calib_dir)

        pipeline = checkpipeline()                  # Pipeline reduced data (True or False)

        # Defining Execution Blocks (EBS) names
        EBsnames = [f for f in os.listdir(path_raw) if f.endswith('.asdm.sdm')]

        if EBexclude is not None:
            if np.isscalar(EBexclude):
                EBexclude = [EBexclude]
            EBsnames = [s for s in EBsnames if s[0:-9] not in EBexclude]

        if len(do_step) == 0:
            do_step = [1,2,3,4,5,6,7,8]

        # Do data reduction for each EB
        for EBs in EBsnames:

            if not pipeline:
                EBs = EBs.replace('.ms.scriptForSDCalibration.py', '.asdm.sdm')
            filename = 'u'+re.search('u(.+?).asdm.sdm', EBs).group(1)+'.ms'
            file_exists = check_exists(filename)                             # Check weather the raw data exists

            if file_exists:

                found_any_ebs = True

                # Import and split data by antenna
                if 1 in do_step:
                    import_and_split_ant(filename,
                                         doplots=doplots,
                                         dosplitants=dosplitants,
                                         )

                # Read source name directly from MS
                source = get_sourcename(filename,
                                        source=in_sources[ms_idx],
                                        )

                # If we don't find the expected source in the EB, continue
                if source is None:
                    continue

                vec_ants_t = read_ants_names(filename)                       # Read vector with name of all antennas
                vec_ants   = [s for s in vec_ants_t if any(xs in s for xs in ['PM','DV'])] # Get only 12m antennas.

                # For antenna with a significant fraction of data masked, future steps will crash
                # so get those here
                vec_ants_to_remove = []
                for ant in vec_ants:
                    s = casaStuff.flagdata(filename + "." + ant + ".ms", mode="summary")
                    frac_flagged = s["flagged"] / s["total"]

                    if frac_flagged > max_flag_frac:
                        logger.warning(f"{filename}.{ant} has {frac_flagged * 100}% of data flagged. Will not process")
                        vec_ants_to_remove.append(ant)

                for ant in vec_ants_to_remove:
                    vec_ants.remove(ant)

                # Read source velocity and SPW info
                vel_source = read_vel_source(filename,source)
                spws_info  = read_spw(filename,source)

                if 2 in do_step:
                    spwmap = gen_tsys_and_flag(
                        filename,
                        spws_info,
                        pipeline,
                        flag_dir=os.path.join(ori_path, 'galaxy-specific-scripts', 'flags-folder'),
                        flag_file='',
                        doplots=doplots,
                    )

                if not dosplitants:
                    vec_ants = None

                # Remove known problematic datasets
                if vec_ants is not None:
                    if filename == 'uid___A002_Xd06716_X126.ms':
                        if 'PM02' in vec_ants:
                            print('** removing known problematic dataset: {} {}'.format(filename, 'PM02'))
                            vec_ants.remove('PM02')

                if 3 in do_step:
                    counts2kelvin(
                        filename,
                        ant_list=vec_ants,
                        spws_info=spws_info,
                        spwmap=spwmap,
                        doplots=doplots,
                    )

                if 4 in do_step:
                    extract_cube(
                        filename,
                        source,
                        name_line,
                        ant_list=vec_ants,
                        freq_rest=freq_rest,
                        spws_info=spws_info,
                        vel_source=vel_source,
                        vel_cube=vel_cube,
                        doplots=doplots,
                    )

                if 5 in do_step:
                    baseline(
                        filename,
                        source,
                        name_line,
                        ant_list=vec_ants,
                        freq_rest=freq_rest,
                        spws_info=spws_info,
                        vel_source=vel_source,
                        vel_line=vel_line,
                        bl_order=bl_order,
                        doplots=doplots,
                    )

                if 6 in do_step:
                    # concat ants and convert flux unit to Jy
                    concat_ants(
                        filename,
                        name_line,
                        ant_list=vec_ants,
                        freq_rest=freq_rest,
                        spws_info=spws_info,
                        vel_source=vel_source,
                        pipeline=pipeline,
                    )

                filenames.append(os.path.join(ms_dir,
                                              "calibration",
                                              filename,
                                              )
                                 )
                sourcenames.append(source)

        if not found_any_ebs:
            raise Exception("No EBs found to process")

    # Change back to the galaxy path
    os.chdir(path_galaxy)

    # Pull out the first file/source to get velocity and any outstanding info
    filename = filenames[0]
    source = sourcenames[0]

    vel_source = read_vel_source(filename, source)

    if 7 in do_step:
        imaging(source,
                name_line,
                phase_center,
                vel_source,
                source_vel_kms,
                vwidth_kms,
                chan_dv_kms,
                freq_rest_im,
                joint_imaging_dirs=joint_imaging_dirs,
                joint_imaging_suffix=joint_imaging_suffix,
                doplots=doplots,
                )

    if 8 in do_step:
        export_fits(name_line,
                    source,
                    output_file,
                    joint_imaging_suffix=joint_imaging_suffix,
                    )

    logger.info("> Changing directory to "+ori_path)
    os.chdir(ori_path)



