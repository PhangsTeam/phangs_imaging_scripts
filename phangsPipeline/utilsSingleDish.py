
import os

import analysisUtils as au
import matplotlib
import numpy as np
import pylab as pb
from analysisUtils import mjdSecondsListToDateTime, mjdsecToUT

from .casaStuff import casa_version, tbtool, msmdtool, metool, qatool

casaVersion = "{0}.{1}.{2}".format(*casa_version)

def get_first_arr_val(arr):
    """Get the first value from a numpy array"""

    while isinstance(arr, np.ndarray):
        arr = arr[0]

    return arr

def setXaxisTimeTicks(adesc, t0, t1, verbose=True):
    """
    Sets sensible major and minor tick intervals for a plot_date plot
    based on the start and end times.
    Inputs: t0 (startTime in seconds)
        and t1 (endTime in seconds)
        Only the difference matters.
   -Todd Hunter
    """
    timeRange = t1-t0
    if (verbose):
        print("timeRange = %f sec" % (timeRange))
    if (timeRange > 200000):
        #<DZLIU># if timeRange > 55 hours, show days in xaxis, this solves the error 'RRuleLocator estimated to generate %d ticks from %s to %s: exceeds Locator.MAXTICKS * 2 (%d) '
        print("timeRange = %f sec, setting xaxis in days" % (timeRange))
        adesc.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
        adesc.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=1))
        adesc.xaxis.set_minor_locator(matplotlib.dates.HourLocator(interval=12))
        adesc.fmt_xdata = matplotlib.dates.DateFormatter('%Y-%m-%d')
    elif (timeRange > 20000):
        adesc.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
        adesc.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0,int(60*np.floor((t1-t0)/3600)),60)))
        adesc.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,15)))
        adesc.fmt_xdata = matplotlib.dates.DateFormatter('%H')
    elif (timeRange > 2000):
        adesc.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        adesc.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,15)))
        adesc.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,5)))
        adesc.fmt_xdata = matplotlib.dates.DateFormatter('%H:%M')
    elif (timeRange > 600):
        adesc.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        adesc.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,5)))
        adesc.xaxis.set_minor_locator(matplotlib.dates.SecondLocator(bysecond=range(0,60,30)))
        adesc.fmt_xdata = matplotlib.dates.DateFormatter('%H:%M')
    elif (timeRange > 200):
        adesc.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        adesc.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,3)))
        adesc.xaxis.set_minor_locator(matplotlib.dates.SecondLocator(bysecond=range(0,60,10)))
        adesc.fmt_xdata = matplotlib.dates.DateFormatter('%H:%M')
    elif (timeRange > 40):
        adesc.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
        adesc.xaxis.set_major_locator(matplotlib.dates.SecondLocator(bysecond=range(0,60,20)))
        adesc.xaxis.set_minor_locator(matplotlib.dates.SecondLocator(bysecond=range(0,60,5)))
        adesc.fmt_xdata = matplotlib.dates.DateFormatter('%H:%M:%S')
    else:
        adesc.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
        adesc.xaxis.set_major_locator(matplotlib.dates.SecondLocator(bysecond=range(0,60,10)))
        adesc.xaxis.set_minor_locator(matplotlib.dates.SecondLocator(bysecond=range(0,60,1)))
        adesc.fmt_xdata = matplotlib.dates.DateFormatter('%H:%M:%S')

def getTPSampling(vis, obsid=0, showplot=False, plotfile='', debug=False,
                  labelFirstNSamples=0, labelIncrement=1, convert=True,
                  field='auto', plotrange=[0,0,0,0], antenna=0, scan=None,
                  refractionCorrection=False, nutationCorrection=False,
                  timerange=None, trimPointingData=True, showComponents=False,
                  pickFirstRaster=True, useApparentPositionForPlanets=False,
                  connectDots=False, intent='OBSERVE_TARGET', findSubscans=False,
                  magnification=0.3, ymodeThreshold=0.075,
                  forceCardinalScanAngle=False, source=None, cofa='',
                  useAntennaAsCOFA=False, overlayOTPointings=''):
    """
    This function reads the main table of the ms, finds the
    start and end time of the specified observation ID, then
    read the POINTING table and extracts the pointing directions
    within that timerange.  It computes successive differences
    on both axes to determine which is the scan direction (RA or Dec)
    and what the sampling is in each axes, which it returns in
    units of arcseconds on-the-sky, i.e. corrected for the
    right ascension axis by multiplying by cos(declination).
    It also generates a png file showing the observed points
    in relative coordinates in units of arc seconds.
    Note: This function does not support arbitrary scanning angles!

    vis: the name of the measurement set
    obsid: the number of the OBSERVATION_ID to analyze
    showplot: set to True to generate an interactive plot of sampled positions
    plotfile: default='' -->  '<vis>.obsid0.sampling.png'
    labelFirstNSamples: put labels on the first N samples
    labelIncrement: the number of samples between labels
    convert: if True, convert AZELGEO coords to RA Dec in the plot
    field: plot the offset coordinates relative to this field ID (or name)
           default = 'auto' == the first field with the specified intent
    source: if specified, then use this source ID as (0,0) (int or string int)
    plotrange: set the plot axes ranges [x0,x1,y0,y1]
    pickFirstRaster: automatically set to True for major planets
        This avoids confusion due to the slowly changing source position.
    antenna: which antenna ID (or name) to use to determine the scan parameters
    scan: default: use all scans on the first target with the specified intent
    timerange: limit the data to this timerange,  e.g. '05:00:00 06:00:00'
            or  '05:00~06:00' or '2011/10/15-05:00:00 2011/10/15-06:00:00'
    trimPointingData: if False, then don't exclude data outside of scan times
    showComponents: if True, then also show X and Y vs. time on upper plot
    intent: the intent for which to pick the scans to use (up to the # sign)
    findSubscans: if True, then run class Atmcal to find subscan times
    cofa: if specifed as [x,y,z], use this ITRF position as array center
    useAntennaAsCOFA: if True, then use the position of the pad holding the
         specified antenna as the COFA instead of the value in CASA for the
         observatory, or the value specified by cofa parameter.
    overlayOTPpositions: if filename is specified, then assume it is an ASCII
        pointings file exported from the OT in relative coordinates (arcsec)

    Returns:
    xSampling, ySampling, largestDimension (all in units of arcsec)

    -Todd Hunter
    """
    if (os.path.exists(vis) == False):
        print("The ms does not exist.")
        return
    if (os.path.exists(vis+'/table.dat') == False):
        print("No table.dat.  This does not appear to be an ms.")
        return
    if overlayOTPointings != '':
        if not os.path.exists(overlayOTPointings):
            print("OT positions file not found.")
            return

    if field == '':
        field = None

    mytb = au.createCasaTool(tbtool)
    mytb.open(vis)
    allObsIDs = mytb.getcol('OBSERVATION_ID')
    obsIDs = np.unique(allObsIDs)
    nObsIDs = len(obsIDs)
    print("There are %d rows and %d OBSERVATION_IDs in this dataset = %s" % (len(allObsIDs), nObsIDs, str(obsIDs)))

    subtable = mytb.query('OBSERVATION_ID == %d' % (obsid))
    mytb.close()
    times = subtable.getcol('TIME')
    subtable.close()
    if (len(times) < 1):
        print("No rows found for OBSERVATION_ID=%d" % (obsid))
        return

    startTime = np.min(times)
    stopTime = np.max(times)
    if (casaVersion <= '4.1.0'):
        print("This function requires casa >= 4.1.0.")
        return

    mymsmd = au.createCasaTool(msmdtool)
    mymsmd.open(vis)
    timeranges = {}
    timecenters = []
    intent = intent.split('#')[0]
    if (intent+'#ON_SOURCE' not in mymsmd.intents()):
        if ('MAP_ANTENNA_SURFACE#ON_SOURCE' in mymsmd.intents()):
            intent = 'MAP_ANTENNA_SURFACE'
        else:
            print("%s#ON_SOURCE is not an intent in this measurement set." % (intent))
            print("Available intents = ", mymsmd.intents())
            mymsmd.close()
            return

    scansOnSource = list(mymsmd.scansforintent(intent+'#ON_SOURCE'))
    if (intent+'#OFF_SOURCE' in mymsmd.intents()):
        scansOnSource += list(mymsmd.scansforintent(intent+'#OFF_SOURCE'))
    scansOnSource = np.unique(scansOnSource)
    if (scan is not None and scan != ''):
        scansToUse = [int(s) for s in str(scan).split(',')]
        for scan in scansToUse:
            if (scan not in scansOnSource):
                print("Scan %d is not an %s scan.  Available scans = %s" % (scan,intent,str(scansOnSource)))
                mymsmd.close()
                return

        print("Using only the specified scans: %s" % (str(scansToUse)))
    else:
        # should default to the first scan with specified intent
        if (len(scansOnSource) == 0):
            print("There are no scans on the with intent=%s to use." % (intent))
            print("Available intents = ", mymsmd.intents())
            print("Use the intent parameter to select one.")
            mymsmd.close()
            return

        myfield = mymsmd.fieldsforscan(scansOnSource[0])[0]
        scansToUse = scansOnSource # mymsmd.scansforfield(myfield)
        print("Using only the %s#ON_SOURCE scans on the science target: %s" % (intent,str(scansToUse)))

    calAtmTimes = []
    firstCalAtmScan = -1
    if ('CALIBRATE_ATMOSPHERE#ON_SOURCE' in mymsmd.intents()):
        calAtmFieldNames = mymsmd.namesforfields(mymsmd.fieldsforintent(intent+'*'))
        calAtmFields = fieldsfornames(mymsmd, calAtmFieldNames)
        if debug:
            print("calAtmFields = ", calAtmFields)
        scans1 = mymsmd.scansforintent('CALIBRATE_ATMOSPHERE#ON_SOURCE')
        scans2 = np.unique(scansforfields(mymsmd,calAtmFields))
        if debug:
            print("scans1 = %s, scans2 = %s" % (scans1, scans2))
        calAtmScans = np.intersect1d(scans1,scans2)
        if (len(calAtmScans) > 0):
            firstCalAtmScan = calAtmScans[0]
            firstCalAtmScanMeanMJD = np.mean(mymsmd.timesforscan(firstCalAtmScan))/86400.
            calAtmField = mymsmd.fieldsforscan(firstCalAtmScan)[0]
            calAtmTimes = mymsmd.timesforscan(firstCalAtmScan)
            calAtmTimesMax = np.max(calAtmTimes)
            calAtmTimesMin = np.min(calAtmTimes)
            print("first AtmCal scan on target = %d has %d times (%s-%s)" % (firstCalAtmScan,len(calAtmTimes),
                                                                             plotbp3.utstring(calAtmTimesMin,3),
                                                                             plotbp3.utstring(calAtmTimesMax,3)))
            if (findSubscans):
                ac = au.Atmcal(vis)
                skyTimes = ac.timestamps[firstCalAtmScan][ac.skysubscan]
                print("sky subscan has times (%s-%s)" % (plotbp3.utstring(np.min(skyTimes),3),
                                                         plotbp3.utstring(np.max(skyTimes),3)))
        else:
            print("No AtmCals were done on a field with intent %s (%s)" % (intent,calAtmField))
            print("scans1 = ", scans1)
            print("scans2 = ", scans2)
    else:
        print("No AtmCals in this dataset.")

    timesforscan = {}
    if (timerange is not None and timerange != ''):
        timerange = au.parseTimerangeArgument(timerange,vis)
        if (debug):
            print("timerange[0] = ", str(timerange[0]))
            print("timerange[1] = ", str(timerange[1]))
    for i in scansToUse:
        mytimes = mymsmd.timesforscan(i)
        if (timerange is not None and timerange != ''):
            idx1 = mytimes > timerange[0]
            idx2 = mytimes < timerange[1]
            mytimes = mytimes[np.logical_and(idx1,idx2)]

        if (len(mytimes) > 0):
            timesforscan[i] = {'begin': np.min(mytimes), 'end': np.max(mytimes)}
            timeranges[i] = [np.min(mytimes),np.max(mytimes)]
            timecenters += list(mytimes)
        else:
            scansToUse = np.delete(scansToUse,list(scansToUse).index(i))
            if (debug):
                print("No times found for scan %d" % (i))

    timecenters = np.array(timecenters)
    tSuccessiveDifferences = timecenters[1:] - timecenters[:-1]
    medianSamplingIntervalDataTable = np.median(tSuccessiveDifferences)
    print("median sampling interval in data table with %s (scans=%s) = %f sec" % (intent, str(scansOnSource), medianSamplingIntervalDataTable))
    if (casaVersion >= '4.2.0'):
        try:
            spw = np.intersect1d(mymsmd.spwsforintent(intent+'#ON_SOURCE'),
                                 mymsmd.almaspws(fdm=True,tdm=True))[0]
            pol = 0
            exposureTime = mymsmd.exposuretime(scan=scansToUse[0], spwid=spw, polid=pol)['value']
            print("Exposure time = %g in spw %d, pol %d" % (exposureTime,spw,pol))
            medianSamplingIntervalDataTable = exposureTime
        except:
            pass

    # find the name of the science field, and (if present) the scan numbers on it
    fieldName = None
    if (field == 'auto'):
        intent = intent + '#ON_SOURCE'
        if (intent not in mymsmd.intents()):
            print("No science target found.  Checking for amplitude or flux calibrator")
            if ('CALIBRATE_FLUX#ON_SOURCE' in mymsmd.intents()):
                intent = 'CALIBRATE_FLUX#ON_SOURCE'
            elif ('CALIBRATE_AMPLI#ON_SOURCE' in mymsmd.intents()):
                intent = 'CALIBRATE_AMPLI#ON_SOURCE'
            else:
                field = None
        if (field is not None):
            field = mymsmd.fieldsforintent(intent)
            if (len(field) < 1):
                field = None
            else:
                field = field[0]
                fieldName = mymsmd.namesforfields(field)[0]
                print("Found first source with %s intent = %d = %s" % (intent, field, fieldName))
    elif (field is not None):
        if (field in mymsmd.namesforfields(range(mymsmd.nfields()))):
            fieldName = field
            field = mymsmd.fieldsforname(fieldName)[0]
        elif (str(field) in [str(a) for a in range(mymsmd.nfields())]):
            fieldName = mymsmd.namesforfields(int(field))[0]
        else:
            print("Field %s is not in this dataset.  Available fields = %s" % (field,str(mymsmd.namesforfields(range(mymsmd.nfields())))))
            mymsmd.close()
            return

    if (field is not None):
        scansforfield = mymsmd.scansforfield(field)
    nantennas = mymsmd.nantennas()
    antennanames = np.array(mymsmd.antennanames(range(nantennas)))

    mytb.open(vis+'/POINTING')
    coordSys = mytb.getcolkeyword('DIRECTION',1)['Ref']
    originalCoordSys = coordSys
    print("Map coordinate system from the POINTING table = ", coordSys)
    alltimes = mytb.getcol('TIME')
    direction = mytb.getcol('DIRECTION')
    antenna_ids = mytb.getcol('ANTENNA_ID')
    uniqueAntennas = len(np.unique(antenna_ids))
    mytb.close()

    if (antenna not in antenna_ids):
        if (antenna not in antennanames):
            print("Antenna %s is not in the POINTING table" % (str(antenna)))
            mymsmd.close()
            return
        else:
            antenna = mymsmd.antennaids(antenna)

    if useAntennaAsCOFA:
        cofa = au.getAntennaPadXYZ(vis, antenna)
        print("Using antenna %d pad position as COFA: " % (antenna), cofa)
    matches = antenna_ids == antenna
    print("Picked %d/%d rows, which correspond to antenna %d=%s (of %d)" % (len(matches), len(antenna_ids), antenna, antennanames[antenna], uniqueAntennas))
    times = alltimes[matches]
    xdirection = direction[0][0][matches]
    ydirection = direction[1][0][matches]

    # make sure RA/azimuth is in positive units
    negatives = xdirection < 0
    print("RA was negative at %d points" % (len(negatives)))
    xdirection[negatives] += 2*np.pi

    uniqueTimes, uniqueTimesIndices = np.unique(times, return_index=True)
    if (debug):
        print("Mean time: %f, MJD = %f =  %s" % (np.mean(uniqueTimes),np.mean(uniqueTimes)/86400.,
                                                 au.mjdSecondsToMJDandUT(np.mean(uniqueTimes))[1]))
    timeSortedIndices = np.argsort(uniqueTimes)

    # single-antenna times (sorted by time)
    pointingTime = times[uniqueTimesIndices[timeSortedIndices]]
    xdirection = xdirection[uniqueTimesIndices[timeSortedIndices]]
    ydirection = ydirection[uniqueTimesIndices[timeSortedIndices]]
    tdirection = times[uniqueTimesIndices[timeSortedIndices]]

    totalTimes = len(pointingTime)

    # Keep only the points from the requested ObsID
    matches1 = pointingTime >= startTime
    matches2 = pointingTime <= stopTime
    matches = np.logical_and(matches1,matches2)
    matchesObsID = matches[:]
    myTimes = len(matches)
    if (debug):
        print("Selected %d of %d unique times (i.e. from one spw) matching OBSERVATION_ID=%d" % (myTimes, totalTimes, obsid))
    if (myTimes == 0):
        mymsmd.close()
        return

    if (trimPointingData):
      # Trim off any pointing table entries not associated with an integration in the selected scan(s)
        scanmatches = []
        scanfieldmatches = []
        pointingTime = np.array(pointingTime)
        for match in np.where(matches)[0]:
          for i in list(timeranges.keys()):
              if (pointingTime[match] > timeranges[i][0] and pointingTime[match] < timeranges[i][1]):
                  scanmatches.append(match)
                  if (field is not None):
                      if (i in scansforfield):
                          scanfieldmatches.append(match)
                  break
        matches = np.array(scanmatches)  # times within any scan
        scanfieldmatches = np.array(scanfieldmatches)  # times within scans on the field
        myTimes = len(matches)
        if (debug):
            print("Selected %d times as being within a data scan (%s)" % (myTimes, scansToUse))

        if (myTimes == 0):
            mymsmd.close()
            return

        x = xdirection[matches]  # These are the longitude values in the pointing table for the selected scan
        y = ydirection[matches]  # These are the latitude values in the pointing table for the selected scan
        times = tdirection[matches]  # These are the timestamps in the pointing table for the selected scan
        if (firstCalAtmScan != -1):
            scanmatches = []
            for match in matchesObsID:
                if (pointingTime[match] > calAtmTimesMin and pointingTime[match] < calAtmTimesMax):
                    scanmatches.append(match)
            matches = np.array(scanmatches)
            xCalAtm = xdirection[matches]
            yCalAtm = ydirection[matches]
            timesCalAtmPointing = tdirection[matches]

            if (findSubscans):
                scanmatches = []
                for match in matchesObsID:
                    if (pointingTime[match] > np.min(skyTimes) and pointingTime[match] < np.max(skyTimes)):
                        scanmatches.append(match)
                matches = np.array(scanmatches)
                xCalAtmSky = xdirection[matches]
                yCalAtmSky = ydirection[matches]
                timesCalAtmSkyPointing = tdirection[matches]
    else:
        x = xdirection
        y = ydirection
        times = tdirection
    tSuccessiveDifferences = times[1:] - times[:-1]
    medianSamplingIntervalPointingTable = np.median(tSuccessiveDifferences)
    print("median sampling interval in POINTING table = %f sec" % (medianSamplingIntervalPointingTable))

    if (False):
        # Try to speed up the calculation by reducing the number of points.
        # Group the pointing table values into the correlator data time bins.
        # Does not yet work right.
        pointsPerIntegration = {}
        print("Assigning %d pointing table values to %d integrations" % (len(times),len(timecenters)))
        for t in range(len(times)):
            mindiff = 1e30
            for integration in range(len(timecenters)):
                absdiff = abs(times[t]-timecenters[integration])
                if (absdiff < mindiff):
                    mindiff = absdiff
                    whichIntegration = integration
            if (whichIntegration not in list(pointsPerIntegration.keys())):
                pointsPerIntegration[whichIntegration] = {}
                pointsPerIntegration[whichIntegration]['x'] = []
                pointsPerIntegration[whichIntegration]['y'] = []
            pointsPerIntegration[whichIntegration]['x'].append(x[t])
            pointsPerIntegration[whichIntegration]['y'].append(y[t])
        xnew = []
        ynew = []
        times = timecenters
        for key in list(pointsPerIntegration.keys()):
            xnew.append(np.median(pointsPerIntegration[key]['x']))
            ynew.append(np.median(pointsPerIntegration[key]['y']))
        x = np.array(xnew[:])
        y = np.array(ynew[:])

    # Keep a copy of the original values in radians
    xrad = x[:]
    yrad = y[:]

    if (source is not None):
        radec = au.getRADecForSource(vis, int(source))
        print("Getting RA,Dec for source %s: %s" % (str(source),radec))
        rightAscension, declination = au.radec2rad(radec)
    elif (field==None):
        rightAscension = np.median(x)
        declination = np.median(y)
        if (debug):
            print("Median coordinates = ", rightAscension, declination)
    else:
        rightAscension, declination = au.getRADecForField(vis, field, forcePositiveRA=True, usemstool=True)
        if (debug):
            print("field coordinates in radians = %f, %f = %s" % (rightAscension, declination,
                                                                  au.rad2radec(rightAscension, declination)))
        if (coordSys.find('AZEL') >= 0 and convert==False):
            rightAscension, declination = au.computeAzElFromRADecMJD([rightAscension,declination],
                                                                  mjd=np.median(times)/86400.,
                                                                  verbose=False,
                                                                  frame='AZELGEO', cofa=cofa)
            if (debug):
                print("field coordinates in az/el = ", rightAscension, declination)

    apparentCoordinates = False
    offSourceTimes = au.getOffSourceTimes(vis, intent)
    if (len(offSourceTimes) < 1):
        print("No off source intent found in the MS.")

    offSourceRA = []
    offSourceDec = []
    calAtmRA = []
    calAtmDec = []
    if (coordSys.find('AZEL') >= 0 and convert):
        coordSys = 'J2000'
        print("Converting %d coordinates from AZELGEO to J2000..." % (len(x)))
        my_metool = au.createCasaTool(metool)
        for i in range(len(x)):
            if ((i+1) % 10000 == 0): print("%d/%d" % (i+1,len(x)))
            ra, dec = au.computeRADecFromAzElMJD([xrad[i],yrad[i]], mjd=times[i]/86400.,
                                                 verbose=False,my_metool=my_metool,
                                                 refractionCorrection=refractionCorrection,
                                                 nutationCorrection=nutationCorrection,
                                                 frame='AZELGEO', cofa=cofa)
            if (ra < 0):
                ra += 2*np.pi
            x[i] = ra
            y[i] = dec
            if (times[i] in offSourceTimes):
                offSourceRA.append(ra)
                offSourceDec.append(dec)

        if (len(offSourceRA) > 0):
            xOffSource = np.median(offSourceRA)
            yOffSource = np.median(offSourceDec)
            print("     Field source position = %s" % (au.rad2radec(rightAscension, declination,verbose=False)))
            separation = np.degrees(au.angularSeparationRadians(rightAscension, declination, xOffSource, yOffSource))
            print("Median off source position = %s (separation = %.1farcmin = %.1fdeg)" % (au.rad2radec(xOffSource, yOffSource, verbose=False), separation*60, separation))
        elif (len(offSourceTimes) > 0):
            print("No times in the POINTING table match the times of off source integrations.")


        medianRA = np.median(x)
        medianDec = np.median(y)
        separation = []
        for i in range(len(x)):
            separation.append(au.angularSeparationRadians(medianRA,medianDec,x[i],y[i]))
        madSeparation = au.MAD(separation)
        if (debug):
            print("MAD of the separation of points from median (%s) = %f arcsec" % (au.rad2radec(medianRA,medianDec),madSeparation*180*3600/np.pi))

        if (firstCalAtmScan != -1):
            print("Converting %d coordinates (for AtmCal scan %d) from AZELGEO to J2000..." % (len(xCalAtm),firstCalAtmScan))
            for i in range(len(xCalAtm)):
                if ((i+1) % 10000 == 0): print("%d/%d" % (i+1,len(x)))
                ra, dec = au.computeRADecFromAzElMJD([xCalAtm[i],yCalAtm[i]], mjd=timesCalAtmPointing[i]/86400.,
                                                     verbose=False,my_metool=my_metool,
                                                     refractionCorrection=refractionCorrection,
                                                     nutationCorrection=nutationCorrection,
                                                     frame='AZELGEO', cofa=cofa)
                calAtmRA.append(ra)
                calAtmDec.append(dec)
            if (len(calAtmRA) > 0):
                xCalAtm = np.median(calAtmRA)
                yCalAtm = np.median(calAtmDec)
                print("Median atm calib. position = %s" % (au.rad2radec(xCalAtm, yCalAtm, verbose=False)))
                xCalAtm = np.mean(calAtmRA)
                yCalAtm = np.mean(calAtmDec)
                print("  Mean atm calib. position = %s" % (au.rad2radec(xCalAtm, yCalAtm, verbose=False)))
            elif (len(calAtmTimes) > 0):
                print("No times in the POINTING table match the times of AtmCal integrations.")
            else:
                print("There are no times found on AtmCal scans.")

            if (findSubscans):
                calAtmSkyRA = []
                calAtmSkyDec = []
                for i in range(len(xCalAtmSky)):
                    if ((i+1) % 10000 == 0):
                        print("%d/%d" % (i+1,len(x)))
                    ra, dec = au.computeRADecFromAzElMJD([xCalAtmSky[i],yCalAtmSky[i]], mjd=timesCalAtmSkyPointing[i]/86400.,
                                                         verbose=False,my_metool=my_metool,
                                                         refractionCorrection=refractionCorrection,
                                                         nutationCorrection=nutationCorrection, frame='AZELGEO', cofa=cofa)
                    calAtmSkyRA.append(ra)
                    calAtmSkyDec.append(dec)
                if (len(calAtmSkyRA) > 0):
                    xCalAtmSky = np.median(calAtmSkyRA)
                    yCalAtmSky = np.median(calAtmSkyDec)
                    print("Median atmcal sky position = %s" % (au.rad2radec(xCalAtmSky, yCalAtmSky, verbose=False)))
                    xCalAtmSky = np.mean(calAtmSkyRA)
                    yCalAtmSky = np.mean(calAtmSkyDec)
                    print("  Mean atmcal sky position = %s" % (au.rad2radec(xCalAtmSky, yCalAtmSky, verbose=False)))

        my_metool.done()
        if (source is not None):
            radec = au.getRADecForSource(vis, int(source))
            print("Getting RA,Dec for source %s: %s" % (str(source),radec))
            rightAscension, declination = au.radec2rad(radec)
        elif (field == None):
            rightAscension = np.median(x)
            declination = np.median(y)
        else:
            newEphemerisTechnique = False
            if (casaVersion >= '4.5'):
                if (au.getEphemeris(vis,verbose=False) != {}):
                     newEphemerisTechnique = True
            if (useApparentPositionForPlanets==False and fieldName.upper() in au.majorPlanets and
                not newEphemerisTechnique):
                try:
                    rightAscension, declination = au.planet(fieldName,mjd=times[0]/86400.)['directionRadians']
                    if (firstCalAtmScan != -1):
                        rightAscensionCalAtm, declinationCalAtm = au.planet(fieldName,mjd=firstCalAtmScanMeanMJD)['directionRadians']
                except:
                    if (casaVersion >= '4.0.0'):
                        print("Failed to contact JPL, reverting to using CASA ephemerides to get the J2000 position.")
                        rightAscension, declination = au.planet(fieldName,mjd=times[0]/86400.,useJPL=False)['directionRadians']
                        if (firstCalAtmScan != -1):
                            rightAscensionCalAtm, declinationCalAtm = au.planet(fieldName,mjd=firstCalAtmScanMeanMJD, useJPL=False)['directionRadians']
                    else:
                        print("Failed to contact JPL, reverting to showing apparent position (from the ms).")
                        rightAscension, declination = au.getRADecForField(vis, field, forcePositiveRA=True, usemstool=True)
                        apparentCoordinates = True
                        if (firstCalAtmScan != -1):
                            rightAscensionCalAtm, declinationCalAtm = au.getRADecForField(vis, calAtmField, forcePositiveRA=True, usemstool=True)
            else:
                rightAscension, declination = au.getRADecForField(vis, field, forcePositiveRA=True, usemstool=True)
                if fieldName.upper() in au.majorPlanets:
                    apparentCoordinates = True
                if (firstCalAtmScan != -1):
                    rightAscensionCalAtm, declinationCalAtm = au.getRADecForField(vis, calAtmField, forcePositiveRA=True, usemstool=True)
        if (len(offSourceRA) > 0):
            separation, deltaRaRadians, deltaDecRadians, deltaRaRadiansCosdec = au.angularSeparationRadians(xOffSource, yOffSource,
                                                                                       rightAscension, declination,
                                                                                       returnComponents=True)
            print("Separation from the off source position in relative coordinates = (%+.1f, %+.1f) arcsec" % (deltaRaRadiansCosdec*3600*180/np.pi,
                                                                        deltaDecRadians*3600*180/np.pi))
        if (len(calAtmRA) > 0):
            separation, deltaRaRadians, deltaDecRadians, deltaRaRadiansCosdec = au.angularSeparationRadians(xCalAtm, yCalAtm,
                                                                                       rightAscensionCalAtm, declinationCalAtm,
                                                                                       returnComponents=True)
            print("Separation from mean atm calib position in relative coordinates = (%+.1f, %+.1f) arcsec" % (deltaRaRadiansCosdec*3600*180/np.pi,
                                                                        deltaDecRadians*3600*180/np.pi))
            if (findSubscans):
                separation, deltaRaRadians, deltaDecRadians, deltaRaRadiansCosdec = au.angularSeparationRadians(xCalAtmSky, yCalAtmSky,
                                                                                       rightAscensionCalAtm, declinationCalAtm,
                                                                                       returnComponents=True)
                print("Separation from mean atmcal sky position in relative coordinates = (%+.1f, %+.1f) arcsec" % (deltaRaRadiansCosdec*3600*180/np.pi,
                                                                        deltaDecRadians*3600*180/np.pi))


    # convert absolute coordinates to relative arcsec on-the-sky
    for i in range(len(x)):
        if (x[i] < 0):
            print("x is negative")
        if (rightAscension < 0):
            print("rightAscension is negative")
        separation, dx, dy, dxcosdec = au.angularSeparationRadians(x[i],y[i],rightAscension,declination,True)
#        if i == 0:
#            print "x-rA=%f y-dec=%f, dx=%f dy=%f dxcosdec=%f" % (x[i]-rightAscension, y[i]-declination,dx,dy,dxcosdec)

        dx = get_first_arr_val(dx)
        dy = get_first_arr_val(dy)
        dxcosdec = get_first_arr_val(dxcosdec)

        x[i] = dxcosdec*180*3600/np.pi
        y[i] = dy*180*3600/np.pi
    totalOffset = (x**2 + y**2)**0.5

    # determine largest dimension of the map
    if (casaVersion < '4.3.0'):
        idx1_ignoreOffPosition, idx1_ignoreOffPositionX, idx1_ignoreOffPositionY = au.ignoreMostCommonPosition(x,y)
    else:
        onSourceTimes = mymsmd.timesforintent(intent)
        try:
            idx1_ignoreOffPosition = np.nonzero(np.in1d(times, onSourceTimes))
        except AttributeError:
            idx1_ignoreOffPosition = np.nonzero(np.isin(times, onSourceTimes))
        idx1_ignoreOffPositionX = idx1_ignoreOffPosition
        idx1_ignoreOffPositionY = idx1_ignoreOffPosition
        if (len(idx1_ignoreOffPosition[0]) > 0):
#            print "idx1_ignoreOffPosition = ", idx1_ignoreOffPosition
            xOnSource = x[idx1_ignoreOffPosition]
            yOnSource = y[idx1_ignoreOffPosition]
            xmad = au.MAD(xOnSource)
            ymad = au.MAD(yOnSource)
            xExtremeOffset = np.max(np.abs(xOnSource-np.median(xOnSource)))
            yExtremeOffset = np.max(np.abs(yOnSource-np.median(yOnSource)))
            if (xExtremeOffset > 5*xmad or yExtremeOffset > 5*ymad):
                # Resort to the old method if the identification by time has failed. - 2015-08-07
                # For example, for uid___A002_X9fa4e2_Xca8
                if (debug):
                    print("*********** xmad=%f, ymad=%f, xExtreme=%f, yExtreme=%f ********" % (xmad, ymad, xExtremeOffset, yExtremeOffset))
                idx1_ignoreOffPosition, idx1_ignoreOffPositionX, idx1_ignoreOffPositionY = au.ignoreMostCommonPosition(x,y)

            xOnSource = x[idx1_ignoreOffPosition]
            yOnSource = y[idx1_ignoreOffPosition]
        else:
            xOnSource = x
            yOnSource = y
    mymsmd.close()
    if (len(yOnSource) > 0):
        ymaxloc = yOnSource.argmax()
        yminloc = yOnSource.argmin()
        xmaxloc = xOnSource.argmax()
        xminloc = xOnSource.argmin()
        locs = [xminloc, xmaxloc, yminloc, ymaxloc]
        distances = []
        for l in range(3): # 0; 1; 2
            for i in range(l+1,4): # 1,2,3;  2,3;  3
                distance = ((xOnSource[locs[l]]-xOnSource[locs[i]])**2 + (yOnSource[locs[l]]-yOnSource[locs[i]])**2)**0.5
                distances.append(distance)
        largestDimension = round(np.max(distances))
    else:
        largestDimension = np.max([np.max(y)-np.min(y), np.max(x)-np.min(x)])

    # Determine the scan direction and sampling
    # It does not seem to be necessary to trim off the OFF position times for this
    # algorithm to get the right answer
    xSuccessiveDifferences = x[1:] - x[:-1]
    ySuccessiveDifferences = y[1:] - y[:-1]
    successiveDifferences = (xSuccessiveDifferences**2 + ySuccessiveDifferences**2)**0.5

    # Find the axis with most rapidly changing values
    mybeam = au.primaryBeamArcsec(vis, showEquation=False)
    finestSampling = np.median(np.abs(successiveDifferences))
    xSampling = np.median(np.abs(xSuccessiveDifferences))
    ySampling = np.median(np.abs(ySuccessiveDifferences))
    if debug:
        print("xSuccessiveDifferences[:500]=", xSuccessiveDifferences[:500])
        print("ySuccessiveDifferences[:500]=", ySuccessiveDifferences[:500])
        print("xSampling = ", xSampling)
        print("ySampling = ", ySampling)

    arbitraryScanAngle = False
    if (np.round(abs(xSampling),2) < np.round(abs(finestSampling),2) and
        np.round(abs(ySampling),2) < np.round(abs(finestSampling),2)):
        print("WARNING: This raster was scanned at some arbitrary position angle, not in RA or Dec.")
        print("xSampling=%f, ySampling=%f, finestSampling=%f" % (xSampling,ySampling, finestSampling))
        xSampling = finestSampling * medianSamplingIntervalDataTable / medianSamplingIntervalPointingTable
        raScan = False
        arbitraryScanAngle = True
        defaultXSampling = xSampling
        defaultYSampling = mybeam/3.0
        print("Will return the default values of X=%f, Y=%f." % (defaultXSampling, defaultYSampling))
    elif (abs(xSampling-finestSampling) < abs(ySampling-finestSampling)):
        print("This raster was scanned along RA.")
        raScan = True
    else:
        print("This raster was scanned along Dec.")
        raScan = False
        swap = x[:]
        x = y[:]
        y = swap[:]
        swap = xSuccessiveDifferences[:]
        xSuccessiveDifferences = ySuccessiveDifferences[:]
        ySuccessiveDifferences = swap[:]
    print("The finest sampling (in the pointing table) = %.3f arcsec." % (finestSampling))

    xSampling = finestSampling
    # correct for the ratio of pointing table data interval (0.048s) to correlator data rate (0.144s)
    print("Scaling sampling from pointing table interval to visibility data table interval: *%f" % (medianSamplingIntervalDataTable / medianSamplingIntervalPointingTable))
    xSampling *= medianSamplingIntervalDataTable / medianSamplingIntervalPointingTable

    # Find where the scan reverses direction
    xReversalPoints = (np.diff(np.sign(np.round(magnification*xSuccessiveDifferences)/magnification)) != 0)*1
    yReversalPoints = (np.diff(np.sign(np.round(magnification*ySuccessiveDifferences)/magnification)) != 0)*1
    reversalPoints = xReversalPoints + yReversalPoints
    indices = np.where(reversalPoints > 0)[0]
    rowChanges = indices[1::2]
    if (debug):
        print("%d xReversalPoints = %s" % (len(np.where(xReversalPoints>0)[0]), str(xReversalPoints)))
        print("%d yReversalPoints = %s" % (len(np.where(yReversalPoints>0)[0]), str(yReversalPoints)))
        print("Found %d row changes = %s" % (len(rowChanges), str(rowChanges)))

    successiveRowDifferences = (xSuccessiveDifferences[rowChanges]**2 + ySuccessiveDifferences[rowChanges]**2)**0.5
    if (debug):
        print("a) successiveRowDifferences = ", successiveRowDifferences)
        print("a) median = ", np.median(successiveRowDifferences))
    ySampling = np.median(successiveRowDifferences)
    xAtRowChanges = x[rowChanges]
    yAtRowChanges = y[rowChanges]
    if (originalCoordSys.find('AZEL') >= 0):
        roundedRowChanges = list(np.round(yAtRowChanges))
        try:
            ymode = max(set(roundedRowChanges), key=roundedRowChanges.count)
        except:
            ymode = np.median(yAtRowChanges) # old method, did not always work
        threshold = max(1,abs(ymode*ymodeThreshold))
        if (debug):
            print("abs(yAtRowChanges-ymode) = ", abs(yAtRowChanges - ymode))
            print("ymode = ", ymode)

        keeprows = np.where(abs(yAtRowChanges - ymode) > threshold)[0]
        if (debug):
            print("1) yAtRowChanges = ", str(yAtRowChanges))

        yAtRowChanges = yAtRowChanges[keeprows]
        xAtRowChanges = xAtRowChanges[keeprows]
        if (debug):
            print("Kept %d/%d rows" % (len(yAtRowChanges), len(x[rowChanges])))
            print("2) yAtRowChanges = ", str(yAtRowChanges))

        successiveRowDifferences = abs(yAtRowChanges[1:] - yAtRowChanges[:-1])
        if (debug):
            print("b) successiveRowDifferences = ", successiveRowDifferences)
            print("b) median = ", np.median(successiveRowDifferences))

        ySampling = np.median(successiveRowDifferences)

    if (fieldName is not None):
        if (fieldName.upper() in au.majorPlanets):
            if (pickFirstRaster == False):
                print("Setting pickFirstRaster=True")
            pickFirstRaster = True

    if (pickFirstRaster and intent.find('TARGET')>=0):

        # Check if idx1_ignoreOffPositionY is actually empty. In that case,
        # the slice fails.
        if idx1_ignoreOffPositionY[0].size == 0:
            idx1_ignoreOffPositionY = slice(None)
        if idx1_ignoreOffPositionX[0].size == 0:
            idx1_ignoreOffPositionX = slice(None)

        # Recalculate the y Sampling over only the first raster
        # Find the range of the raster rows

        maxYoffset = np.max(y[idx1_ignoreOffPositionY])
        minYoffset = np.min(y[idx1_ignoreOffPositionY])
        medianYoffset = np.median(y[idx1_ignoreOffPositionY])
        yrange = maxYoffset-minYoffset
        # idx1 = points within the upper part of the raster
        idx1 = np.where(y > maxYoffset-0.1*yrange)[0]
        if not isinstance(idx1_ignoreOffPositionY, slice):
            idx1 = np.intersect1d(idx1, idx1_ignoreOffPositionY[0])
        if not isinstance(idx1_ignoreOffPositionX, slice):
            idx1 = np.intersect1d(idx1, idx1_ignoreOffPositionX[0])

        if (debug):
            print("y[:400]=", y[:400])
            print("maxYoffset=%f, minYoffset=%f, yrange=%f, maxY-0.1*yrange=%f" % (maxYoffset, minYoffset,yrange,maxYoffset-0.1*yrange))
            print("idx1=upper_portion_of_map=%s" % (str(idx1)))

        # idx2 = points within the lower part of raster
        idx2 = np.where(y < minYoffset+0.1*yrange)[0]
        if not isinstance(idx1_ignoreOffPositionY, slice):
            idx2 = np.intersect1d(idx2, idx1_ignoreOffPositionY[0])

        if (debug):
            print("idx2=lower_portion_of_map=%s" % (str(idx2)))
        # idx3 = overlap of upper with lower
        idx3 = np.where(idx2 > idx1[0])[0]  # assumes that idx1 is increasing on the first raster row
        if (debug):
            print("idx3=%s" % (str(idx3)))
            if (len(idx3) > 0):
                print("idx2[idx3[0]] = ", idx2[idx3[0]])
        myx = x
        myy = y
        mytimes = times
        if (len(idx3) > 0):
            if (debug): print("We found a second raster. Clipping data to first raster.")
            if (len(idx2)>idx3[0]):
                endOfFirstRaster = idx2[idx3[0]]
                if (debug):
                    print("endOfFirstRaster = %d [(%f,%f),(%f,%f),(%f,%f)]" % (endOfFirstRaster,
                        x[endOfFirstRaster-1],y[endOfFirstRaster-1],
                        x[endOfFirstRaster],y[endOfFirstRaster],
                        x[endOfFirstRaster+1],y[endOfFirstRaster+1]))
                myx = x[:endOfFirstRaster]
                myy = y[:endOfFirstRaster]
                mytimes = times[:endOfFirstRaster]
        else:
            if (debug): print("There is only one raster in this dataset.")

        # determine the scan direction and sampling
        xSuccessiveDifferences = myx[1:] - myx[:-1]
        ySuccessiveDifferences = myy[1:] - myy[:-1]
        successiveDifferences = (xSuccessiveDifferences**2 + ySuccessiveDifferences**2)**0.5

        # Find where the scan reverses direction
        xReversalPoints = (np.diff(np.sign(np.round(magnification*xSuccessiveDifferences)/magnification)) != 0)*1
        yReversalPoints = (np.diff(np.sign(np.round(magnification*ySuccessiveDifferences)/magnification)) != 0)*1
        reversalPoints = xReversalPoints + yReversalPoints
        indices = np.where(reversalPoints > 0)[0]
        rowChanges = indices[1::2]
        if (debug):
            print("%d xReversalPoints = %s" % (len(np.where(xReversalPoints>0)[0]), str(xReversalPoints)))
            print("%d yReversalPoints = %s" % (len(np.where(yReversalPoints>0)[0]), str(yReversalPoints)))
            print("Found %d row changes = %s" % (len(rowChanges), str(rowChanges)))

        successiveRowDifferences = (xSuccessiveDifferences[rowChanges]**2 + ySuccessiveDifferences[rowChanges]**2)**0.5
        if (debug):
            print("c) successiveRowDifferences at rowChanges = ", successiveRowDifferences)

        myxAtRowChanges = myx[rowChanges]
        myyAtRowChanges = myy[rowChanges]
        if (not arbitraryScanAngle or forceCardinalScanAngle):
          if (originalCoordSys.find('AZEL') >= 0):
            roundedRowChanges = list(np.round(myyAtRowChanges))
            try:
                ymode = max(set(roundedRowChanges), key=roundedRowChanges.count)
            except:
                ymode = np.median(myyAtRowChanges)  # old method, did not always work

            if (debug):
                print("ymode = ", ymode)

            threshold = max(1,abs(ymode*ymodeThreshold))
            keeprows = np.where(abs(myyAtRowChanges - ymode) > threshold)[0]
            if (debug):
                print("3) yAtRowChanges = ", str(myyAtRowChanges))

            myyAtRowChanges = np.array(sorted(myyAtRowChanges[keeprows]))  # note the sorted() which is essential
            myxAtRowChanges = myxAtRowChanges[keeprows]
            if (debug):
                print("4) yAtRowChanges = ", str(myyAtRowChanges))

            successiveRowDifferences = abs(myyAtRowChanges[1:] - myyAtRowChanges[:-1])
            if (debug):
                print("d) successiveRowDifferences = ", successiveRowDifferences)

            roundedDiff = list(np.round(successiveRowDifferences))
            if len(roundedDiff) > 0:
                x_mode = max(set(roundedDiff), key=roundedDiff.count)
                print("x_mode = ", x_mode)
                idx = np.where(np.round(successiveRowDifferences) == x_mode)
            else:
                idx = [[]]
            print("matches to the mode = ", len(idx[0]))

            if (len(idx[0]) > len(successiveRowDifferences)/5):
                try:
                    ySampling = np.median(successiveRowDifferences[idx])
                except:
                    ySampling = np.median(successiveRowDifferences)  # old method which did not always work
            else:
                # if the most common value (the mode) does not appear more than 20% of the time, then
                ySampling = np.median(successiveRowDifferences) # fail over to the old method
          else:
            roundedDiff = list(np.round(successiveRowDifferences))
            x_mode = max(set(roundedDiff), key=roundedDiff.count)
            try:
                ySampling = np.median(successiveRowDifferences[np.where(np.round(successiveRowDifferences) == x_mode)])
            except:
                ySampling = np.median(successiveRowDifferences)  # old method which did not always work

    if (not raScan) and (not arbitraryScanAngle):
        swap = xSampling
        xSampling = ySampling
        ySampling = swap
        swap = x
        x = y
        y = swap
        swap = xAtRowChanges
        xAtRowChanges = yAtRowChanges
        yAtRowChanges = swap
    print("xSampling = %.3f arcsec (%.3f points per beam)" % (xSampling, mybeam/xSampling))
    print("ySampling = %.3f arcsec (%.3f points per beam)" % (ySampling, mybeam/ySampling))

    if (showplot or plotfile!=''):
        if (showplot == False):
            pb.ioff()
        mjdsec = au.getObservationStart(vis)
        obsdateString = au.mjdToUT(mjdsec/86400.)
        pb.clf()
        adesc = pb.subplot(211)
        lineStyleUpperPlot = '.'
        if (connectDots):
            lineStyle = '-'
        else:
            lineStyle = '.'
        pb.plot_date(pb.date2num(mjdSecondsListToDateTime(times)),totalOffset,
                     'k'+lineStyleUpperPlot,
                     markeredgecolor='k',markerfacecolor='k',markersize=2.0)
#        pb.hold(True) # not needed
        if (showComponents):
            pb.plot_date(pb.date2num(mjdSecondsListToDateTime(times)),x,'r-')
            pb.plot_date(pb.date2num(mjdSecondsListToDateTime(times)),y,'g-')
        pb.xlabel('Universal Time on %s' % (mjdsecToUT(times[0]).split()[0]))
        pb.ylabel('Angle from origin (arcsec)')
        adesc.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
        #au.setXaxisTimeTicks(adesc, np.min(times), np.max(times))
        setXaxisTimeTicks(adesc, np.min(times), np.max(times)) #<DZLIU>#
        y0,y1 = pb.ylim()
        for s in list(timesforscan.keys()):
            b = timesforscan[s]['begin']
            pb.plot_date(pb.date2num(mjdSecondsListToDateTime([b,b])), [y0,y1], 'k-')
            e = timesforscan[s]['end']
            pb.plot_date(pb.date2num(mjdSecondsListToDateTime([e,e])), [y0,y1], 'k--')
            pb.text(pb.date2num(mjdSecondsListToDateTime([0.5*(b+e)]))[0], 0.8*(y1-y0)+y0,
                    'Scan '+str(s),size=8,rotation='vertical')

        if (timerange == None or timerange == ''):
            newStartTime = timesforscan[scansToUse[0]]['begin'] - 20
            newEndTime = timesforscan[scansToUse[-1]]['end'] + 20
            newlimits = pb.date2num(mjdSecondsListToDateTime([newStartTime, newEndTime]))
            pb.xlim(newlimits)
            #au.setXaxisTimeTicks(adesc, newStartTime, newEndTime)
            setXaxisTimeTicks(adesc, newStartTime, newEndTime) #<DZLIU>#
        else:
            pb.xlim(pb.date2num(mjdSecondsListToDateTime(timerange)))
            #au.setXaxisTimeTicks(adesc, timerange[0], timerange[1])
            setXaxisTimeTicks(adesc, timerange[0], timerange[1]) #<DZLIU>#

        adesc.xaxis.grid(True,which='major')
        adesc.yaxis.grid(True,which='major')
        if (field is not None):
            fieldString = "  (0,0)=%s" % (fieldName)
        else:
            fieldString = ""
        mytitle = ', '.join([os.path.basename(vis), antennanames[antenna][0], obsdateString, 'OBS_ID=%d'%obsid, fieldString])
        pb.title(mytitle, fontsize=10)
        if (debug):
            print("subplot 211 : xlim=%s, ylim=%s" % (str(pb.xlim()),str(pb.ylim())))

        adesc = pb.subplot(212)
        pb.plot(x,y,'b.')
        pb.plot(x,y,'b'+lineStyle)
        xlimits = pb.xlim()
        ylimits = pb.ylim()
#        pb.hold(True) # not needed
        pb.plot(xAtRowChanges, yAtRowChanges, 'r.', x[0], y[0], 'ro')
        pb.xlim(xlimits)
        pb.ylim(ylimits)
        if (plotrange != [0,0,0,0]):
            if (plotrange[0] != 0 or plotrange[1] != 0):
                pb.xlim([plotrange[0],plotrange[1]])
            if (plotrange[2] != 0 or plotrange[3] != 0):
                pb.ylim([plotrange[2],plotrange[3]])
        else:
            pb.xlim([np.max(x), np.min(x)])
            pb.axis('equal')

        for i in range(np.min([labelFirstNSamples, len(x)//labelIncrement])):
            sample = i*labelIncrement
            pb.text(x[sample],y[sample],str(sample),size=8)

        myqa = au.createCasaTool(qatool)
        if (coordSys.find('AZEL') >= 0):
            azimString = myqa.formxxx('%frad'%(rightAscension), format='deg', prec=5)
            elevString = myqa.formxxx('%frad'%(declination), format='deg', prec=5)
            pb.xlabel('Azimuth offset (arcsec) from %s deg' % azimString)
            pb.ylabel('Elevation offset (arcsec) from %s deg' % elevString)
            pb.xlim([np.min(x), np.max(x)])
        else:
            raString = myqa.formxxx('%frad'%(rightAscension), format='hms', prec=2)
            decString = myqa.formxxx('%frad'%(declination), format='dms', prec=0).replace('.',':',2).replace('-0','-').replace('+0','+')
            if (apparentCoordinates):
                basis = '(apparent)'
            else:
                basis = '(J2000)'

            pb.ylabel('Dec offset (arcsec) from %s' % (decString), size=10)
            if (convert):
                pb.xlabel('Right Ascension offset (arcsec) from %s %s' % (raString,basis))
        myqa.done()

        adesc.xaxis.grid(True,which='major')
        adesc.yaxis.grid(True,which='major')

        if overlayOTPointings != '':
            f = open(overlayOTPointings,'r')
            lines = f.readlines()
            otx = []; oty = []
            for line in lines:
                token = line.split(',')
                if token[0] == 'RA ' or len(token)<4: continue
                otx.append(float(token[0]))
                oty.append(float(token[1]))
                if token[3] == 'ARCMINS':
                    otx[-1] *= 60
                    oty[-1] *= 60
            f.close()

            pb.plot(otx, oty, 'g+', ms=12)

        xlim = pb.xlim();  ylim = pb.ylim()
        pb.plot([0.025], [0.96], 'ro', transform=adesc.transAxes)
        pb.plot([0.025], [0.91], 'r.', transform=adesc.transAxes)
        pb.text(0.05, 0.93, 'start point', transform=adesc.transAxes)
        pb.text(0.05, 0.86, 'end stroke', transform=adesc.transAxes)
        # The following is require to restore the tight limits
        pb.xlim(xlim)
        pb.ylim(ylim)

        if (showplot):
            pb.draw()
        if (plotfile != ''):
            if (plotfile == True):
                plotfile = vis+'.obsid%d.sampling.png' % (obsid)
            if (os.path.exists(plotfile)):
                if (os.access(plotfile, os.W_OK) == False):
                    plotfile = '/tmp/' + os.path.basename(plotfile)
            else:
                mydir = os.path.dirname(plotfile)
                if (mydir == ''):
                    mydir = os.getcwd()
                if (os.access(mydir, os.W_OK) == False):
                    plotfile = '/tmp/' + os.path.basename(plotfile)
            pb.savefig(plotfile)
            print("Saved plot in %s" % (plotfile))

        if (showplot == False):
            pb.ion()
    else:
        print("To show a plot, re-run with parameter showplot=True")
    if (convert==False and coordSys.find('AZEL')>=0):
        print("To determine the reference position, re-run with convert=True")
    if (arbitraryScanAngle):
        print("Due to arbitraryScanAngle, returning the default values of X=%f, Y=%f." % (defaultXSampling, defaultYSampling))
        return defaultXSampling, defaultYSampling, largestDimension
    else:
        return xSampling, ySampling, largestDimension
    # end of getTPSampling
