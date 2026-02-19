# CASA imports
import casashell
import casatasks
import casatools
import casaplotms
from almahelpers_localcopy import tsysspwmap
from casatasks import (
    casalog,
    concat,
    exportfits,
    feather,
    flagcmd,
    flagdata,
    gencal,
    imhead,
    immath,
    impbcor,
    importasdm,
    importfits,
    imrebin,
    imregrid,
    imsmooth,
    imstat,
    imsubimage,
    imtrans,
    imval,
    listobs,
    makemask,
    mstransform,
    plotbandpass,
    rmtables,
    sdbaseline,
    sdcal,
    split,
    statwt,
    tclean,
    tsdimaging,
    uvcontsub,
    visstat,
)
from casatasks.private import sdint_helper
from casatools import (
    table,
    image,
    imager,
    msmetadata,
    synthesisimager,
    synthesisutils,
    regionmanager,
    measures,
    quanta,
)

try:
    import casaviewer
except (ImportError, ModuleNotFoundError):
    casaviewer = None
    print("Could not import casaviewer")

# TODO: Move back to CASA task
from .taskSDIntImaging import sdintimaging

# Get CASA version
casa_version = (
    casatools.version()[0],
    casatools.version()[1],
    casatools.version()[2],
)
casa_version_str = ".".join(
    [str(casa_version_no) for casa_version_no in casa_version]
)

print(f"CASA version: {casa_version_str}")

iatool = image
rgtool = regionmanager
imtool = imager
msmdtool = msmetadata
tbtool = table
metool = measures
qatool = quanta

plotms = casaplotms.plotms
if casaviewer is not None:
    viewer = casaviewer.imview
