import pytest
import sys
from inspect import getfullargspec

try:
    import analysisUtils as au
except ModuleNotFoundError:
    au = None

try:
    import casatasks
except ModuleNotFoundError:
    casatasks = None


def check_args_exist(in_args, function):
    """Check if arguments exist within a function.

    Args:
        in_args (list): list of expected arguments
        function (function): function to check for arguments
    """

    func_args = getfullargspec(function).args
    all_args_exist = set(in_args).issubset(set(func_args))

    return all_args_exist


class TestCASAImports:
    """Suite of tests for various CASA imports."""

    casa_imports = [
        "casashell",
        "casaplotms",
        pytest.param(
            "casaviewer",
            marks=pytest.mark.xfail(
                sys.platform == "darwin",
                reason="casaviewer not available on Mac OS X",
            ),
        ),
        "casatasks.private.sdint_helper",
    ]

    @pytest.mark.casa
    @pytest.mark.parametrize("casa_module", casa_imports)
    def test_import_casa(
        self,
        casa_module,
    ):
        """Test CASA imports.

        Args:
            casa_module: CASA module to import
        """

        success = False
        try:
            __import__(casa_module)
            success = True
        except ModuleNotFoundError:
            pass

        assert success


class TestCASATaskArguments:
    """Suite of tests for testing CASA task arguments exists."""

    casatask_funcs = [
        (
            "concat",
            [
                "vis",
                "dirtol",
                "concatvis",
                "copypointing",
                "freqtol",
                "respectname",
            ],
        ),
        (
            "exportfits",
            [
                "bitpix",
                "dropdeg",
                "dropstokes",
                "fitsimage",
                "imagename",
                "overwrite",
                "velocity",
            ],
        ),
        (
            "feather",
            [
                "highres",
                "imagename",
                "lowpassfiltersd",
                "lowres",
                "sdfactor",
            ],
        ),
        (
            "flagcmd",
            [
                "action",
                "inpmode",
                "plotfile",
                "useapplied",
                "vis",
            ],
        ),
        (
            "flagdata",
            [
                "action",
                "antenna",
                "mode",
                "spw",
                "vis",
            ],
        ),
        (
            "gencal",
            [
                "caltable",
                "caltype",
                "vis",
            ],
        ),
        (
            "imhead",
            [
                "mode",
            ],
        ),
        (
            "immath",
            [
                "expr",
                "imagemd",
                "imagename",
                "mode",
                "outfile",
                "stokes",
            ],
        ),
        (
            "impbcor",
            [
                "cutoff",
                "imagename",
                "mode",
                "outfile",
                "pbimage",
            ],
        ),
        (
            "importasdm",
            [
                "asis",
                "bdfflags",
                "process_caldevice",
                "with_pointing_correction",
            ],
        ),
        (
            "importfits",
            [
                "defaultaxes",
                "defaultaxesvalues",
                "fitsimage",
                "imagename",
                "overwrite",
                "zeroblanks",
            ],
        ),
        (
            "imrebin",
            [
                "crop",
                "dropdeg",
                "factor",
                "imagename",
                "outfile",
                "overwrite",
            ],
        ),
        (
            "imsmooth",
            [
                "major",
                "minor",
                "outfile",
                "overwrite",
                "pa",
                "targetres",
                "imagename",
            ],
        ),
        (
            "imstat",
            [
                "imagename",
                "mask",
            ],
        ),
        (
            "imsubimage",
            [
                "box",
                "chans",
                "dropdeg",
                "imagename",
                "mask",
                "outfile",
            ],
        ),
        (
            "imtrans",
            [
                "imagename",
                "order",
                "outfile",
            ],
        ),
        (
            "imval",
            [
                "box",
                "stokes",
            ],
        ),
        (
            "listobs",
            [
                "listfile",
                "vis",
            ],
        ),
        (
            "mstransform",
            [
                "datacolumn",
                "outputvis",
                "spw",
                "vis",
            ],
        ),
        (
            "plotbandpass",
            [
                "buildpdf",
                "caltable",
                "chanrange",
                "field",
                "figfile",
                "interactive",
                "overlay",
                "pwv",
                "showatm",
                "showfdm",
                "subplot",
                "xaxis",
                "yaxis",
            ],
        ),
        (
            "sdbaseline",
            [
                "blfunc",
                "datacolumn",
                "infile",
                "maskmode",
                "order",
                "outfile",
                "overwrite",
                "spw",
            ],
        ),
        (
            "sdcal",
            [
                "calmode",
                "infile",
                "outfile",
                "overwrite",
                "spw",
                "spwmap",
            ],
        ),
        (
            "split",
            [
                "antenna",
                "datacolumn",
                "field",
                "intent",
                "keepflags",
                "outputvis",
                "spw",
                "timebin",
                "vis",
                "width",
            ],
        ),
        (
            "statwt",
            [
                "chanbin",
                "datacolumn",
                "excludechans",
                "fitspw",
                "slidetimebin",
                "statalg",
                "timebin",
                "vis",
            ],
        ),
        (
            "tsdimaging",
            [
                "cell",
                "convsupport",
                "field",
                "gridfunction",
                "imsize",
                "infiles",
                "mode",
                "nchan",
                "outfile",
                "outframe",
                "overwrite",
                "phasecenter",
                "restfreq",
                "start",
                "veltype",
                "width",
            ],
        ),
        (
            "uvcontsub",
            [
                "fitmethod",
                "fitorder",
                "fitspec",
                "outputvis",
                "vis",
            ],
        ),
        (
            "visstat",
            [
                "axis",
                "spw",
                "vis",
            ],
        ),
    ]

    @pytest.mark.casa
    @pytest.mark.parametrize("func,in_args", casatask_funcs)
    def test_casatask_args(
        self,
        func,
        in_args,
    ):
        """Check arguments exist within casatask function.

        Args:
            func: function to check for arguments
            in_args (list): list of expected arguments
        """

        if casatasks is None:
            raise ModuleNotFoundError("Could not import casatasks")

        casatask_func = getattr(casatasks, func)

        success = check_args_exist(in_args, casatask_func)

        assert success


class TestAnalysisUtilsImports:
    """Suite of tests for various AnalysisUtils imports."""

    au_funcs = [
        "Atmcal",
        "MAD",
        "ValueMapping",
        "angularSeparationRadians",
        "clearPointingTable",
        "computeAzElFromRADecMJD",
        "computeRADecFromAzElMJD",
        "createCasaTool",
        "dateStringToMJD",
        "getAntennaPadXYZ",
        "getBaselineStats",
        "getChanWidths",
        "getObservationStart",
        "getObservationStartDate",
        "getOffSourceTimes",
        "getRADecForField",
        "getRADecForSource",
        "getScienceSpws",
        "lsrkToTopo",
        "mjdSecondsListToDateTime",
        "mjdSecondsToMJDandUT",
        "parseTimerangeArgument",
        "pickCellSize",
        "radec2rad",
        "stuffForScienceDataReduction",
        "timeOnSource",
    ]

    @pytest.mark.casa
    @pytest.mark.parametrize("func", au_funcs)
    def test_import_au_func(
        self,
        func,
    ):
        """Test analysisUtils import.

        Args:
            func: function to check imports
        """

        if au is None:
            raise ModuleNotFoundError("Could not import analysisUtils")

        if hasattr(au, func):
            success = True
        else:
            success = False

        assert success


class TestAnalysisUtilsArguments:
    """Suite of tests for testing analysisUtils function arguments exists."""

    au_funcs = [
        (
            "angularSeparationRadians",
            [
                "returnComponents",
            ],
        ),
        (
            "computeAzElFromRADecMJD",
            [
                "verbose",
            ],
        ),
        (
            "computeRADecFromAzElMJD",
            [
                "cofa",
                "frame",
                "my_metool",
                "nutationCorrection",
                "refractionCorrection",
                "verbose",
            ],
        ),
        (
            "getRADecForField",
            [
                "forcePositiveRA",
                "usemstool",
            ],
        ),
        (
            "getScienceSpws",
            [
                "intent",
                "returnFreqRanges",
            ],
        ),
        (
            "pickCellSize",
            [
                "imsize",
                "intent",
                "npix",
                "pblevel",
            ],
        ),
    ]

    @pytest.mark.casa
    @pytest.mark.parametrize("func,in_args", au_funcs)
    def test_analysisutils_args(
        self,
        func,
        in_args,
    ):
        """Check arguments exist within analysisUtils function.

        Args:
            func: function to check for arguments
            in_args (list): list of expected arguments
        """

        if au is None:
            raise ModuleNotFoundError("Could not import analysisUtils")

        au_func = getattr(au, func)

        success = check_args_exist(in_args, au_func)

        assert success
