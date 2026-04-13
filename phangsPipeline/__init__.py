from importlib.metadata import PackageNotFoundError, version

# Ensure CASA is installed
from .check_imports import is_casa_installed, is_spectral_cube_installed

casa_enabled = is_casa_installed()
spectral_cube_enabled = is_spectral_cube_installed()

try:
    from .version import version as __version__
except ImportError:
    __version__ = "dev"

from .handlerAlmaDownload import AlmaDownloadHandler
from .handlerKeys import KeyHandler
from .handlerRelease import ReleaseHandler
from .phangsLogger import setup_logger

__all__ = [
    "AlmaDownloadHandler",
    "KeyHandler",
    "ReleaseHandler",
    "setup_logger",
]

# Modules that require CASA to be installed
if casa_enabled:
    from .handlerImagingChunked import ImagingChunkedHandler
    from .handlerImaging import ImagingHandler
    from .handlerPostprocess import PostProcessHandler
    from .handlerSingleDish import SingleDishHandler
    from .handlerTestImaging import TestImagingHandler
    from .handlerVis import VisHandler

    __all__.extend([
        "ImagingChunkedHandler",
        "ImagingHandler",
        "PostProcessHandler",
        "SingleDishHandler",
        "TestImagingHandler",
        "VisHandler",
    ])

# Modules that require spectral-cube to be installed
if spectral_cube_enabled:
    from .handlerDerived import DerivedHandler
    __all__.extend([
        "DerivedHandler",
    ])
