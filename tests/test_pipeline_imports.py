import pytest

import phangsPipeline


def _import_handler(
    handler,
):
    """Import handler.

    Args:
        handler: Name for Handler to import
    """

    if hasattr(phangsPipeline, handler):
        success = True
    else:
        success = False

    return success


class TestPipelineImports:
    """Suite of tests for various pipeline imports."""

    # Some handlers need CASA, some don't
    non_casa_handlers = [
        "AlmaDownloadHandler",
        "KeyHandler",
        "ReleaseHandler",
        "DerivedHandler",
        "setup_logger",
    ]

    casa_handlers = [
        "ImagingChunkedHandler",
        "ImagingHandler",
        "PostProcessHandler",
        "SingleDishHandler",
        "TestImagingHandler",
        "VisHandler",
    ]

    @pytest.mark.parametrize("handler", non_casa_handlers)
    def test_import_non_casa_handler(
        self,
        handler,
    ):
        """Test non-CASA Handler import.

        Args:
            handler: Name for Handler to import
        """

        success = _import_handler(handler)

        assert success

    @pytest.mark.casa
    @pytest.mark.parametrize("handler", casa_handlers)
    def test_import_casa_handler(
        self,
        handler,
    ):
        """Test CASA Handler import.

        Args:
            handler: Name for Handler to import
        """

        success = _import_handler(handler)

        assert success
