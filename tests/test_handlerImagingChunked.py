import importlib.util
import os

import pytest

# Load the key writer directly, since importing the phangsPipelineTests package
# pulls in the full (slow, side-effectful) legacy test suite
_spec = importlib.util.spec_from_file_location(
    "minimal_key_writer",
    os.path.join(os.path.dirname(__file__), "..", "phangsPipelineTests", "minimal_key_writer.py"),
)
_mkw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mkw)
MinimalKeyWriter = _mkw.MinimalKeyWriter

_TARGET = "testgal_1"
_MOSAIC = "testgal"
_CONFIG = "C1"
_PRODUCT = "co21"


@pytest.fixture
def mosaic_key_handler(tmp_path):
    """KeyHandler where _MOSAIC is a linear mosaic of _TARGET, with no MS of its own."""

    from phangsPipeline import handlerKeys as kh

    writer = MinimalKeyWriter(
        key_dir=tmp_path / "keys",
        data_dir=tmp_path / "data",
        ms_root=tmp_path / "uvdata",
    )
    master_key = writer.write_all(
        target=_TARGET,
        config=_CONFIG,
        product=_PRODUCT,
        ms_filename="missing.ms",
        restfreq_GHz=230.538,
        ra="12h00m00.0s",
        dec="-30d00m00.0s",
    )

    # Add the mosaic as a target, and define it as a linear mosaic
    with open(os.path.join(writer.key_dir, "target_definitions.txt"), "a") as f:
        f.write(f"{_MOSAIC}\t12h00m00.0s\t-30d00m00.0s\t0\t15\n")
    with open(os.path.join(writer.key_dir, "linearmosaic_definitions.txt"), "a") as f:
        f.write(f"{_MOSAIC}\t{_TARGET}\n")

    cwd = os.getcwd()
    yield kh.KeyHandler(master_key=master_key)
    os.chdir(cwd)


@pytest.mark.casa
def test_chunked_handler_skips_linmos_target(mosaic_key_handler):
    """ImagingChunkedHandler should skip linear mosaic targets, rather than crash."""

    from phangsPipeline.handlerImagingChunked import ImagingChunkedHandler

    assert mosaic_key_handler.is_target_linmos(_MOSAIC)

    im_handler = ImagingChunkedHandler(
        target=_MOSAIC,
        config=_CONFIG,
        product=_PRODUCT,
        key_handler=mosaic_key_handler,
        make_temp_dir=False,
    )

    assert im_handler.skip_target
    assert im_handler.nchunks == 0

    # Neither of these should do anything, or raise
    im_handler.run_imaging(do_all=True)
    im_handler.task_complete_gather_into_cubes(root_name="all")
