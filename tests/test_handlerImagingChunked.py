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


class _StubKeyHandler:
    """Minimal stand-in for KeyHandler, returning a fixed clean mask filename."""

    def __init__(self, cleanmask):
        self._cleanmask = cleanmask

    def get_cleanmask_filename(self, target=None, product=None):
        return self._cleanmask


class _StubCleanCall:
    """Minimal stand-in for CleanCall, supporting only get/set of parameters."""

    def __init__(self, **params):
        self._params = params

    def get_param(self, key):
        return self._params.get(key)

    def set_param(self, key, value, nowarning=False):
        self._params[key] = value


def _bare_chunked_handler(key_handler, imaging_dir, chunk_ranges, imaging_method="tclean"):
    """ImagingChunkedHandler with just enough state to exercise the mask tasks, without an MS."""

    from phangsPipeline.handlerImagingChunked import ImagingChunkedHandler

    handler = ImagingChunkedHandler.__new__(ImagingChunkedHandler)
    handler._kh = key_handler
    handler._dry_run = False
    handler.target = _TARGET
    handler.config = _CONFIG
    handler.product = _PRODUCT
    handler.imaging_method = imaging_method
    handler.chunk_params = {}
    for chunk_num, (chan_start, chan_end) in enumerate(chunk_ranges):
        image_name = f"cube_chan{chan_start}_{chan_end}"
        handler.chunk_params[chunk_num] = {
            "channel_range": [chan_start, chan_end],
            "image_name": image_name,
            "full_imagename": os.path.join(imaging_dir, image_name),
        }
    handler.nchunks = len(handler.chunk_params)
    return handler


@pytest.mark.casa
def test_chunked_read_clean_mask_aligns_per_chunk(tmp_path):
    """Each chunk gets the matching channel slice of the clean mask, on its own grid."""

    import numpy as np
    from astropy.io import fits

    from phangsPipeline import casaStuff

    nx, ny, nchan = 24, 20, 12
    chunk_ranges = [(0, 5), (6, 11)]

    def write_fits(filename, data, stokes=False):
        """Write a cube on a realistic CO(2-1) grid, optionally with a Stokes axis like tclean output."""
        header = fits.Header()
        header.update(
            CTYPE1="RA---SIN", CRVAL1=180.0, CDELT1=-0.5 / 3600, CRPIX1=nx // 2 + 1, CUNIT1="deg",
            CTYPE2="DEC--SIN", CRVAL2=-30.0, CDELT2=0.5 / 3600, CRPIX2=ny // 2 + 1, CUNIT2="deg",
            RESTFRQ=230.538e9, SPECSYS="LSRK", RADESYS="FK5", EQUINOX=2000.0,
        )
        if stokes:
            header.update(CTYPE3="STOKES", CRVAL3=1.0, CDELT3=1.0, CRPIX3=1.0)
            header.update(CTYPE4="FREQ", CRVAL4=230.5e9, CDELT4=1e6, CRPIX4=1.0, CUNIT4="Hz")
        else:
            header.update(CTYPE3="FREQ", CRVAL3=230.5e9, CDELT3=1e6, CRPIX3=1.0, CUNIT3="Hz")
        fits.PrimaryHDU(data=data, header=header).writeto(filename, overwrite=True)

    # Full-cube image to act as the dirty image, with axes ordered as tclean writes them
    full_fits = str(tmp_path / "full.fits")
    write_fits(full_fits, np.ones((nchan, 1, ny, nx), dtype=np.float32), stokes=True)
    full_image = str(tmp_path / "full.image")
    casaStuff.importfits(fitsimage=full_fits, imagename=full_image, overwrite=True)

    # Clean mask: a different box per channel, so a channel mix-up would show.
    # Expected values are in CASA (RA, Dec, Stokes, Freq) order
    expected = np.zeros((nx, ny, 1, nchan), dtype=np.float32)
    for chan in range(nchan):
        expected[chan:chan + 5, 2:8 + chan // 2, 0, chan] = 1.0
    mask_fits = str(tmp_path / "cleanmask.fits")
    write_fits(mask_fits, np.transpose(expected[:, :, 0, :]))

    handler = _bare_chunked_handler(_StubKeyHandler(mask_fits), str(tmp_path), chunk_ranges)

    # Per-chunk "dirty" images, cut from the full cube
    for params in handler.chunk_params.values():
        chan_start, chan_end = params["channel_range"]
        casaStuff.imsubimage(imagename=full_image, outfile=params["full_imagename"] + ".image",
                             chans=f"{chan_start}~{chan_end}", overwrite=True)

    handler.task_read_clean_mask()

    for params in handler.chunk_params.values():
        chan_start, chan_end = params["channel_range"]
        mask_name = params["full_imagename"] + ".mask"
        assert os.path.isdir(mask_name)

        myia = casaStuff.iatool()
        myia.open(mask_name)
        shape = list(myia.shape())
        data = myia.getchunk()
        myia.close()

        assert shape == [nx, ny, 1, chan_end - chan_start + 1]
        np.testing.assert_array_equal(data, expected[..., chan_start:chan_end + 1])


@pytest.mark.casa
def test_chunked_read_clean_mask_no_mask(tmp_path):
    """With no clean mask defined, nothing is written and nothing is raised."""

    handler = _bare_chunked_handler(_StubKeyHandler(None), str(tmp_path), [(0, 5)])
    handler.task_read_clean_mask()

    assert not os.path.exists(handler.chunk_params[0]["full_imagename"] + ".mask")


@pytest.mark.casa
def test_chunked_read_clean_mask_skips_missing_template(tmp_path):
    """A chunk without an image to align to is skipped, rather than raising."""

    mask_fits = tmp_path / "cleanmask.fits"
    mask_fits.write_text("")

    handler = _bare_chunked_handler(_StubKeyHandler(str(mask_fits)), str(tmp_path), [(0, 5)])
    handler.task_read_clean_mask()

    assert not os.path.exists(handler.chunk_params[0]["full_imagename"] + ".mask")


@pytest.mark.casa
@pytest.mark.parametrize(
    "mask_exists, params, expected_usemask, expected_pbmask",
    [
        # No mask on disk and no user mask: fall back to a pb mask at pblimit
        (False, {"usemask": "user", "mask": "", "pblimit": 0.2}, "pb", 0.2),
        # ... or at 0.5 if pblimit is not set
        (False, {"usemask": "user", "mask": "", "pblimit": None}, "pb", 0.5),
        # A user mask from the clean parameter file is kept
        (False, {"usemask": "user", "mask": "user.mask", "pblimit": 0.2}, "user", None),
        # A mask on disk (e.g. an aligned clean mask) is kept
        (True, {"usemask": "user", "mask": "", "pblimit": 0.2}, "user", None),
    ],
)
def test_chunked_pb_mask_fallback(tmp_path, mask_exists, params, expected_usemask, expected_pbmask):
    """Mirror ImagingHandler: use a pb mask only when no mask of any kind is in place."""

    handler = _bare_chunked_handler(_StubKeyHandler(None), str(tmp_path), [(0, 5)])
    imagename = handler.chunk_params[0]["full_imagename"]
    if mask_exists:
        os.makedirs(imagename + ".mask")

    clean_call = _StubCleanCall(imagename=imagename, **params)
    handler._use_pb_mask_if_no_mask(clean_call)

    assert clean_call.get_param("usemask") == expected_usemask
    assert clean_call.get_param("pbmask") == expected_pbmask
