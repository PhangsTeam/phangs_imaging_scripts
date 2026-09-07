import os
import tempfile

import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from spectral_cube import Projection, SpectralCube

from phangsPipeline.scMoments import moment_generator


class TestMomentGenerator:
    """Suite of tests for moment_generator"""

    # Set up a temporary directory to keep things clean
    tmp_path = tempfile.TemporaryDirectory()

    # Set up a *very basic* cube
    data = np.ones((10, 100, 100)) * u.K

    # Create a basic World Coordinate System (WCS)
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "VRAD"]
    wcs.wcs.crval = [0.0, 0.0, 0.0]  # Reference coordinates and velocity (m/s)
    wcs.wcs.cdelt = [-0.01, 0.01, 2500]  # Step size
    wcs.wcs.crpix = [0.0, 0.0, 0.0]  # Reference pixel

    # Construct the SpectralCube
    cube = SpectralCube(data=data,
                        wcs=wcs,
                        )
    cube_file = os.path.join(tmp_path.name, "scMoments_test_cube.fits")
    cube.write(cube_file, overwrite=True)

    # Create a mask that has some masked values
    mask = np.ones_like(data)
    mask[2, 5:8, 2:4] = 0
    mask = SpectralCube(data=np.ones_like(data),
                        wcs=wcs,
                        )
    mask_file = os.path.join(tmp_path.name, "scMoments_test_mask.fits")
    mask.write(mask_file, overwrite=True)

    noise = SpectralCube(data=np.random.normal(loc=0, scale=0.3, size=data.shape),
                         wcs=wcs,
                         )
    noise_file = os.path.join(tmp_path.name, "scMoments_test_noise.fits")
    noise.write(noise_file, overwrite=True)

    @pytest.mark.xfail(raises=ValueError, reason="No moment tag passed")
    def test_pass_non_moment(self):
        """Test passing no moment tag, which should fail"""

        moment_generator(
            self.cube,
            moment=None,
        )

    @pytest.mark.xfail(raises=TypeError, reason="Incorrect moment tag type passed")
    def test_pass_incorrect_moment_type(self):
        """Test passing bad moment tag, which should fail"""

        moment_generator(
            self.cube,
            moment=12345,
        )

    @pytest.mark.xfail(raises=ValueError, reason="Unrecognised moment tag passed")
    def test_pass_unrecognised_moment(self):
        """Test passing an unrecognised moment tag, which should fail"""

        moment = "bad_moment"

        moment_generator(
            self.cube,
            moment=moment,
        )

    @pytest.mark.xfail(raises=TypeError, reason="kwargs not a dictionary")
    def test_pass_non_dict_kwargs(self):
        """Test passing a moment case where the default kwargs are not a dictionary, which should fail"""

        moment = "failure_case"

        moment_generator(
            self.cube,
            moment=moment,
        )

    @pytest.mark.xfail(raises=TypeError, reason="momkwargs not a dictionary")
    def test_pass_non_dict_momkwargs(self):
        """Test passing momkwargs that is not a dictionary, which should fail"""

        momkwargs = "not_a_dict"

        moment_generator(
            self.cube,
            moment="mom0",
            momkwargs=momkwargs,
        )

    @pytest.mark.xfail(raises=TypeError, reason="Not a cube")
    def test_pass_non_cube(self):
        """Test passing a non-cube, which should fail"""

        cubein = 12345

        moment_generator(
            cubein,
            moment="mom0",
        )

    @pytest.mark.xfail(raises=TypeError, reason="Not a mask")
    def test_pass_non_mask(self):
        """Test passing a non-mask, which should fail"""

        mask = 12345

        moment_generator(
            self.cube,
            mask=mask,
            moment="mom0",
        )

    @pytest.mark.xfail(raises=TypeError, reason="Not a noise cube")
    def test_pass_non_noise(self):
        """Test passing a non-noise cube, which should fail"""

        noise = 12345

        moment_generator(
            self.cube,
            noise=noise,
            moment="mom0",
        )

    def test_pass_linewidth(self):
        """Test passing a linewidth in momkwargs"""

        momkwargs = {"line_width": 5 * u.km / u.s}

        moment_map, error_map = moment_generator(
            self.cube,
            moment="mom0",
            momkwargs=momkwargs,
        )
        assert isinstance(moment_map, Projection)

    def test_pass_cube_spectralcube(self):
        """Test passing a SpectralCube"""

        moment_map, error_map = moment_generator(
            self.cube,
            moment="mom0",
        )
        assert isinstance(moment_map, Projection)

    def test_pass_cube_file(self):
        """Test passing a cube file"""

        moment_map, error_map = moment_generator(
            self.cube_file,
            moment="mom0",
        )
        assert isinstance(moment_map, Projection)

    def test_pass_mask_spectralcube(self):
        """Test passing a SpectralCube mask"""

        moment_map, error_map = moment_generator(
            self.cube,
            mask=self.mask,
            moment="mom0",
        )
        assert isinstance(moment_map, Projection)

    def test_pass_mask_file(self):
        """Test passing a mask file"""

        moment_map, error_map = moment_generator(
            self.cube,
            mask=self.mask_file,
            moment="mom0",
        )
        assert isinstance(moment_map, Projection)

    def test_pass_noise_spectralcube(self):
        """Test passing a SpectralCube noise cube"""

        moment_map, error_map = moment_generator(
            self.cube,
            mask=self.mask,
            noise=self.noise,
            moment="mom0",
        )
        assert isinstance(moment_map, Projection)

    def test_pass_noise_file(self):
        """Test passing a noise file"""

        moment_map, error_map = moment_generator(
            self.cube,
            mask=self.mask,
            noise=self.noise_file,
            moment="mom0",
        )
        assert isinstance(moment_map, Projection)
