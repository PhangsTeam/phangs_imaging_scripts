import astropy.units as u
import numpy as np
import pytest

from phangsPipeline import utilsResolutions


class TestUtilsResolutions:
    """Suite of tests for the utilsResolutions module"""

    test_angular_resolutions = [
        ("5.0arcsec", True),
        ("5p0arcsec", True),
        ("5.0arcmin", True),
        ("5.0deg", True),
        ("5.0rad", True),
        (5 * u.arcsec, True),
        (5 * u.arcmin, True),
        (5 * u.deg, True),
        (5 * u.rad, True),
        ("5.0pc", False),
        ("5.0kpc", False),
        ("5.0Mpc", False),
        (5 * u.pc, False),
        (5 * u.kpc, False),
        (5 * u.Mpc, False),
        (5.0, True),
        (5, True),
        (-5.0, True),
        (-5, True),
        pytest.param(
            (1, 2, 3),
            False,
            marks=pytest.mark.xfail(
                raises=TypeError,
                reason="Not a string, float, int, or u.Quantity",
            ),
        ),
    ]

    test_physical_resolutions = [
        ("5.0pc", True),
        ("5p0pc", True),
        ("5.0kpc", True),
        ("5.0Mpc", True),
        (5 * u.pc, True),
        (5 * u.kpc, True),
        (5 * u.Mpc, True),
        ("5.0arcsec", False),
        ("5.0arcmin", False),
        ("5.0deg", False),
        ("5.0rad", False),
        (5 * u.arcsec, False),
        (5 * u.arcmin, False),
        (5 * u.deg, False),
        (5 * u.rad, False),
        (5.0, True),
        (5, True),
        (-5.0, True),
        (-5, True),
        pytest.param(
            (1, 2, 3),
            False,
            marks=pytest.mark.xfail(
                raises=TypeError,
                reason="Not a string, float, int, or u.Quantity",
            ),
        ),
    ]

    @pytest.mark.parametrize("angular_resolution,output", test_angular_resolutions)
    def test_angular_resolution(
        self,
        angular_resolution: str | float | int | u.Quantity,
        output: bool,
    ):
        """Test the is_angular_resolution function

        Args:
            angular_resolution (str|float|int|u.Quantity): Input angular resolution
            output (bool): Expected output. Should be True if a valid angular resolution,
                or False otherwise
        """

        res = utilsResolutions.is_angular_resolution(angular_resolution)
        assert res == output

    def test_return_angular_resolution(self):
        """Test the is_angular_resolution function with return_value=True

        This test checks that the function returns the correct value in arcsec
        when the return_value parameter is set to True.
        """

        is_angular_resolution, ang_res_arcsec = utilsResolutions.is_angular_resolution(
            res="5.0arcsec",
            return_value=True,
        )
        assert ang_res_arcsec == 5.0

    @pytest.mark.parametrize("physical_resolution,output", test_physical_resolutions)
    def test_physical_resolution(
        self,
        physical_resolution: str | float | int | u.Quantity,
        output: bool,
    ):
        """Test the is_physical_resolution function

        Args:
            physical_resolution (str|float|int|u.Quantity): Input physical resolution
            output (bool): Expected output. Should be True if a valid physical resolution,
                or False otherwise
        """

        res = utilsResolutions.is_physical_resolution(physical_resolution)
        assert res == output

    def test_return_physical_resolution(self):
        """Test the is_physical_resolution function with return_value=True

        This test checks that the function returns the correct value in pc
        when the return_value parameter is set to True.
        """

        is_physical_resolution, phys_res_pc = utilsResolutions.is_physical_resolution(
            res="5.0pc",
            return_value=True,
        )
        assert phys_res_pc == 5.0

    @pytest.mark.parametrize("dist,output", test_physical_resolutions)
    def test_distances(
        self,
        dist: str | float | int | u.Quantity,
        output: bool,
    ):
        """Test the is_distance function

        Args:
            dist (str|float|int|u.Quantity): Input distance
            output (bool): Expected output. Should be True if a valid distance,
                or False otherwise
        """

        res = utilsResolutions.is_distance(dist)
        assert res == output

    def test_return_distance(self):
        """Test the is_distance function with return_value=True

        This test checks that the function returns the correct value in Mpc
        when the return_value parameter is set to True.
        """

        is_dist, dist_mpc = utilsResolutions.is_distance(
            distance="5.0Mpc",
            return_value=True,
        )
        assert dist_mpc == 5.0

    def test_get_tag_for_angular_resolution(self):
        """Test the get_tag_for_angular_resolution function.

        This test checks that the function returns the correct tag string for a given angular resolution.
        """

        tag = utilsResolutions.get_tag_for_angular_resolution(
            res="5.0arcsec",
            sig_figs=2,
        )
        assert tag == "5p00"

    @pytest.mark.xfail(raises=ValueError, reason="Not an angular resolution")
    def test_fail_get_tag_for_angular_resolution(self):
        """Test the fail case for get_tag_for_angular_resolution function when non-angular resolution is passed."""

        utilsResolutions.get_tag_for_angular_resolution(
            res="5.0pc",
            sig_figs=2,
        )

    def test_get_tag_for_physical_resolution(self):
        """Test the get_tag_for_physical_resolution function.

        This test checks that the function returns the correct tag string for a given physical resolution.
        """

        tag = utilsResolutions.get_tag_for_physical_resolution(
            res="120.0pc",
        )
        assert tag == "120pc"

    @pytest.mark.xfail(raises=ValueError, reason="Not a physical resolution")
    def test_fail_get_tag_for_physical_resolution(self):
        """Test the fail case for get_tag_for_physical_resolution function when non-physical resolution is passed."""

        utilsResolutions.get_tag_for_physical_resolution(
            res="5.0arcsec",
        )

    def test_get_tag_for_res_angular(self):
        """Test the get_tag_for_res function for angular resolution.

        This test checks that the function returns the correct tag string for a given angular resolution.
        """

        tag = utilsResolutions.get_tag_for_res(
            res="5.0arcsec",
        )
        assert tag == "5p00"

    def test_get_tag_for_res_physical(self):
        """Test the get_tag_for_res function for physical resolution.

        This test checks that the function returns the correct tag string for a given physical resolution.
        """

        tag = utilsResolutions.get_tag_for_res(
            res="25pc",
        )
        assert tag == "25pc"

    @pytest.mark.xfail(raises=ValueError, reason="Not a valid resolution")
    def test_fail_get_tag_for_res(self):
        """Test the fail case for the get_tag_for_res function for invalid resolution."""

        utilsResolutions.get_tag_for_res(
            res="25V",
        )

    def test_get_angular_resolution_from_physical_resolution(self):
        """Test the get_angular_resolution_from_physical_resolution function."""

        result = ((100 * u.pc) / (10 * u.Mpc) * u.rad).to(u.arcsec).value

        ang_res = utilsResolutions.get_angular_resolution_from_physical_resolution(
            res="100pc",
            distance="10Mpc",
        )
        assert np.isclose(ang_res, result)

    @pytest.mark.xfail(raises=ValueError, reason="Not a valid resolution")
    def test_get_angular_resolution_from_physical_resolution_invalid_res(self):
        """Test the fail case for get_angular_resolution_from_physical_resolution function with non-valid resolution."""

        utilsResolutions.get_angular_resolution_from_physical_resolution(
            res="5.0arcsec",
            distance="10Mpc",
        )

    @pytest.mark.xfail(raises=ValueError, reason="Not a valid resolution")
    def test_get_angular_resolution_from_physical_resolution_invalid_distance(self):
        """Test the fail case for get_angular_resolution_from_physical_resolution function with non-valid distance."""

        utilsResolutions.get_angular_resolution_from_physical_resolution(
            res="100pc",
            distance="10rad",
        )

    def test_get_angular_resolution_for_res_angular_resolution(self):
        """Test the get_angular_resolution_for_res function for passing an angular resolution."""

        ang_res = utilsResolutions.get_angular_resolution_for_res(
            res="5arcsec",
        )

        assert ang_res == 5
        
    def test_get_angular_resolution_for_res_physical_resolution(self):
        """Test the get_angular_resolution_for_res function for passing a physical resolution."""

        result = ((100 * u.pc) / (10 * u.Mpc) * u.rad).to(u.arcsec).value

        ang_res = utilsResolutions.get_angular_resolution_for_res(
            res="100pc",
            distance="10Mpc",
        )

        assert np.isclose(ang_res, result)

    @pytest.mark.xfail(raises=ValueError, reason="No distance passed")
    def test_get_angular_resolution_for_res_invalid_distance(self):
        """Test the get_angular_resolution_for_res function for not passing a distance."""

        utilsResolutions.get_angular_resolution_for_res(
            res="100pc",
        )
