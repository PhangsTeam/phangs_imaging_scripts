import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from phangsPipeline import utilsLines


class TestUtilsLines:
    """Suite of tests for utilsLines"""

    def test_all_lines_in_families_in_line_list(self):
        """Test that all the lines present in line families are also in the line list"""

        # Get all the lines in the line families
        lines_in_families = []
        for family in utilsLines.line_families.values():
            lines_in_families.extend(family)

        # Check that all the lines in the families are also in the line list
        for line in lines_in_families:
            assert line in utilsLines.line_list, f"{line} is in a family but not in the line list"

    def test_no_repeat_frequencies_in_line_list(self):
        """Test that there are no repeat frequencies in the line list"""

        # Get all the frequencies in the line list
        frequencies = np.array([freq for freq in utilsLines.line_list.values()])

        # Check that there are no repeat frequencies
        assert len(frequencies) == len(np.unique(frequencies)), (
            "There are repeat frequencies in the line list"
        )

    def test_get_line_name_exact_match(self):
        """Test that get_line_name returns the correct line name for an exact match"""

        # Test with an exact match
        line_name = "co21"
        frequency = 230.53800
        matched_line_name, matched_frequency = utilsLines.get_line_name_and_frequency(line_name)

        assert matched_line_name is not None, f"{line_name} not found in line list"
        assert matched_frequency == frequency, f"{line_name} frequency does not match"

    def test_get_line_name_uppercase_match(self):
        """Test that get_line_name returns the correct line name for an uppercase match"""

        # Test with an uppercase match
        line_name = "CO21"
        frequency = 230.53800
        matched_line_name, matched_frequency = utilsLines.get_line_name_and_frequency(line_name)

        assert matched_line_name is not None, f"{line_name} not found in line list"
        assert matched_frequency == frequency, f"{line_name} frequency does not match"

    def test_get_line_name_extra_symbol_match(self):
        """Test that get_line_name returns the correct line name for a match with extra symbols"""

        # Test with an extra symbol match
        line_name = "co21+"
        frequency = 230.53800
        matched_line_name, matched_frequency = utilsLines.get_line_name_and_frequency(line_name)

        assert matched_line_name is not None, f"{line_name} not found in line list"
        assert matched_frequency == frequency, f"{line_name} frequency does not match"

    @pytest.mark.xfail(raises=ValueError, reason="Line name not found")
    def test_get_line_name_no_match(self):
        """Test that get_line_name fails for a line name not in the list"""

        # Test with a line name not in the list
        line_name = "notaline"
        utilsLines.get_line_name_and_frequency(line_name)

    @pytest.mark.xfail(raises=TypeError, reason="Line name not string")
    def test_get_line_name_non_string(self):
        """Test that get_line_name fails for a line name that's not a string"""

        # Test with a line name in the wrong type
        line_name = 123
        utilsLines.get_line_name_and_frequency(line_name)

    @pytest.mark.xfail(raises=TypeError, reason="Line frequency not float")
    def test_get_line_name_non_float_freq(self):
        """Test that get_line_name fails for a line name with a non-float frequency"""

        # Test with a line name in the wrong type
        line_name = "failure_case"
        utilsLines.get_line_name_and_frequency(line_name)

    def test_get_line_names_in_line_family(self):
        """Test that get_line_family returns the correct family for a line"""

        # Test with a line in a family
        line_family = "hi"
        expected_line_names = ["hi21cm"]
        line_names = utilsLines.get_line_names_in_line_family(line_family)

        assert line_names == expected_line_names, f"{line_family} line names do not match"

    @pytest.mark.xfail(raises=ValueError, reason="Line family does not exist")
    def test_get_line_names_in_line_family_no_match(self):
        """Test that get_line_family fails for a non-existent family"""

        # Test with a line in a family
        line_family = "failure_case"
        utilsLines.get_line_names_in_line_family(line_family)

    def test_is_line_family(self):
        """Test that is_line_family returns the correct boolean for a line family"""

        # Test with a valid line family
        line_family = "hi"
        is_family = utilsLines.is_line_family(line_family)

        assert is_family is True, f"{line_family} is not recognized as a line family"

    def test_is_line_family_nonexistent(self):
        """Test that is_line_family returns False for a non-existent line family"""

        # Test with a non-existent line family
        line_family = "failure_case"
        is_family = utilsLines.is_line_family(line_family)

        assert is_family is False, f"{line_family} is incorrectly recognized as a line family"

    def test_get_ghz_range_for_line_vlow_vhigh(self):
        """Test that get_ghz_range_for_line returns the correct frequency range for a line with vlow and vhigh"""

        # Test with a line and vlow/vhigh
        line = "hi21cm"
        vlow_kms = -100.0
        vhigh_kms = 100.0
        rest_freq = 1.420405751
        sol_kms = const.c.to(u.km / u.s)

        expected_low = rest_freq - rest_freq * (vhigh_kms / sol_kms.value)
        expected_high = rest_freq - rest_freq * (vlow_kms / sol_kms.value)
        low, high = utilsLines.get_ghz_range_for_line(
            line=line,
            vlow_kms=vlow_kms,
            vhigh_kms=vhigh_kms,
        )

        assert np.isclose(low, expected_low), f"{line} low frequency does not match"
        assert np.isclose(high, expected_high), f"{line} high frequency does not match"

    def test_get_ghz_range_for_line_vsys_vwidth(self):
        """Test that get_ghz_range_for_line returns the correct frequency range for a line with vsys and vwidth"""

        # Test with a line and vsys/vwidth
        line = "hi21cm"
        vsys_kms = 100.0
        vwidth_kms = 50.0
        rest_freq = 1.420405751
        sol_kms = const.c.to(u.km / u.s)

        expected_low = rest_freq - rest_freq * ((vsys_kms + vwidth_kms / 2) / sol_kms.value)
        expected_high = rest_freq - rest_freq * ((vsys_kms - vwidth_kms / 2) / sol_kms.value)
        low, high = utilsLines.get_ghz_range_for_line(
            line=line,
            vsys_kms=vsys_kms,
            vwidth_kms=vwidth_kms,
        )

        assert np.isclose(low, expected_low), f"{line} low frequency does not match"
        assert np.isclose(high, expected_high), f"{line} high frequency does not match"

    def test_get_ghz_range_for_line_all_info(self):
        """Test that get_ghz_range_for_line falls to vlow/vhigh when both vsys/vwidth and vlow/vhigh are provided"""

        # Test with a line and vsys/vwidth
        line = "hi21cm"
        vsys_kms = 100.0
        vwidth_kms = 50.0
        vlow_kms = -100.0
        vhigh_kms = 100.0
        rest_freq = 1.420405751
        sol_kms = const.c.to(u.km / u.s)

        expected_low = rest_freq - rest_freq * (vhigh_kms / sol_kms.value)
        expected_high = rest_freq - rest_freq * (vlow_kms / sol_kms.value)
        low, high = utilsLines.get_ghz_range_for_line(
            line=line,
            vsys_kms=vsys_kms,
            vwidth_kms=vwidth_kms,
            vlow_kms=vlow_kms,
            vhigh_kms=vhigh_kms,
        )

        assert np.isclose(low, expected_low), f"{line} low frequency does not match"
        assert np.isclose(high, expected_high), f"{line} high frequency does not match"

    def test_get_ghz_range_for_line_vsys_vwidth_rest_freq(self):
        """Test that get_ghz_range_for_line returns the correct frequency range when rest frequency is provided"""

        # Test with a line and vsys/vwidth
        line = "hi21cm"
        vsys_kms = 100.0
        vwidth_kms = 50.0
        rest_freq = 1.56
        sol_kms = const.c.to(u.km / u.s)

        expected_low = rest_freq - rest_freq * ((vsys_kms + vwidth_kms / 2) / sol_kms.value)
        expected_high = rest_freq - rest_freq * ((vsys_kms - vwidth_kms / 2) / sol_kms.value)
        low, high = utilsLines.get_ghz_range_for_line(
            line=line,
            vsys_kms=vsys_kms,
            vwidth_kms=vwidth_kms,
            restfreq_ghz=rest_freq,
        )

        assert np.isclose(low, expected_low), f"{line} low frequency does not match"
        assert np.isclose(high, expected_high), f"{line} high frequency does not match"

    @pytest.mark.xfail(raises=ValueError, reason="No information provided")
    def test_get_ghz_range_for_line_no_info(self):
        """Test that get_ghz_range_for_line fails when no information is provided"""

        # Test with a line and no additional information
        line = "hi21cm"

        utilsLines.get_ghz_range_for_line(line=line)

    def test_get_ghz_range_for_list_of_lines(self):
        """Test that get_ghz_range_for_line returns the correct frequency range for a list of lines"""

        # Test with a list of lines
        lines = ["hi"]
        vlow_kms = -100.0
        vhigh_kms = 100.0
        rest_freq = 1.420405751
        sol_kms = const.c.to(u.km / u.s)

        expected_low = rest_freq - rest_freq * (vhigh_kms / sol_kms.value)
        expected_high = rest_freq - rest_freq * (vlow_kms / sol_kms.value)
        expected = [expected_low, expected_high]

        res = utilsLines.get_ghz_range_for_list(
            lines=lines,
            vlow_kms=vlow_kms,
            vhigh_kms=vhigh_kms,
        )

        assert np.isclose(res, expected).all(), "Frequencies do not match"

    def test_get_ghz_range_for_list_of_lines_empty_list(self):
        """Test that get_ghz_range_for_line returns an empty list when an empty list is provided"""

        # Test with an empty list
        expected = []

        res = utilsLines.get_ghz_range_for_list()

        assert res == expected, "Non-empty list returned for empty input"

    def test_get_ghz_range_for_list_of_lines_non_list(self):
        """Test that get_ghz_range_for_line correctly handles string input"""

        lines = "hi"
        vlow_kms = -100.0
        vhigh_kms = 100.0
        rest_freq = 1.420405751
        sol_kms = const.c.to(u.km / u.s)

        expected_low = rest_freq - rest_freq * (vhigh_kms / sol_kms.value)
        expected_high = rest_freq - rest_freq * (vlow_kms / sol_kms.value)
        expected = [expected_low, expected_high]

        res = utilsLines.get_ghz_range_for_list(
            lines=lines,
            vlow_kms=vlow_kms,
            vhigh_kms=vhigh_kms,
        )

        assert np.isclose(res, expected).all(), "Frequencies do not match"

    def test_get_ghz_range_for_list_of_lines_single_line(self):
        """Test that get_ghz_range_for_line correctly handles a single line"""

        lines = "hi21cm"
        vlow_kms = -100.0
        vhigh_kms = 100.0
        rest_freq = 1.420405751
        sol_kms = const.c.to(u.km / u.s)

        expected_low = rest_freq - rest_freq * (vhigh_kms / sol_kms.value)
        expected_high = rest_freq - rest_freq * (vlow_kms / sol_kms.value)
        expected = [expected_low, expected_high]

        res = utilsLines.get_ghz_range_for_list(
            lines=lines,
            vlow_kms=vlow_kms,
            vhigh_kms=vhigh_kms,
        )

        assert np.isclose(res, expected).all(), "Frequencies do not match"
