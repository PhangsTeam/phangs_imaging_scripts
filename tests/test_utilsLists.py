import pytest

from phangsPipeline.utilsLists import select_from_list, merge_pairs

class TestSelectFromList:
    """Suite of tests for utilsLists.select_from_list"""

    test_list = [
        "ngc0000",
        "ngc0123",
        "ngc2345",
        "ngc3456",
        "ngc4567",
    ]

    @pytest.mark.xfail(raises=TypeError, reason="Not a list")
    def test_pass_non_list(self):
        """Test passing a non-list, which should fail"""

        test_list = "not a list"

        select_from_list(
            test_list,
        )
        

    def test_default_selection(self):
        """Test the default selection"""

        filtered_test_list = select_from_list(
            self.test_list,
        )

        result = self.test_list

        assert filtered_test_list == result

    def test_first_selection(self):
        """Test passing a first element"""

        first = "ngc0123"

        filtered_test_list = select_from_list(
            self.test_list,
            first=first,
        )

        result = [
            "ngc0123",
            "ngc2345",
            "ngc3456",
            "ngc4567",
        ]

        assert filtered_test_list == result

    def test_last_selection(self):
        """Test passing a last element"""

        last = "ngc3456"

        filtered_test_list = select_from_list(
            self.test_list,
            last=last,
        )

        result = [
            "ngc0000",
            "ngc0123",
            "ngc2345",
            "ngc3456",
        ]

        assert filtered_test_list == result

    def test_first_last_selection(self):
        """Test passing a first element and last element"""

        first = "ngc0123"
        last = "ngc3456"

        filtered_test_list = select_from_list(
            self.test_list,
            first=first,
            last=last,
        )

        result = [
            "ngc0123",
            "ngc2345",
            "ngc3456",
        ]

        assert filtered_test_list == result

    def test_single_skip(self):
        """Test passing a single skip element"""

        skip = "ngc0123"

        filtered_test_list = select_from_list(
            self.test_list,
            skip=skip,
        )

        result = [
            "ngc0000",
            "ngc2345",
            "ngc3456",
            "ngc4567",
        ]

        assert filtered_test_list == result

    def test_multiple_skip(self):
        """Test passing multiple skip elements"""

        skip = [
            "ngc0123",
            "ngc2345",
        ]

        filtered_test_list = select_from_list(
            self.test_list,
            skip=skip,
        )
        
        result = [
            "ngc0000",
            "ngc3456",
            "ngc4567",
        ]

        assert filtered_test_list == result

    def test_single_only(self):
        """Test passing a single only element"""

        only = "ngc0123"

        filtered_test_list = select_from_list(
            self.test_list,
            only=only,
        )

        result = [
            "ngc0123"
        ]

        assert filtered_test_list == result

    def test_multiple_only(self):
        """Test passing multiple only elements"""

        only = [
            "ngc0123",
            "ngc2345",
        ]

        filtered_test_list = select_from_list(
            self.test_list,
            only=only,
        )

        result = [
            "ngc0123",
            "ngc2345",
        ]

        assert filtered_test_list == result

    def test_loose_skip(self):
        """Test loose works as expected for skips"""

        skip = ["NGC0123"]

        filtered_test_list = select_from_list(
            self.test_list,
            skip=skip,
            loose=False,
        )

        result = self.test_list

        assert filtered_test_list == result

    def test_loose_lowercase_skip(self):
        """Test loose works as expected for lowercase skips"""

        skip = ["ngc0123"]

        filtered_test_list = select_from_list(
            self.test_list,
            skip=skip,
            loose=False,
        )

        result = [
            "ngc0000",
            "ngc2345",
            "ngc3456",
            "ngc4567",
        ]

        assert filtered_test_list == result

    def test_loose_only(self):
        """Test loose works as expected for only"""

        only = ["NGC0123"]

        result = []

        filtered_test_list = select_from_list(
            self.test_list,
            only=only,
            loose=False,
        )

        assert filtered_test_list == result

    def test_loose_lowercase_only(self):
        """Test loose works as expected for lowercase only"""

        only = ["ngc0123"]

        filtered_test_list = select_from_list(
            self.test_list,
            only=only,
            loose=False,
        )

        result = [
            "ngc0123"
        ]

        assert filtered_test_list == result


class TestMergePairs:
    """Suite of tests for utilsLists.merge_pairs"""

    @pytest.mark.xfail(raises=TypeError, reason="Not a list")
    def test_pass_non_list(self):
        """Test passing a non-list, which should fail"""
        
        test_pairs = "not a list"

        merge_pairs(test_pairs)

    def test_sorted_merge(self):
        """Test values are merged for an already sorted list"""

        test_pairs = [
            [0, 1],
            [2, 3],
            [1, 2],
        ]

        result = [
            (0, 3),
        ]

        merged_pairs = merge_pairs(test_pairs)

        assert merged_pairs == result

    def test_unsorted_merge(self):
        """Test values are merged for an unsorted list"""

        test_pairs = [
            [0, 1],
            [1, 2],
            [2, 3],
        ]

        result = [
            (0, 3),
        ]

        merged_pairs = merge_pairs(test_pairs)

        assert merged_pairs == result

    def test_sorted_non_merge(self):
        """Test values are not merged for an already sorted list"""

        test_pairs = [
            [0, 1],
            [2, 3],
            [4, 5],
        ]

        result = [
            (0, 1),
            (2, 3),
            (4, 5),
        ]

        merged_pairs = merge_pairs(test_pairs)

        assert merged_pairs == result

    def test_unsorted_non_merge(self):
        """Test values are not merged for an unsorted list"""

        test_pairs = [
            [2, 3],
            [0, 1],
            [4, 5],
        ]

        result = [
            (0, 1),
            (2, 3),
            (4, 5),
        ]

        merged_pairs = merge_pairs(test_pairs)

        assert merged_pairs == result
