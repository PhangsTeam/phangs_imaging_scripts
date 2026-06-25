def select_from_list(
    master_list: list,
    first: str | None = None,
    last: str | None = None,
    skip: str | list | None = None,
    only: str | list | None = None,
    loose: bool = True,
) -> list:
    """
    Select only part of a list.

    Args:
        master_list (list): List of elements
        first (str): First element to select in list. Defaults to None,
            which will run from start of list
        last (str): Last element to select in list. Defaults to None,
            which will run to end of list
        skip (str, list): List of elements to skip. Defaults
            to None, which will not filter
        only (str, list): List of only elements to include. Defaults
            to None, which will not filter
        loose (bool): If False, for skip/only, will match on case.
            Otherwise, will be case agnostic. Defaults to True

    Returns:
        list: Sorted list of filtered elements
    """

    if not isinstance(master_list, list):
        raise TypeError("master_list must be a list")

    # Start by sorting the list
    sorted_list = sorted(
        master_list,
        key=lambda s: s.lower(),
    )

    if first is not None:
        before_first = True
    else:
        before_first = False

    if last is not None:
        after_last = False
    else:
        after_last = False

    if skip is None:
        skip = []
    if isinstance(skip, str):
        skip = [skip]

    if only is None:
        only = []
    if isinstance(only, str):
        only = [only]

    if loose:
        skip = [s.lower() for s in skip]
        only = [s.lower() for s in only]

    sub_list = []

    for element in sorted_list:

        # Check if we're before or after first/last element
        if first is not None:
            if element.lower() >= first.lower():
                before_first = False

        if last is not None:
            if element.lower() > last.lower():
                after_last = True

        # If we're before first or after last, skip
        if before_first or after_last:
            continue

        # Check skip
        if len(skip) > 0:
            match = False

            # If we're loose, we've already lowercase-d skip
            if loose:
                if element.lower() in skip:
                    match = True

            # Else, match exactly
            else:
                if element in skip:
                    match = True

            # If we've found a match, skip
            if match:
                continue

        # Check only
        if len(only) > 0:
            match = False

            # If we're loose, we've already lowercase-d only
            if loose:
                if element.lower() in only:
                    match = True

            # Else, match exactly
            else:
                if element in only:
                    match = True

            # If we haven't found a match, skip
            if not match:
                continue

        sub_list.append(element)

    return sub_list


def merge_pairs(
    pairs: list,
) -> list:
    """
    Accept a matched list lo/hi pairs and returns the list of pairs
    merged until convergence.

    Args:
        pairs (list): list of pairs

    Returns:
        list: list of pairs, as tuples
    """

    if not isinstance(pairs, list):
        raise TypeError("pairs must be a list")

    # Sort on the x coordinate
    pairs = sorted(pairs)

    # Start the list of new pairs
    new_pairs = [tuple(pairs[0])]
    i = 1

    # Iterate
    while i <= len(pairs) - 1:

        # Pick the last new pair for comparison
        x1, y1 = new_pairs[-1]

        # Compare to the current pair
        x2, y2 = pairs[i]

        # Since we are sorted by x, we know x2>=x1. If y2 is less than
        # y1, the current window already spans the range.
        if y2 <= y1:
            i += 1  # included
            continue

        # If x2 is less than y1, the comparison window begins inside
        # the end point, then we might extend the window. Span to the
        # maximum of y1, y2.
        if x2 <= y1:
            new_pairs[-1] = (x1, max(y1, y2))  # grow
            continue
        else:
            # Otherwise, the new window begins outside the previous
            # window. In that case, we move to a new part of the
            # sequence. Now use that pair for comparison instead.
            new_pairs.append((x2, y2))  # new
            i += 1
            continue

    return new_pairs
