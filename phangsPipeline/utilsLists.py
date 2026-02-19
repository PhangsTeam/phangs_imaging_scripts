"""
General purpose utilities.
"""


def select_from_list(
    master_list, 
    first=None,
    last=None,
    skip=[],
    only=[],
    loose=True,
    ):
    """
    Select only part of a list.
    """

    sorted_list = sorted(master_list,key=lambda s: s.lower())

    sub_list = []

    if first is not None:
        before_first = True
    else:
        before_first = False

    if last is not None:
        after_last = False
    else:
        after_last = False

    for element in sorted_list:

        if first is not None:
            if loose:
                if element.lower() >= first.lower():
                    before_first = False
            else:
                if element.lower() == first.lower():
                    before_first = False

        if last is not None:
            if loose:
                if element.lower() > last.lower():
                    after_last = True

        if before_first:
            continue

        if after_last:
            continue

        if skip is not None:
            if len(skip) > 0:
                match = False
                if loose:
                    for this_skip in skip:
                        if element.lower() == this_skip.lower():
                            match = True                        
                else:
                    if element in skip:
                        match = True
                if match:
                    continue

        if only is not None:
            if len(only) > 0:
                match = False
                if loose:
                    for this_only in only:
                        if element.lower() == this_only.lower():
                            match = True                        
                else:
                    if element in only:
                        match = True
                if not match:
                    continue
            
        sub_list.append(element)

        if last is not None:
            if not loose:
                if element.lower() == last.lower():
                    after_last = True

    return(sub_list)
    
def lohi_vecs_to_pairs(lovec, hivec):
    """
    Convert a low and hi
    """

    # Add some error checking

    pairs = []

    for ii in range(lovec):
        pairs.append((lovec[ii], hivec[ii]))

    return(pairs)

def merge_pairs(pairs):
    """
    Accepted a matched list lo/hi pairs and returns the list of pairs
    merged until convergence.
    """
    
    # Add some error checking

    # Sort on the x coordinate
    pairs = sorted(pairs)

    # Start the list of new pairs
    new_pairs = [pairs[0]]
    i = 1

    # Iterate
    while i <= len(pairs)-1:

        # Pick the last new pair for comparison
        x1,y1 = new_pairs[-1]

        # Compare to the current pair
        x2,y2 = pairs[i]

        # Since we are sorted by x, we know x2>=x1. If y2 is less than
        # y1, the current window already spans the range.
        if y2 <= y1:
            i += 1 # included
            continue

        # If x2 is less than y1, the comparison window begins inside
        # the end point, then we might extend the window. Span to the
        # maximum of y1, y2.
        if x2 <= y1:
            new_pairs[-1] = (x1,max(y1,y2)) # grow
            continue
        else:
            # Otherwise, the new window begins outside the previous
            # window. In that case, we move to a new part of the
            # sequence. Now use that pair for comparison instead.
            new_pairs.append((x2,y2)) # new
            i += 1
            continue

    return(new_pairs)
