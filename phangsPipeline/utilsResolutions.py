import re

import astropy.units as u

regex_psep = re.compile(
    pattern=r"([0-9eE.+-]+)(p)([0-9eE.+-]+)(.*)",
    flags=re.IGNORECASE,
)


def is_astropy_unit_equivalent(
    res: str | float | int | u.Quantity,
    unit: u.Unit,
):
    """Check if a string is a valid generic unit equivalent.

    Args:
        res (str|float|int|u.Quantity): Input resolution.
            Can be a string with units like '5.0arcsec', '5.0pc', or a float/int which is assumed to be
            in the specified unit.
        unit (u.Unit): The unit to check against.

    Returns:
        bool, float | None: A tuple where the first element is True if the input is equivalent to
            the specified unit, and False otherwise. The second element is the value converted to
            the specified unit if equivalent, or None otherwise.
    """

    # Modify the string so astropy units is happy with it
    if isinstance(res, str):
        # Strip out whitespace
        res = res.strip()

        # If the input is like "5p00", we convert it to "5.00"
        if regex_psep.match(res):
            res = regex_psep.sub(
                repl=r"\1.\3\4",
                string=res,
            )

        # Convert to astropy unit
        res = u.Quantity(res)

    # If float or int, assume the specified unit
    elif isinstance(res, float) or isinstance(res, int):
        res = res * unit

    # If already a quantity, do nothing
    elif isinstance(res, u.Quantity):
        pass

    else:
        raise TypeError("Input must be a string, float, int, or u.Quantity.")

    if res.unit.is_equivalent(unit):
        is_unit_equivalent = True
        res_value = res.to(unit).value

    else:
        is_unit_equivalent = False
        res_value = None

    return is_unit_equivalent, res_value


def is_angular_resolution(
    res: str | float | int | u.Quantity,
    return_value: bool = False,
):
    """Check if a string is a valid angular resolution.

    Args:
        res (str|float|int|u.Quantity): Input resolution.
            Can be a string with units like '5.0arcsec', '5.0arcmin', '5.0deg', or a float/int value in arcsec.
        return_value (bool): If True, return the value in arcsec
            Default is False.
    """

    is_ang_res, ang_res_arcsec = is_astropy_unit_equivalent(
        res=res,
        unit=u.arcsec,
    )

    if return_value:
        return is_ang_res, ang_res_arcsec
    else:
        return is_ang_res


def is_physical_resolution(
    res: str | float | int | u.Quantity,
    return_value: bool = False,
):
    """Check if a string is a physical resolution.

    Args:
        res (str|float|int|u.Quantity): Input resolution.
            Can be a string with units like '5.0pc', '5.0kpc', '5.0Mpc', or a float/int value in parsec.
        return_value (bool): If True, return the value in parsec
            Default is False.
    """

    is_phys_res, phys_res_pc = is_astropy_unit_equivalent(
        res=res,
        unit=u.pc,
    )

    if return_value:
        return is_phys_res, phys_res_pc
    else:
        return is_phys_res


def is_distance(
    distance: str | float | int | u.Quantity,
    return_value: bool = False,
):
    """Check if a string is a distance ending with parsec units.

    Args:
        distance (str|float|int|u.Quantity): Input distance.
            Can be a string with units like '5.0pc', '5.0kpc', '5.0Mpc', or a float/int value in Mpc.
        return_value (bool): If True, return the value in Mpc
            Default is False.
    """

    is_dist, dist_mpc = is_astropy_unit_equivalent(
        res=distance,
        unit=u.Mpc,
    )

    if return_value:
        return is_dist, dist_mpc
    else:
        return is_dist


def get_tag_for_angular_resolution(
    res: str | float | int | u.Quantity,
    sig_figs: int = 2,
):
    """Input an angular resolution string or value, output a formatted string tag to be used in filenames.

    Args:
        res (str|float|int|u.Quantity): Input angular resolution.
        sig_figs (int): Number of significant figures to use in the output.
    """

    is_ang_res, ang_res_value = is_angular_resolution(
        res=res,
        return_value=True,
    )
    if not is_ang_res:
        raise ValueError(
            f"Input resolution string {res} it not a valid angular resolution"
        )

    res_str = f"{ang_res_value:.{sig_figs}f}".replace(".", "p")

    return res_str


def get_tag_for_physical_resolution(
    res: str | float | int | u.Quantity,
    sig_figs: int = 0,
):
    """Input a physical resolution string or value, output a formatted string tag to be used in filenames.

    Args:
        res (str|float|int|u.Quantity): Input physical resolution.
        sig_figs (int): Number of significant figures to use in the output.
    """

    is_phys_res, phys_res_value = is_physical_resolution(
        res=res,
        return_value=True,
    )
    if not is_phys_res:
        raise ValueError(
            f"Input resolution string {res} is not a valid physical resolution"
        )

    res_str = f"{phys_res_value:.{sig_figs}f}pc".replace(".", "p")

    return res_str


def get_tag_for_res(
    res: str | float | int | u.Quantity,
    angular_res_sig_figs: int = 2,
    physical_res_sig_figs: int = 0,
):
    """Return a tag string to be used in filenames given a resolution string

    This is the general task that distinguishes between angular and physical resolutions.
    The input resolution can be a string like either '5.0arcsec' or '80pc', an u.Quantity,
    or a float/int. If a float/int, this is assumed to be in arcsec.

    By default, the returned tag string is formatted like '5p00' for an angular resolution, or
    like '80pc' for a physical resolution.

    Args:
        res (str|float|int|u.Quantity): Input resolution.
        angular_res_sig_figs (int): Number of significant figures to use in the output for angular resolutions.
            Defaults to 2.
        physical_res_sig_figs (int): Number of significant figures to use in the output for physical resolutions.
            Defaults to 0.
    """
    if is_angular_resolution(res=res):
        tag = get_tag_for_angular_resolution(
            res=res,
            sig_figs=angular_res_sig_figs,
        )
    elif is_physical_resolution(res=res):
        tag = get_tag_for_physical_resolution(
            res=res,
            sig_figs=physical_res_sig_figs,
        )
    else:
        raise ValueError(
            f"Input resolution string {res} is not a valid physical or angular resolution"
        )

    return tag


def get_angular_resolution_from_physical_resolution(
    res: str | float | int | u.Quantity,
    distance: str | float | int | u.Quantity,
):
    """Return the angular resolution in arcsec, given a physical resolution and a distance.

    Args:
        res (str|float|int|u.Quantity): Input physical resolution.
        Can be a string with units like '5.0pc', '5.0kpc', '5.0Mpc', or a float/int value in parsec.
        distance (str|float|int|u.Quantity): Input distance.
            Can be a string with units like '5.0Mpc', '5.0kpc', '5.0pc', or a float/int value in Mpc.
    """

    is_phys_res, phys_res_pc = is_physical_resolution(
        res,
        return_value=True,
    )
    if not is_phys_res:
        raise ValueError(
            f"Input resolution string {res} is not a valid physical resolution"
        )

    is_dist, dist_mpc = is_distance(
        distance,
        return_value=True,
    )
    if not is_dist:
        raise ValueError(f"Input distance string {distance} is not a valid distance")

    ang_res_arcsec = (phys_res_pc / dist_mpc / 1e6 * u.rad).to(u.arcsec).value

    return ang_res_arcsec


def get_angular_resolution_for_res(
    res: str | float | int | u.Quantity,
    distance: str | float | int | u.Quantity | None = None,
):
    """Return an angular resolution value in units of arcsec

    There are two paths here: The first is passing an angular resolution, in which case that will be returned.
    The second is passing a physical resolution, in which case a distance must also be provided, and the angular
    resolution will be calculated from the physical resolution and distance.

    Args:
        res (str|float|int|u.Quantity): Input resolution, either angular or physical.
        distance (str|float|int|u.Quantity|None): Input distance, required if res is a physical resolution.
            Defaults to None

    """
    is_ang_res, ang_res_arcsec = is_angular_resolution(res, return_value=True)

    # If we have an angular resolution, do nothing
    if is_ang_res:
        pass

    # Otherwise, calculate from physical distance and physical resolution
    else:

        if distance is None:
            raise ValueError(
                "Distance must be provided if res is a physical resolution"
            )

        ang_res_arcsec = get_angular_resolution_from_physical_resolution(res, distance)

    return ang_res_arcsec
