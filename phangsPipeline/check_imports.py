def is_casa_installed():
    """Check if CASA is installed."""

    casa_enabled = False
    try:
        import casatasks
        import casatools
        casa_enabled = True
    except (ImportError, ModuleNotFoundError):
        pass

    return casa_enabled

def is_spectral_cube_installed():
    """Check if spectral-cube is installed."""

    spectral_cube_enabled = False
    try:
        import spectral_cube
        spectral_cube_enabled = True
    except (ImportError, ModuleNotFoundError):
        pass

    return spectral_cube_enabled

