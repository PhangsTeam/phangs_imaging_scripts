"""utilsTestImagingPlots

Utility functions for creating diagnostic plots from test imaging results.
Creates 1D radial PSF profile galleries, 2D PSF image galleries, and
2D dirty image galleries for comparing different imaging parameters.

These functions work with the results list produced by
TestImagingHandler.get_results().

This module requires matplotlib and numpy. It can read CASA images
(when running inside CASA) or FITS files (via astropy).

Example:
    from phangsPipeline import handlerTestImaging as tih
    from phangsPipeline import utilsTestImagingPlots as tip

    # After running test imaging:
    results = this_tih.get_results()

    # Create galleries
    tip.make_psf_radial_profile_gallery(
        results, imaging_dir='/path/to/imaging/',
        output_file='psf_profiles.png')

    tip.make_psf_image_gallery(
        results, imaging_dir='/path/to/imaging/',
        output_file='psf_gallery.png')

    tip.make_image_gallery(
        results, imaging_dir='/path/to/imaging/',
        output_file='image_gallery.png')
"""

import os
import logging
import math
import base64
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available, plotting functions will not work")

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

from .check_imports import is_casa_installed
casa_enabled = is_casa_installed()

if casa_enabled:
    import analysisUtils as au
    from . import casaStuff

from . import utilsFilenames



import os

from scipy.interpolate import UnivariateSpline


fwhm_factor = 2 * np.sqrt(2 * np.log(2))  # FWHM to sigma conversion factor



# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------


def _read_image_data(image_file):
    """Read 2D image data from a CASA image or FITS file.

    Tries CASA image tools first (if available), then falls back to
    astropy FITS reading. Squeezes any degenerate axes to return a
    2D array.

    Parameters
    ----------
    image_file : str
        Path to a CASA image directory or a FITS file.

    Returns
    -------
    data_2d : numpy.ndarray or None
        2D image data array (y, x ordering for matplotlib).
    pixel_scale_arcsec : float or None
        Pixel scale in arcseconds.
    """
    if image_file is None:
        return None, None

    # Try CASA image
    if casa_enabled and os.path.isdir(image_file):
        try:
            myia = au.createCasaTool(casaStuff.iatool)
            myia.open(image_file)
            data = myia.getchunk()
            csys = myia.coordsys()
            increments = csys.increment()['numeric']
            # RA increment is in radians; take absolute value
            pixel_scale_arcsec = abs(increments[0]) * 206264.806
            myia.close()
            # Squeeze degenerate axes (Stokes, Freq) to get 2D
            data_2d = np.squeeze(data)
            if data_2d.ndim != 2:
                logger.warning(f"Image has {data_2d.ndim} dimensions after squeeze, "
                               f"expected 2: {image_file}")
                return None, None
            # CASA images are (x, y) order; transpose for matplotlib (y, x)
            data_2d = data_2d.T
            return data_2d, pixel_scale_arcsec
        except Exception as e:
            logger.warning(f"Failed to read CASA image {image_file}: {e}")

    # Try FITS file
    fits_file = image_file
    if not fits_file.endswith('.fits'):
        fits_file = image_file + '.fits'
    if HAS_ASTROPY and os.path.isfile(fits_file):
        try:
            hdu = fits.open(fits_file)[0]
            data_2d = np.squeeze(hdu.data)
            if data_2d.ndim != 2:
                logger.warning(f"FITS image has {data_2d.ndim} dimensions after squeeze: "
                               f"{fits_file}")
                return None, None
            # Pixel scale from CDELT2 (Dec axis), in degrees -> arcsec
            pixel_scale_arcsec = abs(hdu.header.get('CDELT2', 1.0)) * 3600.0
            return data_2d, pixel_scale_arcsec
        except Exception as e:
            logger.warning(f"Failed to read FITS file {fits_file}: {e}")

    logger.warning(f"Could not read image: {image_file}")
    return None, None


def _compute_radial_profile(data_2d, pixel_scale_arcsec, center=None,
                            max_radius_arcsec=None, bin_width_arcsec=None):
    """Compute an azimuthally averaged radial profile of a 2D image.

    Parameters
    ----------
    data_2d : numpy.ndarray
        2D image array.
    pixel_scale_arcsec : float
        Pixel scale in arcseconds.
    center : tuple of int, optional
        (y, x) center pixel. If None, uses the peak pixel.
    max_radius_arcsec : float, optional
        Maximum radius to compute the profile out to. If None, extends
        to the shorter image half-dimension.
    bin_width_arcsec : float, optional
        Radial bin width in arcseconds. The actual width used is
        ``max(bin_width_arcsec, pixel_scale_arcsec)`` so that each bin
        is at least one pixel wide and never empty due to under-sampling.
        If None, defaults to one pixel (``pixel_scale_arcsec``).

    Returns
    -------
    radius_arcsec : numpy.ndarray
        Bin centers in arcseconds.
    profile : numpy.ndarray
        Azimuthally averaged profile values.
    """
    ny, nx = data_2d.shape

    if center is None:
        # Find peak pixel
        peak_idx = np.unravel_index(np.nanargmax(data_2d), data_2d.shape)
        cy, cx = peak_idx
    else:
        cy, cx = center

    # Build distance array in arcsec
    y_arr, x_arr = np.mgrid[0:ny, 0:nx]
    dist_pix = np.sqrt((x_arr - cx)**2 + (y_arr - cy)**2)
    dist_arcsec = dist_pix * pixel_scale_arcsec

    # Determine max radius
    if max_radius_arcsec is None:
        max_radius_pix = min(cx, cy, nx - cx - 1, ny - cy - 1)
        max_radius_arcsec = max_radius_pix * pixel_scale_arcsec

    # Enforce minimum bin width of one pixel
    if bin_width_arcsec is None:
        bin_width_arcsec = pixel_scale_arcsec
    else:
        bin_width_arcsec = max(bin_width_arcsec, pixel_scale_arcsec)

    n_bins = max(1, int(np.ceil(max_radius_arcsec / bin_width_arcsec)))

    # Create radial bins
    bin_edges = np.linspace(0, max_radius_arcsec, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    profile = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (dist_arcsec >= bin_edges[i]) & (dist_arcsec < bin_edges[i + 1])
        if np.any(mask):
            profile[i] = np.nanmean(data_2d[mask])
        else:
            profile[i] = np.nan

    return bin_centers, profile


def _build_image_root(result):
    """Reconstruct the image root filename from a result dictionary.

    Parameters
    ----------
    result : dict
        A result dictionary from TestImagingHandler.get_results().

    Returns
    -------
    str
        The image root path (without CASA extension like .psf or .image).
    """
    return utilsFilenames.get_cube_filename(
        target=result['target'],
        product=result['product'],
        config=result['config'],
        ext='test_' + result['test_name'],
        casa=True,
        casaext='',
    )


def _get_grid_layout(n_panels):
    """Compute a grid layout (nrows, ncols) for n_panels.

    Parameters
    ----------
    n_panels : int
        Number of panels.

    Returns
    -------
    nrows : int
    ncols : int
    """
    if n_panels <= 0:
        return 0, 0
    ncols = math.ceil(math.sqrt(n_panels))
    nrows = math.ceil(n_panels / ncols)
    return nrows, ncols


def _filter_results(results, target=None, config=None, product=None):
    """Filter results list by target, config, and/or product.

    Parameters
    ----------
    results : list of dict
        Results from TestImagingHandler.get_results().
    target : str, optional
        Filter to this target only.
    config : str, optional
        Filter to this config only.
    product : str, optional
        Filter to this product only.

    Returns
    -------
    list of dict
        Filtered results.
    """
    filtered = results
    if target is not None:
        filtered = [r for r in filtered if r.get('target') == target]
    if config is not None:
        filtered = [r for r in filtered if r.get('config') == config]
    if product is not None:
        filtered = [r for r in filtered if r.get('product') == product]
    return filtered


def _get_max_beam_arcsec(results):
    """Return the largest beam major axis from a list of results.

    Parameters
    ----------
    results : list of dict
        Filtered results list.

    Returns
    -------
    float or None
        Largest bmaj_arcsec, or None if no valid beams.
    """
    bmajs = [r['bmaj_arcsec'] for r in results
             if r.get('bmaj_arcsec') is not None]
    if len(bmajs) == 0:
        return None
    return max(bmajs)


def _paginated_output_file(output_file, page_idx, n_pages):
    """Generate an enumerated output filename for multi-page galleries.

    Parameters
    ----------
    output_file : str or None
        Base output filename.
    page_idx : int
        Zero-based page index.
    n_pages : int
        Total number of pages.

    Returns
    -------
    str or None
        For a single page, returns the original filename unchanged.
        For multiple pages, inserts ``_N`` before the extension
        (e.g. ``gallery.png`` -> ``gallery_1.png``, ``gallery_2.png``).
    """
    if output_file is None:
        return None
    if n_pages <= 1:
        return output_file
    base, ext = os.path.splitext(output_file)
    return f'{base}_{page_idx + 1}{ext}'


def _make_label(result):
    """Build a concise label string from a result dictionary.

    Parameters
    ----------
    result : dict
        A single result dictionary.

    Returns
    -------
    str
        Label string, e.g. "briggs_r0.5 (1.23x0.98 arcsec)".
    """
    label = result.get('test_name', '?')
    bmaj = result.get('bmaj_arcsec')
    bmin = result.get('bmin_arcsec')
    if bmaj is not None and bmin is not None:
        label += f' ({bmaj:.2f}x{bmin:.2f}")'
    return label


# ---------------------------------------------------------------
# Public gallery functions
# ---------------------------------------------------------------


def make_psf_radial_profile_gallery(
    results,
    imaging_dir=None,
    output_file=None,
    target=None,
    config=None,
    product=None,
    max_radius_arcsec=None,
    bin_width_arcsec=None,
    log_scale=False,
    figsize=None,
    dpi=150,
    title=None,
):
    """Create a gallery figure showing 1D radial PSF profiles.

    Produces a single figure with all PSF radial profiles overlaid for
    direct comparison. Each profile is azimuthally averaged. PSF data
    are expected to already be peak-normalized (as produced by tclean).
    The default x-axis range is 3x the largest beam major axis to
    focus on inner PSF structure.

    Parameters
    ----------
    results : list of dict
        Results from TestImagingHandler.get_results().
    imaging_dir : str, optional
        Base directory containing the test images. If None, uses the
        current working directory.
    output_file : str, optional
        Output filename for the figure. Set to None to skip saving.
    target : str, optional
        Filter results to this target.
    config : str, optional
        Filter results to this config.
    product : str, optional
        Filter results to this product.
    max_radius_arcsec : float, optional
        Maximum radius for the profile. If None, defaults to 3x the
        largest beam major axis in the result set.
    bin_width_arcsec : float, optional
        Radial bin width in arcseconds. Enforced to be at least one pixel
        wide. If None, defaults to one pixel.
    log_scale : bool
        If True, use a log scale on the y-axis.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    dpi : int
        Figure resolution.
    title : str, optional
        Figure title. If None, auto-generated.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib is required for plotting")
        return None

    filtered = _filter_results(results, target=target, config=config, product=product)
    if len(filtered) == 0:
        logger.warning("No results match the filter criteria")
        return None

    if imaging_dir is None:
        imaging_dir = '.'

    # Determine default radius from beam sizes (3x largest beam)
    if max_radius_arcsec is None:
        max_beam = _get_max_beam_arcsec(filtered)
        if max_beam is not None:
            max_radius_arcsec = 3.0 * max_beam
            logger.info(f"Auto max_radius_arcsec = {max_radius_arcsec:.2f} "
                        f"(3 x {max_beam:.2f})")

    if figsize is None:
        figsize = (10, 7)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    profiles_plotted = 0
    for result in filtered:
        # Use cached radial profile when available to avoid re-reading image
        if '_psf_radii' in result and '_psf_profile' in result:
            radius = np.asarray(result['_psf_radii'])
            profile = np.asarray(result['_psf_profile'])
            # Optionally truncate to the requested x-axis range
            if max_radius_arcsec is not None and radius[-1] > max_radius_arcsec:
                mask = radius <= max_radius_arcsec
                radius = radius[mask]
                profile = profile[mask]
        else:
            image_root = _build_image_root(result)
            if image_root is None:
                continue
            psf_file = os.path.join(imaging_dir, image_root + '.psf')
            data_2d, pix_scale = _read_image_data(psf_file)
            if data_2d is None:
                logger.warning(f"Could not read PSF: {psf_file}")
                continue
            radius, profile = _compute_radial_profile(
                data_2d, pix_scale,
                max_radius_arcsec=max_radius_arcsec,
                bin_width_arcsec=bin_width_arcsec,
            )
            del data_2d

        label = _make_label(result)
        line, = ax.plot(radius, profile, label=label, linewidth=1.5)

        # Overlay the Gaussian fit profile using the beam parameters.
        # The azimuthal average of a 2D elliptical Gaussian with
        # FWHM (bmaj, bmin) is a 1D Gaussian with sigma equal to the
        # geometric mean: sigma_eff = sqrt(sigma_maj * sigma_min).
        bmaj = result.get('bmaj_arcsec')
        bmin = result.get('bmin_arcsec')
        if bmaj is not None and bmin is not None:
            fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            sigma_eff = np.sqrt(bmaj * bmin) * fwhm_to_sigma
            gauss_profile = np.exp(-radius**2 / (2.0 * sigma_eff**2))
            ax.plot(radius, gauss_profile, color=line.get_color(),
                    linewidth=1.0, linestyle='--', alpha=0.7)

        profiles_plotted += 1

    if profiles_plotted == 0:
        logger.warning("No PSF profiles could be plotted")
        plt.close(fig)
        return None

    ax.set_xlabel('Radius (arcsec)')
    ax.set_ylabel('PSF amplitude')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(fontsize='small', loc='upper right')
    if max_radius_arcsec is not None:
        ax.set_xlim(0, max_radius_arcsec)
    else:
        ax.set_xlim(left=0)

    if log_scale:
        ax.set_yscale('symlog', linthresh=0.01)
        ax.set_ylabel('PSF amplitude (symlog)')

    if title is None:
        parts = []
        if target or (len(set(r['target'] for r in filtered)) == 1):
            parts.append(filtered[0]['target'])
        if config or (len(set(r['config'] for r in filtered)) == 1):
            parts.append(filtered[0]['config'])
        if product or (len(set(r['product'] for r in filtered)) == 1):
            parts.append(filtered[0]['product'])
        title = 'PSF Radial Profiles'
        if parts:
            title += ' - ' + ' / '.join(parts)
    ax.set_title(title)

    fig.tight_layout()

    if output_file is not None:
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved PSF radial profile gallery to {output_file}")

    return fig


def make_psf_image_gallery(
    results,
    imaging_dir=None,
    output_file=None,
    target=None,
    config=None,
    product=None,
    zoom_radius_arcsec=None,
    vmin=-0.1,
    vmax=1.0,
    cmap='RdBu_r',
    show_beam_ellipse=True,
    max_panels_per_page=9,
    figsize=None,
    dpi=150,
    title=None,
):
    """Create a gallery of 2D PSF images as a grid of panels.

    Each panel shows the central region of a PSF image with a
    consistent color scale. PSF data are expected to already be
    peak-normalized (as produced by tclean). The default zoom level
    is 3x the largest beam major axis to focus on inner PSF structure,
    with all panels sharing the same angular scale.

    If more than ``max_panels_per_page`` results are provided, the
    gallery is split across multiple enumerated figures/files.

    Parameters
    ----------
    results : list of dict
        Results from TestImagingHandler.get_results().
    imaging_dir : str, optional
        Base directory containing the test images. If None, uses the
        current working directory.
    output_file : str, optional
        Output filename for the figure. Set to None to skip saving.
        When multiple pages are produced, a page number is inserted
        before the extension (e.g. ``gallery_1.png``, ``gallery_2.png``).
    target : str, optional
        Filter results to this target.
    config : str, optional
        Filter results to this config.
    product : str, optional
        Filter results to this product.
    zoom_radius_arcsec : float, optional
        Half-width of the zoomed region in arcseconds. If None,
        defaults to 3x the largest beam major axis in the result set.
    vmin : float
        Minimum color scale value (PSFs are normalized to peak=1).
    vmax : float
        Maximum color scale value.
    cmap : str
        Matplotlib colormap name.
    show_beam_ellipse : bool
        If True, draw the restoring beam ellipse in the lower-left corner.
    max_panels_per_page : int
        Maximum number of panels per figure page (default 9, i.e. 3x3).
    figsize : tuple, optional
        Figure size. If None, auto-computed from number of panels.
    dpi : int
        Figure resolution.
    title : str, optional
        Figure supertitle. If None, auto-generated.

    Returns
    -------
    list of matplotlib.figure.Figure
        List of figure objects (one per page).
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib is required for plotting")
        return []

    filtered = _filter_results(results, target=target, config=config, product=product)
    if len(filtered) == 0:
        logger.warning("No results match the filter criteria")
        return []

    if imaging_dir is None:
        imaging_dir = '.'

    # Determine default zoom from beam sizes (3x largest beam)
    if zoom_radius_arcsec is None:
        max_beam = _get_max_beam_arcsec(filtered)
        if max_beam is not None:
            zoom_radius_arcsec = 3.0 * max_beam
            logger.info(f"Auto zoom_radius_arcsec = {zoom_radius_arcsec:.2f} "
                        f"(3 x {max_beam:.2f})")

    # Auto-generate base title
    if title is None:
        parts = []
        if target or (len(set(r['target'] for r in filtered)) == 1):
            parts.append(filtered[0]['target'])
        if config or (len(set(r['config'] for r in filtered)) == 1):
            parts.append(filtered[0]['config'])
        if product or (len(set(r['product'] for r in filtered)) == 1):
            parts.append(filtered[0]['product'])
        title = 'PSF Image Gallery'
        if parts:
            title += ' - ' + ' / '.join(parts)

    # Split into pages
    n_pages = math.ceil(len(filtered) / max_panels_per_page)
    figs = []

    for page_idx in range(n_pages):
        page_results = filtered[page_idx * max_panels_per_page:
                                (page_idx + 1) * max_panels_per_page]
        n = len(page_results)
        nrows, ncols = _get_grid_layout(n)

        if figsize is None:
            this_figsize = (4.5 * ncols, 4.0 * nrows + 0.6)
        else:
            this_figsize = figsize

        fig, axes = plt.subplots(nrows, ncols, figsize=this_figsize, squeeze=False)

        panels_plotted = 0
        for idx, result in enumerate(page_results):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col]

            image_root = _build_image_root(result)
            if image_root is None:
                ax.set_visible(False)
                continue
            psf_file = os.path.join(imaging_dir, image_root + '.psf')

            data_2d, pix_scale = _read_image_data(psf_file)
            if data_2d is None:
                logger.warning(f"Could not read PSF: {psf_file}")
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center')
                ax.set_title(result.get('test_name', '?'), fontsize=9)
                continue

            ny, nx = data_2d.shape

            # Build coordinate arrays in arcsec offset from center
            cy, cx = ny // 2, nx // 2
            extent_x = (np.arange(nx) - cx) * pix_scale
            extent_y = (np.arange(ny) - cy) * pix_scale
            extent = [extent_x[0], extent_x[-1], extent_y[0], extent_y[-1]]

            ax.imshow(data_2d, origin='lower', extent=extent,
                      vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')

            # Zoom (common scale for all panels)
            if zoom_radius_arcsec is not None:
                ax.set_xlim(-zoom_radius_arcsec, zoom_radius_arcsec)
                ax.set_ylim(-zoom_radius_arcsec, zoom_radius_arcsec)

            # Crop data to zoom region before contouring (contour processes
            # the full array unlike imshow, so this avoids huge memory/CPU
            # cost on large images).
            zoom_r = zoom_radius_arcsec if zoom_radius_arcsec is not None else None
            if zoom_r is not None:
                margin_pix = int(zoom_r / pix_scale) + 2
                x0 = max(cx - margin_pix, 0)
                x1 = min(cx + margin_pix + 1, nx)
                y0 = max(cy - margin_pix, 0)
                y1 = min(cy + margin_pix + 1, ny)
            else:
                x0, x1, y0, y1 = 0, nx, 0, ny
            crop = np.array(data_2d[y0:y1, x0:x1])
            del data_2d
            x_crop = (np.arange(x0, x1) - cx) * pix_scale
            y_crop = (np.arange(y0, y1) - cy) * pix_scale

            # Contour at the 0.5 level (half-power)
            ax.contour(x_crop, y_crop, crop, levels=[0.5],
                       colors='black', linewidths=1.0)

            # Gaussian fit ellipse at center (FWHM from beam parameters)
            bmaj = result.get('bmaj_arcsec')
            bmin = result.get('bmin_arcsec')
            bpa = result.get('bpa_deg', 0)
            if bmaj is not None and bmin is not None:
                gauss_ellipse = Ellipse(
                    (0, 0), width=bmin, height=bmaj, angle=-bpa + 90,
                    edgecolor='lime', facecolor='none',
                    linewidth=1.2, linestyle='--')
                ax.add_patch(gauss_ellipse)

            # Beam ellipse
            if show_beam_ellipse:
                bmaj = result.get('bmaj_arcsec')
                bmin = result.get('bmin_arcsec')
                bpa = result.get('bpa_deg', 0)
                if bmaj is not None and bmin is not None:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    # Position in lower-left corner
                    ex = xlim[0] + 0.15 * (xlim[1] - xlim[0])
                    ey = ylim[0] + 0.15 * (ylim[1] - ylim[0])
                    ellipse = Ellipse(
                        (ex, ey), width=bmin, height=bmaj, angle=-bpa + 90,
                        edgecolor='black', facecolor='gold', alpha=0.7,
                        linewidth=1.0)
                    ax.add_patch(ellipse)

            # Panel label
            label = result.get('test_name', '?')
            bmaj = result.get('bmaj_arcsec')
            bmin = result.get('bmin_arcsec')
            if bmaj is not None and bmin is not None:
                label += f'\n{bmaj:.2f}x{bmin:.2f}"'
            ax.set_title(label, fontsize=9)
            ax.set_xlabel('Offset (arcsec)', fontsize=8)
            ax.set_ylabel('Offset (arcsec)', fontsize=8)
            ax.tick_params(labelsize=7)
            panels_plotted += 1

        # Hide unused panels
        for idx in range(n, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row][col].set_visible(False)

        if panels_plotted == 0:
            logger.warning("No PSF images could be plotted on page "
                           f"{page_idx + 1}")
            plt.close(fig)
            continue

        # Supertitle
        page_title = title
        if n_pages > 1:
            page_title += f' ({page_idx + 1}/{n_pages})'
        fig.suptitle(page_title, fontsize=12, y=1.01)

        fig.tight_layout()

        page_file = _paginated_output_file(output_file, page_idx, n_pages)
        if page_file is not None:
            fig.savefig(page_file, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved PSF image gallery to {page_file}")

        figs.append(fig)

    return figs


def make_image_gallery(
    results,
    imaging_dir=None,
    output_file=None,
    target=None,
    config=None,
    product=None,
    zoom_radius_arcsec=None,
    vmin=None,
    vmax=None,
    cmap='inferno',
    show_beam_ellipse=True,
    symmetric_color=False,
    sigma_clip=3.0,
    max_panels_per_page=9,
    figsize=None,
    dpi=150,
    title=None,
):
    """Create a gallery of 2D dirty images as a grid of panels.

    Each panel shows a dirty image from a different test parameter
    combination. Color scales are determined per-panel or can be
    set globally.

    If more than ``max_panels_per_page`` results are provided, the
    gallery is split across multiple enumerated figures/files.

    Parameters
    ----------
    results : list of dict
        Results from TestImagingHandler.get_results().
    imaging_dir : str, optional
        Base directory containing the test images. If None, uses the
        current working directory.
    output_file : str, optional
        Output filename for the figure. Set to None to skip saving.
        When multiple pages are produced, a page number is inserted
        before the extension (e.g. ``gallery_1.png``, ``gallery_2.png``).
    target : str, optional
        Filter results to this target.
    config : str, optional
        Filter results to this config.
    product : str, optional
        Filter results to this product.
    zoom_radius_arcsec : float, optional
        Half-width of the displayed region in arcseconds. If None,
        shows the full image.
    vmin : float, optional
        Minimum color scale value. If None, auto-determined per panel.
    vmax : float, optional
        Maximum color scale value. If None, auto-determined per panel.
    cmap : str
        Matplotlib colormap name.
    show_beam_ellipse : bool
        If True, draw the restoring beam ellipse on each panel.
    symmetric_color : bool
        If True and vmin/vmax are not set, use a symmetric color range
        centered on zero and switch to a diverging colormap.
    sigma_clip : float
        When auto-determining color range, clip at this many sigma
        above/below the median.
    max_panels_per_page : int
        Maximum number of panels per figure page (default 9, i.e. 3x3).
    figsize : tuple, optional
        Figure size. If None, auto-computed from number of panels.
    dpi : int
        Figure resolution.
    title : str, optional
        Figure supertitle. If None, auto-generated.

    Returns
    -------
    list of matplotlib.figure.Figure
        List of figure objects (one per page).
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib is required for plotting")
        return []

    filtered = _filter_results(results, target=target, config=config, product=product)
    if len(filtered) == 0:
        logger.warning("No results match the filter criteria")
        return []

    if imaging_dir is None:
        imaging_dir = '.'

    # Auto-generate base title
    if title is None:
        parts = []
        if target or (len(set(r['target'] for r in filtered)) == 1):
            parts.append(filtered[0]['target'])
        if config or (len(set(r['config'] for r in filtered)) == 1):
            parts.append(filtered[0]['config'])
        if product or (len(set(r['product'] for r in filtered)) == 1):
            parts.append(filtered[0]['product'])
        title = 'Dirty Image Gallery'
        if parts:
            title += ' - ' + ' / '.join(parts)

    # Split into pages
    n_pages = math.ceil(len(filtered) / max_panels_per_page)
    figs = []

    for page_idx in range(n_pages):
        page_results = filtered[page_idx * max_panels_per_page:
                                (page_idx + 1) * max_panels_per_page]
        n = len(page_results)
        nrows, ncols = _get_grid_layout(n)

        if figsize is None:
            this_figsize = (4.5 * ncols, 4.0 * nrows + 0.6)
        else:
            this_figsize = figsize

        fig, axes = plt.subplots(nrows, ncols, figsize=this_figsize,
                                 squeeze=False)

        panels_plotted = 0
        for idx, result in enumerate(page_results):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col]

            image_root = _build_image_root(result)
            if image_root is None:
                ax.set_visible(False)
                continue
            image_file = os.path.join(imaging_dir, image_root + '.image')

            data_2d, pix_scale = _read_image_data(image_file)
            if data_2d is None:
                logger.warning(f"Could not read image: {image_file}")
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center')
                ax.set_title(result.get('test_name', '?'), fontsize=9)
                continue

            ny, nx = data_2d.shape

            # Build coordinate arrays in arcsec offset from center
            cy, cx = ny // 2, nx // 2
            extent_x = (np.arange(nx) - cx) * pix_scale
            extent_y = (np.arange(ny) - cy) * pix_scale
            extent = [extent_x[0], extent_x[-1], extent_y[0], extent_y[-1]]

            # Determine color scale
            this_vmin = vmin
            this_vmax = vmax
            this_cmap = cmap
            if this_vmin is None or this_vmax is None:
                finite_data = data_2d[np.isfinite(data_2d)]
                if len(finite_data) > 0:
                    med = np.median(finite_data)
                    std = np.std(finite_data)
                    if symmetric_color:
                        absmax = max(abs(med - sigma_clip * std),
                                     abs(med + sigma_clip * std))
                        this_vmin = -absmax if this_vmin is None else this_vmin
                        this_vmax = absmax if this_vmax is None else this_vmax
                        this_cmap = 'RdBu_r'
                    else:
                        this_vmin = (med - sigma_clip * std) if this_vmin is None else this_vmin
                        this_vmax = (med + sigma_clip * std) if this_vmax is None else this_vmax

            im = ax.imshow(data_2d, origin='lower', extent=extent,
                           vmin=this_vmin, vmax=this_vmax, cmap=this_cmap,
                           aspect='equal')
            del data_2d

            # Zoom
            if zoom_radius_arcsec is not None:
                ax.set_xlim(-zoom_radius_arcsec, zoom_radius_arcsec)
                ax.set_ylim(-zoom_radius_arcsec, zoom_radius_arcsec)

            # Colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)

            # Beam ellipse
            if show_beam_ellipse:
                bmaj = result.get('bmaj_arcsec')
                bmin = result.get('bmin_arcsec')
                bpa = result.get('bpa_deg', 0)
                if bmaj is not None and bmin is not None:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    ex = xlim[0] + 0.10 * (xlim[1] - xlim[0])
                    ey = ylim[0] + 0.10 * (ylim[1] - ylim[0])
                    ellipse = Ellipse(
                        (ex, ey), width=bmin, height=bmaj, angle=-bpa,
                        edgecolor='white', facecolor='none',
                        linewidth=1.5)
                    ax.add_patch(ellipse)

            # Panel label
            label = result.get('test_name', '?')
            bmaj = result.get('bmaj_arcsec')
            bmin = result.get('bmin_arcsec')
            rms = result.get('rms')
            sublabel_parts = []
            if bmaj is not None and bmin is not None:
                sublabel_parts.append(f'{bmaj:.2f}x{bmin:.2f}"')
            if rms is not None:
                sublabel_parts.append(f'rms={rms:.2e}')
            if sublabel_parts:
                label += '\n' + ', '.join(sublabel_parts)
            ax.set_title(label, fontsize=9)
            ax.set_xlabel('Offset (arcsec)', fontsize=8)
            ax.set_ylabel('Offset (arcsec)', fontsize=8)
            ax.tick_params(labelsize=7)
            panels_plotted += 1

        # Hide unused panels
        for idx in range(n, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row][col].set_visible(False)

        if panels_plotted == 0:
            logger.warning(f"No images could be plotted on page "
                           f"{page_idx + 1}")
            plt.close(fig)
            continue

        # Supertitle
        page_title = title
        if n_pages > 1:
            page_title += f' ({page_idx + 1}/{n_pages})'
        fig.suptitle(page_title, fontsize=12, y=1.01)

        fig.tight_layout()

        page_file = _paginated_output_file(output_file, page_idx, n_pages)
        if page_file is not None:
            fig.savefig(page_file, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved image gallery to {page_file}")

        figs.append(fig)

    return figs


# ---------------------------------------------------------------
# Metrics-vs-robust gallery
# ---------------------------------------------------------------


# Default metrics shown in the gallery.
# Each entry is either:
#   (result_key, y_label)                       — single line per series
#   ((key1, key2), y_label, (name1, name2))     — paired lines (solid + dashed)
_DEFAULT_METRIC_DEFS = [
    (('bmaj_arcsec', 'bmin_arcsec'), 'Beam size (arcsec)', ('major', 'minor')),
    ('bpa_deg',      'Beam PA (deg)'),
    ('rms',          'RMS'),
    ('kappa',        'Kappa'),
    ('epsilon',          'epsilon'),
    ('skirt_level',  'Skirt level'),
]


def _parse_metric_def(metric_def):
    """Normalise a metric definition into (keys, ylabel, key_labels).

    Parameters
    ----------
    metric_def : tuple
        Either ``(key, ylabel)`` or ``((key1, key2, ...), ylabel, (name1, name2, ...))``.

    Returns
    -------
    keys : tuple of str
    ylabel : str
    key_labels : tuple of str or None
    """
    if len(metric_def) == 3:
        keys, ylabel, key_labels = metric_def
        if isinstance(keys, str):
            keys = (keys,)
            key_labels = (key_labels,)
        return tuple(keys), ylabel, tuple(key_labels)
    else:
        key, ylabel = metric_def
        if isinstance(key, str):
            return (key,), ylabel, None
        return tuple(key), ylabel, None


def make_metrics_vs_robust_gallery(
    results,
    output_file=None,
    target=None,
    config=None,
    product=None,
    metrics=None,
    figsize=None,
    dpi=150,
    title=None,
):
    """Create a multi-panel figure of imaging metrics vs. robust parameter.

    Each panel shows one metric on the y-axis and the robust parameter
    on the x-axis.  Results with different weightings (e.g. natural,
    uniform) that lack a robust value are plotted as isolated markers
    at the edges.  Different taper values are shown as separate
    series.

    Panels may show paired quantities (e.g. beam major and minor axis)
    on the same y-axis using solid and dashed lines.

    Parameters
    ----------
    results : list of dict
        Results from ``TestImagingHandler.get_results()``.
    output_file : str, optional
        Output filename.  Set to None to skip saving.
    target : str, optional
        Filter results to this target.
    config : str, optional
        Filter results to this config.
    product : str, optional
        Filter results to this product.
    metrics : list of tuple, optional
        Metric definitions for each panel.  Each entry is either
        ``(result_key, y_label)`` for a single quantity, or
        ``((key1, key2), y_label, (name1, name2))`` to show two
        quantities on the same panel with solid/dashed lines.
        Defaults to beam size (major + minor), beam PA, RMS, kappa,
        epsilon, and skirt level.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.
    dpi : int
        Figure resolution.
    title : str, optional
        Figure super-title.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib is required for plotting")
        return None

    filtered = _filter_results(results, target=target, config=config,
                               product=product)
    if len(filtered) == 0:
        logger.warning("No results match the filter criteria")
        return None

    if metrics is None:
        metrics = list(_DEFAULT_METRIC_DEFS)

    parsed = [_parse_metric_def(m) for m in metrics]
    n_panels = len(parsed)
    if n_panels == 0:
        return None

    if figsize is None:
        figsize = (8, 3.0 * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True,
                             squeeze=False)
    axes = axes[:, 0]

    # Group results by taper value for separate series
    taper_set = sorted(set(r.get('taper_arcsec', 0.0) or 0.0
                           for r in filtered))

    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    linestyles = ['-', '--', ':', '-.']

    for panel_idx, (keys, ylabel, key_labels) in enumerate(parsed):
        ax = axes[panel_idx]

        for tap_idx, taper in enumerate(taper_set):
            tap_results = [r for r in filtered
                           if (r.get('taper_arcsec', 0.0) or 0.0) == taper]

            marker = markers[tap_idx % len(markers)]
            tap_label = f'taper {taper:.1f}"' if taper > 0 else 'no taper'

            for k_idx, key in enumerate(keys):
                ls = linestyles[k_idx % len(linestyles)]

                # Build legend label
                if key_labels is not None:
                    label = f'{tap_label}, {key_labels[k_idx]}'
                else:
                    label = tap_label

                # Separate Briggs (has robust) from non-Briggs
                briggs = [r for r in tap_results
                          if r.get('robust') is not None
                          and r.get(key) is not None]
                other = [r for r in tap_results
                         if r.get('robust') is None
                         and r.get(key) is not None]

                if briggs:
                    briggs_sorted = sorted(briggs, key=lambda r: r['robust'])
                    x = [r['robust'] for r in briggs_sorted]
                    y = [r[key] for r in briggs_sorted]
                    ax.plot(x, y, marker=marker, label=label,
                            linewidth=1.2, markersize=6, linestyle=ls)

                for r in other:
                    weighting = r.get('weighting', '?')
                    yval = r[key]
                    if weighting == 'natural':
                        xpos = 2.5
                    elif weighting == 'uniform':
                        xpos = -2.5
                    else:
                        xpos = 3.0
                    ax.plot(xpos, yval, marker=marker, markersize=8,
                            linestyle='none', color='gray', alpha=0.7)
                    ann = weighting
                    if key_labels is not None:
                        ann += f' ({key_labels[k_idx]})'
                    ax.annotate(ann, (xpos, yval), fontsize=7,
                                textcoords='offset points', xytext=(5, 3))

        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        if panel_idx == 0:
            ax.legend(fontsize='small', loc='best')

    axes[-1].set_xlabel('Robust parameter', fontsize=10)

    if title is None:
        parts = []
        if target or (len(set(r['target'] for r in filtered)) == 1):
            parts.append(filtered[0]['target'])
        if config or (len(set(r.get('config', '') for r in filtered)) == 1):
            parts.append(filtered[0].get('config', ''))
        title = ' / '.join(parts) if parts else 'Metrics vs. Robust'
    fig.suptitle(title, fontsize=12, y=1.01)

    fig.tight_layout()

    if output_file is not None:
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved metrics-vs-robust gallery to {output_file}")

    return fig


# ---------------------------------------------------------------
# Metrics-vs-beam gallery
# ---------------------------------------------------------------


_DEFAULT_BEAM_METRIC_DEFS = [
    ('bpa_deg',      'Beam PA (deg)'),
    ('rms',          'RMS'),
    ('kappa',        'Kappa'),
    ('epsilon',          'epsilon'),
    ('skirt_level',  'Skirt level'),
]


def _series_label(weighting, taper):
    """Build a concise legend label from weighting and taper."""
    parts = [weighting]
    if taper and taper > 0:
        parts.append(f'taper {taper:.1f}"')
    return ', '.join(parts)


def make_metrics_vs_beam_gallery(
    results,
    output_file=None,
    target=None,
    config=None,
    product=None,
    metrics=None,
    beam_key='bmaj_arcsec',
    beam_label='Beam major axis (arcsec)',
    figsize=None,
    dpi=150,
    title=None,
):
    """Create a multi-panel figure of imaging metrics vs. beam size.

    Each panel shows one metric on the y-axis and the beam size on
    the x-axis.  All results — regardless of weighting scheme (Briggs,
    natural, uniform) or UV taper — are plotted together, with each
    weighting+taper combination as a distinct series.  Points are
    labelled with the robust value or weighting name.

    Panels may show paired quantities using solid and dashed lines
    (see ``metrics`` parameter).

    Parameters
    ----------
    results : list of dict
        Results from ``TestImagingHandler.get_results()``.
    output_file : str, optional
        Output filename.  Set to None to skip saving.
    target : str, optional
        Filter results to this target.
    config : str, optional
        Filter results to this config.
    product : str, optional
        Filter results to this product.
    metrics : list of tuple, optional
        Metric definitions for each panel.  Each entry is either
        ``(result_key, y_label)`` for a single quantity, or
        ``((key1, key2), y_label, (name1, name2))`` to show two
        quantities on the same panel.
        Defaults to beam PA, RMS, kappa, epsilon, and skirt level.
    beam_key : str
        Result dictionary key for the x-axis beam size.
    beam_label : str
        X-axis label.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.
    dpi : int
        Figure resolution.
    title : str, optional
        Figure super-title.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib is required for plotting")
        return None

    filtered = _filter_results(results, target=target, config=config,
                               product=product)
    filtered = [r for r in filtered if r.get(beam_key) is not None]
    if len(filtered) == 0:
        logger.warning("No results with beam measurements match the filter")
        return None

    if metrics is None:
        metrics = list(_DEFAULT_BEAM_METRIC_DEFS)

    parsed = [_parse_metric_def(m) for m in metrics]
    n_panels = len(parsed)
    if n_panels == 0:
        return None

    if figsize is None:
        figsize = (8, 3.0 * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True,
                             squeeze=False)
    axes = axes[:, 0]

    # Group by (weighting, taper) for distinct series
    series_keys = []
    seen = set()
    for r in filtered:
        w = r.get('weighting', 'unknown')
        t = r.get('taper_arcsec', 0.0) or 0.0
        k = (w, t)
        if k not in seen:
            seen.add(k)
            series_keys.append(k)

    markers_list = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyles = ['-', '--', ':', '-.']

    for panel_idx, (keys, ylabel, key_labels) in enumerate(parsed):
        ax = axes[panel_idx]

        for s_idx, (weighting, taper) in enumerate(series_keys):
            marker = markers_list[s_idx % len(markers_list)]
            color = colors[s_idx % len(colors)]
            base_label = _series_label(weighting, taper)

            for k_idx, key in enumerate(keys):
                ls = linestyles[k_idx % len(linestyles)]
                if key_labels is not None:
                    label = f'{base_label}, {key_labels[k_idx]}'
                else:
                    label = base_label

                s_results = [r for r in filtered
                             if r.get('weighting', 'unknown') == weighting
                             and (r.get('taper_arcsec', 0.0) or 0.0) == taper
                             and r.get(key) is not None]
                if not s_results:
                    continue

                s_results = sorted(s_results, key=lambda r: r[beam_key])
                x = [r[beam_key] for r in s_results]
                y = [r[key] for r in s_results]

                ax.plot(x, y, marker=marker, color=color, label=label,
                        linewidth=1.2, markersize=6, linestyle=ls)

                # Annotate only the primary key to avoid clutter
                if k_idx == 0:
                    for r, xi, yi in zip(s_results, x, y):
                        rob = r.get('robust')
                        ann = f'r={rob}' if rob is not None else weighting
                        ax.annotate(ann, (xi, yi), fontsize=6,
                                    textcoords='offset points', xytext=(4, 4),
                                    color=color, alpha=0.8)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        if panel_idx == 0:
            ax.legend(fontsize='small', loc='best')

    axes[-1].set_xlabel(beam_label, fontsize=10)

    if title is None:
        parts = []
        if target or (len(set(r['target'] for r in filtered)) == 1):
            parts.append(filtered[0]['target'])
        if config or (len(set(r.get('config', '') for r in filtered)) == 1):
            parts.append(filtered[0].get('config', ''))
        title = ' / '.join(parts) if parts else 'Metrics vs. Beam Size'
    fig.suptitle(title, fontsize=12, y=1.01)

    fig.tight_layout()

    if output_file is not None:
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved metrics-vs-beam gallery to {output_file}")

    return fig


# ---------------------------------------------------------------
# HTML summary report
# ---------------------------------------------------------------

_METRIC_DESCRIPTIONS = {
    'kappa': (
        'Kappa',
        'The sum of the difference between the azimuthally averaged '
        'radial PSF profile and the best-fit Gaussian (from the beam), '
        'evaluated within the FWHM and normalised by the Gaussian area '
        'inside the FWHM. Kappa measures how close the PSF within the '
        'FWHM is to a Gaussian. Kappa &lt; 0 means the PSF is more peaked '
        'relative to the Gaussian model. Kappa &gt; 0 means the PSF is '
        'flatter relative to the Gaussian model. '
        'See Koch+2018 Eq. 7.'
    ),
    'epsilon': (
        'Epsilon',
        'Ratio of the summed ideal Gaussian beam to the summed dirty PSF '
        'within a window around the peak, after masking regions beyond '
        'the first null. Values close to 1 indicate a well-behaved PSF.'
        'We use the definition from Czekala+2021.'
    ),
    'skirt_level': (
        'Skirt Level',
        'The PSF amplitude at 2 sqrt(2 log 2) sigma (i.e., FWHM in radius), '
        'estimated via spline interpolation of the azimuthally averaged '
        'radial profile. For clarity, "FWHM" means a radius at the FWHM, '
        'not the true FWHM converted to a radius. If the PSF is a Gaussian, '
        'the skirt level should be exp(&minus;4 log(2)) &asymp; 0.0625.'
    ),
}


def make_html_report(
    results,
    plot_files=None,
    output_file='report.html',
    target=None,
    title=None,
):
    """Generate a self-contained HTML summary report.

    Produces an HTML file that embeds all diagnostic plots as base64
    images and includes a metrics summary table with definitions of
    the key PSF quality metrics.

    Parameters
    ----------
    results : list of dict
        Results from ``TestImagingHandler.get_results()``, typically
        pre-filtered to a single target.
    plot_files : dict, optional
        Dictionary mapping plot names (e.g. ``'psf_profiles'``,
        ``'psf_gallery'``) to file paths (str) or lists of file
        paths.  Each image file is embedded in the report.
    output_file : str, optional
        Output HTML filename.
    target : str, optional
        Target name for the report title.
    title : str, optional
        Custom report title.  If None, auto-generated from target.

    Returns
    -------
    str or None
        Path to the written HTML file.
    """

    if plot_files is None:
        plot_files = {}

    if title is None:
        title = f'Test Imaging Report: {target}' if target else 'Test Imaging Report'

    # --- Build the metrics table rows ---
    table_columns = [
        ('test_name', 'Test'),
        ('config', 'Config'),
        ('weighting', 'Weighting'),
        ('robust', 'Robust'),
        ('taper_arcsec', 'Taper (")'),
        ('cell_arcsec', 'Cell (")'),
        ('bmaj_arcsec', 'B<sub>maj</sub> (")'),
        ('bmin_arcsec', 'B<sub>min</sub> (")'),
        ('bpa_deg', 'BPA (&deg;)'),
        ('rms', 'RMS'),
        ('kappa', 'Kappa'),
        ('epsilon', 'Epsilon'),
        ('skirt_level', 'Skirt'),
    ]

    def _fmt(val, key):
        if val is None:
            return '&mdash;'
        if key == 'rms':
            return f'{val:.2e}'
        if isinstance(val, float):
            return f'{val:.4f}'
        return str(val)

    header_row = ''.join(f'<th>{label}</th>' for _, label in table_columns)
    data_rows = []
    for r in results:
        cells = ''.join(
            f'<td>{_fmt(r.get(k), k)}</td>' for k, _ in table_columns
        )
        data_rows.append(f'<tr>{cells}</tr>')

    # --- Embed plot images ---
    plot_sections = []
    plot_display_names = {
        'psf_profiles': 'PSF Radial Profiles',
        'psf_gallery': 'PSF Image Gallery',
        'image_gallery': 'Dirty Image Gallery',
        'metrics_vs_robust': 'Metrics vs. Robust',
        'metrics_vs_beam': 'Metrics vs. Beam Size',
    }

    for plot_key, paths in plot_files.items():
        if paths is None:
            continue
        if isinstance(paths, str):
            paths = [paths]
        # Handle dict returns (per-target) by extracting the values
        if isinstance(paths, dict):
            all_paths = []
            for v in paths.values():
                if isinstance(v, list):
                    all_paths.extend(v)
                elif v is not None:
                    all_paths.append(v)
            paths = all_paths

        section_title = plot_display_names.get(plot_key, plot_key)
        img_tags = []
        for p in paths:
            if isinstance(p, str) and os.path.isfile(p):
                with open(p, 'rb') as f:
                    data = base64.b64encode(f.read()).decode('ascii')
                img_tags.append(
                    f'<img src="data:image/png;base64,{data}" '
                    f'alt="{plot_key}" '
                    f'style="max-width:100%; margin:10px 0;">'
                )

        if img_tags:
            plot_sections.append(
                f'<h2>{section_title}</h2>\n' + '\n'.join(img_tags)
            )

    # --- Metric definitions ---
    metric_defs_html = []
    for _, (name, description) in _METRIC_DESCRIPTIONS.items():
        metric_defs_html.append(
            f'<dt><strong>{name}</strong></dt>\n<dd>{description}</dd>'
        )

    # --- Assemble HTML ---
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
         Roboto, Helvetica, Arial, sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px;
         color: #333; background: #fafafa; }}
  h1 {{ border-bottom: 2px solid #2c5f8a; padding-bottom: 8px; color: #2c5f8a; }}
  h2 {{ color: #2c5f8a; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0;
           font-size: 0.9em; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
  th {{ background: #2c5f8a; color: #fff; text-align: center; }}
  tr:nth-child(even) {{ background: #eef3f8; }}
  tr:hover {{ background: #d6e4f0; }}
  dl {{ margin: 15px 0; }}
  dt {{ margin-top: 12px; font-size: 1.0em; }}
  dd {{ margin-left: 20px; color: #555; line-height: 1.5; }}
  .timestamp {{ color: #999; font-size: 0.85em; margin-top: 40px;
                border-top: 1px solid #ddd; padding-top: 8px; }}
  img {{ border: 1px solid #ddd; border-radius: 4px; }}
</style>
</head>
<body>
<h1>{title}</h1>

<h2>Summary Table</h2>
<table>
<thead><tr>{header_row}</tr></thead>
<tbody>
{''.join(data_rows)}
</tbody>
</table>

{''.join(plot_sections)}

<h2>Metric Definitions</h2>
<dl>
{''.join(metric_defs_html)}
</dl>

<p class="timestamp">Report generated {timestamp}</p>
</body>
</html>"""

    if output_file is not None:
        with open(output_file, 'w') as f:
            f.write(html)
        logger.info(f"Saved HTML report to {output_file}")

    return output_file


# ---------------------------------------------------------------
# PSF metric functions
# ---------------------------------------------------------------


def measure_kappa(radii, psf_radial, bmaj, bmin):
    """Compute kappa, the normalised PSF deviation from a Gaussian.

    Kappa is the sum of the difference between the azimuthally averaged
    radial PSF profile and the best-fit Gaussian (from the beam),
    evaluated within the FWHM and normalised by the Gaussian area
    inside the FWHM.

    Kappa measures how close the PSF within the FWHM is to a Gaussian.
    Kappa < 0 means the PSF is more peaked relative to the Gaussian model.
    Kappa > 0 means the PSF is flatter relative to the Gaussian model.

    See Koch+2018 Equation 7.

    Parameters
    ----------
    radii : numpy.ndarray
        Radial bin centres in arcseconds.
    psf_radial : numpy.ndarray
        Azimuthally averaged PSF profile values at each radius.
    bmaj : float
        Beam major axis FWHM in arcseconds.
    bmin : float
        Beam minor axis FWHM in arcseconds.

    Returns
    -------
    float or None
        The kappa metric value.
    """
    radii = np.asarray(radii, dtype=float)
    psf_radial = np.asarray(psf_radial, dtype=float)

    # Remove NaN bins (empty annuli at large radii)
    valid = ~np.isnan(psf_radial)
    radii = radii[valid]
    psf_radial = psf_radial[valid]

    if len(radii) == 0:
        return None

    sigma = np.sqrt(bmaj * bmin) / fwhm_factor
    hwhm = np.sqrt(bmaj * bmin) / 2.0  # half-width at half maximum

    # Gaussian model evaluated at the radial bin centres
    gauss_profile = np.exp(-radii**2 / (2.0 * sigma**2))

    # Mask to within the half-max radius (FWHM/2)
    hwhm_mask = radii <= hwhm

    if not np.any(hwhm_mask):
        return None

    # Normalise by the Gaussian sum within the same region
    gauss_sum = np.sum(gauss_profile[hwhm_mask])
    kappa = np.sum(psf_radial[hwhm_mask] - gauss_profile[hwhm_mask]) / gauss_sum

    return float(kappa)


def measure_skirt_level(radii, psf_radial, bmaj, bmin):
    """Estimate the PSF amplitude at 2 sqrt(2 log 2) sigma (i.e., FWHM in radius).

    For clarity, "FWHM" means a radius at the FWHM, not the true FWHM converted
    to a radius.

    Uses a spline interpolation of the azimuthally averaged radial
    profile to evaluate the PSF at the geometric-mean FWHM radius
    derived from the beam parameters.

    If the PSF is a Gaussian, the skirt level should be exp(- 4 log(2)) = 0.0625.

    Parameters
    ----------
    radii : numpy.ndarray
        Radial bin centres in arcseconds.
    psf_radial : numpy.ndarray
        Azimuthally averaged PSF profile values at each radius.
    bmaj : float
        Beam major axis FWHM in arcseconds.
    bmin : float
        Beam minor axis FWHM in arcseconds.

    Returns
    -------
    float or None
        PSF value at the FWHM radius.
    """
    radii = np.asarray(radii, dtype=float)
    psf_radial = np.asarray(psf_radial, dtype=float)

    # Remove NaN bins (empty annuli at large radii)
    valid = ~np.isnan(psf_radial)
    radii = radii[valid]
    psf_radial = psf_radial[valid]

    if len(radii) == 0:
        return None

    fwhm = np.sqrt(bmaj * bmin)

    if fwhm > radii[-1]:
        logger.warning("FWHM exceeds radial profile extent; cannot compute skirt level")
        return None

    spline = UnivariateSpline(radii, psf_radial, s=0)
    skirt_val = float(spline(fwhm))
    return skirt_val


def measure_epsilon(radii, psf_radial, bmaj, bmin):
    """Compute epsilon, the ratio of clean beam to dirty beam flux.

    Evaluates the ratio of the azimuthally integrated ideal Gaussian
    beam to the azimuthally integrated dirty PSF, truncated at the
    first null (zero crossing) of the radial profile. Values close
    to 1 indicate a well-behaved PSF.

    Parameters
    ----------
    radii : numpy.ndarray
        Radial bin centres in arcseconds.
    psf_radial : numpy.ndarray
        Azimuthally averaged PSF profile values at each radius.
    bmaj : float
        Beam major axis FWHM in arcseconds.
    bmin : float
        Beam minor axis FWHM in arcseconds.

    Returns
    -------
    float or None
        The epsilon metric value.
    """
    radii = np.asarray(radii, dtype=float)
    psf_radial = np.asarray(psf_radial, dtype=float)

    # Remove NaN bins
    valid = ~np.isnan(psf_radial)
    radii = radii[valid]
    psf_radial = psf_radial[valid]

    if len(radii) == 0:
        return None

    sigma = np.sqrt(bmaj * bmin) / fwhm_factor

    # Truncate at the first null (zero crossing)
    null_idx = np.where(psf_radial <= 0)[0]
    if len(null_idx) > 0:
        radii = radii[:null_idx[0]]
        psf_radial = psf_radial[:null_idx[0]]

    if len(radii) == 0:
        return None

    # Gaussian model evaluated at the radial bin centres
    gauss_profile = np.exp(-radii**2 / (2.0 * sigma**2))

    # Azimuthal integration weight: 2*pi*r*dr
    dr = np.gradient(radii)
    psf_integral = np.sum(psf_radial * radii * dr)
    gauss_integral = np.sum(gauss_profile * radii * dr)

    if psf_integral == 0:
        logger.warning("PSF integrates to zero; cannot compute epsilon")
        return None

    epsilon = float(gauss_integral / psf_integral)

    return epsilon
