# Routines relates to python (not CASA) image handling.

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.units import Quantity
from astropy.wcs import WCS


# ------------------------------------------------------------------------
# Make new headers
# ------------------------------------------------------------------------

def make_simple_header_from_box(
        tlc_coord = None, brc_coord = None,
        pix_scale_deg = None, pad_arcsec = 0.0):
    """Wraps make_simple_header.

    tlc_coord and brc_coord : tuples of floats or SkyCoords giving
    corners of the image to be made.
    
    pix_scale_deg : pixel scale of the resulting image float in
    degrees or Quantity that can be converted to deg.

    pad_arcsec : padding to go beyond the tlc and brc coord in the
    resulting header. Float in arcseconds or Quantity that can be
    converted to arcsec. Default 0.0.

    """

    # Check the type of the inputs    
    if isinstance(pix_scale_deg, Quantity):
        pix_scale_deg.to("deg")
    else:
        pix_scale_deg = pix_scale_deg*u.deg

    if isinstance(pad_arcsec, Quantity):
        pad_arcsec.to("arcsec")
    else:
        pad_arcsec = pad_arcsec*u.arcsec
        
    # If not SkyCoords convert
    if isinstance(tlc_coord, SkyCoord) == False:
        
        # Assume tlc_coord[0] and [1] have the same type
        if isinstance(tlc_coord[0], Quantity):            
            tlc_coord = SkyCoord(
                tlc_coord[0], tlc_coord[1], frame='icrs')        
        else:
            tlc_coord = SkyCoord(
                tlc_coord[0]*u.deg, tlc_coord[1]*u.deg, frame='icrs')
            
    if isinstance(brc_coord, SkyCoord) == False:
        # Assume brc_coord[0] and [1] have the same type
        if isinstance(brc_coord[0], Quantity):            
            brc_coord = SkyCoord(
                brc_coord[0], brc_coord[1], frame='icrs')        
        else:
            brc_coord = SkyCoord(
                brc_coord[0]*u.deg, brc_coord[1]*u.deg, frame='icrs')

    # Find the midpoint
    pa = tlc_coord.position_angle(brc_coord)
    sep = tlc_coord.separation(brc_coord)
    midpoint = tlc_coord.directional_offset_by(pa, sep/2.0)
    
    # Find the edge
    extent_deg = sep.to("deg")/np.sqrt(2)
    extent_deg += 2.0*pad_arcsec.to("deg")

    # Call the other routine
    hdr = make_simple_header(
        ra_ctr = midpoint.ra.deg, dec_ctr = midpoint.dec.deg,
        extent_x = extent_deg, extent_y = extent_deg,
        pix_scale = pix_scale_deg)

    return(hdr)
    
def make_simple_header(
        ra_ctr = None, dec_ctr = None, pix_scale = None,
        extent_x = None, extent_y = None,
        nx = None, ny = None):
    """
    Make a simple FITS header centered on some locations.

    
    Parameters
    ----------
    pix_scale : required. Size in decimal degrees of a pixel.

    extent_x : the angular extent of the image along the x coordinate
    extent_y : the angular extent of the image along the y coordinate

    nx : the number of x pixels (not needed with extent_x and pix_scale)
    ny : the number of y pixels (not needed with extent_y and pix_scale)
    
    """

    # Check types of inputs
    if isinstance(ra_ctr, Quantity):
        ra_ctr = ra_ctr.to("deg")
    else:
        ra_ctr = ra_ctr*u.deg

    if isinstance(dec_ctr, Quantity):
        dec_ctr = dec_ctr.to("deg")
    else:
        dec_ctr = dec_ctr*u.deg

    # Only one of extent and nx can be set. Given one work out the
    # other giving priority to nx and ny
    if nx is not None and ny is not None:
        extent_x = pix_scale * nx
        extent_y = pix_scale * ny
    elif extent_x is not None and extent_y is not None:
        nx = int(np.ceil(extent_x*0.5 / pix_scale) * 2 + 1)
        ny = int(np.ceil(extent_y*0.5 / pix_scale) * 2 + 1)

    new_hdr = fits.Header()
    new_hdr['NAXIS'] = 2
    new_hdr['NAXIS1'] = nx
    new_hdr['NAXIS2'] = ny
    
    new_hdr['CTYPE1'] = 'RA---SIN'
    new_hdr['CRVAL1'] = ra_ctr.to("deg").value
    new_hdr['CRPIX1'] = np.float16((nx / 2) * 1 - 0.5)
    new_hdr['CDELT1'] = -1.0 * pix_scale.to("deg").value
    
    new_hdr['CTYPE2'] = 'DEC--SIN'
    new_hdr['CRVAL2'] = dec_ctr.to("deg").value
    new_hdr['CRPIX2'] = np.float16((ny / 2) * 1 - 0.5)
    new_hdr['CDELT2'] = 1.0 * pix_scale.to("deg").value
    
    new_hdr['EQUINOX'] = 2000.0
    new_hdr['RADESYS'] = 'FK5'

    return (new_hdr)

# ------------------------------------------------------------------------
# Coordinate handling
# ------------------------------------------------------------------------

def make_axes(
        header=None, wcs=None, naxis=None
        , ra_axis=None, dec_axis=None
        , with_units=True):
    """Accept a header, WCS object, or pair of vectors to return an RA and
    Dec image.

    """

    # Deal with a common previous-generation radio astrometry issue,
    # probably not needed but leaving for now.

    if wcs is not None:
        if 'GLS' in wcs.wcs.ctype[0]:
            wcs.wcs.ctype[0] = wcs.wcs.ctype[0].replace('GLS', 'SFL')
            print(f"Replaced GLS with SFL; ctype[0] now = {wcs.wcs.ctype[0]}")
            
    if header is not None:

        # Extract WCS
        wcs_cel = WCS(header).celestial
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2']
        
    elif wcs is not None and naxis is not None:

        # Get the relevant data from the WCS
        wcs_cel = wcs.celestial
        naxis1, naxis2 = naxis
        
    elif ra_axis is not None and dec_axis is not None:
        
        if ra.ndim == 1:
            ra_deg, dec_deg = np.broadcast_arrays(ra, dec)
        else:
            ra_deg, dec_deg = ra, dec
            if verbose:
                print("ra ndim != 1")
        if hasattr(ra, 'unit'):
            ra_deg = ra.to(u.deg).value
            dec_deg = dec.to(u.deg).value

        if with_units:
            return(ra_deg*u.deg, dec_deg*u.deg)
        else:
            return(ra_deg, dec_deg)
            
    else:
        # Throw error message
        return(None)

    # If we get here we have naxis and a WCS, proceed ...
        
    ix = np.arange(naxis1)
    iy = np.arange(naxis2).reshape(-1, 1)
    ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)

    if with_units:
        return(ra_deg*u.deg, dec_deg*u.deg)
    else:
        return(ra_deg, dec_deg)
