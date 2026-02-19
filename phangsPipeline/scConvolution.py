import logging

import astropy.units as u
import numpy as np
from astropy.convolution import Box1DKernel
from astropy.io import fits
from radio_beam import Beam
from spectral_cube import SpectralCube, LazyMask, Projection

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def coverage_collapser(coveragecube,
                       coverage2dfile=None,
                       overwrite=False):
    coverage2d = coveragecube.sum(axis=0)
    coverage2darray = (np.array(coverage2d, dtype=np.float32)
                       / coveragecube.shape[0])
    hdr = coverage2d.header
    hdr['DATAMIN'] = np.nanmin(coverage2darray)
    hdr['DATAMAX'] = np.nanmax(coverage2darray)
    hdu = fits.PrimaryHDU(coverage2darray, hdr)
    hdu.writeto(coverage2dfile, overwrite=overwrite)


def smooth_cube(
        incube=None,
        outfile=None,
        angular_resolution=None,
        linear_resolution=None,
        distance=None,
        velocity_resolution=None,
        nan_treatment='interpolate', # can also be 'fill'
        tol=None,
        make_coverage_cube=False,
        collapse_coverage=False,
        coveragefile=None,
        coverage2dfile=None,
        dtype=np.float32,
        overwrite=True
    ):
    """
    Smooth an input cube to coarser angular or spectral
    resolution. This lightly wraps spectral cube and some of the error
    checking is left to that.

    tol is a fraction. When the target beam is within tol of the
    original beam, we just copy.

    Optionally, also calculate a coverage footprint in which original
    (finite) cube coverage starts at 1.0 and the output cube shows the
    fraction of finite pixels.
    """

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Error checking
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # Require a valid cube or map input
    twod = False
    if type(incube) is SpectralCube:
        cube = incube
    elif type(incube) == type("hello"):
        hdulist = fits.open(incube)
        if hdulist[0].header['NAXIS'] == 2:
            cube = Projection.from_hdu(hdulist)
            twod = True
        else:
            cube = SpectralCube.read(incube)
    else:
        logger.error("Input must be a SpectralCube object or a filename.")

    # Allow huge operations. If the speed or segfaults become a huge
    # problem, we will adjust our strategy here.

    cube.allow_huge_operations = True

    # Check that only one target scale is set
    if (angular_resolution is not None) and (linear_resolution is not None):
        logger.error('Only one of angular_resolution or ',
                     'linear_resolution can be set')
        return(None)

    # Work out the target angular resolution
    if angular_resolution is not None:
        if type(angular_resolution) is str:
            angular_resolution = u.Quantity(angular_resolution)

    if linear_resolution is not None:
        if distance is None:
            logger.error('Convolution to linear resolution requires a distance.')
            return(None)

        if type(distance) is str:
            distance = u.Quantity(distance)
        if type(linear_resolution) is str:
            linear_resolution = u.Quantity(linear_resolution)
        angular_resolution = (linear_resolution / distance * u.rad).to(u.arcsec)
        dist_mpc_val = float(distance.to(u.pc).value) / 1e6
        cube._header.append(('DIST_MPC',dist_mpc_val,'Used in convolution'))

    if tol is None:
        tol = 0.0

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Convolution to coarser beam
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    
    if angular_resolution is not None:
        logger.info("... convolving from beam: "+str(cube.beam))
        target_beam = Beam(major=angular_resolution,
                           minor=angular_resolution,
                           pa=0 * u.deg)
        logger.info("... convolving to beam: "+str(target_beam))

        new_major = float(target_beam.major.to(u.arcsec).value)
        old_major = float(cube.beam.major.to(u.arcsec).value)        
        delta = (new_major-old_major)/old_major

        logger.info("... fractional change: "+str(delta))
        
        if make_coverage_cube:
            if twod:
                coverage = Projection(np.isfinite(hdulist[0].data)*1.0,
                                      wcs=cube.wcs.celestial, header=cube.header,
                                      beam=cube.beam)
            else:
                coverage = SpectralCube(
                    np.isfinite(cube.unmasked_data[:])*1.0,
                    wcs=cube.wcs,
                    header=cube.header,
                    meta={'BUNIT': ' ', 'BTYPE': 'Coverage'})
                coverage = \
                    coverage.with_mask(LazyMask(np.isfinite,cube=coverage))            
            
            # Allow huge operations. If the speed or segfaults become a huge
            # problem, we will adjust our strategy here.

            coverage.allow_huge_operations = True

        if delta > tol:
            logger.info("... proceeding with convolution.")
            if twod:
                cube = cube.convolve_to(target_beam,
                                        nan_treatment=nan_treatment,
                                        allow_huge=True)
            else: 
                cube = cube.convolve_to(target_beam,
                                        nan_treatment=nan_treatment)
            if make_coverage_cube:
                if twod:
                    coverage = coverage.convolve_to(target_beam,
                                                    nan_treatment=nan_treatment,
                                                    allow_huge=True)
                else:
                    coverage = coverage.convolve_to(target_beam,
                                                    nan_treatment=nan_treatment)

        if np.abs(delta) < tol:
            logger.info("... current resolution meets tolerance.")

        if delta < -1.0*tol:
            logger.info("... resolution cannot be matched. Returning")
            return(None)

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Spectral convolution
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    # This is only a boxcar smooth right now and does not downsample
    # or update the header.

    if velocity_resolution is not None and twod == False:
        if type(velocity_resolution) is str:
            velocity_resolution = u.Quantity(velocity_resolution)

        dv = scdr.channel_width(cube)
        nChan = (velocity_resolution / dv).to(u.dimensionless_unscaled).value
        if nChan > 1:
            cube = cube.spectral_smooth(Box1DKernel(nChan))
            if make_coverage_cube:
                coverage = coverage.spectral_smooth(Box1DKernel(nChan))

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Write or return as requested
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    if outfile is not None:
        # cube.write(outfile, overwrite=overwrite)
        hdu = fits.PrimaryHDU(np.array(cube.filled_data[:], dtype=dtype),
                              header=cube.header)
        hdu.writeto(outfile, overwrite=overwrite)
        if make_coverage_cube:
            if coveragefile is not None:
                hdu = fits.PrimaryHDU(np.array(coverage.filled_data[:], dtype=dtype),
                                      header=coverage.header)
                hdu.writeto(coveragefile, overwrite=overwrite)
            if collapse_coverage and twod==False:
                if coveragefile and not coverage2dfile:
                    coverage2dfile = coveragefile.replace('.fits','2d.fits')
                coverage_collapser(coverage,
                                   coverage2dfile=coverage2dfile,
                                   overwrite=overwrite)
                # coverage.write(coveragefile, overwrite=overwrite)

    return(cube)
