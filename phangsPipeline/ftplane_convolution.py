import copy
import warnings

import astropy.units as u
import astropy.utils.console as console
import numpy as np
from radio_beam import Beam
from spectral_cube import SpectralCube, VaryingResolutionSpectralCube


def ftconvolve(ImageIn, major = 1.0, minor = 1.0,
               angle = 0.0):
    NanMaskFlag = False
    nanmask = np.isnan(ImageIn)
    image = np.copy(ImageIn)
    if np.any(nanmask):
        wtimg = np.ones_like(ImageIn)
        NanMaskFlag = True
        image[nanmask] = 0.0
        wtimg[nanmask] = 0.0
        ftwtimg = np.fft.fftn(wtimg)

    ftimg = np.fft.fftn(image)

    if major == 0.0:
        sigmau = np.inf
    else:
        sigmau = 0.5/(np.pi * major)

    if minor == 0.0:
        sigmav = np.inf
    else:
        sigmav = 0.5/(np.pi * minor)

    FTPA = angle + np.pi

    a = 0.5 * (np.cos(FTPA)**2 / sigmau**2 +
               np.sin(FTPA)**2 / sigmav**2)
    c = 0.5 * (np.sin(FTPA)**2 / sigmau**2 +
               np.cos(FTPA)**2 / sigmav**2)
    b = 0.25 * np.sin(2 * FTPA) * (1.0 / sigmav**2 - 1.0 / sigmau**2)

    vv, uu = np.meshgrid(np.fft.fftfreq(ftimg.shape[0]),
                         np.fft.fftfreq(ftimg.shape[1]),
                         indexing='ij')

    FTkernel = np.exp(-a*(uu)**2 -c*(vv)**2 +2*b*(uu*vv))
    ConvolvedImage = (np.fft.ifftn(ftimg * FTkernel)).real

    if NanMaskFlag:
        ConvolvedMask = (np.fft.ifftn(ftwtimg * FTkernel)).real
        ConvolvedImage /= ConvolvedMask
        ConvolvedImage[nanmask] = np.nan
    return(ConvolvedImage)

def MakeRoundBeam(incube,
                  outfile=None,
                  overwrite=True):

    '''
    This takes a FITS file or a SpectralCube and outputs

    Parameters
    ----------
    filename : `string` or `SpectralCube`
       Input spectral cube

    Returns
    -------
    cube : `SpectralCube`

    '''
    if isinstance(incube,str):
        cube = SpectralCube.read(incube)

    if isinstance(incube, VaryingResolutionSpectralCube):
        cube = incube

    if not isinstance(cube, VaryingResolutionSpectralCube):
        warnings.warn("No information about multiple beams")
        return(None)

    beams = cube.beams
    major_axes = np.array([bm.major.to(u.deg).value for bm in beams])
    target_beamsize = np.array(major_axes.max())
    target_beam = Beam(major=target_beamsize*u.deg,
                       minor=target_beamsize*u.deg,
                       pa=0.0*u.deg)
    print("Target beam is : {}".format(target_beam))

    # Let's assume square pixels
    pixsize = cube.wcs.pixel_scale_matrix[1,1]
    fwhm2sigma = np.sqrt(8*np.log(2))

    output = np.zeros(cube.shape)

    with console.ProgressBar(cube.shape[0]) as bar:

        for ii, plane in enumerate(cube.filled_data[:]):
            this_beam = beams[ii]
            conv_beam = target_beam - this_beam

            majpix = conv_beam.major.value / pixsize / fwhm2sigma
            minpix = conv_beam.minor.value / pixsize / fwhm2sigma

            output[ii,:,:] = ftconvolve(plane,
                                        major = majpix,
                                        minor = minpix,
                                        angle = conv_beam.pa.value)

            bar.update()

    hdr = copy.copy(cube.header)
    hdr['CASAMBM'] = False
    hdr['BMAJ'] = float(target_beam.major.value)
    hdr['BMIN'] = float(target_beam.major.value)
    hdr['BPA'] = 0.0
    outcube = SpectralCube(output, cube.wcs, header=hdr)
    if outfile:
        outcube.write(outfile,overwrite=overwrite)
        return None
    return(outcube)


