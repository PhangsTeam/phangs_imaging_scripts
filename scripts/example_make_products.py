# This script demonstrates generation of a "PHANGS product suite" for
# a data set that doesn't have a key file suite set up.

from spectral_cube import SpectralCube, BooleanArrayMask
from astropy.io import fits
from astropy.wcs import wcs
import scipy.ndimage as nd
import numpy as np
import scConvolution as scc
import scNoiseRoutines as scn
import scMaskingRoutines as scm
import scStackingRoutines as scs
import scMoments as sco
from astropy import units as u

# region Definitions/tuning

# ####################################################
# Definitions/tuning
# ####################################################

working_dir = '../working_data/'
vfield_dir = '../velocity_fields/'

infile = 'ngc4321_12m+7m+tp_co21.fits'

dist = "15.21Mpc"
target_angular = ['8arcsec','15arcsec','21arcsec']
target_physical = ['150pc','500pc','1000pc','1500pc']

# velocity vfield map used for shuffling
vfield_infile = 'ngc4321_vfield.fits'
window = '50km/s' # velocity window for masking

# Use boolean flags to set the steps to be performed when the pipeline
# is called. See descriptions below (but only edit here).

do_convolve = True
do_noise = True
do_strictmask = True
do_broadmask = True
do_moments = True
# new DR5 routines for shuffling and flat maps
do_vfield = False # run this if you don't have a velocity field
do_shuffling = True
do_flatmask = True
do_flatmaps = True

# endregion

# region Convolution

# ####################################################
# Convolution
# ####################################################

if do_convolve:

    this_infile = working_dir+infile

    # angular resolutions
    for this_ext in target_angular:
        print("Convolution to "+this_ext)
        this_outfile = this_infile.replace('.fits','_'+this_ext+'.fits')
        scc.smooth_cube(
            incube=this_infile,
            outfile=this_outfile,
            angular_resolution=this_ext,
            linear_resolution=None,
            distance=None,
            velocity_resolution=None,
            nan_treatment='interpolate',
            tol=None,
            make_coverage_cube=True,
            collapse_coverage=True,
            coveragefile=this_outfile.replace('.fits','_coverage.fits'),
            coverage2dfile=this_outfile.replace('.fits','_coverage2d.fits'),
            dtype=np.float32,
            overwrite=True
        )

    # physical resolutions
    for this_ext in target_physical:
        print("Convolution to "+this_ext)
        this_outfile = this_infile.replace('.fits','_'+this_ext+'.fits')
        scc.smooth_cube(
            incube=this_infile,
            outfile=this_outfile,
            angular_resolution=None,
            linear_resolution=this_ext,
            distance=dist,
            velocity_resolution=None,
            nan_treatment='interpolate',
            tol=None,
            make_coverage_cube=True,
            collapse_coverage=True,
            coveragefile=this_outfile.replace('.fits','_coverage.fits'),
            coverage2dfile=this_outfile.replace('.fits','_coverage2d.fits'),
            dtype=np.float32,
            overwrite=True
        )

# endregion

# region Noise Estimation

# ####################################################
# Noise Estimation
# ####################################################

if do_noise:
        
    # native resolution
    this_base_infile = working_dir+infile
    print("Noise native res for ", this_base_infile)
    scn.recipe_phangs_noise(
        incube=this_base_infile,
        outfile=this_base_infile.replace('.fits','_noise.fits'),
        overwrite=True)
        
    # angular resolutions
    for this_ext in target_angular:
        this_infile = this_base_infile.replace('.fits','_'+this_ext+'.fits')
        print("Noise for "+this_ext, this_infile)
        scn.recipe_phangs_noise(
            incube=this_infile,
            outfile=this_infile.replace('.fits','_noise.fits'),
            overwrite=True)

    # physical resolutions
    for this_ext in target_physical:
        this_infile = this_base_infile.replace('.fits','_'+this_ext+'.fits')
        print("Noise for "+this_ext, this_infile)
        scn.recipe_phangs_noise(
            incube=this_infile,
            outfile=this_infile.replace('.fits','_noise.fits'),
            overwrite=True)

# endregion

# region Strict Mask Creation

# ####################################################
# Strict Mask Creation
# ####################################################

if do_strictmask:

    mask_kwargs = {'hi_thresh':5.0, 'lo_thresh':2.0, 
                'hi_nchan':2, 'lo_nchan':2}

    this_base_infile = working_dir+infile

    # native resolution
    this_infile = this_base_infile
    print("Strict mask for ", this_infile)
    scm.recipe_phangs_strict_mask(
        this_infile, 
        this_infile.replace('.fits','_noise.fits'),
        outfile=this_infile.replace('.fits','_strictmask.fits'),
        mask_kwargs=mask_kwargs,
        overwrite=True)

    # angular resolutions
    for this_ext in target_angular:
        this_infile = this_base_infile.replace('.fits','_'+this_ext+'.fits')
        print("Strict mask for ", this_infile)
        scm.recipe_phangs_strict_mask(
            this_infile,
            this_infile.replace('.fits','_noise.fits'),
            outfile=this_infile.replace('.fits','_strictmask.fits'),
            mask_kwargs=mask_kwargs,
            overwrite=True)

    # physical resolutions
    for this_ext in target_physical:
        this_infile = this_base_infile.replace('.fits','_'+this_ext+'.fits')
        print("Strict mask for ", this_infile)
        scm.recipe_phangs_strict_mask(
            this_infile, 
            this_infile.replace('.fits','_noise.fits'),
            outfile=this_infile.replace('.fits','_strictmask.fits'),
            mask_kwargs=mask_kwargs,
            overwrite=True)
    
# endregion

# region Broad Mask Creation

# ####################################################
# Broad Mask Creation
# ####################################################

if do_broadmask:

    this_base_infile = working_dir+infile
    template_mask = this_base_infile.replace('.fits','_strictmask.fits')
    broad_mask_file = this_base_infile.replace('.fits','_broadmask.fits')

    list_of_masks = [template_mask]

    for this_ext in target_angular:
        this_infile = this_base_infile.replace('.fits','_'+this_ext+'.fits')
        this_mask = this_infile.replace('.fits','_strictmask.fits')
        list_of_masks.append(this_mask)
        
    for this_ext in target_physical:
        this_infile = this_base_infile.replace('.fits','_'+this_ext+'.fits')
        this_mask = this_infile.replace('.fits','_strictmask.fits')
        list_of_masks.append(this_mask)
        
    scm.recipe_phangs_broad_mask(
        template_mask, 
        outfile=broad_mask_file, 
        list_of_masks = list_of_masks,
        grow_xy = 2, 
        grow_v = 2,
        return_spectral_cube=False, overwrite=True,
        recipe='anyscale', fraction_of_scales=0.25)

# endregion

# region Moments

# ####################################################
# Moments
# ####################################################

if do_moments:

    mom_list = {
        'strictmom0':{
            'algorithm':'mom0',
            'mask':'_strictmask.fits',
            'ext':'_strict_mom0.fits',
            'ext_error':'_strict_emom0.fits',
            'kwargs':{},
            },
        'broadmom0':{
            'algorithm':'mom0',
            'mask':'_broadmask.fits',
            'ext':'_broad_mom0.fits',
            'ext_error':'_broad_emom0.fits',
            'kwargs':{},
            },
        'strictmom1':{
            'algorithm':'mom1',
            'mask':'_strictmask.fits',
            'ext':'_strict_mom1.fits',
            'ext_error':'_strict_emom1.fits',
            'kwargs':{},
            },
        'broadmom1':{
            'algorithm':'mom1',
            'mask':'_broadmask.fits',
            'ext':'_broad_mom1.fits',
            'ext_error':'_broad_emom1.fits',
            'kwargs':{},
            },
        'strictmom2':{
            'algorithm':'mom2',
            'mask':'_strictmask.fits',
            'ext':'_strict_mom2.fits',
            'ext_error':'_strict_emom2.fits',
            'kwargs':{},
            },
        'strictew':{
            'algorithm':'ew',
            'mask':'_strictmask.fits',
            'ext':'_strict_ew.fits',
            'ext_error':'_strict_eew.fits',
            'kwargs':{},
            },
        'tpeak':{
            'algorithm':'tpeak',
            'mask':'_broadmask.fits',
            'ext':'_broad_tpeak.fits',
            'ext_error':'_broad_etpeak.fits',
            'kwargs':{},
            },
        }

    # native resolution
    this_base_infile = working_dir+infile
    this_infile = this_base_infile
    print("Moments for ", this_infile)

    for this_mom_key in mom_list:      
        this_mom = mom_list[this_mom_key]
        if this_mom['mask'] == '_broadmask.fits':
            this_mask_file = this_base_infile.replace('.fits',this_mom['mask'])
        else:
            this_mask_file = this_infile.replace('.fits',this_mom['mask'])
        this_noise_file = this_infile.replace('.fits','_noise.fits')
        this_out_file = this_infile.replace('.fits',this_mom['ext'])
        this_error_file = this_infile.replace('.fits',this_mom['ext_error'])
        sco.moment_generator(
            this_infile,
            mask=this_mask_file, noise=this_noise_file,
            moment=this_mom['algorithm'], momkwargs=this_mom['kwargs'],
            outfile=this_out_file, errorfile=this_error_file,
            channel_correlation=None,
        )

    # angular resolutions
    for this_ext in target_angular:
        this_infile = this_base_infile.replace('.fits','_'+this_ext+'.fits')
        for this_mom_key in mom_list:      
            this_mom = mom_list[this_mom_key]
            if this_mom['mask'] == '_broadmask.fits':
                this_mask_file = this_base_infile.replace('.fits',this_mom['mask'])
            else:
                this_mask_file = this_infile.replace('.fits',this_mom['mask'])
            this_noise_file = this_infile.replace('.fits','_noise.fits')
            this_out_file = this_infile.replace('.fits',this_mom['ext'])
            this_error_file = this_infile.replace('.fits',this_mom['ext_error'])
            sco.moment_generator(
                this_infile,
                mask=this_mask_file, noise=this_noise_file,
                moment=this_mom['algorithm'], momkwargs=this_mom['kwargs'],
                outfile=this_out_file, errorfile=this_error_file,
                channel_correlation=None,
            )

    # physical resolutions
    for this_ext in target_physical:
        this_infile = this_base_infile.replace('.fits','_'+this_ext+'.fits')
        for this_mom_key in mom_list:      
            this_mom = mom_list[this_mom_key]
            if this_mom['mask'] == '_broadmask.fits':
                this_mask_file = this_base_infile.replace('.fits',this_mom['mask'])
            else:
                this_mask_file = this_infile.replace('.fits',this_mom['mask'])
            this_noise_file = this_infile.replace('.fits','_noise.fits')
            this_out_file = this_infile.replace('.fits',this_mom['ext'])
            this_error_file = this_infile.replace('.fits',this_mom['ext_error'])
            sco.moment_generator(
                this_infile,
                mask=this_mask_file, noise=this_noise_file,
                moment=this_mom['algorithm'], momkwargs=this_mom['kwargs'],
                outfile=this_out_file, errorfile=this_error_file,
                channel_correlation=None,
            )
            
# endregion

# region Velocity Field

# ####################################################
# Velocity Field
# ####################################################

if do_vfield:

    this_base_infile = working_dir+infile
    template_vfield = this_base_infile.replace('.fits','_strict_mom1.fits')
    vfield_file = this_base_infile.replace('.fits','_vfield.fits')

    list_of_vfields = [template_vfield]

    for this_ext in target_angular:
        this_infile = this_base_infile.replace('.fits','_'+this_ext+'.fits')
        this_vfield = this_infile.replace('.fits','_strict_mom1.fits')
        list_of_vfields.append(this_vfield)
        
    for this_ext in target_physical:
        this_infile = this_base_infile.replace('.fits','_'+this_ext+'.fits')
        this_vfield = this_infile.replace('.fits','_strict_mom1.fits')
        list_of_vfields.append(this_vfield)
        
    scs.recipe_phangs_vfield(
        template_vfield, 
        outfile=vfield_file, 
        list_of_vfields = list_of_vfields,
        overwrite=True)
       
    
# endregion

# region Shuffling

# ####################################################
# Shuffling
# ####################################################

if do_shuffling:

    # native resolution
    this_base_infile = working_dir+infile
    this_vfield_infile = vfield_dir+vfield_infile
    print("Shuffling ", this_base_infile)
    scs.recipe_shuffle_cube(
        incube=this_base_infile,
        invfield=this_vfield_infile,
        outfile=this_base_infile.replace('.fits','_shuffled.fits'),
        overwrite=True)

# endregion


# region Flat Mask Creation

# ####################################################
# Flat Mask Creation
# ####################################################


if do_flatmask:

    mask_kwargs = {'window':window}
    this_base_infile = working_dir+infile
    this_vfield_infile = vfield_dir+vfield_infile

    # native resolution
    this_infile = this_base_infile

    # just flat mask, with no join    
    print("Flat only mask for ", this_infile)
    scm.recipe_phangs_flat_mask(
        cube_in=this_infile, 
        vfield_in=this_vfield_infile, 
        mask_in=None, 
        outfile=this_infile.replace('.fits','_flatstrictmask.fits'),
        mask_kwargs=mask_kwargs,
        return_spectral_cube=False,
        overwrite=True)
    # strict version
    this_mask_in = this_infile.replace('.fits','_strictmask.fits')
    print("Flat strict mask for ", this_infile)
    scm.recipe_phangs_flat_mask(
        cube_in=this_infile, 
        vfield_in=this_vfield_infile, 
        mask_in=this_mask_in, 
        outfile=this_infile.replace('.fits','_flatstrictmask.fits'),
        mask_kwargs=mask_kwargs,
        return_spectral_cube=False,
        overwrite=True)
    # broad version
    this_mask_in = this_infile.replace('.fits','_broadmask.fits')
    print("Flat broad mask for ", this_infile)
    scm.recipe_phangs_flat_mask(
        cube_in=this_infile, 
        vfield_in=this_vfield_infile, 
        mask_in=this_mask_in, 
        outfile=this_infile.replace('.fits','_flatbroadmask.fits'),
        mask_kwargs=mask_kwargs,
        return_spectral_cube=False,
        overwrite=True)
    
    # angular resolutions
    for this_ext in target_angular:
        this_infile = this_base_infile.replace('.fits','_'+this_ext+'.fits')
        # just flat mask, with no join    
        print("Flat only mask for ", this_infile)
        scm.recipe_phangs_flat_mask(
            cube_in=this_infile, 
            vfield_in=this_vfield_infile, 
            mask_in=None, 
            outfile=this_infile.replace('.fits','_flatmask.fits'),
            mask_kwargs=mask_kwargs,
            return_spectral_cube=False,
            overwrite=True)
        # strict version
        this_mask_in = this_infile.replace('.fits','_strictmask.fits')
        print("Flat strict mask for ", this_infile)
        scm.recipe_phangs_flat_mask(
            cube_in=this_infile, 
            vfield_in=this_vfield_infile, 
            mask_in=this_mask_in, 
            outfile=this_infile.replace('.fits','_flatstrictmask.fits'),
            mask_kwargs=mask_kwargs,
            return_spectral_cube=False,
            overwrite=True)
        # broad version
        this_mask_in = this_base_infile.replace('.fits','_broadmask.fits')
        print("Flat broad mask for ", this_infile)
        scm.recipe_phangs_flat_mask(
            cube_in=this_infile, 
            vfield_in=this_vfield_infile, 
            mask_in=this_mask_in, 
            outfile=this_infile.replace('.fits','_flatbroadmask.fits'),
            mask_kwargs=mask_kwargs,
            return_spectral_cube=False,
            overwrite=True)
        
    # physical resolutions
    for this_ext in target_physical:
        this_infile = this_base_infile.replace('.fits','_'+this_ext+'.fits')
        # just flat mask, with no join    
        print("Flat only mask for ", this_infile)
        scm.recipe_phangs_flat_mask(
            cube_in=this_infile, 
            vfield_in=this_vfield_infile, 
            mask_in=None, 
            outfile=this_infile.replace('.fits','_flatmask.fits'),
            mask_kwargs=mask_kwargs,
            return_spectral_cube=False,
            overwrite=True)
        # strict version
        this_mask_in = this_infile.replace('.fits','_strictmask.fits')
        print("Flat strict mask for ", this_infile)
        scm.recipe_phangs_flat_mask(
            cube_in=this_infile, 
            vfield_in=this_vfield_infile, 
            mask_in=this_mask_in, 
            outfile=this_infile.replace('.fits','_flatstrictmask.fits'),
            mask_kwargs=mask_kwargs,
            return_spectral_cube=False,
            overwrite=True)
        # broad version
        this_mask_in = this_base_infile.replace('.fits','_broadmask.fits')
        print("Flat broad mask for ", this_infile)
        scm.recipe_phangs_flat_mask(
            cube_in=this_infile, 
            vfield_in=this_vfield_infile, 
            mask_in=this_mask_in, 
            outfile=this_infile.replace('.fits','_flatbroadmask.fits'),
            mask_kwargs=mask_kwargs,
            return_spectral_cube=False,
            overwrite=True)

# endregion

# region Flat Maps

# ####################################################
# Flat Maps
# ####################################################

if do_flatmaps:

    mom_list = {
        'flatstrictmom0':
            {
            'algorithm':'mom0',
            'mask':'_flatstrictmask.fits',
            'ext':'_flatstrict_mom0.fits',
            'ext_error':'_flatstrict_emom0.fits',
            'kwargs':{},
            },
        'flatbroadmom0':
            {
            'algorithm':'mom0',
            'mask':'_flatbroadmask.fits',
            'ext':'_flatbroad_mom0.fits',
            'ext_error':'_flatbroad_emom0.fits',
            'kwargs':{},
            },
        }

    # native resolution
    this_base_infile = working_dir+infile
    this_infile = this_base_infile
    print("Flat maps for ", this_infile)

    for this_mom_key in mom_list:      
        this_mom = mom_list[this_mom_key]
        this_mask_file = this_infile.replace('.fits',this_mom['mask'])
        this_noise_file = this_infile.replace('.fits','_noise.fits')
        this_out_file = this_infile.replace('.fits',this_mom['ext'])
        this_error_file = this_infile.replace('.fits',this_mom['ext_error'])
        sco.moment_generator(
            this_infile,
            mask=this_mask_file, noise=this_noise_file,
            moment=this_mom['algorithm'], momkwargs=this_mom['kwargs'],
            outfile=this_out_file, errorfile=this_error_file,
            channel_correlation=None,
        )

    # angular resolutions
    for this_ext in target_angular:
        this_infile = this_base_infile.replace('.fits','_'+this_ext+'.fits')
        for this_mom_key in mom_list:      
            this_mom = mom_list[this_mom_key]
            this_mask_file = this_infile.replace('.fits',this_mom['mask'])
            this_noise_file = this_infile.replace('.fits','_noise.fits')
            this_out_file = this_infile.replace('.fits',this_mom['ext'])
            this_error_file = this_infile.replace('.fits',this_mom['ext_error'])
            sco.moment_generator(
                this_infile,
                mask=this_mask_file, noise=this_noise_file,
                moment=this_mom['algorithm'], momkwargs=this_mom['kwargs'],
                outfile=this_out_file, errorfile=this_error_file,
                channel_correlation=None,
            )

    # physical resolutions
    for this_ext in target_physical:
        this_infile = this_base_infile.replace('.fits','_'+this_ext+'.fits')
        for this_mom_key in mom_list:      
            this_mom = mom_list[this_mom_key]
            this_mask_file = this_infile.replace('.fits',this_mom['mask'])
            this_noise_file = this_infile.replace('.fits','_noise.fits')
            this_out_file = this_infile.replace('.fits',this_mom['ext'])
            this_error_file = this_infile.replace('.fits',this_mom['ext_error'])
            sco.moment_generator(
                this_infile,
                mask=this_mask_file, noise=this_noise_file,
                moment=this_mom['algorithm'], momkwargs=this_mom['kwargs'],
                outfile=this_out_file, errorfile=this_error_file,
                channel_correlation=None,
            )
            
# endregion