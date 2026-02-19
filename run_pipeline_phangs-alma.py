import sys

from casatasks import casalog

# Add analysisUtils to the path. Make sure to set this to where you have analysisUtils downloaded!
au_path = "path/to/analysis_scripts"
sys.path.append(au_path)

import phangsPipeline as ppl
from phangsPipeline.check_imports import is_casa_installed

casa_enabled = is_casa_installed()

# YOU SHOULD EDIT THINGS BELOW THIS #

# Path to your master key
master_key_file = "path/to/master_key.txt"

# Steps to run
do_singledish = False
do_staging = True
do_imaging = True
do_postprocess = True
do_derived = True
do_release = False

# Targets to process
targets = [
    "some_exciting_galaxy",
]

line_products = [
    "a_thrilling_line",
]
interf_configs = [
    "7m",
    "12m",
]
feather_configs = [
    "7m+tp",
    '12m+7m+tp',
]

no_cont = True

imaging_method = "tclean"

# Switches for derived products
do_convolve = True
do_noise = True
do_strictmask = True
do_broadmask = True
do_moments = True
do_secondary = True

# new DR5 routines (shuffling and flat maps)
do_vfield = False  # creates a velocity field for shuffling
do_shuffling = True  # runs independently from other tasks
do_flatmask = True  # requires noise and broad/strict masks to join
do_flatmaps = True  # requires flat masks

# You should not need to edit below here

# Setup logger
ppl.setup_logger(level="DEBUG", logfile=None)
casalog.filter("INFO")
casalog.showconsole(True)

# Initialize the KeyHandler, which reads the master key and the files linked in the
# master key.
key_handler = ppl.KeyHandler(master_key=master_key_file)

# Initialise other handlers
sd_handler = None
uv_handler = None
im_handler = None
pp_handler = None
derived_handler = None
release_handler = None

if casa_enabled:
    if do_singledish:
        sd_handler = ppl.SingleDishHandler(key_handler=key_handler)
        sd_handler.set_targets(only=targets)
        sd_handler.set_line_products(only=line_products)
        sd_handler.set_no_cont_products(no_cont)
    if do_staging:
        uv_handler = ppl.VisHandler(key_handler=key_handler)
        uv_handler.set_targets(only=targets)
        uv_handler.set_interf_configs(only=interf_configs)
        uv_handler.set_line_products(only=line_products)
        uv_handler.set_no_cont_products(no_cont)
    if do_imaging:
        im_handler = ppl.ImagingHandler(key_handler=key_handler)
        im_handler.set_targets(only=targets)
        im_handler.set_interf_configs(only=interf_configs)
        im_handler.set_line_products(only=line_products)
        im_handler.set_no_cont_products(no_cont)
    if do_postprocess:
        pp_handler = ppl.PostProcessHandler(key_handler=key_handler)
        pp_handler.set_targets(only=targets)
        pp_handler.set_interf_configs(only=interf_configs)
        pp_handler.set_line_products(only=line_products)
        pp_handler.set_feather_configs(only=feather_configs)
        pp_handler.set_no_cont_products(no_cont)
if do_derived:
    derived_handler = ppl.DerivedHandler(key_handler=key_handler)
    derived_handler.set_targets(only=targets)
    derived_handler.set_interf_configs(only=interf_configs)
    derived_handler.set_feather_configs(only=feather_configs)
    derived_handler.set_line_products(only=line_products)
    derived_handler.set_no_cont_products(no_cont)
if do_release:
    release_handler = ppl.ReleaseHandler(key_handler=key_handler)
    release_handler.set_targets(only=targets)
    release_handler.set_interf_configs(only=interf_configs)
    release_handler.set_feather_configs(only=feather_configs)
    release_handler.set_line_products(only=line_products)
    release_handler.set_no_cont_products(no_cont)

# Run things
key_handler.make_missing_directories(
    imaging=do_staging,
    postprocess=do_postprocess,
    derived=do_derived,
    release=do_release,
)

if casa_enabled:
    ##############################################################################
    # Run singledish pipeline
    ##############################################################################
    # Run TP data through the pipeline, from calibration to imaging.

    if do_singledish:
        sd_handler.loop_singledish(do_all=True)

    ##############################################################################
    # Run staging
    ##############################################################################

    # "Stage" the visibility data. This involves copying the original
    # calibrated measurement set, continuum subtracting (if requested),
    # extraction of requested lines and continuum data, regridding and
    # concatenation into a single measurement set. The overwrite=True flag
    # is needed to ensure that previous runs can be overwritten.

    if do_staging:
        uv_handler.loop_stage_uvdata(
            do_copy=True,
            do_contsub=True,
            do_extract_line=False,
            do_extract_cont=False,
            require_full_line_coverage=True,
            do_remove_staging=False,
            overwrite=True,
        )

        uv_handler.loop_stage_uvdata(
            do_copy=False,
            do_contsub=False,
            do_extract_line=True,
            do_extract_cont=False,
            require_full_line_coverage=True,
            do_remove_staging=False,
            overwrite=True,
        )

        uv_handler.loop_stage_uvdata(
            do_copy=False,
            do_contsub=False,
            do_extract_line=False,
            do_extract_cont=True,
            require_full_line_coverage=True,
            do_remove_staging=False,
            overwrite=True,
        )

        uv_handler.loop_stage_uvdata(
            do_copy=False,
            do_contsub=False,
            do_extract_line=False,
            do_extract_cont=False,
            require_full_line_coverage=True,
            do_remove_staging=True,
            overwrite=True,
        )

    ##############################################################################
    # Step through imaging
    ##############################################################################

    # Image the concatenated, regridded visibility data. The full loop
    # involves applying any user-supplied clean mask, multiscale imaging,
    # mask generation for the single scale clean, and single scale
    # clean. The individual parts can be turned on or off with flags to
    # the imaging loop call but this call does everything.

    if do_imaging:
        high_snr = 4.0
        low_snr = 2.0
        absolute = True

        convergence_fracflux = 0.01
        singlescale_threshold_value = 1

        im_handler.loop_imaging(
            imaging_method=imaging_method,
            do_dirty_image=True,
            do_revert_to_dirty=True,
            do_read_clean_mask=True,
            do_multiscale_clean=True,
            do_revert_to_multiscale=True,
            do_singlescale_mask=True,
            singlescale_mask_absolute=absolute,
            singlescale_mask_high_snr=high_snr,
            singlescale_mask_low_snr=low_snr,
            do_singlescale_clean=True,
            do_revert_to_singlescale=True,
            convergence_fracflux=convergence_fracflux,
            singlescale_threshold_value=singlescale_threshold_value,
            do_export_to_fits=True,
            export_dirty=False,
            export_multiscale=False,
            overwrite=True,
        )

    ##############################################################################
    # Step through postprocessing
    ##############################################################################

    # Postprocess the data in CASA after imaging. This involves primary
    # beam correction, linear mosaicking, feathering, conversion to Kelvin
    # units, and some downsampling to save space.

    if do_postprocess:
        pp_handler.loop_postprocess(
            do_prep=True,
            do_feather=True,
            feather_before_mosaic=True,
            do_mosaic=True,
            do_cleanup=True,
            imaging_method=imaging_method,
        )

##############################################################################
# Step through derived product creation
##############################################################################

# Run the calculations requested by the user. The steps are annotated
# here, but in general, do not change anything below this line. Just
# use the flags above to steer the calculation.
# 1) Convolve the post-processed data products to the various angular and
# physical resolutions specified in the keys.
# 2) Estimate the noise from the signal-free regions of the data to
# produce a three-dimensional noise model for each cube.
# 3) Construct "strict masks" for each cube at each resolution.
# 4) Combine the strict masks across all linked resolutions to form
# "broad masks" that have high completeness.
# 5) Apply the masks and use the cubes and noise models to produce moment
# maps with associated uncertainty.
# 6) Run a second round of moment calculations. This enables calculation
# of moments that depend on other, earlier moment map calculations
# 7) Create velocity field. This creates a combined velocity field from
# a list of derived mom1 maps.
# 8) Create shuffled cubes. This shuffles the processed and derived cubes
# by a velocity offset defined by an input velocity field.
# 9) Construct flat masks. This combines the existing signal masks
# with a velocity slab based on a input velocity field and velocity
# window.
# 10) Produce flat moment-0 maps. This uses the flat masks to create
# moment-0 maps.

if do_derived:
    derived_handler.loop_derive_products(
        do_convolve=do_convolve,
        do_noise=do_noise,
        do_strictmask=do_strictmask,
        do_broadmask=do_broadmask,
        do_moments=do_moments,
        do_secondary=do_secondary,
        do_vfield=do_vfield,
        do_shuffling=do_shuffling,
        do_flatmask=do_flatmask,
        do_flatmaps=do_flatmaps,
        overwrite=True,
    )

if do_release:
    release_handler.loop_build_release()

print("Complete!")
