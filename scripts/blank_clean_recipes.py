# This is a very simple script that will generate blank
# clean input files. These can be used as the basis of clean recipes.

# The main use is to generate a new complete set of default-value
# clean fields for use building a clean call when a new version of
# CASA comes out. The resulting clean recipes can be fed as input to
# the CleanCall object for pipeline imaging.

default(tclean)
specmode='mfs'
deconvolver='mtmfs'
gridder='mosaic'
weighting='briggs'
niter=1
saveinputs(outfile='continuum_mosaic.clean')

default(tclean)
specmode='cube'
deconvolver='multiscale'
gridder='mosaic'
weighting='briggs'
niter=1
saveinputs(outfile='cube_mosaic_multiscale.clean')
