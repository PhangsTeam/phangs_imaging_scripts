
# Example script demonstrating how to create and parallelize jobs on a cluster

This example demonstrates how to deploy individual targets with a scheduler to complete
the staging and imaging with the PHANGS pipeline.

1. Run `create_imaging_job_config_files.py` to produce config files for each target.
2. Adjust the number of jobs based on the config files in `jobarray_field_line_staging_imaging_job.sh`. Submit the job array.
3. Proceed using the normal postprocessing step in `run_casa_pipeline_phangs-alma.py` to linearly mosaic the targets.
