
## Minimal example showing how to image a single target in chunks in a cluster

1. Prior to running the imaging scripts, stage the uv-data using the normal approach with the pipeline in `run_casa_pipeline_phangs-alma.py``
2. Specify the target, config, and line name, and the number of channels per chunk via `chunksize`. Update these in all the python scripts.
3. Run `run_casa_find_nchunks.py` to find the number of chunks that imaging will be split into based on the given `chunksize`.
4. Update the number of jobs in `jobarray_imaging_per_chunk.sh` to match the number of chunks. Adjust the requested resources per job. Submit the job array.
5. When all chunks are complete, submit `job_gather_chunks_to_cube.sh` to gather the sub-cubes together and run postprocessing.
6. Proceed with the derived products creation in `run_derived_pipeline_phangs-alma.py`.


Note that `chunksize` must currently be larger than 1.

Note that the clean thresholds may moderately differ between each chunk as the noise
and cleaning heuristics are performed on each chunk. This is unlikely to have a measurable
effect on the final results for spectral lines with similar spectral noise properties and
where the noise is sensitivity limited (not dynamic range limited).
