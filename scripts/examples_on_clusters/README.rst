
Scripts in this folder provide examples on deploying the PHANGS pipeline
in a cluster environment.

- `imaging_per_target` demonstrates how to stage and image each target as a job array. The example scripts show how 173 seperate mosaics of M33 in Band 6 with the ACA can be deployed on a cluster running torque.
- `imaging_in_chunks` demonstrates how to use `ImagingChunkedHandler` to parallelize imaging large cubes into separate jobs.