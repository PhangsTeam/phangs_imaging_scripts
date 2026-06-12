######
MS key
######

The MS key defines the location of input visibility data be staged, processed,
and imaged by the pipeline.

The pipeline begins with calibrated visibility data, which referenced in this file.

In this key file, each measurement set is mapped to a target, project, array tag,
and numerical tag. These are used to group the measurement sets together and stage
them for imaging.

This file should be space or tab delimited. Each entry should have **6** columns,
as described below.

- **Column 1:** Target name associated with the measurement set - should be defined
  in the target definition key.
- **Column 2:** Project tag associated with the measurement set - suggested convention
  is the numeric code associated with the project by the observatory.
- **Column 3:** Science field name - "all" will select all science targets.
  If not "all", then this value is selected on during the staging of the data.
  Should map to the field names in the measurement set.
- **Column 4:** Array tag (7m, C, etc.). This is a tag that indicates the
  interferometric configuration associated with the measurement
  set. This is set by the user - that is, this is not a formal
  definition based on baseline lengths, it's a tag. This should match array
  tags in the config definitions file.
- **Column 5:** Observation number. This is just a numeric flag indicating
  the number of the observation among other observations of the same
  target, project, and array tag.
- **Column 6:** The file name of a calibrated measurement set. The paths
  are relative to ANY ms_root directory defined in the master key.

.. note::

    If you are using the SingleDishPipeline to process TP data, then
    the file name is somewhat different. You should point this to the
    member.uid* directory, i.e. the one that looks like:

    member.uid*/
        - calibration
        - log
        - qa
        - raw
        - script

===================
Example MS key file
===================

.. include:: ../../../phangs-alma_keys/ms_file_key.txt
    :literal: