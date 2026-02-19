'''
Example of how to produce config files that enable easy use of
slurm job array submissions.
'''

import os
import sys
from copy import copy

from pathlib import Path

# Give the line names for imaging via the command line:
line_name = sys.argv[-1]

overwrite = True

# Location of the master key. Set this to the master key that points
# to all of the keys for your project.

# key_path = Path("/home/erickoch/M33/full_aca_band6/keys/")
key_path = Path("/home/erickoch/full_aca_band6/keys_hydra/")

key_file = str(key_path / 'master_key.txt')

# Import the logger and initialize the logging. You can change the
# level of message that you want to see by changing "level" here or
# save to a logfile with the keyword.

from phangsPipeline import phangsLogger as pl
pl.setup_logger(level='DEBUG', logfile=None)

from phangsPipeline import handlerKeys as kh

this_kh = kh.KeyHandler(master_key=key_file)

# Keep all the mosaic targets. Idea is to image 1 per job.
mosaic_targets = this_kh.get_all_targets()
whole_targets = this_kh.get_whole_targets()

mosaic_targets = list(set(mosaic_targets) - set(whole_targets))
mosaic_targets.sort()

# I want to order them by project to keep better track of which job ids
# correspond to which setup.
mosaic_target_aca_orig = [target for target in mosaic_targets if 'row' in target]
mosaic_target_aca_2018 = [target for target in mosaic_targets if 'M_33' in target]
mosaic_target_aca_2022 = [target for target in mosaic_targets if 'brick' in target]

mosaic_targets = mosaic_target_aca_orig
mosaic_targets.extend(mosaic_target_aca_2018)
mosaic_targets.extend(mosaic_target_aca_2022)

# Write for line definition from config_definitions.txt
all_lines = this_kh.get_line_products()

if line_name.lower() == 'all':
    line_list = all_lines
else:
    if line_name not in all_lines:
        raise ValueError(f"Line {line_name} not found in the defined line products: {all_lines}")

    line_list = [line_name]

config_files_path = key_path / "config_lines"

if not config_files_path.exists():
    config_files_path.mkdir()

for ii, this_target in enumerate(mosaic_targets):
    this_config_file = config_files_path / f"line_staging_imaging.{line_name}.{ii+1}.jobconfig.txt"

    if this_config_file.exists() and overwrite:
        this_config_file.unlink()

    with open(this_config_file, 'w') as f:
        f.write(this_target + "\n")

        # Then list the line names to image.
        for this_line in line_list:
            f.write(this_line + "\n")



###########################
# Make a matching set of continuum config files:
config_files_path = key_path / "config_cont"

if not config_files_path.exists():
    config_files_path.mkdir()

for ii, this_target in enumerate(mosaic_targets):
    this_config_file = config_files_path / f"cont_staging_imaging.{ii+1}.jobconfig.txt"

    if this_config_file.exists() and overwrite:
        this_config_file.unlink()

    with open(this_config_file, 'w') as f:
        f.write(this_target + "\n")
        f.write("cont" + "\n")

