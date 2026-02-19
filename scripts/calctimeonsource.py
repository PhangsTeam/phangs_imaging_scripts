import os

import analysisUtils as au

from phangsPipeline import handlerKeys as kh
from phangsPipeline import phangsLogger as pl

# Set the logging level
pl.setup_logger(level='DEBUG', logfile=None)

# Instantiate handlers
this_kh = kh.KeyHandler()

summary = {}
summary['7m'] = {
    'clock_time':0.0,
    'minutes_on_science':0.0,
    }
summary['12m'] = {
    'clock_time':0.0,
    'minutes_on_science':0.0,
    }

targets = this_kh.get_targets_in_ms_key()
for this_target in targets:
    this_dict = this_kh._ms_dict[this_target]
    for this_project in this_dict.keys():
        this_sub_dict = this_dict[this_project]
        for this_label in this_sub_dict.keys():
            this_config = (this_label.split('_'))[0]
            this_ms = this_sub_dict[this_label]
            for this_dir in this_kh._ms_roots:
                if os.path.isdir(this_dir + this_ms):
                    use_this_dir = this_dir
                    break
            tos = au.timeOnSource(use_this_dir+this_ms)
            summary[this_config]['clock_time'] = \
                summary[this_config]['clock_time'] + tos['clock_time']
            summary[this_config]['minutes_on_science'] = \
                summary[this_config]['minutes_on_science'] + tos['minutes_on_science']

