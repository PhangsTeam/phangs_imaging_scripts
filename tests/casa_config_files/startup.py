import os
import sys

envdir = os.environ.get("ENVDIR", os.getcwd())

au_dir = os.path.join(envdir, "analysis_scripts")
sys.path.append(au_dir)

import analysisUtils as au
aU = au
