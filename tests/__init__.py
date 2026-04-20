import os
import sys

# Add the analysisUtils directory to PATH
envdir = os.environ.get("ENVDIR", os.getcwd())
au_dir = os.path.join(envdir, "analysis_scripts")
sys.path.append(au_dir)
