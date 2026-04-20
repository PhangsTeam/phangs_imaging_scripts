import os

casadir = os.environ.get("CASADIR", os.getcwd())

datapath = [os.path.join(casadir, "data")]
measurespath = datapath[0]
measures_auto_update = True
data_auto_update = True
