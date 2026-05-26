############
Installation
############

The pipeline is distributed as the Python package ``phangsPipeline``.

============
Requirements
============

The PHANGS pipeline requires Python 3.12, and uses CASA >= 6.7.3.
We recommend installing the pipeline in a separate `Conda <https://www.anaconda.com/>`_ environment.

============================
Install directly from GitHub
============================

Standard install (without CASA extras):

.. code-block:: bash

   pip install "phangsPipeline @ git+https://github.com/phangsTeam/phangs_imaging_scripts.git"

Install including CASA optional dependencies:

.. code-block:: bash

   pip install "phangsPipeline[casa] @ git+https://github.com/phangsTeam/phangs_imaging_scripts.git"

================================
Install from a cloned repository
================================

Clone and enter the repository:

.. code-block:: bash

   git clone https://github.com/phangsTeam/phangs_imaging_scripts.git
   cd phangs_imaging_scripts

Standard local install:

.. code-block:: bash

   pip install -e .

Local install including CASA optional dependencies:

.. code-block:: bash

   pip install -e ".[casa]"

