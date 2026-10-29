Installation
============

Prerequisites
-------------

Install Python 3.10 or higher. Check your version with ``python --version``.

.. _installation-environment:

Create a virtual environment
----------------------------

While not strictly required, it is recommended to install massphoto in a virtual environment
to avoid conflicts with other Python packages.
Create an environment in your working directory, then activate it using the
commands for your shell. The environment name ``mp-venv`` is arbitrary.

.. tab-set::

   .. tab-item:: macOS / Linux (bash or zsh)

      .. code-block:: bash

          python -m venv mp-venv
          source mp-venv/bin/activate

   .. tab-item:: Windows (PowerShell)

      .. code-block:: powershell

          python -m venv mp-venv
          .\mp-venv\Scripts\Activate.ps1

   .. tab-item:: Conda

      If you use Anaconda, Miniconda, or Miniforge:

      .. code-block:: bash

          conda create -n massphoto python=3.11
          conda activate massphoto

Install massphoto
-----------------

Run these commands in the activated environment on any platform:

.. code-block:: console

    python -m pip install --upgrade pip
    python -m pip install massphoto

The package's declared runtime dependencies are installed automatically,
including Numpy, Dask, Numba, and file-format libraries such as
``h5py``, ``hdf5plugin``, ``nptdms``, and ``tifffile``.

Verify the installation:

.. code-block:: console

    python -c "import massphoto as mp; print(mp.__version__)"

Upgrade massphoto
-----------------

Activate the same environment using the appropriate command above, then run:

.. code-block:: console

    python -m pip install --upgrade massphoto
    python -m pip show massphoto

Install from a local source checkout
------------------------------------

For an editable development installation, open a terminal in the repository
directory containing ``pyproject.toml``, activate your environment, and run:

.. code-block:: console

    python -m pip install -e .

Install additional dependencies for testing or documentation as needed:

.. code-block:: console

    python -m pip install -e ".[testing]"
    python -m pip install -e ".[docs]"

For development including documentation, install both extras together:

.. code-block:: console

    python -m pip install -e ".[dev,docs]"
