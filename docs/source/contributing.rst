Contributing
============

Reporting Bugs & Suggesting Features
-------------------------------------

Community feedback and suggestions are warmly encouraged! Whether you found an issue or have
an idea, please start by opening an issue on the `GitHub Issue Tracker <https://github.com/thielec/massphoto/issues>`_:

- **Bug Reports**: If you experience an error or unexpected output, please share the issue, ideally with a minimal reproducible example and your environment details.
- **New File Formats**: File formats change frequently. If you encounter mass photometry recordings that are not supported, please open an issue and, if possible, share a small sample file.
- **Feature Requests & Suggestions**: Ideas for alternative detection filters, fitting methods, or downstream analysis are always welcome. If you are interested in contributing code, please open an issue first so we can coordinate before you submit a pull request.

Development Setup
-----------------

To contribute code or documentation to massphoto, set up your development environment as follows.

Prerequisites
~~~~~~~~~~~~~

- Python 3.10 or higher
- Git

Clone the repository:

.. code-block:: console

    git clone https://github.com/thielec/massphoto.git
    cd massphoto

Create and activate a virtual environment using the
:ref:`commands for your platform <installation-environment>`. Then install the
development and documentation dependencies from the repository root:

.. code-block:: console

    python -m pip install -e ".[dev,docs]"

Versioning
~~~~~~~~~~

Versioning is managed dynamically using `setuptools_scm`, which derives versions from git tags. To create a new version, tag the commit with ``git tag vX.Y.Z``.

Pre-commit Hooks
~~~~~~~~~~~~~~~~

This project uses `pre-commit <https://pre-commit.com/>`_ to enforce code quality and formatting.

Install pre-commit:

.. code-block:: console

    python -m pip install pre-commit
    python -m pre_commit install

The hooks include:

- **end-of-file-fixer**: Ensures files end with a newline.
- **trailing-whitespace**: Removes trailing whitespace.
- **check-yaml**: Validates YAML files.
- **ruff**: Lints and fixes Python code.
- **black**: Formats Python code to a consistent style (line length 120).
- **codespell**: Checks code and documentation for common spelling mistakes.

Run pre-commit on all files:

.. code-block:: console

    python -m pre_commit run --all-files

Or, it runs automatically on commits. To skip, use ``git commit --no-verify``.

Testing
-------

Run tests with pytest:

.. code-block:: console

    python -m pytest

Building Documentation
----------------------

The documentation is built with Sphinx.

Install dependencies:

.. code-block:: console

    python -m pip install -e ".[docs]"

Build from the repository root using the same activated environment. This
command works on Windows, macOS, and Linux without requiring ``make``:

.. code-block:: console

    python -m sphinx -b html docs/source docs/build/html

The standard build executes the gallery examples, which download sample data.
For a documentation build without executing examples, add ``-D plot_gallery=0``.
This requires previously generated gallery output, including the homepage
panels; a clean checkout must first run an executing build.

View at ``docs/build/html/index.html``.
