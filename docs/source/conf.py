import os
import re
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version
from pathlib import Path

from docutils import nodes

# Add the path to your package for autodoc to find it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import massphoto  # noqa: E402

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "massphoto"
copyright = "2026, Jan Christoph Thiele"
author = "Jan Christoph Thiele"

# The full version, including alpha/beta/rc tags
try:
    release = _get_version("massphoto")
except PackageNotFoundError:
    release = getattr(massphoto, "__version__", "0.0.0")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # For parsing NumPy and Google style docstrings
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.mathjax",  # Support for math equations
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "sphinx.ext.githubpages",  # Creates .nojekyll for GitHub Pages
    # 'sphinx_autodoc_typehints',   # renders typing.Annotated extras and nicer hints
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",  # Adds "copy" buttons to code blocks
    "notfound.extension",
    "sphinx_sitemap",
]

# Show type hints in the description area rather than the signature
autodoc_typehints = "description"
# autodoc_typehints_format = "short"
# python_display_short_literal_types = True

# Make NewType / alias names prettier in generated docs
# map fully-qualified names or simple names to readable strings
autodoc_type_aliases = {
    "massphoto.utils.NativeMovie": "NativeMovie - np.ndarray (marker)",
    "massphoto.utils.RatiometricMovie": "RatiometricMovie - np.ndarray (marker)",
    "massphoto.utils.EventList": "EventList - pd.DataFrame (marker)",
}

# Add mappings for intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_examples/GALLERY_HEADER.rst",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme'
html_theme = "pydata_sphinx_theme"
html_title = f"{project} {release}"
html_baseurl = "https://thielec.github.io/massphoto/"
sitemap_url_scheme = "{link}"
html_static_path = ["_static"]
html_css_files = ["home.css"]
html_js_files = ["home.js"]

# Hide empty primary navigation sidebar on single-page sections
html_sidebars = {
    "index": [],
    "installation": [],
    "usage": [],
    "contributing": [],
}


# -- Options for sphinx-gallery ------------------------------------------------
# Builds executed examples (plots + downloadable scripts/notebooks) from text-only
# Python files in `docs/source/_examples`.
sphinx_gallery_conf = {
    "examples_dirs": ["_examples"],
    "gallery_dirs": ["auto_examples"],
    # Follow the numbered filenames instead of sorting by code length.
    "within_subsection_order": "FileNameSortKey",
    "filename_pattern": r".*",
    "download_all_examples": False,
    "remove_config_comments": True,
    # Render DataFrames as styled HTML tables when they end an example cell.
    "capture_repr": ("_repr_html_", "__repr__"),
    # Enable linking of identifiers inside rendered example code blocks.
    "doc_module": ("massphoto",),
    "backreferences_dir": "gen_modules/backreferences",
    "abort_on_example_error": False,
}

# -- Options for autodoc -----------------------------------------------------
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}


def _remove_empty_gallery_code_blocks(app, doctree):
    """Remove cells left empty by hidden plotting code or gallery directives."""
    if not app.env.docname.startswith("auto_examples/"):
        return
    for node in list(doctree.findall(nodes.literal_block)):
        if not node.astext().strip():
            node.parent.remove(node)


def _ensure_homepage_panels(app):
    """Re-execute only example 2 if one of its auxiliary outputs is missing."""
    gallery = app.config.sphinx_gallery_conf
    source = Path(app.srcdir) / "_examples" / "02-mass_distribution.py"
    if not gallery["plot_gallery"] or not re.search(gallery["filename_pattern"], str(source)):
        return
    generated = Path(app.srcdir) / "auto_examples"
    panels = generated / "images" / "homepage"
    required = ("raw.png", "events.png", "profile.svg", "histogram.svg", "provenance.json")
    if any(not (panels / name).is_file() for name in required):
        (generated / "02-mass_distribution.py.md5").unlink(missing_ok=True)


# Build mapping between top-level exports and submodule functions
TOP_LEVEL_TO_MODULE: dict[str, str] = {}
MODULE_TO_TOP_LEVEL: dict[str, str] = {}

for _name in getattr(massphoto, "__all__", []):
    _obj = getattr(massphoto, _name, None)
    if _obj is not None and hasattr(_obj, "__module__") and hasattr(_obj, "__name__"):
        _mod = _obj.__module__
        _orig_name = _obj.__name__
        if _mod and _mod.startswith("massphoto."):
            _canonical = f"{_mod}.{_orig_name}"
            _top_level = f"massphoto.{_name}"
            TOP_LEVEL_TO_MODULE[_top_level] = _canonical
            MODULE_TO_TOP_LEVEL[_canonical] = _top_level


def _add_export_alias_note(app, what, name, obj, options, lines):
    """Mention both top-level export and module-level name in the doc entry."""
    if what not in ("function", "class", "method"):
        return
    top_level = MODULE_TO_TOP_LEVEL.get(name)
    canonical = TOP_LEVEL_TO_MODULE.get(name)
    note_text = None
    if top_level and top_level != name:
        note_text = f"Available as both ``{top_level}`` and ``{name}``."
    elif canonical and canonical != name:
        note_text = f"Available as both ``{name}`` and ``{canonical}``."

    if note_text:
        # Insert after the first summary paragraph so autosummary can extract
        # the leading summary sentence for summary tables.
        first_blank = next((i for i, l in enumerate(lines) if not l.strip()), len(lines))
        note_block = ["", ".. note::", f"   {note_text}", ""]
        lines[first_blank:first_blank] = note_block


def _register_top_level_aliases(app, env):
    """Register top-level package exports as aliases in the Python domain."""
    objects = env.domaindata["py"]["objects"]
    for top_level, canonical in TOP_LEVEL_TO_MODULE.items():
        if canonical in objects:
            objects[top_level] = objects[canonical]._replace(aliased=True)


def setup(app):
    # Gallery configuration is normalised at priority 10; execution starts at 500.
    app.connect("builder-inited", _ensure_homepage_panels, priority=20)
    app.connect("doctree-read", _remove_empty_gallery_code_blocks)
    app.connect("autodoc-process-docstring", _add_export_alias_note)
    app.connect("env-updated", _register_top_level_aliases)
