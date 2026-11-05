"""
Sphinx configuration for pism-emulator.

Layout follows the pism-terra documentation, which in turn mirrors xDEM
(https://xdem.readthedocs.io/): sphinx-book-theme, sphinx-design cards, MyST
Markdown pages, autosummary API tables and a BibTeX bibliography.
"""

from __future__ import annotations

from importlib import metadata

# -- Project information -----------------------------------------------------

project = "pism-emulator"
author = "Andy Aschwanden and Douglas Brinkerhoff"
copyright = f"2021-2026, {author}"  # pylint: disable=redefined-builtin

try:
    release = metadata.version("pism-emulator")
except metadata.PackageNotFoundError:
    release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "myst_nb",
    "numpydoc",
]

# -- sphinxcontrib-bibtex ---------------------------------------------------

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"

templates_path = ["_templates"]
exclude_patterns: list[str] = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

# -- Theme -------------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_logo = "_static/pism_logo_transp.png"
html_favicon = None
html_title = "pism-emulator"

html_theme_options = {
    "logo": {
        "alt_text": "pism-emulator - Home",
    },
    "repository_url": "https://github.com/pism/pism-emulator",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_edit_page_button": False,
    # Always show the full nav tree in the left sidebar, expanded.
    "navigation_depth": 4,
    "show_navbar_depth": 2,
    "collapse_navbar": False,
    "show_toc_level": 2,
    "home_page_in_toc": True,
    "path_to_docs": "docs/source",
}

# -- autodoc / autosummary / numpydoc ---------------------------------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
# Methods without their own docstring (e.g. Lightning hooks) would otherwise show
# Lightning's, whose reST numpydoc cannot parse.
autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# numpydoc validation lives in pyproject.toml / pre-commit; only suppress the
# class-member table here, which duplicates autodoc's member listing.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# -- MyST / MyST-NB ---------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# Don't execute notebooks at build time — data paths aren't portable.
nb_execution_mode = "off"

# -- intersphinx ------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable", None),
    "arviz": ("https://python.arviz.org/en/stable", None),
}

# -- copybutton -------------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Warning suppression ----------------------------------------------------

suppress_warnings = [
    # autosummary :toctree: warns until generated/ exists at first build.
    "autosummary",
    # myst-nb tolerated warnings on unconfigured cell-metadata keys.
    "mystnb.unknown_mime_type",
]
