# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from importlib.metadata import version as get_version

project = "BayesHalvingSearchCV"
copyright = "2026, Rodrigo Pinto Coelho"
author = "Rodrigo Pinto Coelho"
release = get_version("BayesHalvingSearchCV")
version = ".".join(release.split(".")[:3])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "external_links": [],
    "github_url": "https://github.com/rpcoelho17/PatternSearch_HyperParam_Opt__V2",
    "use_edit_page_button": False,
    "show_toc_level": 1,
}

html_context = {
    "github_user": "rpcoelho17",
    "github_repo": "PatternSearch_HyperParam_Opt__V2",
    "github_version": "main",
    "doc_path": "doc",
}

# -- Options for autodoc ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
}

# generate autosummary even if no references
autosummary_generate = True

# -- Options for numpydoc -----------------------------------------------------

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False

# -- Options for intersphinx --------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "scikit-learn": ("https://scikit-learn.org/stable", None),
}
