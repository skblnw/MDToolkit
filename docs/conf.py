"""Sphinx configuration for MDToolkit documentation."""

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Project information
project = 'MDToolkit'
copyright = '2024, MDToolkit Contributors'
author = 'MDToolkit Contributors'
release = '1.0.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'nbsphinx',
]

# Theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
}

# Paths
templates_path = ['_templates']
exclude_patterns = []
html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'MDAnalysis': ('https://docs.mdanalysis.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Autodoc
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'show-inheritance': True,
}