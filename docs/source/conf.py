# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'aiSFX'
copyright = '2022, A.B. Ma and A. Lerch'
author = 'Alison B. Ma'
release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import sys
import os
import six
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../../src'))

# Mock modules......: Credit - https://github.com/marl/openl3/blob/main/docs/conf.py
if six.PY3:
    from unittest.mock import MagicMock
else:
    from mock import Mock as MagicMock
class Mock(MagicMock):
    @classmethod
    def getattr(cls, name):
        return MagicMock()
sys.modules.update((mod_name, Mock()) for mod_name in ["torch", "torch.nn", "torch.nn.functional"])

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_rtd_theme",
    "nbsphinx", #MyST-NB
    "sphinx_issues",
    "numpydoc"
]

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
