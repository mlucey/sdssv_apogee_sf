import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

project   = "sdssv-apogee-sf"
author    = "Maddie Lucey"
copyright = "2026, Maddie Lucey"
release   = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",     # Google/NumPy docstring styles
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python"  : ("https://docs.python.org/3", None),
    "numpy"   : ("https://numpy.org/doc/stable", None),
    "astropy" : ("https://docs.astropy.org/en/stable", None),
    "healpy"  : ("https://healpy.readthedocs.io/en/latest", None),
}

autodoc_default_options = {
    "members"          : True,
    "undoc-members"    : False,
    "show-inheritance" : True,
}
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"
html_static_path = []
