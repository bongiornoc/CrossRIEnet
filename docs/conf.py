"""Sphinx configuration for CRIENT."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "CRIENT"
author = "Efstratios Manolakis, Christian Bongiorno, Rosario N. Mantegna"
copyright = "2026, CRIENT authors"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "numpydoc",
]

autodoc_typehints = "description"
numpydoc_show_class_members = False
numpydoc_validation_checks = set()

html_theme = "alabaster"
exclude_patterns = ["_build"]
