"""Sphinx configuration for ``protea-backends``."""

from __future__ import annotations

import os
import sys
from importlib.metadata import version as _pkg_version

# Make the source tree importable so autodoc can resolve symbols
# without an installed wheel.
sys.path.insert(0, os.path.abspath("../../src"))

# ── Project info ─────────────────────────────────────────────
project = "protea-backends"
author = "Francisco Miguel Pérez Canales"
copyright = "2026, Francisco Miguel Pérez Canales"

try:
    release = _pkg_version("protea-backends")
except Exception:  # noqa: BLE001  -- fallback when not installed
    release = "0.0.1"
version = release

# ── Extensions ───────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "exclude-members": "__weakref__,__init_subclass__,__subclasshook__",
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Heavy ML deps live behind extras and are not installed in the docs
# build environment. autodoc would otherwise crash trying to import
# them; mock them out here.
autodoc_mock_imports = [
    "torch",
    "transformers",
    "sentencepiece",
    "esm",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# ── HTML output ──────────────────────────────────────────────
html_theme = "shibuya"
html_title = "protea-backends"
html_static_path: list[str] = []

templates_path = ["_templates"]
exclude_patterns: list[str] = []

master_doc = "index"
