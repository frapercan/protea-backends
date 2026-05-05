"""Smoke tests for protea-backends bootstrap."""

from __future__ import annotations

from importlib.metadata import entry_points

import protea_backends
from protea_backends import ankh, esm, esm3c, t5


def test_version_is_string() -> None:
    assert isinstance(protea_backends.__version__, str)


def test_submodules_importable() -> None:
    assert hasattr(esm, "plugin")
    assert hasattr(t5, "plugin")
    assert hasattr(ankh, "plugin")
    assert hasattr(esm3c, "plugin")


def test_entry_points_registered() -> None:
    eps = entry_points(group="protea.backends")
    names = {ep.name for ep in eps}
    assert "esm" in names
    assert "t5" in names
    assert "ankh" in names
    assert "esm3c" in names


def test_no_platform_imports_leak() -> None:
    import sys

    forbidden = {"sqlalchemy", "fastapi", "protea_core"}
    leaked = forbidden & set(sys.modules)
    assert not leaked, f"Forbidden modules leaked: {leaked}"
