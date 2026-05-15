"""Contract-compliance suite for the protea-backends plugin pack (F6.4).

Sibling slice T1.7 (PROTEA PR #371) shipped
``tests/test_contracts_invariants.py`` in the platform repo. F6.4 is the
plugin-side equivalent: this module enumerates every entry point under
the ``protea.backends`` group and asserts that the loaded object honours
the :class:`protea_contracts.EmbeddingBackend` contract. The check is
deliberately group-driven (not hard-coded plugin names) so a new backend
added in ``pyproject.toml`` is automatically covered, and a regression
where someone forgets to subclass the ABC fails the suite without any
test edit.

The entry-point target for every plugin is a *singleton instance* of
the implementing class (see ``esm/__init__.py``:
``plugin = EsmBackend()``). The tests therefore use :func:`isinstance`
rather than :func:`issubclass`.
"""

from __future__ import annotations

import importlib
import re
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path

import pytest
from protea_contracts import EmbeddingBackend

ENTRY_GROUP = "protea.backends"


def _discovered_entry_points() -> list[EntryPoint]:
    eps = list(entry_points(group=ENTRY_GROUP))
    if not eps:
        pytest.fail(
            f"No entry points found under group {ENTRY_GROUP!r}; "
            "protea-backends is not installed or pyproject.toml is misconfigured."
        )
    return eps


def test_every_entry_point_is_importable() -> None:
    """Each declared entry_point target must resolve without ImportError.

    The plugin ``__init__`` modules are deliberately import-cheap (no
    torch / transformers / esm at top level) so this test runs without
    the heavy ML extras installed.
    """
    for ep in _discovered_entry_points():
        ep.load()


def test_every_entry_point_satisfies_embedding_backend_abc() -> None:
    """Every loaded plugin must be an :class:`EmbeddingBackend` instance.

    Mirrors T1.7's PROTEA-side invariant: any plugin registered under
    the ``protea.backends`` group MUST honour the ABC contract so that
    ``protea-core`` can dispatch by ``EmbeddingConfig.model_backend``
    through ``isinstance`` checks.
    """
    for ep in _discovered_entry_points():
        plugin = ep.load()
        assert isinstance(plugin, EmbeddingBackend), (
            f"Entry point {ep.name!r} (target={ep.value!r}) loaded as {type(plugin)!r}, "
            "which is not a protea_contracts.EmbeddingBackend subclass."
        )


def test_every_plugin_declares_name() -> None:
    """``name`` class attribute is required by the ABC contract.

    ``protea-core`` matches a plugin against
    ``EmbeddingConfig.model_backend`` by ``name`` so the attribute must
    be a non-empty string.
    """
    for ep in _discovered_entry_points():
        plugin = ep.load()
        assert isinstance(plugin.name, str) and plugin.name, (
            f"{ep.name!r} has empty or non-string 'name' attribute."
        )


def test_every_plugin_implements_required_methods() -> None:
    """``load_model`` / ``embed_batch`` must be concretely implemented.

    The ABC marks these as ``@abstractmethod``; instantiation already
    fails if any is missing, but the explicit assertion documents the
    contract and catches accidental shadowing by a non-callable
    attribute. ``embed_batch_per_residue`` is intentionally NOT checked
    here because the ABC ships a default ``NotImplementedError`` body
    that backends opt into (ESM in MIL.1a; T5 / Ankh / ESM-C in MIL.1b),
    so non-overriding backends still legitimately satisfy the ABC.
    """
    for ep in _discovered_entry_points():
        plugin = ep.load()
        for method_name in ("load_model", "embed_batch"):
            method = getattr(plugin, method_name, None)
            assert callable(method), (
                f"{ep.name!r} does not expose a callable {method_name!r}."
            )


def test_pyproject_pins_protea_contracts() -> None:
    """``protea-contracts`` must be declared as a runtime dependency.

    F6.4 acceptance: each plugin repo's CI runs contract tests against a
    pinned protea-contracts version. The pin here is the git-branch dep
    (the canonical shape across the plugin stack while the contracts
    package iterates on ``main``); a future bookkeeping commit promotes
    it to a semver tag, at which point this test continues to pass.
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    # Regex-based scan keeps the test py3.10-compatible (``tomllib`` is
    # py3.11+; the mypy strict pass targets py3.10 per pyproject.toml).
    # The dependency line is canonical TOML so we match on the bare key
    # within the ``[tool.poetry.dependencies]`` section.
    text = pyproject.read_text(encoding="utf-8")
    section_match = re.search(
        r"\[tool\.poetry\.dependencies\](?P<body>.*?)(?=\n\[|\Z)",
        text,
        flags=re.DOTALL,
    )
    assert section_match, "pyproject.toml has no [tool.poetry.dependencies] section."
    body = section_match.group("body")
    assert re.search(r'^\s*"?protea-contracts"?\s*=', body, flags=re.MULTILINE), (
        "pyproject.toml [tool.poetry.dependencies] is missing 'protea-contracts'."
    )


def test_protea_contracts_version_is_resolvable() -> None:
    """The installed ``protea-contracts`` must expose ``__version__``.

    A successful import + version probe proves the CI lane resolved a
    real wheel/sdist, not a phantom path-dep that silently no-ops.
    """
    contracts = importlib.import_module("protea_contracts")
    assert isinstance(contracts.__version__, str) and contracts.__version__, (
        "protea_contracts.__version__ is missing or empty; "
        "the runtime dependency did not resolve to a valid package."
    )
