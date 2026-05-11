"""Smoke tests for the Ankh backend plugin (F2A.3 of master plan v3)."""

from __future__ import annotations

from importlib.metadata import entry_points

import pytest
from protea_contracts import EmbeddingBackend

from protea_backends.ankh import AnkhBackend, plugin


def test_plugin_is_ankh_backend_instance() -> None:
    assert isinstance(plugin, AnkhBackend)


def test_plugin_implements_embedding_backend_abc() -> None:
    assert isinstance(plugin, EmbeddingBackend)


def test_plugin_name_is_ankh() -> None:
    assert plugin.name == "ankh"


def test_plugin_resolvable_via_entry_points() -> None:
    eps = entry_points(group="protea.backends")
    ankh_eps = [ep for ep in eps if ep.name == "ankh"]
    assert len(ankh_eps) == 1
    resolved = ankh_eps[0].load()
    assert resolved is plugin


def test_load_model_signature_present() -> None:
    assert callable(plugin.load_model)
    assert callable(plugin.embed_batch)


def test_embed_batch_per_residue_raises_not_implemented() -> None:
    """MIL.1a defers Ankh per-residue wiring to MIL.1b."""
    with pytest.raises(NotImplementedError, match=r"MIL\.1b"):
        plugin.embed_batch_per_residue(
            model=object(),
            tokenizer=object(),
            sequences=["MSEQ"],
            emit=lambda *a, **kw: None,
        )
