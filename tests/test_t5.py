"""Smoke tests for the T5 backend plugin (F2A.2 of master plan v3).

Same shape as test_esm.py: assert plugin discoverable + contract
compliant without requiring torch/transformers in CI. Heavy
load_model/embed_batch behaviour is exercised in protea-core
integration runs.
"""

from __future__ import annotations

from importlib.metadata import entry_points

import pytest
from protea_contracts import EmbeddingBackend

from protea_backends.t5 import T5Backend, plugin


def test_plugin_is_t5_backend_instance() -> None:
    assert isinstance(plugin, T5Backend)


def test_plugin_implements_embedding_backend_abc() -> None:
    assert isinstance(plugin, EmbeddingBackend)


def test_plugin_name_is_t5() -> None:
    assert plugin.name == "t5"


def test_plugin_resolvable_via_entry_points() -> None:
    eps = entry_points(group="protea.backends")
    t5_eps = [ep for ep in eps if ep.name == "t5"]
    assert len(t5_eps) == 1
    resolved = t5_eps[0].load()
    assert resolved is plugin


def test_load_model_signature_present() -> None:
    assert callable(plugin.load_model)
    assert callable(plugin.embed_batch)


def test_embed_batch_per_residue_raises_not_implemented() -> None:
    """MIL.1a defers T5 per-residue wiring to MIL.1b."""
    with pytest.raises(NotImplementedError, match=r"MIL\.1b"):
        plugin.embed_batch_per_residue(
            model=object(),
            tokenizer=object(),
            sequences=["MSEQ"],
            emit=lambda *a, **kw: None,
        )
