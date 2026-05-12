"""Smoke tests for the ESM-C backend plugin (F2A.4 of master plan v3).

Same shape as test_esm.py / test_t5.py / test_ankh.py: assert plugin
discoverable + contract compliant without requiring torch/esm in
CI. Heavy load_model/embed_batch behaviour is exercised in
protea-core integration runs.
"""

from __future__ import annotations

from importlib.metadata import entry_points

import pytest
from protea_contracts import EmbeddingBackend

from protea_backends.esm3c import EsmcBackend, plugin


def test_plugin_is_esmc_backend_instance() -> None:
    assert isinstance(plugin, EsmcBackend)


def test_plugin_implements_embedding_backend_abc() -> None:
    assert isinstance(plugin, EmbeddingBackend)


def test_plugin_name_is_esm3c() -> None:
    assert plugin.name == "esm3c"


def test_plugin_resolvable_via_entry_points() -> None:
    eps = entry_points(group="protea.backends")
    esm3c_eps = [ep for ep in eps if ep.name == "esm3c"]
    assert len(esm3c_eps) == 1
    resolved = esm3c_eps[0].load()
    assert resolved is plugin


def test_load_model_signature_present() -> None:
    assert callable(plugin.load_model)
    assert callable(plugin.embed_batch)


def test_embed_batch_per_residue_raises_not_implemented() -> None:
    """MIL.1a defers ESM-C per-residue wiring to MIL.1b."""
    with pytest.raises(NotImplementedError, match=r"MIL\.1b"):
        plugin.embed_batch_per_residue(
            model=object(),
            tokenizer=None,
            sequences=["MSEQ"],
            emit=lambda *a, **kw: None,
        )
