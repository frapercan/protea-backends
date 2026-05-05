"""Smoke tests for the ESM backend plugin (F2A.1 of master plan v3).

These tests run **without** torch/transformers installed: they assert
the plugin is discoverable and contract-compliant. Heavy
load_model/embed_batch behaviour is exercised in protea-core's
integration suite and live `study_v_thesis` runs.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from protea_contracts import EmbeddingBackend

from protea_backends.esm import EsmBackend, plugin


def test_plugin_is_esm_backend_instance() -> None:
    assert isinstance(plugin, EsmBackend)


def test_plugin_implements_embedding_backend_abc() -> None:
    assert isinstance(plugin, EmbeddingBackend)


def test_plugin_name_is_esm() -> None:
    assert plugin.name == "esm"


def test_plugin_resolvable_via_entry_points() -> None:
    eps = entry_points(group="protea.backends")
    esm_eps = [ep for ep in eps if ep.name == "esm"]
    assert len(esm_eps) == 1
    resolved = esm_eps[0].load()
    # entry_point points at ``protea_backends.esm:plugin`` so the
    # loaded object is the plugin instance itself.
    assert resolved is plugin


def test_load_model_signature_present() -> None:
    # Subclassing the EmbeddingBackend ABC enforces the method
    # exists; this just pins the expected callable shape.
    assert callable(plugin.load_model)
    assert callable(plugin.embed_batch)
