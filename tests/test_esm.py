"""Smoke tests for the ESM backend plugin (F2A.1 of master plan v3).

These tests run **without** torch/transformers installed: they assert
the plugin is discoverable and contract-compliant. Heavy
load_model/embed_batch behaviour is exercised in protea-core's
integration suite and live `study_v_thesis` runs.

MIL.1a adds a contract round-trip for the per-residue path: the empty
sequence shortcut + (when torch is available) a stub-driven end-to-end
test that checks shape, dtype and CLS/EOS stripping.
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import entry_points
from typing import Any

import numpy as np
import pytest
from protea_contracts import EmbeddingBackend, EmbeddingPayload

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


def test_embed_batch_per_residue_method_exists() -> None:
    """MIL.1a: ESM overrides the default-raise contract method."""
    assert callable(plugin.embed_batch_per_residue)
    # Method is overridden on the subclass (not the base ABC's
    # default ``NotImplementedError`` body).
    assert (
        type(plugin).embed_batch_per_residue
        is not EmbeddingBackend.embed_batch_per_residue
    )


def test_embed_batch_per_residue_empty_sequences() -> None:
    """Empty input returns an empty per-residue payload (no torch needed)."""
    payload = plugin.embed_batch_per_residue(
        model=object(),
        tokenizer=object(),
        sequences=[],
        emit=lambda *a, **kw: None,
    )
    assert isinstance(payload, EmbeddingPayload)
    assert payload.granularity == "per_residue"
    assert payload.residues == []
    assert payload.attention_mask == []


_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class _StubTokens(dict):  # type: ignore[type-arg]
    """Dict-like tokens carrier matching the HF tokenizer return type."""


class _StubTokenizer:
    """Tokenizer stub that returns deterministic ids + mask.

    Produces ``L + 2`` tokens per sequence (CLS + residues + EOS),
    matching real HF EsmTokenizer behaviour with ``add_special_tokens=True``.
    """

    def __call__(
        self,
        seq: str,
        *,
        return_tensors: str,
        truncation: bool,
        add_special_tokens: bool,
    ) -> _StubTokens:
        import torch

        del return_tensors, truncation, add_special_tokens
        n = len(seq) + 2
        ids = torch.zeros((1, n), dtype=torch.long)
        mask = torch.ones((1, n), dtype=torch.long)
        out = _StubTokens()
        out["input_ids"] = ids
        out["attention_mask"] = mask
        return out


class _StubModel:
    """Model stub exposing the surface used by ESM's residue extractor."""

    def __init__(self, dim: int = 8) -> None:
        import torch

        self._dim = dim
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self) -> Any:
        yield self._param

    def __call__(self, **tokens: Any) -> Any:
        import torch

        n = int(tokens["input_ids"].shape[1])
        # Single layer's hidden state, shape (1, n, dim). Distinct per
        # position so the test can verify CLS / EOS stripping.
        hs = torch.arange(n * self._dim, dtype=torch.float32).reshape(1, n, self._dim)
        return type("Out", (), {"hidden_states": (hs,)})()


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_batch_per_residue_roundtrip_with_stubs() -> None:
    """MIL.1a contract test: ESM per-residue path round-trips end-to-end.

    Uses tokenizer + model stubs so the heavy ESM checkpoint is not
    required. Asserts the resulting :class:`EmbeddingPayload` validates,
    has the correct shapes for each input sequence (CLS and EOS
    stripped), float16 dtype, all-ones masks, and survives a
    ``as_matrix`` round-trip to a ``(B, D)`` matrix.
    """
    sequences = ["MSEQ", "GG"]
    payload = plugin.embed_batch_per_residue(
        model=_StubModel(dim=8),
        tokenizer=_StubTokenizer(),
        sequences=sequences,
        emit=lambda *a, **kw: None,
    )
    assert isinstance(payload, EmbeddingPayload)
    assert payload.granularity == "per_residue"
    assert payload.residues is not None and len(payload.residues) == 2
    assert payload.attention_mask is not None and len(payload.attention_mask) == 2

    for seq, residues, mask in zip(
        sequences, payload.residues, payload.attention_mask, strict=True
    ):
        assert residues.shape == (len(seq), 8)
        assert residues.dtype == np.float16
        assert mask.shape == (len(seq),)
        assert mask.dtype == bool
        assert mask.all()

    matrix = payload.as_matrix()
    assert matrix.shape == (2, 8)
    assert matrix.dtype == np.float16


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_batch_mean_pool_still_returns_matrix() -> None:
    """MIL.1a retro-compat: mean-pool path still returns a ``(B, D)`` ndarray."""
    out = plugin.embed_batch(
        model=_StubModel(dim=8),
        tokenizer=_StubTokenizer(),
        sequences=["MSEQ", "GG"],
        emit=lambda *a, **kw: None,
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 8)
    assert out.dtype == np.float16
