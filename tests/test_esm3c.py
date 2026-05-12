"""Smoke tests for the ESM-C backend plugin (F2A.4 of master plan v3).

MIL.1b adds a contract round-trip for the per-residue path: the empty
sequence shortcut + (when torch is available) a stub-driven end-to-end
test that checks shape, dtype and BOS/EOS stripping. The ``esm`` SDK
itself is stubbed via ``sys.modules`` so the test does not require the
heavy ESMC checkpoint.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

import numpy as np
import pytest
from protea_contracts import EmbeddingBackend, EmbeddingPayload

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


def test_embed_batch_per_residue_method_exists() -> None:
    """MIL.1b: ESM-C overrides the default-raise contract method."""
    assert callable(plugin.embed_batch_per_residue)
    assert (
        type(plugin).embed_batch_per_residue
        is not EmbeddingBackend.embed_batch_per_residue
    )


def test_embed_batch_per_residue_empty_sequences() -> None:
    """Empty input returns an empty per-residue payload (no torch needed)."""
    payload = plugin.embed_batch_per_residue(
        model=object(),
        tokenizer=None,
        sequences=[],
        emit=lambda *a, **kw: None,
    )
    assert isinstance(payload, EmbeddingPayload)
    assert payload.granularity == "per_residue"
    assert payload.residues == []
    assert payload.attention_mask == []


_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@dataclass
class _ESMProteinStub:
    sequence: str


@dataclass
class _LogitsConfigStub:
    sequence: bool = True
    return_hidden_states: bool = True


class _LogitsOutputStub:
    def __init__(self, hidden_states: Any) -> None:
        self.hidden_states = hidden_states


def _install_esm_sdk_stubs() -> None:
    """Register a fake ``esm.sdk.api`` exposing ``ESMProtein`` and ``LogitsConfig``.

    The backend imports these lazily inside ``_compute_residue_tensors``;
    plugging them via ``sys.modules`` lets the test bypass the real
    ``esm`` package while keeping the import path inside the backend
    untouched.
    """
    esm_root = types.ModuleType("esm")
    esm_sdk = types.ModuleType("esm.sdk")
    esm_sdk_api = types.ModuleType("esm.sdk.api")
    esm_sdk_api.ESMProtein = _ESMProteinStub  # type: ignore[attr-defined]
    esm_sdk_api.LogitsConfig = _LogitsConfigStub  # type: ignore[attr-defined]
    sys.modules.setdefault("esm", esm_root)
    sys.modules.setdefault("esm.sdk", esm_sdk)
    sys.modules["esm.sdk.api"] = esm_sdk_api


class _StubModel:
    """ESM-C model stub exposing ``encode`` + ``logits`` surface.

    Generates a deterministic hidden-state tensor of shape
    ``(1, len(seq) + 2, dim)`` (BOS + residues + EOS) so the test can
    verify the backend strips BOS / EOS correctly.
    """

    def __init__(self, dim: int = 8) -> None:
        import torch

        self._dim = dim
        self._param = torch.nn.Parameter(torch.zeros(1))
        self._current_len: int = 0

    def parameters(self) -> Any:
        yield self._param

    def encode(self, protein: Any) -> Any:
        # ``encode`` is opaque to the backend; we only use its return as a
        # token-ids carrier. Stash the sequence length so ``logits`` knows
        # the residue count to model.
        self._current_len = len(protein.sequence) + 2
        return protein

    def logits(self, protein_tensor: Any, config: Any) -> _LogitsOutputStub:
        del protein_tensor, config
        import torch

        n = self._current_len
        # Single hidden state tensor (1, n, dim) — matches the SDK return
        # type when ``return_hidden_states=True`` and only one layer is
        # exposed. Distinct values per position so CLS/EOS stripping is
        # verifiable.
        hs = torch.arange(n * self._dim, dtype=torch.float32).reshape(1, n, self._dim)
        return _LogitsOutputStub(hidden_states=hs)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_batch_per_residue_roundtrip_with_stubs() -> None:
    """MIL.1b contract test: ESM-C per-residue path round-trips end-to-end."""
    _install_esm_sdk_stubs()
    sequences = ["MSEQ", "GG"]
    payload = plugin.embed_batch_per_residue(
        model=_StubModel(dim=8),
        tokenizer=None,
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
    """MIL.1b retro-compat: mean-pool path still returns a ``(B, D)`` ndarray."""
    _install_esm_sdk_stubs()
    out = plugin.embed_batch(
        model=_StubModel(dim=8),
        tokenizer=None,
        sequences=["MSEQ", "GG"],
        emit=lambda *a, **kw: None,
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 8)
    assert out.dtype == np.float16
