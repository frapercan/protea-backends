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


# ---------------------------------------------------------------------------
# T2A.4 — embed_chunks tests (chunked output for PROTEA's _dispatch_embed)
# ---------------------------------------------------------------------------


class _ChunkedStubModel:
    """ESM-C model stub for the ``embed_chunks`` path.

    ``logits`` returns ``hidden_states`` as a list of per-layer tensors
    of shape ``(1, len(seq) + 2, dim)`` (BOS + residues + EOS), matching
    the ESM SDK return shape when ``return_hidden_states=True``. Multiple
    layers let the chunked path exercise the ``hs[-(li + 1)]`` selection.
    """

    def __init__(self, dim: int = 8, num_layers: int = 2) -> None:
        import torch

        self._dim = dim
        self._num_layers = num_layers
        self._param = torch.nn.Parameter(torch.zeros(1))
        self._current_len: int = 0

    def parameters(self) -> Any:
        yield self._param

    def encode(self, protein: Any) -> Any:
        self._current_len = len(protein.sequence) + 2
        return protein

    def logits(self, protein_tensor: Any, config: Any) -> _LogitsOutputStub:
        del protein_tensor, config
        import torch

        n = self._current_len
        layers = [
            torch.arange(n * self._dim, dtype=torch.float32).reshape(1, n, self._dim)
            + float(k)
            for k in range(self._num_layers)
        ]
        return _LogitsOutputStub(hidden_states=layers)


def _chunked_cfg(
    *,
    pooling: str = "mean",
    use_chunking: bool = False,
    chunk_size: int = 512,
    chunk_overlap: int = 0,
    normalize: bool = False,
    normalize_residues: bool = False,
    max_length: int = 1024,
    layer_indices: list[int] | None = None,
    layer_agg: str = "mean",
) -> Any:
    """Build a duck-typed config object for ``embed_chunks`` tests."""
    from types import SimpleNamespace

    return SimpleNamespace(
        model_name="esmc_300m",
        max_length=max_length,
        layer_indices=layer_indices if layer_indices is not None else [0],
        layer_agg=layer_agg,
        pooling=pooling,
        normalize=normalize,
        normalize_residues=normalize_residues,
        use_chunking=use_chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_returns_chunk_embedding_per_sequence() -> None:
    """T2A.4 contract: ``embed_chunks`` mirrors PROTEA's legacy ``_embed_esm3c``.

    Residue-level mean pooling (no chunking) returns one
    ``ChunkEmbedding`` per sequence with vector shape ``(D,)``, and the
    chunk window covers the full sequence (``chunk_index_s=0``,
    ``chunk_index_e=None``).
    """
    from protea_backends._chunk_helpers import ChunkEmbedding

    _install_esm_sdk_stubs()
    cfg = _chunked_cfg()
    sequences = ["MSEQ", "GG"]
    out = plugin.embed_chunks(
        model=_ChunkedStubModel(dim=8),
        tokenizer=None,
        sequences=sequences,
        config=cfg,
        device="cpu",
    )
    assert len(out) == 2
    for seq, chunks in zip(sequences, out, strict=True):
        assert len(chunks) == 1
        chunk = chunks[0]
        assert isinstance(chunk, ChunkEmbedding)
        assert chunk.chunk_index_s == 0
        assert chunk.chunk_index_e is None
        assert chunk.vector.shape == (8,)
        assert seq  # silence unused-loop-var lint while keeping the iteration
        assert np.isfinite(chunk.vector).all()


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_cls_pool_returns_single_vector() -> None:
    """CLS pooling path returns one ``ChunkEmbedding`` whose vector is the BOS row."""
    _install_esm_sdk_stubs()
    cfg = _chunked_cfg(pooling="cls")
    out = plugin.embed_chunks(
        model=_ChunkedStubModel(dim=4),
        tokenizer=None,
        sequences=["AC"],
        config=cfg,
        device="cpu",
    )
    assert len(out) == 1
    assert len(out[0]) == 1
    assert out[0][0].chunk_index_s == 0
    assert out[0][0].chunk_index_e is None
    assert out[0][0].vector.shape == (4,)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_chunking_splits_long_sequences() -> None:
    """Chunked path emits one ``ChunkEmbedding`` per overlapping window.

    The stub model emits hidden states of shape ``(1, len(seq) + 2, dim)``
    (BOS + residues + EOS); after stripping BOS and EOS the residue tensor
    is length ``len(seq)``. With ``chunk_size=3, chunk_overlap=1`` over a
    length-10 input that yields spans (0,3), (2,5), (4,7), (6,9), (8,10).
    """
    _install_esm_sdk_stubs()
    cfg = _chunked_cfg(use_chunking=True, chunk_size=3, chunk_overlap=1)
    out = plugin.embed_chunks(
        model=_ChunkedStubModel(dim=4),
        tokenizer=None,
        sequences=["A" * 10],
        config=cfg,
        device="cpu",
    )
    assert len(out) == 1
    chunks = out[0]
    assert len(chunks) == 5
    starts = [c.chunk_index_s for c in chunks]
    ends = [c.chunk_index_e for c in chunks]
    assert starts == [0, 2, 4, 6, 8]
    assert ends == [3, 5, 7, 9, 10]


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_truncates_to_max_length() -> None:
    """T2A.4 parity: sequences longer than ``config.max_length`` are
    truncated before being encoded, matching PROTEA's legacy
    ``_embed_esm3c_one`` (``seq_str[: config.max_length]``).

    The stub model's ``encode`` records the post-truncation sequence
    length so we can assert the backend never feeds raw long sequences
    through the SDK.
    """
    _install_esm_sdk_stubs()
    cfg = _chunked_cfg(max_length=5)
    model = _ChunkedStubModel(dim=4)
    out = plugin.embed_chunks(
        model=model,
        tokenizer=None,
        sequences=["A" * 20],
        config=cfg,
        device="cpu",
    )
    # encode stashed sequence length (post-truncation) + 2 for BOS/EOS
    assert model._current_len == 5 + 2
    assert len(out[0]) == 1
    assert out[0][0].vector.shape == (4,)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_tokenizer_argument_is_ignored() -> None:
    """T2A.4 contract: ESM-C ignores the ``tokenizer`` slot (always
    ``None`` in production)."""
    _install_esm_sdk_stubs()
    cfg = _chunked_cfg()
    out_none = plugin.embed_chunks(
        model=_ChunkedStubModel(dim=4),
        tokenizer=None,
        sequences=["MSEQ"],
        config=cfg,
        device="cpu",
    )
    out_garbage = plugin.embed_chunks(
        model=_ChunkedStubModel(dim=4),
        tokenizer="not-a-tokenizer",
        sequences=["MSEQ"],
        config=cfg,
        device="cpu",
    )
    np.testing.assert_array_equal(out_none[0][0].vector, out_garbage[0][0].vector)
