"""Smoke tests for the T5 backend plugin (F2A.2 of master plan v3).

Same shape as test_esm.py: assert plugin discoverable + contract
compliant without requiring torch/transformers in CI. Heavy
load_model/embed_batch behaviour is exercised in protea-core
integration runs.

MIL.1b adds a contract round-trip for the per-residue path: the empty
sequence shortcut + (when torch is available) a stub-driven end-to-end
test that checks shape, dtype and AA2fold/EOS stripping for both
ProtT5 and ProstT5.
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import entry_points
from typing import Any

import numpy as np
import pytest
from protea_contracts import EmbeddingBackend, EmbeddingPayload

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


def test_embed_batch_per_residue_method_exists() -> None:
    """MIL.1b: T5 overrides the default-raise contract method."""
    assert callable(plugin.embed_batch_per_residue)
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


class _StubConfig:
    """Minimal config carrier exposing ``name_or_path`` for AA2fold detection."""

    def __init__(self, name_or_path: str) -> None:
        self.name_or_path = name_or_path


class _StubBatch(dict):  # type: ignore[type-arg]
    """Dict-like batch carrier matching the HF tokenizer return type."""


class _StubTokenizer:
    """T5 tokenizer stub returning deterministic padded ids + mask.

    Mirrors :meth:`T5Tokenizer.batch_encode_plus` with ``padding="longest"``
    and ``add_special_tokens=True``: each sequence is space-joined first
    (the backend prefixes ``<AA2fold>`` for ProstT5), so token count per
    sequence is ``n_words + 1`` (words + trailing EOS). The shorter
    sequence is right-padded so ``attention_mask`` masks the difference.
    """

    def batch_encode_plus(
        self,
        processed: list[str],
        *,
        padding: str,
        truncation: bool,
        add_special_tokens: bool,
        return_tensors: str,
    ) -> _StubBatch:
        import torch

        del padding, truncation, add_special_tokens, return_tensors
        token_counts = [len(s.split()) + 1 for s in processed]
        max_len = max(token_counts)
        ids = torch.zeros((len(processed), max_len), dtype=torch.long)
        mask = torch.zeros((len(processed), max_len), dtype=torch.long)
        for i, count in enumerate(token_counts):
            mask[i, :count] = 1
        out = _StubBatch()
        out["input_ids"] = ids
        out["attention_mask"] = mask
        return out


class _StubModel:
    """T5 encoder stub exposing the surface used by T5's residue extractor.

    The fake forward returns a single hidden-state tensor of shape
    ``(B, L, D)`` with values derived from the position so the test can
    verify AA2fold prefix and EOS stripping.
    """

    def __init__(self, name_or_path: str, dim: int = 8) -> None:
        import torch

        self.config = _StubConfig(name_or_path)
        self._dim = dim
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self) -> Any:
        yield self._param

    def __call__(self, **inputs: Any) -> Any:
        import torch

        ids = inputs["input_ids"]
        b, n = int(ids.shape[0]), int(ids.shape[1])
        hs = torch.arange(b * n * self._dim, dtype=torch.float32).reshape(b, n, self._dim)
        return type("Out", (), {"hidden_states": (hs,)})()


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_batch_per_residue_roundtrip_prott5() -> None:
    """MIL.1b contract test: ProtT5 per-residue path round-trips end-to-end.

    No AA2fold prefix; only the trailing EOS is stripped so each residue
    aligns with an amino-acid position.
    """
    sequences = ["MSEQ", "GG"]
    payload = plugin.embed_batch_per_residue(
        model=_StubModel("Rostlab/prot_t5_xl_uniref50", dim=8),
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
def test_embed_batch_per_residue_roundtrip_prostt5() -> None:
    """MIL.1b contract test: ProstT5 strips the AA2fold prefix on top of EOS."""
    sequences = ["MSEQ", "GG"]
    payload = plugin.embed_batch_per_residue(
        model=_StubModel("Rostlab/ProstT5", dim=8),
        tokenizer=_StubTokenizer(),
        sequences=sequences,
        emit=lambda *a, **kw: None,
    )
    assert payload.residues is not None
    for seq, residues in zip(sequences, payload.residues, strict=True):
        # AA2fold prefix + EOS both stripped -> back to len(seq).
        assert residues.shape == (len(seq), 8)
        assert residues.dtype == np.float16


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_batch_mean_pool_still_returns_matrix() -> None:
    """MIL.1b retro-compat: mean-pool path still returns a ``(B, D)`` ndarray."""
    out = plugin.embed_batch(
        model=_StubModel("Rostlab/prot_t5_xl_uniref50", dim=8),
        tokenizer=_StubTokenizer(),
        sequences=["MSEQ", "GG"],
        emit=lambda *a, **kw: None,
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 8)
    assert out.dtype == np.float16


# ---------------------------------------------------------------------------
# T2A.2 — embed_chunks tests (chunked output for PROTEA's _dispatch_embed)
# ---------------------------------------------------------------------------


class _ChunkedStubTokenizer:
    """Tokenizer stub for the ``embed_chunks`` path.

    Adds the ``max_length`` and ``is_split_into_words`` kwargs that
    :func:`_t5_tokenise` forwards (per the legacy ``_embed_t5``
    contract). Mirrors :class:`_StubTokenizer` otherwise: each sequence
    is tokenised to ``n_words + 1`` tokens (words + EOS), shorter
    sequences right-padded with ``attention_mask=0``.
    """

    def batch_encode_plus(
        self,
        processed: Any,
        *,
        padding: str,
        truncation: bool,
        max_length: int,
        add_special_tokens: bool,
        is_split_into_words: bool = False,
        return_tensors: str = "pt",
    ) -> _StubBatch:
        import torch

        del padding, truncation, max_length, add_special_tokens, return_tensors
        if is_split_into_words:
            token_counts = [len(c) + 1 for c in processed]  # residues + EOS
        else:
            token_counts = [len(s.split()) + 1 for s in processed]  # words + EOS
        max_len = max(token_counts)
        ids = torch.zeros((len(processed), max_len), dtype=torch.long)
        mask = torch.zeros((len(processed), max_len), dtype=torch.long)
        for i, count in enumerate(token_counts):
            mask[i, :count] = 1
        out = _StubBatch()
        out["input_ids"] = ids
        out["attention_mask"] = mask
        return out


def _chunked_cfg(
    model_name: str = "Rostlab/prot_t5_xl_uniref50",
    *,
    pooling: str = "mean",
    use_chunking: bool = False,
    chunk_size: int = 512,
    chunk_overlap: int = 0,
    normalize: bool = False,
    normalize_residues: bool = False,
    layer_indices: list[int] | None = None,
    layer_agg: str = "mean",
) -> Any:
    """Build a duck-typed config object for ``embed_chunks`` tests."""
    from types import SimpleNamespace

    return SimpleNamespace(
        model_name=model_name,
        max_length=1024,
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
    """T2A.2 contract: ``embed_chunks`` mirrors PROTEA's legacy ``_embed_t5``.

    Residue-level mean pooling (no chunking) returns one
    ``ChunkEmbedding`` per sequence with vector shape ``(D,)``, and the
    chunk window covers the full sequence (``chunk_index_s=0``,
    ``chunk_index_e=None``).
    """
    from protea_backends._chunk_helpers import ChunkEmbedding

    cfg = _chunked_cfg()
    sequences = ["MSEQ", "GG"]
    out = plugin.embed_chunks(
        model=_StubModel("Rostlab/prot_t5_xl_uniref50", dim=8),
        tokenizer=_ChunkedStubTokenizer(),
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
    """CLS pooling path returns one ``ChunkEmbedding`` whose vector is the CLS row."""
    cfg = _chunked_cfg(pooling="cls")
    out = plugin.embed_chunks(
        model=_StubModel("Rostlab/prot_t5_xl_uniref50", dim=4),
        tokenizer=_ChunkedStubTokenizer(),
        sequences=["AC"],
        config=cfg,
        device="cpu",
    )
    assert len(out) == 1
    assert len(out[0]) == 1
    assert out[0][0].vector.shape == (4,)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_chunking_splits_long_sequences() -> None:
    """Chunked path emits one ``ChunkEmbedding`` per overlapping window.

    The stub tokenizer pads ``"A" * 10`` to 11 tokens (10 residues + EOS);
    after stripping the trailing EOS the residue tensor is length 10, so
    spans (0,3), (2,5), (4,7), (6,9), (8,10) match the legacy ``_embed_t5``
    output.
    """
    cfg = _chunked_cfg(use_chunking=True, chunk_size=3, chunk_overlap=1)
    out = plugin.embed_chunks(
        model=_StubModel("Rostlab/prot_t5_xl_uniref50", dim=4),
        tokenizer=_ChunkedStubTokenizer(),
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
def test_embed_chunks_prostt5_strips_aa2fold_prefix() -> None:
    """ProstT5 auto-detect: residue slice begins at AA 0 (prefix stripped)."""
    # Use a length-1 sequence so we exercise the slice ``[1:actual_len-1]``
    # branch: with ProstT5 the actual tokens are AA2fold + A + EOS = 3,
    # so the residue range is [1:2] (1 residue). Without ProstT5 the
    # tokens are A + EOS = 2, residue range [0:1] (still 1 residue).
    cfg = _chunked_cfg(model_name="Rostlab/ProstT5")
    out = plugin.embed_chunks(
        model=_StubModel("Rostlab/ProstT5", dim=4),
        tokenizer=_ChunkedStubTokenizer(),
        sequences=["A"],
        config=cfg,
        device="cpu",
    )
    assert len(out) == 1
    assert len(out[0]) == 1
    assert out[0][0].vector.shape == (4,)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_with_mode_split_into_words_path() -> None:
    """Ankh-style mode: ``split_into_words=True`` routes through the list tokeniser.

    This is the entry point T2A.3 (Ankh) will call. The output shape is
    the same as the default path; the test guards against accidental
    drift in :func:`embed_chunks_with_mode` while T2A.3 is still in
    flight.
    """
    from protea_backends.t5 import T5Mode, embed_chunks_with_mode

    cfg = _chunked_cfg(model_name="ElnaggarLab/ankh-base")
    out = embed_chunks_with_mode(
        model=_StubModel("ElnaggarLab/ankh-base", dim=4),
        tokenizer=_ChunkedStubTokenizer(),
        sequences=["AC"],
        config=cfg,
        device="cpu",
        mode=T5Mode(use_aa2fold=False, split_into_words=True),
    )
    assert len(out) == 1
    assert len(out[0]) == 1
    assert out[0][0].vector.shape == (4,)
