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
        max_length: int | None = None,
    ) -> _StubTokens:
        import torch

        del return_tensors, truncation, add_special_tokens, max_length
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
def test_embed_chunks_returns_chunk_embedding_per_sequence() -> None:
    """T2A.1 contract: ``embed_chunks`` mirrors PROTEA's legacy ``_embed_esm``.

    With the residue-level mean pooling path (no chunking), one
    ``ChunkEmbedding`` is returned per sequence. The vector shape is
    ``(hidden_dim,)`` and the chunk window covers the full sequence
    (``chunk_index_s=0``, ``chunk_index_e=None``).
    """
    from types import SimpleNamespace

    from protea_backends._chunk_helpers import ChunkEmbedding

    cfg = SimpleNamespace(
        max_length=1024,
        layer_indices=[0],
        layer_agg="mean",
        pooling="mean",
        normalize=False,
        normalize_residues=False,
        use_chunking=False,
        chunk_size=512,
        chunk_overlap=0,
    )
    sequences = ["MSEQ", "GG"]
    out = plugin.embed_chunks(
        model=_StubModel(dim=8),
        tokenizer=_StubTokenizer(),
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
        # Sanity: the stub model produces deterministic per-position
        # tensors, so a non-empty sequence yields a non-zero mean.
        assert seq  # silence unused-loop-var lint while keeping the iteration
        assert np.isfinite(chunk.vector).all()


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_cls_pool_returns_single_vector() -> None:
    """CLS pooling path returns one ``ChunkEmbedding`` whose vector is the CLS row."""
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        max_length=1024,
        layer_indices=[0],
        layer_agg="mean",
        pooling="cls",
        normalize=False,
        normalize_residues=False,
        use_chunking=False,
        chunk_size=512,
        chunk_overlap=0,
    )
    out = plugin.embed_chunks(
        model=_StubModel(dim=4),
        tokenizer=_StubTokenizer(),
        sequences=["AC"],
        config=cfg,
        device="cpu",
    )
    assert len(out) == 1
    assert len(out[0]) == 1
    assert out[0][0].vector.shape == (4,)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_chunking_splits_long_sequences() -> None:
    """Chunked path emits one ``ChunkEmbedding`` per overlapping window."""
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        max_length=1024,
        layer_indices=[0],
        layer_agg="mean",
        pooling="mean",
        normalize=False,
        normalize_residues=False,
        use_chunking=True,
        chunk_size=3,
        chunk_overlap=1,
    )
    # 10-residue sequence yields spans (0,3),(2,5),(4,7),(6,9),(8,10) (5 chunks).
    out = plugin.embed_chunks(
        model=_StubModel(dim=4),
        tokenizer=_StubTokenizer(),
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


class _CountingStubModel:
    """Stub with several layers that records how many forward passes it ran.

    The count is the point. A shared-pass implementation that is correct but
    still runs one pass per config produces identical vectors and saves
    nothing, so equality alone cannot tell the two apart.
    """

    def __init__(self, dim: int = 8, layers: int = 4) -> None:
        import torch

        self._dim = dim
        self._layers = layers
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.passes = 0

    def parameters(self) -> Any:
        yield self._param

    def __call__(self, **tokens: Any) -> Any:
        import torch

        self.passes += 1
        n = int(tokens["input_ids"].shape[1])
        base = torch.arange(n * self._dim, dtype=torch.float32).reshape(1, n, self._dim)
        # Each layer offset by its index so selecting a different layer is
        # visible in the output rather than a no-op.
        hs = tuple(base + float(k) for k in range(self._layers))
        return type("Out", (), {"hidden_states": hs})()


def _multi_cfg(layer: int, max_length: int | None = 1024) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        max_length=max_length,
        layer_indices=[layer],
        layer_agg="mean",
        pooling="mean",
        normalize=False,
        normalize_residues=False,
        use_chunking=False,
        chunk_size=512,
        chunk_overlap=0,
    )


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_multi_runs_one_pass_per_sequence_not_per_config() -> None:
    """The saving exists, and this is the only test that can show it.

    Three configs over two sequences: two forward passes, not six. Every other
    assertion here passes just as happily against an implementation that loops
    ``embed_chunks`` per config, which is exactly the implementation this
    change exists to replace.
    """
    model = _CountingStubModel(dim=8, layers=4)
    out = plugin.embed_chunks_multi(
        model=model,
        tokenizer=_StubTokenizer(),
        sequences=["MSEQ", "GG"],
        configs=[_multi_cfg(0), _multi_cfg(1), _multi_cfg(2)],
        device="cpu",
    )
    assert model.passes == 2, f"expected one pass per sequence, ran {model.passes}"
    assert len(out) == 3
    assert all(len(per_config) == 2 for per_config in out)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_multi_matches_separate_passes_exactly() -> None:
    """One pass reduced N times equals N passes reduced once, byte for byte.

    Both paths call the same ``_reduce_one``, so this is a regression guard on
    the split rather than a discovery: it fails if someone gives the shared
    path a reduction of its own.
    """
    import numpy as np

    configs = [_multi_cfg(0), _multi_cfg(2)]
    sequences = ["MSEQ", "GG"]
    shared = plugin.embed_chunks_multi(
        model=_CountingStubModel(dim=8, layers=4),
        tokenizer=_StubTokenizer(),
        sequences=sequences,
        configs=configs,
        device="cpu",
    )
    for idx, cfg in enumerate(configs):
        separate = plugin.embed_chunks(
            model=_CountingStubModel(dim=8, layers=4),
            tokenizer=_StubTokenizer(),
            sequences=sequences,
            config=cfg,
            device="cpu",
        )
        for shared_seq, separate_seq in zip(shared[idx], separate, strict=True):
            for shared_chunk, separate_chunk in zip(shared_seq, separate_seq, strict=True):
                assert np.array_equal(shared_chunk.vector, separate_chunk.vector)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_multi_returns_configs_in_the_order_they_arrived() -> None:
    """Slot ``i`` holds config ``i``, which is what the caller indexes by."""
    import numpy as np

    ascending = plugin.embed_chunks_multi(
        model=_CountingStubModel(dim=8, layers=4),
        tokenizer=_StubTokenizer(),
        sequences=["MSEQ"],
        configs=[_multi_cfg(0), _multi_cfg(3)],
        device="cpu",
    )
    descending = plugin.embed_chunks_multi(
        model=_CountingStubModel(dim=8, layers=4),
        tokenizer=_StubTokenizer(),
        sequences=["MSEQ"],
        configs=[_multi_cfg(3), _multi_cfg(0)],
        device="cpu",
    )
    assert np.array_equal(ascending[0][0][0].vector, descending[1][0][0].vector)
    assert np.array_equal(ascending[1][0][0].vector, descending[0][0][0].vector)
    assert not np.array_equal(ascending[0][0][0].vector, ascending[1][0][0].vector)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_multi_refuses_configs_that_do_not_share_a_pass() -> None:
    """``max_length`` is read before the tokeniser, so it cannot be grouped.

    Refusing is the whole point: accepting would embed both configs under the
    first one's limit and report success, which is the silent-wrong-answer
    shape this repository's guards exist to prevent.
    """
    model = _CountingStubModel(dim=8, layers=4)
    with pytest.raises(ValueError, match="max_length"):
        plugin.embed_chunks_multi(
            model=model,
            tokenizer=_StubTokenizer(),
            sequences=["MSEQ"],
            configs=[_multi_cfg(0, max_length=1024), _multi_cfg(1, max_length=512)],
            device="cpu",
        )
    assert model.passes == 0, "refused before spending a forward pass, not after"


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_multi_refuses_an_empty_config_list() -> None:
    """No config means no output slot to write into; say so rather than return []."""
    with pytest.raises(ValueError, match="at least one config"):
        plugin.embed_chunks_multi(
            model=_CountingStubModel(dim=8, layers=4),
            tokenizer=_StubTokenizer(),
            sequences=["MSEQ"],
            configs=[],
            device="cpu",
        )
