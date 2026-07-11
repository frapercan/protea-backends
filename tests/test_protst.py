"""Smoke + contract tests for the ProtST-ESM1b backend plugin.

ProtST is a whole-protein text-aligned backend: its value is the 512-d
``protein_feature`` projection, not a residue-level tensor. These tests
verify plugin registration, the ABC surface, and (when torch is
available) that ``embed_batch`` / ``embed_chunks`` round-trip the
``protein_feature`` head through a stubbed model without needing the
heavy ProtST checkpoint or ``trust_remote_code`` download.
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import entry_points
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from protea_contracts import EmbeddingBackend

from protea_backends._chunk_helpers import ChunkEmbedding
from protea_backends.protst import PROTEIN_FEATURE_DIM, ProtstBackend, plugin


def test_plugin_is_protst_backend_instance() -> None:
    assert isinstance(plugin, ProtstBackend)


def test_plugin_implements_embedding_backend_abc() -> None:
    assert isinstance(plugin, EmbeddingBackend)


def test_plugin_name_is_protst() -> None:
    assert plugin.name == "protst"


def test_plugin_resolvable_via_entry_points() -> None:
    eps = entry_points(group="protea.backends")
    protst_eps = [ep for ep in eps if ep.name == "protst"]
    assert len(protst_eps) == 1
    resolved = protst_eps[0].load()
    assert resolved is plugin


def test_load_and_embed_methods_present() -> None:
    assert callable(plugin.load_model)
    assert callable(plugin.embed_batch)
    assert callable(plugin.embed_chunks)


def test_embed_batch_per_residue_left_at_contract_default() -> None:
    """ProtST has no meaningful per-residue projection: it keeps the
    default-raise contract method rather than overriding it."""
    assert (
        type(plugin).embed_batch_per_residue is EmbeddingBackend.embed_batch_per_residue
    )
    with pytest.raises(NotImplementedError):
        plugin.embed_batch_per_residue(
            model=object(),
            tokenizer=object(),
            sequences=["MSEQ"],
            emit=lambda *a, **kw: None,
        )


def test_embed_batch_empty_sequences() -> None:
    """Empty input returns an empty matrix (no torch / model needed)."""
    out = plugin.embed_batch(
        model=object(),
        tokenizer=object(),
        sequences=[],
        emit=lambda *a, **kw: None,
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (0, 0)


def test_embed_chunks_empty_sequences() -> None:
    out = plugin.embed_chunks(
        model=object(),
        tokenizer=object(),
        sequences=[],
        config=SimpleNamespace(max_length=1024, normalize=False),
        device="cpu",
    )
    assert out == []


_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class _TokenizerStub:
    """Minimal tokenizer stub returning padded ``input_ids`` + mask tensors.

    Records the post-truncation sequences it saw so a test can assert the
    backend truncates to ``max_length`` before the forward pass.
    """

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(
        self,
        sequences: list[str],
        *,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 1024,
    ) -> dict[str, Any]:
        import torch

        self.seen = list(sequences)
        lengths = [min(len(s), max_length) for s in sequences]
        width = max(lengths) if lengths else 0
        ids = torch.zeros((len(sequences), width), dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for i, n in enumerate(lengths):
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _ProtstModelStub:
    """ProtST model stub exposing ``protein_model`` + ``protein_feature``.

    ``protein_model`` returns an object whose ``protein_feature`` is a
    deterministic ``(B, dim)`` tensor so ``embed_batch`` / ``embed_chunks``
    output is verifiable without the real checkpoint.
    """

    def __init__(self, dim: int = PROTEIN_FEATURE_DIM) -> None:
        import torch

        self._dim = dim
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self) -> Any:
        yield self._param

    def protein_model(self, *, input_ids: Any, attention_mask: Any) -> Any:
        import torch

        del attention_mask
        b = int(input_ids.shape[0])
        feat = (
            torch.arange(b * self._dim, dtype=torch.float32).reshape(b, self._dim) + 1.0
        )
        return SimpleNamespace(protein_feature=feat)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_batch_returns_protein_feature_matrix() -> None:
    out = plugin.embed_batch(
        model=_ProtstModelStub(dim=8),
        tokenizer=_TokenizerStub(),
        sequences=["MSEQ", "GG"],
        emit=lambda *a, **kw: None,
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 8)
    assert out.dtype == np.float16


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_returns_single_full_sequence_chunk() -> None:
    cfg = SimpleNamespace(max_length=1024, normalize=False)
    sequences = ["MSEQ", "GG"]
    out = plugin.embed_chunks(
        model=_ProtstModelStub(dim=8),
        tokenizer=_TokenizerStub(),
        sequences=sequences,
        config=cfg,
        device="cpu",
    )
    assert len(out) == 2
    for chunks in out:
        assert len(chunks) == 1
        chunk = chunks[0]
        assert isinstance(chunk, ChunkEmbedding)
        assert chunk.chunk_index_s == 0
        assert chunk.chunk_index_e is None
        assert chunk.vector.shape == (8,)
        assert chunk.vector.dtype == np.float32
        assert np.isfinite(chunk.vector).all()


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_normalize_l2() -> None:
    cfg = SimpleNamespace(max_length=1024, normalize=True)
    out = plugin.embed_chunks(
        model=_ProtstModelStub(dim=8),
        tokenizer=_TokenizerStub(),
        sequences=["MSEQ"],
        config=cfg,
        device="cpu",
    )
    vec = out[0][0].vector
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")
def test_embed_chunks_truncates_to_max_residues() -> None:
    """Sequences are truncated to ``min(max_length, 1022)`` residues before
    tokenisation, matching the ESM-1b positional ceiling."""
    tok = _TokenizerStub()
    plugin.embed_chunks(
        model=_ProtstModelStub(dim=4),
        tokenizer=tok,
        sequences=["A" * 5000],
        config=SimpleNamespace(max_length=4096, normalize=False),
        device="cpu",
    )
    assert len(tok.seen[0]) == 1022
