"""Smoke tests for the Ankh backend plugin (F2A.3 of master plan v3).

MIL.1b adds a contract round-trip for the per-residue path: the empty
sequence shortcut + (when torch is available) a stub-driven end-to-end
test that checks shape, dtype and trailing-EOS stripping.
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import entry_points
from typing import Any

import numpy as np
import pytest
from protea_contracts import EmbeddingBackend, EmbeddingPayload

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


def test_embed_batch_per_residue_method_exists() -> None:
    """MIL.1b: Ankh overrides the default-raise contract method."""
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


class _StubBatch(dict):  # type: ignore[type-arg]
    """Dict-like batch carrier matching the HF tokenizer return type."""


class _StubTokenizer:
    """Ankh tokenizer stub.

    Mirrors :meth:`T5TokenizerFast.batch_encode_plus` with
    ``is_split_into_words=True``: each amino-acid character becomes
    one token plus one trailing EOS. The shorter sequence is
    right-padded so the attention mask correctly masks the difference.
    """

    def batch_encode_plus(
        self,
        chars_lists: list[list[str]],
        *,
        padding: str,
        truncation: bool,
        add_special_tokens: bool,
        is_split_into_words: bool,
        return_tensors: str,
    ) -> _StubBatch:
        import torch

        del padding, truncation, add_special_tokens, is_split_into_words, return_tensors
        token_counts = [len(c) + 1 for c in chars_lists]
        max_len = max(token_counts)
        ids = torch.zeros((len(chars_lists), max_len), dtype=torch.long)
        mask = torch.zeros((len(chars_lists), max_len), dtype=torch.long)
        for i, count in enumerate(token_counts):
            mask[i, :count] = 1
        out = _StubBatch()
        out["input_ids"] = ids
        out["attention_mask"] = mask
        return out


class _StubModel:
    """Ankh encoder stub exposing the surface used by the residue extractor."""

    def __init__(self, dim: int = 8) -> None:
        import torch

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
def test_embed_batch_per_residue_roundtrip_with_stubs() -> None:
    """MIL.1b contract test: Ankh per-residue path round-trips end-to-end."""
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
    """MIL.1b retro-compat: mean-pool path still returns a ``(B, D)`` ndarray."""
    out = plugin.embed_batch(
        model=_StubModel(dim=8),
        tokenizer=_StubTokenizer(),
        sequences=["MSEQ", "GG"],
        emit=lambda *a, **kw: None,
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 8)
    assert out.dtype == np.float16
