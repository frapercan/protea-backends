"""Long sequences go through exact attention in the bank path, and nothing else changes.

WHY. ``embed_batch`` ran every sequence through the Hugging Face forward, whose attention
memory grows with the square of the length. On the GPU node's 12 GB card the bank stopped
at 8,259 residues, and 161 sequences of the corpus, up to 35,991 residues, never entered
it; the per-residue pass reads the bank, so they never got a code either. Sequences above
:data:`EXACT_ABOVE_RESIDUES` now go through :class:`ExactAnkh`, which computes the same
Ankh with memory linear in the length.

WHAT THESE PIN, on a tiny random T5 on the CPU (the threshold lowered so short sequences
count as long): the exact route gives the vectors the Hugging Face forward gives, for one
layer and for several, mean and max pooled, per protein and per residue; PROTEA's backward
``layer_indices`` land on the right forward hidden state; a mixed batch keeps its order;
and below the threshold nothing takes the new route.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torch.nn.attention.flex_attention")

from test_ankh_exact import _CharTokenizer, _tiny  # noqa: E402

import protea_backends.ankh as ankh  # noqa: E402
from protea_backends.ankh import plugin  # noqa: E402


class _Tokenizer(_CharTokenizer):
    """The tiny tokenizer, also answering the Hugging Face call the old forward makes."""

    def batch_encode_plus(self, words: list[list[str]], **kw: Any) -> dict[str, Any]:
        return self(words, **kw)


SEQS = ["MKVLAAGIV" * 6, "ACDEF", "WYKLMNPQRST" * 3, "GHIK"]


def _quiet(*_a: Any, **_k: Any) -> None:
    return None


def _batch(monkeypatch: pytest.MonkeyPatch, threshold: int, **kw: Any) -> np.ndarray:
    monkeypatch.setattr(ankh, "EXACT_ABOVE_RESIDUES", threshold)
    return plugin.embed_batch(_tiny(), _Tokenizer(), SEQS, emit=_quiet, **kw)


@pytest.mark.parametrize(
    "kw",
    [
        {"layers": [0], "layer_agg": "mean", "pooling": "mean"},
        {"layers": [1], "layer_agg": "mean", "pooling": "mean"},
        {"layers": [0, 2], "layer_agg": "mean", "pooling": "max"},
        {"layers": None, "layer_agg": "mean", "pooling": "mean"},
    ],
)
def test_the_exact_route_gives_the_vectors_the_old_forward_gives(
    monkeypatch: pytest.MonkeyPatch, kw: dict[str, Any]
) -> None:
    """Every sequence above 10 residues takes the exact route here; the old forward takes all."""
    old = _batch(monkeypatch, 10**9, **kw)
    new = _batch(monkeypatch, 10, **kw)
    assert new.shape == old.shape and new.dtype == np.float16
    assert np.allclose(new.astype(np.float32), old.astype(np.float32), atol=2e-3), (
        "the backward layer index must land on the same hidden state on both routes"
    )


def test_a_mixed_batch_keeps_its_order_and_routes_by_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ankh, "EXACT_ABOVE_RESIDUES", 10)
    seen: list[list[int]] = []
    original = plugin._exact_residue_tensors

    def spy(model: Any, tokenizer: Any, sequences: list[str], **kw: Any) -> list[Any]:
        seen.append([len(s) for s in sequences])
        return original(model, tokenizer, sequences, **kw)

    monkeypatch.setattr(plugin, "_exact_residue_tensors", spy)
    out = plugin.embed_batch(
        _tiny(), _Tokenizer(), SEQS, emit=_quiet, layers=[0], layer_agg="mean", pooling="mean"
    )
    assert seen == [[54, 33]], "only the sequences above the threshold, in input order"
    alone = [
        plugin.embed_batch(
            _tiny(), _Tokenizer(), [s], emit=_quiet, layers=[0], layer_agg="mean", pooling="mean"
        )[0]
        for s in SEQS
    ]
    assert np.allclose(out.astype(np.float32), np.stack(alone).astype(np.float32), atol=2e-3)


def test_below_the_threshold_nothing_takes_the_new_route(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_a: Any, **_k: Any) -> list[Any]:
        raise AssertionError("a sequence below the threshold took the exact route")

    monkeypatch.setattr(plugin, "_exact_residue_tensors", refuse)
    out = plugin.embed_batch(
        _tiny(), _Tokenizer(), SEQS, emit=_quiet, layers=[0], layer_agg="mean", pooling="mean"
    )
    assert out.shape[0] == len(SEQS)


def test_per_residue_embeddings_take_the_same_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ankh, "EXACT_ABOVE_RESIDUES", 10**9)
    old = plugin.embed_batch_per_residue(_tiny(), _Tokenizer(), SEQS, emit=_quiet, layers=[0])
    monkeypatch.setattr(ankh, "EXACT_ABOVE_RESIDUES", 10)
    new = plugin.embed_batch_per_residue(_tiny(), _Tokenizer(), SEQS, emit=_quiet, layers=[0])
    for a, b in zip(old.residues, new.residues, strict=True):
        assert a.shape == b.shape
        assert np.allclose(
            np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32), atol=2e-3
        )
