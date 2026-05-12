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
