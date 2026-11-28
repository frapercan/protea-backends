"""The padded pass gives every row what the one-sequence pass gives it.

``embed_chunks`` runs ESM-2 and ESM-C one sequence per forward pass, which keeps
a bank byte reproducible and leaves a large card mostly idle.
``embed_chunks_padded`` runs the whole batch in ONE pass, padded on the right,
and reduces each row with the same function over that row's own span. On real
kernels the two differ by floating-point rounding, because the matrices have
other shapes; what must never differ is WHICH positions a row reads. These
stubs compute every position from its own token and place alone, so a row's
values cannot depend on its neighbours or on padding, and the two paths must
agree bit for bit: a padded position that leaked into a span, or a span cut at
the wrong place, would show here as an inequality.

T5 and Ankh always ran a padded batch; their ``embed_chunks_padded`` is that
same call under the shared name, which is all their tests pin.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from protea_backends.ankh import plugin as ankh_plugin
from protea_backends.esm import plugin as esm_plugin
from protea_backends.esm3c import plugin as esm3c_plugin
from protea_backends.t5 import plugin as t5_plugin

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in test env")

#: Sequences of different lengths, so every batch is padded.
SEQUENCES = ["MKTAYIAKQR", "GG", "MSEQWLLV", "A"]
CLS, PAD, EOS = 0, 1, 2


def _cfg(
    *,
    pooling: str = "mean",
    layer_indices: list[int] | None = None,
    normalize: bool = True,
    max_length: int | None = 1022,
) -> Any:
    return SimpleNamespace(
        model_name="stub",
        max_length=max_length,
        layer_indices=layer_indices if layer_indices is not None else [0],
        layer_agg="mean",
        pooling=pooling,
        normalize=normalize,
        normalize_residues=False,
        use_chunking=False,
        chunk_size=512,
        chunk_overlap=0,
    )


def _rowwise_layers(tokens: Any, dim: int, num_layers: int) -> list[Any]:
    """Hidden states where position p of a row depends on its token and p alone."""
    import torch

    batch, length = tokens.shape
    pos = torch.arange(length, dtype=torch.float32).expand(batch, length)
    base = tokens.float() * 0.5 + pos * 0.25
    span = torch.arange(dim, dtype=torch.float32)
    return [base.unsqueeze(-1) + span + 10.0 * k for k in range(num_layers)]


def _same(left: list[Any], right: list[Any]) -> None:
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert (x.chunk_index_s, x.chunk_index_e) == (y.chunk_index_s, y.chunk_index_e)
            assert x.vector.dtype == y.vector.dtype
            assert np.array_equal(x.vector, y.vector)


# --------------------------------------------------------------------------- ESM-2


class _EsmTokens(dict):  # type: ignore[type-arg]
    """The mapping a Hugging Face tokeniser returns."""


class _EsmTokenizer:
    """CLS + one token per residue + EOS, cut at ``max_length``, padded right."""

    def __init__(self, padding_side: str = "right") -> None:
        self.padding_side = padding_side

    def __call__(
        self,
        seqs: str | list[str],
        *,
        return_tensors: str,
        truncation: bool,
        add_special_tokens: bool,
        max_length: int | None = None,
        padding: bool = False,
    ) -> _EsmTokens:
        import torch

        del return_tensors, truncation, add_special_tokens
        batch = [seqs] if isinstance(seqs, str) else seqs
        rows = []
        for seq in batch:
            residues = [ord(c) for c in seq]
            if max_length is not None:
                residues = residues[: max_length - 2]
            rows.append([CLS, *residues, EOS])
        width = max(len(r) for r in rows)
        assert padding or len(rows) == 1
        ids = torch.full((len(rows), width), PAD, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.tensor(row)
            mask[i, : len(row)] = 1
        out = _EsmTokens()
        out["input_ids"] = ids
        out["attention_mask"] = mask
        return out


class _EsmModel:
    """Row-wise stand-in for ``EsmModel``; counts its forward passes."""

    def __init__(self, dim: int = 6, num_layers: int = 3) -> None:
        self.dim, self.num_layers, self.calls = dim, num_layers, 0

    def __call__(self, **tokens: Any) -> Any:
        self.calls += 1
        layers = _rowwise_layers(tokens["input_ids"], self.dim, self.num_layers)
        return SimpleNamespace(hidden_states=tuple(layers))


@needs_torch
@pytest.mark.parametrize(
    "config",
    [
        _cfg(),
        _cfg(layer_indices=[0, 2]),
        _cfg(pooling="cls"),
        _cfg(normalize=False, max_length=6),
    ],
    ids=["last-layer-mean", "two-layers", "cls", "cut-at-max-length"],
)
def test_esm_padded_rows_equal_the_one_sequence_pass(config: Any) -> None:
    one_by_one = esm_plugin.embed_chunks(_EsmModel(), _EsmTokenizer(), SEQUENCES, config, "cpu")
    padded = esm_plugin.embed_chunks_padded(_EsmModel(), _EsmTokenizer(), SEQUENCES, config, "cpu")
    _same(padded, one_by_one)


@needs_torch
def test_esm_padded_runs_one_forward_pass_for_the_batch() -> None:
    model = _EsmModel()
    esm_plugin.embed_chunks_padded(model, _EsmTokenizer(), SEQUENCES, _cfg(), "cpu")
    assert model.calls == 1


def test_esm_padded_refuses_a_tokeniser_that_pads_on_the_left() -> None:
    with pytest.raises(ValueError, match="right padding"):
        esm_plugin.embed_chunks_padded(
            _EsmModel(), _EsmTokenizer(padding_side="left"), SEQUENCES, _cfg(), "cpu"
        )


def test_esm_padded_on_nothing_is_nothing() -> None:
    assert esm_plugin.embed_chunks_padded(object(), object(), [], _cfg(), "cpu") == []


# --------------------------------------------------------------------------- ESM-C


@dataclass
class _ESMProtein:
    sequence: str


@dataclass
class _LogitsConfig:
    sequence: bool = True
    return_hidden_states: bool = True


@dataclass
class _Batched:
    sequence: Any


class _EsmcModel:
    """Row-wise stand-in for ``ESMC``: ``encode``, ``logits`` and its tokeniser."""

    def __init__(self, dim: int = 6, num_layers: int = 3) -> None:
        self.dim, self.num_layers, self.calls = dim, num_layers, 0
        self.tokenizer = SimpleNamespace(pad_token_id=PAD)

    def encode(self, protein: Any) -> Any:
        import torch

        ids = [CLS, *(ord(c) for c in protein.sequence), EOS]
        return SimpleNamespace(sequence=torch.tensor(ids, dtype=torch.long))

    def logits(self, protein_tensor: Any, config: Any) -> Any:
        del config
        self.calls += 1
        tokens = protein_tensor.sequence
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        layers = _rowwise_layers(tokens, self.dim, self.num_layers)
        return SimpleNamespace(hidden_states=layers)


@pytest.fixture
def esm_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two SDK modules the ESM-C paths import, stubbed for this test only."""
    api = types.ModuleType("esm.sdk.api")
    api.ESMProtein = _ESMProtein  # type: ignore[attr-defined]
    api.LogitsConfig = _LogitsConfig  # type: ignore[attr-defined]
    sampling = types.ModuleType("esm.utils.sampling")
    sampling._BatchedESMProteinTensor = _Batched  # type: ignore[attr-defined]
    for name in ("esm", "esm.sdk", "esm.utils"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name, types.ModuleType(name)))
    monkeypatch.setitem(sys.modules, "esm.sdk.api", api)
    monkeypatch.setitem(sys.modules, "esm.utils.sampling", sampling)


@needs_torch
@pytest.mark.parametrize(
    "config",
    [
        _cfg(),
        _cfg(layer_indices=[0, 2]),
        _cfg(pooling="cls"),
        _cfg(normalize=False, max_length=5),
    ],
    ids=["last-layer-mean", "two-layers", "cls", "cut-at-max-length"],
)
def test_esmc_padded_rows_equal_the_one_sequence_pass(esm_sdk: None, config: Any) -> None:
    one_by_one = esm3c_plugin.embed_chunks(_EsmcModel(), None, SEQUENCES, config, "cpu")
    padded = esm3c_plugin.embed_chunks_padded(_EsmcModel(), None, SEQUENCES, config, "cpu")
    _same(padded, one_by_one)


@needs_torch
def test_esmc_padded_runs_one_forward_pass_for_the_batch(esm_sdk: None) -> None:
    model = _EsmcModel()
    esm3c_plugin.embed_chunks_padded(model, None, SEQUENCES, _cfg(), "cpu")
    assert model.calls == 1


def test_esmc_padded_on_nothing_is_nothing() -> None:
    assert esm3c_plugin.embed_chunks_padded(object(), None, [], _cfg(), "cpu") == []


# ------------------------------------------------------------------- T5 and Ankh


@pytest.mark.parametrize("plugin", [t5_plugin, ankh_plugin], ids=["t5", "ankh"])
def test_t5_family_padded_is_its_padded_embed_chunks(
    plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[Any, ...]] = []
    answer = [["sentinel"]]

    def fake(*args: Any) -> Any:
        seen.append(args)
        return answer

    monkeypatch.setattr(plugin, "embed_chunks", fake)
    config = _cfg()
    assert plugin.embed_chunks_padded("model", "tok", SEQUENCES, config, "cuda") is answer
    assert seen == [("model", "tok", SEQUENCES, config, "cuda")]


@pytest.mark.parametrize(
    "plugin", [esm_plugin, esm3c_plugin, t5_plugin, ankh_plugin], ids=["esm", "esm3c", "t5", "ankh"]
)
def test_every_backend_of_the_stage_one_models_has_a_padded_pass(plugin: Any) -> None:
    assert callable(getattr(type(plugin), "embed_chunks_padded", None))
