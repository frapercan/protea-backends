"""Exact-attention Ankh, layer by layer, against Hugging Face's T5 encoder (ADR-D51 in PROTEA).

A tiny random T5 stands in for Ankh, so these run on the CPU with FlexAttention eager; the
compiled path, which is what keeps memory linear, is checked on CUDA when a card is present.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torch.nn.attention.flex_attention")

from protea_backends.ankh import exact as ex  # noqa: E402
from protea_backends.ankh import plugin  # noqa: E402

GPU = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compiled FlexAttention runs on CUDA"
)


def _tiny(device: str = "cpu") -> Any:
    from transformers import T5Config, T5EncoderModel

    torch.manual_seed(0)
    cfg = T5Config(
        vocab_size=40,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_heads=4,
        num_layers=3,
        relative_attention_num_buckets=64,
        relative_attention_max_distance=128,
        feed_forward_proj="gated-gelu",
        dropout_rate=0.0,
    )
    m = T5EncoderModel(cfg).to(device).eval()
    with torch.no_grad():  # a bias large enough to matter
        m.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight.normal_(0, 2)
    return m


class _CharTokenizer:
    """One token per residue plus EOS, as Ankh's tokenizer does with is_split_into_words."""

    pad_token_id, eos_token_id, unk_token_id = 0, 1, 2
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"

    def __call__(self, words: list[list[str]], **kw: Any) -> dict[str, Any]:
        n = max(len(w) for w in words) + 1
        ids = torch.full((len(words), n), self.pad_token_id)
        mask = torch.zeros((len(words), n), dtype=torch.long)
        for i, w in enumerate(words):
            ids[i, : len(w)] = torch.tensor([3 + self.alphabet.index(c) for c in w])
            ids[i, len(w)] = self.eos_token_id
            mask[i, : len(w) + 1] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_the_bias_table_is_the_hf_bias_at_every_offset() -> None:
    m = _tiny()
    table, md = ex.bias_table(m)
    assert md == 128 and table.shape == (4, 257)
    assert ex.offsets_saturate(m, far=20_000)
    sa = m.encoder.block[0].layer[0].SelfAttention
    n = 1500
    hf = sa.compute_bias(n, n)[0]
    q = torch.arange(n)[:, None]
    k = torch.arange(n)[None, :]
    assert torch.equal(table[:, (k - q).clamp(-md, md) + md], hf)


def test_tokenize_refuses_a_count_other_than_l_plus_one() -> None:
    class Doubling:
        unk_token_id, eos_token_id = 2, 1

        def __call__(self, words: list[list[str]], **kw: Any) -> dict[str, Any]:
            n = max(2 * len(w) + 1 for w in words)
            return {
                "input_ids": torch.full((len(words), n), 5),
                "attention_mask": torch.ones(len(words), n, dtype=torch.long),
            }

    with pytest.raises(ValueError, match="tokens for"):
        ex.tokenize(Doubling(), ["MKV"])


def test_tokenize_maps_uzob_to_x_and_ends_with_eos() -> None:
    t = ex.tokenize(_CharTokenizer(), ["ACDUZOB", "MK"])
    assert t.lengths.tolist() == [8, 3]
    u = ex.tokenize(_CharTokenizer(), ["ACDXXXX"])
    assert torch.equal(t.ids[0], u.ids[0])


@pytest.mark.parametrize("n", [5, 129, 300])
def test_every_layer_matches_hf_in_single_precision(n: int) -> None:
    m = _tiny()
    enc = ex.ExactAnkh(m, compiled=False)
    ids = torch.randint(3, 40, (1, n))
    with torch.no_grad():
        ref = m(input_ids=ids, output_hidden_states=True).hidden_states
    out = enc(ex.Tokens(ids, torch.tensor([n])), layers=(0, 1, 2, 3))
    for i in range(4):
        assert torch.allclose(out[i], ref[i], atol=1e-4, rtol=1e-4), (n, i)


def test_a_padded_batch_matches_hf_with_its_attention_mask() -> None:
    m = _tiny()
    enc = ex.ExactAnkh(m, compiled=False)
    lens = [300, 250, 123]
    ids = torch.zeros(3, 300, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, n in enumerate(lens):
        ids[i, :n] = torch.randint(3, 40, (n,))
        mask[i, :n] = 1
    with torch.no_grad():
        ref = m(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[3]
    out = enc(ex.Tokens(ids, torch.tensor(lens)), layers=(3,))[3]
    for i, n in enumerate(lens):
        assert torch.allclose(out[i, :n], ref[i, :n], atol=1e-4, rtol=1e-4), i


def test_a_layer_outside_the_model_is_refused() -> None:
    enc = ex.ExactAnkh(_tiny(), compiled=False)
    with pytest.raises(ValueError, match="layers must be"):
        enc(ex.Tokens(torch.randint(3, 40, (1, 5)), torch.tensor([5])), layers=(4,))


def test_length_batches_keep_the_budget_and_cover_every_sequence() -> None:
    lengths = np.array([10, 3000, 7, 500, 499, 20000, 64])
    batches = ex.length_batches(lengths, budget=2048)
    seen = np.concatenate(batches)
    assert sorted(seen.tolist()) == list(range(len(lengths)))
    for b in batches:
        longest = int(lengths[b].max()) + 1
        assert len(b) == 1 or longest * len(b) <= 2048
    flat = [int(lengths[i]) for b in batches for i in b]
    assert flat == sorted(flat)


def test_encode_per_layer_returns_each_sequence_alone_in_input_order() -> None:
    m = _tiny()
    enc = ex.ExactAnkh(m, compiled=False)
    tok = _CharTokenizer()
    seqs = ["MKVLAAGIV" * 7, "ACD", "WYKLMNPQRST" * 3]
    got = dict(ex.encode_per_layer(enc, tok, seqs, (3, 1), token_budget=4096))
    assert sorted(got) == [0, 1, 2]
    for i, s in enumerate(seqs):
        assert got[i].shape == (len(s), 2, 64) and got[i].dtype == np.float32
        alone = enc(ex.tokenize(tok, [s]), layers=(3, 1))
        assert np.allclose(got[i][:, 0], alone[3][0, : len(s)].numpy(), atol=1e-4)
        assert np.allclose(got[i][:, 1], alone[1][0, : len(s)].numpy(), atol=1e-4)


def test_the_backend_returns_float32_states_per_layer_in_input_order() -> None:
    m = _tiny()
    tok = _CharTokenizer()
    events: list[str] = []
    seqs = ["MKV" * 20, "A", "CDEFG" * 9]
    out = plugin.embed_batch_per_layer(
        m,
        tok,
        seqs,
        emit=lambda name, *a, **k: events.append(name),
        hidden_state_indices=[1, 3],
    )
    assert [o.shape for o in out] == [(60, 2, 64), (1, 2, 64), (45, 2, 64)]
    assert all(o.dtype == np.float32 for o in out)
    assert events == ["backend.ankh.embed_per_layer_done"]
    assert (
        plugin.embed_batch_per_layer(
            m, tok, [], emit=lambda *a, **k: None, hidden_state_indices=[1]
        )
        == []
    )


@GPU
def test_the_compiled_path_matches_hf_on_cuda() -> None:
    m = _tiny("cuda")
    enc = ex.ExactAnkh(m, compiled=True)
    for n in (5, 129, 700):
        ids = torch.randint(3, 40, (1, n), device="cuda")
        with torch.no_grad():
            ref = m(input_ids=ids, output_hidden_states=True).hidden_states
        out = enc(ex.Tokens(ids, torch.tensor([n])), layers=(0, 1, 2, 3))
        for i in range(4):
            assert torch.allclose(out[i], ref[i], atol=1e-4, rtol=1e-4), (n, i)
