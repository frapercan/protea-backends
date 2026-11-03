"""Ankh with exact attention at any length, layer by layer (PROTEA ADR-D51).

Hugging Face's T5 encoder builds a (heads, L, L) score matrix and its position bias, so one
protein of 35991 residues would need about 31 GB per layer for the scores alone, and the
eager path stops near 8300 residues on a 12 GB card. Here every layer's self-attention runs
through FlexAttention. The T5 relative bias enters as a score modifier read from a
(heads, 2 * max_distance + 1) table: the bucket of an offset k - q is constant beyond
+-max_distance (ankh base: 64 buckets, max_distance 128), so the clamped offset indexes it
exactly and no L by L tensor ever exists. Memory is linear in L, compute stays quadratic, and
every residue sees every other, as in Ankh.

Everything else is Hugging Face's own modules with the weights untouched: RMS norms in single
precision, the gated GELU feed forward (``wo`` kept in single precision by
``_keep_in_fp32_modules``, so the residual stream turns single precision after the first block
exactly as there), and no 1/sqrt(d) scale (T5 folds it into the initialisation). Outputs use
Hugging Face's ``hidden_states`` indices: 0 is the embeddings, i in 1..n_layers - 1 the
residual stream after block i - 1, and n_layers the output of the final norm.

FlexAttention accumulates the scores in single precision where Hugging Face rounds them and
the bias to the model's dtype first, so in bfloat16 the two differ by rounding, and the
equivalence tests compare both in single precision.

This is a port of the lab's ``sfjepa.plm.ankh_full`` (sf-jepa repository), which computed the
states the residue vocabulary of ADR-D51 was trained on, and of its length batching.

Triton builds a small C helper the first time a compiled attention runs. If the interpreter
ships no Python.h, point ``PROTEA_PY_INCLUDE`` at an unpacked include directory and it is
added to CPATH.
"""

from __future__ import annotations

import os
import re
import sysconfig
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

#: Default token budget of one batch (rows times padded length), as the lab encoded.
TOKEN_BUDGET = 16384

_COMPILED_FLEX: Any = None


def _python_headers_on_cpath() -> None:
    """Put an unpacked Python include directory on CPATH when the interpreter has none."""
    inc = os.environ.get("PROTEA_PY_INCLUDE")
    if not inc or (Path(sysconfig.get_paths()["include"]) / "Python.h").exists():
        return
    root, v = Path(inc), f"python{sysconfig.get_python_version()}"
    parts = [
        str(root / v),
        str(root / "x86_64-linux-gnu" / v),
        str(root),
        os.environ.get("CPATH", ""),
    ]
    os.environ["CPATH"] = os.pathsep.join(p for p in parts if p)


def flex_attention_fn(compiled: bool) -> Any:
    """FlexAttention, compiled (linear memory, needs Triton) or eager (for tests and the CPU)."""
    global _COMPILED_FLEX
    from torch.nn.attention.flex_attention import flex_attention

    if not compiled:
        return flex_attention
    if _COMPILED_FLEX is None:
        _python_headers_on_cpath()
        _COMPILED_FLEX = torch.compile(flex_attention)
    return _COMPILED_FLEX


def bias_table(model: Any) -> tuple[torch.Tensor, int]:
    """(heads, 2 * max_distance + 1) table of the T5 relative bias by clamped offset k - q.

    Built with Hugging Face's own bucket function, in the model's dtype, so it is exact.
    """
    from transformers.models.t5.modeling_t5 import T5Attention

    sa = model.encoder.block[0].layer[0].SelfAttention
    md = int(sa.relative_attention_max_distance)
    w = sa.relative_attention_bias.weight
    d = torch.arange(-md, md + 1, device=w.device)
    b = T5Attention._relative_position_bucket(
        d, bidirectional=True, num_buckets=sa.relative_attention_num_buckets, max_distance=md
    )
    return w[b].T.contiguous(), md


def offsets_saturate(model: Any, far: int = 200_000) -> bool:
    """Whether every offset beyond +-max_distance falls in the bucket of +-max_distance.

    It is the premise of the clamped table, checked rather than assumed.
    """
    from transformers.models.t5.modeling_t5 import T5Attention

    sa = model.encoder.block[0].layer[0].SelfAttention
    md = int(sa.relative_attention_max_distance)
    nb = sa.relative_attention_num_buckets
    d = torch.arange(md, far)
    pos = T5Attention._relative_position_bucket(
        d, bidirectional=True, num_buckets=nb, max_distance=md
    )
    neg = T5Attention._relative_position_bucket(
        -d, bidirectional=True, num_buckets=nb, max_distance=md
    )
    return bool((pos == pos[0]).all() and (neg == neg[0]).all())


@dataclass
class Tokens:
    """A padded batch: ids ``(B, Lmax)`` and tokens per row (residues plus EOS)."""

    ids: torch.Tensor
    lengths: torch.Tensor


def tokenize(tokenizer: Any, sequences: Sequence[str], max_tokens: int | None = None) -> Tokens:
    """Ankh tokenisation as the backend does it, refusing anything but one token per residue.

    U, Z, O and B become X, the sequence goes in as a list of characters with
    ``is_split_into_words``, and EOS closes it. A count other than L + 1, an ``<unk>`` or a
    missing EOS raises: a tokenizer change once doubled the count.
    """
    cleaned = [re.sub(r"[UZOB]", "X", s) for s in sequences]
    kw: dict[str, Any] = (
        {"truncation": True, "max_length": max_tokens} if max_tokens else {"truncation": False}
    )
    enc = tokenizer(
        [list(c) for c in cleaned],
        padding="longest",
        add_special_tokens=True,
        is_split_into_words=True,
        return_tensors="pt",
        **kw,
    )
    ids, mask = enc["input_ids"], enc["attention_mask"]
    lengths = mask.sum(1)
    for i, c in enumerate(cleaned):
        want = min(len(c) + 1, max_tokens) if max_tokens else len(c) + 1
        if int(lengths[i]) != want:
            raise ValueError(
                f"tokenizer gave {int(lengths[i])} tokens for {len(c)} residues (want {want})"
            )
        if tokenizer.unk_token_id is not None and bool(
            (ids[i, :want] == tokenizer.unk_token_id).any()
        ):
            raise ValueError("tokenizer produced <unk>")
        if int(ids[i, want - 1]) != tokenizer.eos_token_id:
            raise ValueError("last token is not EOS")
    return Tokens(ids, lengths)


class ExactAnkh:
    """``ExactAnkh(model)(tokens, layers)`` gives ``{layer: (B, Lmax, d)}`` by hidden state index.

    ``model`` is a loaded ``T5EncoderModel`` in eval mode. Padded keys are masked; padded query
    rows are computed and meaningless. ``compiled=False`` runs FlexAttention eagerly, which
    materialises the scores and is meant for tests and the CPU.
    """

    def __init__(self, model: Any, compiled: bool = True):
        self.model = model
        enc = model.encoder
        self.n_layers = len(enc.block)
        self.table, self.md = bias_table(model)
        self._lo = torch.tensor(-self.md, device=self.table.device)
        self._hi = torch.tensor(self.md, device=self.table.device)
        sa = enc.block[0].layer[0].SelfAttention
        self.h, self.dk = sa.n_heads, sa.key_value_proj_dim
        self._flex = flex_attention_fn(compiled)

    def _score_mod(self, lengths: torch.Tensor | None) -> Any:
        # The clamp bounds are 0-d tensors: a Python int captured here turns symbolic under the
        # compiler's dynamic shapes and breaks the lowering (torch 2.11).
        table, lo, hi = self.table, self._lo, self._hi

        if lengths is None:

            def mod(
                score: torch.Tensor,
                b: torch.Tensor,
                h: torch.Tensor,
                q: torch.Tensor,
                k: torch.Tensor,
            ) -> torch.Tensor:
                return score + table[h, torch.minimum(torch.maximum(k - q, lo), hi) + hi].to(
                    score.dtype
                )

        else:

            def mod(
                score: torch.Tensor,
                b: torch.Tensor,
                h: torch.Tensor,
                q: torch.Tensor,
                k: torch.Tensor,
            ) -> torch.Tensor:
                s = score + table[h, torch.minimum(torch.maximum(k - q, lo), hi) + hi].to(
                    score.dtype
                )
                return torch.where(k < lengths[b], s, -float("inf"))

        return mod

    def _attention(self, sa: Any, x: torch.Tensor, mod: Any) -> torch.Tensor:
        b, n, _ = x.shape
        q = sa.q(x).view(b, n, self.h, self.dk).transpose(1, 2)
        k = sa.k(x).view(b, n, self.h, self.dk).transpose(1, 2)
        v = sa.v(x).view(b, n, self.h, self.dk).transpose(1, 2)
        o = self._flex(q, k, v, score_mod=mod, scale=1.0)
        return sa.o(o.transpose(1, 2).reshape(b, n, self.h * self.dk))

    @staticmethod
    def _clamp(x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:  # Hugging Face's clamp, only for fp16 (Ankh runs in bf16)
            m = torch.finfo(x.dtype).max
            c = torch.where(torch.isinf(x).any(), m - 1000, m)
            x = torch.clamp(x, min=-c, max=c)
        return x

    def __call__(self, tokens: Tokens, layers: Sequence[int] = (48,)) -> dict[int, torch.Tensor]:
        with torch.no_grad():
            return self._forward(tokens, layers)

    def _forward(self, tokens: Tokens, layers: Sequence[int]) -> dict[int, torch.Tensor]:
        enc = self.model.encoder
        dev = enc.embed_tokens.weight.device
        ids = tokens.ids.to(dev)
        lengths = tokens.lengths.to(dev)
        full = bool((lengths == ids.shape[1]).all())
        mod = self._score_mod(None if full else lengths)
        want = {int(x) for x in layers}
        if not want <= set(range(self.n_layers + 1)):
            raise ValueError(f"layers must be in 0..{self.n_layers}")
        out: dict[int, torch.Tensor] = {}
        h = enc.embed_tokens(ids)
        if 0 in want:
            out[0] = h
        for i, blk in enumerate(enc.block):
            a, f = blk.layer[0], blk.layer[-1]
            h = h + self._attention(a.SelfAttention, a.layer_norm(h), mod)
            h = self._clamp(h)
            h = h + f.DenseReluDense(f.layer_norm(h))
            h = self._clamp(h)
            if i + 1 < self.n_layers and i + 1 in want:
                out[i + 1] = h
        if self.n_layers in want:
            out[self.n_layers] = enc.final_layer_norm(h)
        return out


def length_batches(
    lengths: Sequence[int] | np.ndarray[Any, Any], budget: int = TOKEN_BUDGET
) -> list[np.ndarray[Any, Any]]:
    """Indices grouped by ascending length so that rows times padded length stays within budget.

    A sequence longer than the budget goes alone.
    """
    lengths = np.asarray(lengths)
    order = np.argsort(lengths, kind="stable")
    out: list[np.ndarray[Any, Any]] = []
    cur: list[int] = []
    for i in order:
        n = int(lengths[i]) + 1
        if cur and n * (len(cur) + 1) > budget:
            out.append(np.array(cur))
            cur = []
        cur.append(int(i))
    if cur:
        out.append(np.array(cur))
    return out


def encode_per_layer(
    exact: ExactAnkh,
    tokenizer: Any,
    sequences: Sequence[str],
    hidden_state_indices: Sequence[int],
    token_budget: int = TOKEN_BUDGET,
) -> Iterator[tuple[int, np.ndarray[Any, Any]]]:
    """Every sequence's states, by length batches.

    Yields ``(index into sequences, states)`` with ``states`` a ``(L, n_layers, d)`` float32
    array in the order of ``hidden_state_indices``, EOS stripped. The conversion to float32 is
    exact (bfloat16 and float32 values both fit), so a consumer standardising these values
    computes what it would from the native tensors.
    """
    layers = [int(li) for li in hidden_state_indices]
    lens = np.array([len(s) for s in sequences])
    for b in length_batches(lens, token_budget):
        t = tokenize(tokenizer, [sequences[i] for i in b])
        out = exact(t, layers)
        for j, i in enumerate(b):
            n = int(t.lengths[j]) - 1
            yield int(i), torch.stack([out[li][j, :n].float() for li in layers], 1).cpu().numpy()
