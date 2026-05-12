"""Ankh embedding backend (Ankh-base, Ankh-large).

Ankh is a T5 architecture trained by ElnaggarLab for protein
sequences. It ships through HuggingFace ``T5EncoderModel`` but has
two quirks that distinguish it from generic ProtT5:

* **bfloat16 on CUDA**. Ankh was pre-trained on TPU in bfloat16; its
  LayerNorm overflows in FP16, every forward collapses to NaN. Use
  bfloat16 on CUDA (same VRAM footprint as FP16 but FP32 dynamic
  range) and FP32 on CPU. Verified on ``ElnaggarLab/ankh-base``
  2026-04-10.
* **is_split_into_words=True** for tokenisation. Ankh's SentencePiece
  tokeniser maps a literal space to ``<unk>``, so the space-joined
  path used by ProtT5/ProstT5 produces ~50% ``<unk>`` tokens. Pass
  the sequence as a list of characters with
  ``is_split_into_words=True`` so each residue becomes one word.

Heavy ML deps (torch, transformers) are imported lazily; install the
runtime stack with ``pip install protea-backends[ankh]`` (extras land
in F2A.5 of master plan v3).

Three output paths:

* :meth:`AnkhBackend.embed_batch` returns the historical ``(B, D)``
  mean-pooled ``float16`` matrix.
* :meth:`AnkhBackend.embed_batch_per_residue` returns an
  :class:`protea_contracts.EmbeddingPayload` with ``B`` ragged
  ``(L_i, D)`` ``float16`` tensors and matching attention masks
  (trailing EOS already stripped). Wired in MIL.1b alongside T5 and
  ESM-C.
* :meth:`AnkhBackend.embed_chunks` returns one
  ``list[ChunkEmbedding]`` per sequence and is the bit-exact home of
  PROTEA's legacy ``_embed_ankh`` pipeline (T2A.3): multi-layer
  selection / aggregation, per-residue normalisation, chunking, and
  ``mean`` / ``max`` / ``mean_max`` / ``cls`` pooling. ``protea-core``
  dispatches to it from ``compute_embeddings._dispatch_embed``.
  Internally delegates to :func:`protea_backends.t5.embed_chunks_with_mode`
  with ``use_aa2fold=False, split_into_words=True`` to reuse the
  shared T5 batched pipeline without code duplication.

Example::

    from protea_backends.ankh import plugin

    model, tok = plugin.load_model(
        "ElnaggarLab/ankh-base",
        "cpu",
        emit=lambda *a, **k: None,
    )
    embeddings = plugin.embed_batch(
        model, tok, ["MSEQ"], emit=lambda *a, **k: None,
    )
    per_residue = plugin.embed_batch_per_residue(
        model, tok, ["MSEQ"], emit=lambda *a, **k: None,
    )
"""

from __future__ import annotations

import re
from typing import Any, cast

import numpy as np
from protea_contracts import EmbeddingBackend, EmbeddingPayload

from protea_backends._chunk_helpers import ChunkEmbedding
from protea_backends.t5 import T5Mode, embed_chunks_with_mode

#: Tokenisation mode for Ankh: never inject the ``<AA2fold>`` prefix
#: and tokenise via ``is_split_into_words=True`` (Ankh's SentencePiece
#: tokeniser maps a literal space to ``<unk>``). Module-level constant
#: so the :meth:`AnkhBackend.embed_chunks` default stays a 5-arg shape
#: and ruff B008 does not flag a callable default argument.
_ANKH_MODE = T5Mode(use_aa2fold=False, split_into_words=True)


class AnkhBackend(EmbeddingBackend):
    """Ankh family backend (HuggingFace T5EncoderModel + AutoTokenizer)."""

    name = "ankh"

    def load_model(
        self,
        model_name: str,
        device: str,
        emit: Any,
    ) -> tuple[Any, Any]:
        """Load an Ankh checkpoint + AutoTokenizer, move to ``device``.

        ``model_name`` is a HuggingFace identifier (e.g.
        ``"ElnaggarLab/ankh-base"`` or ``"ElnaggarLab/ankh-large"``).
        Returns ``(model, tokenizer)``. **bfloat16** on CUDA (FP16
        LayerNorm collapses to NaN), FP32 on CPU.

        AutoTokenizer is used (resolves to T5TokenizerFast) instead of
        hardcoding T5Tokenizer; some Ankh revisions changed the
        tokenizer class and AutoTokenizer rides those changes.
        """
        import torch
        from transformers import AutoTokenizer, T5EncoderModel

        emit("backend.ankh.load_start", None, {"model_name": model_name}, "info")
        device_obj = torch.device(device)
        # Ankh-specific: bfloat16 on CUDA, fp32 on CPU.
        dtype = torch.bfloat16 if device_obj.type == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5EncoderModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype=dtype,
        )
        model.eval()
        model.to(device)
        emit(
            "backend.ankh.load_done",
            None,
            {"model_name": model_name, "dtype": str(dtype)},
            "info",
        )
        return model, tokenizer

    def embed_batch(
        self,
        model: Any,
        tokenizer: Any,
        sequences: list[str],
        *,
        emit: Any,
        layers: list[int] | None = None,
        layer_agg: str = "mean",
        pooling: str = "mean",
    ) -> np.ndarray[Any, Any]:
        """Run inference on a batch and return mean-pooled embeddings.

        Tokenises each sequence as a list of characters with
        ``is_split_into_words=True`` so the SentencePiece tokeniser
        treats every residue as one word and never falls back to
        ``<unk>``. Strips trailing EOS to align with the other
        backends. Returns ``(batch_size, hidden_dim)`` float16 ndarray
        (output cast back to fp16 for storage parity even though the
        model runs in bfloat16).
        """
        import torch

        if not sequences:
            return np.zeros((0, 0), dtype=np.float16)

        residue_tensors = self._compute_residue_tensors(
            model, tokenizer, sequences, layers=layers, layer_agg=layer_agg
        )

        out: list[np.ndarray[Any, Any]] = []
        for residues in residue_tensors:
            pooled = _pool_residues(residues, pooling)
            out.append(pooled.cpu().numpy().astype(np.float16))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        emit(
            "backend.ankh.embed_done",
            None,
            {"n_sequences": len(sequences)},
            "info",
        )
        return cast("np.ndarray[Any, Any]", np.stack(out).astype(np.float16))

    def embed_batch_per_residue(
        self,
        model: Any,
        tokenizer: Any,
        sequences: list[str],
        *,
        emit: Any,
        layers: list[int] | None = None,
    ) -> EmbeddingPayload:
        """Run inference and return per-residue embeddings + masks.

        Returns an :class:`EmbeddingPayload` with
        ``granularity="per_residue"``. Each entry in ``residues`` is the
        ``(L_i, hidden_dim)`` ``float16`` tensor for one sequence, with
        the trailing EOS already stripped so ``residues[i][j]`` is the
        embedding of the ``j``-th amino acid of ``sequences[i]`` (Ankh
        has no CLS / no AA2fold prefix). The matching
        ``attention_mask[i]`` is all-ones (kept for forward compat with
        padded-batch consumers).
        """
        if not sequences:
            return EmbeddingPayload(
                granularity="per_residue",
                residues=[],
                attention_mask=[],
            )

        residue_tensors = self._compute_residue_tensors(
            model, tokenizer, sequences, layers=layers, layer_agg="mean"
        )

        residues_np = [r.cpu().numpy().astype(np.float16) for r in residue_tensors]
        masks = [np.ones((r.shape[0],), dtype=bool) for r in residues_np]

        emit(
            "backend.ankh.embed_per_residue_done",
            None,
            {
                "n_sequences": len(sequences),
                "total_residues": int(sum(r.shape[0] for r in residues_np)),
            },
            "info",
        )
        return EmbeddingPayload(
            granularity="per_residue",
            residues=residues_np,
            attention_mask=masks,
        )

    def embed_chunks(
        self,
        model: Any,
        tokenizer: Any,
        sequences: list[str],
        config: Any,
        device: str,
    ) -> list[list[ChunkEmbedding]]:
        """Embed sequences with Ankh and return PROTEA's chunked output.

        Bit-exact port of PROTEA's pre-plugin ``_embed_ankh`` pipeline
        (T2A.3 of master plan v3.2). Delegates to the shared T5
        :func:`protea_backends.t5.embed_chunks_with_mode` helper with
        Ankh's tokenisation mode (no ``<AA2fold>`` prefix, list-of-chars
        tokenisation) so the layer / chunk / pool logic stays in one
        place.

        ``config`` is duck-typed to PROTEA's ``EmbeddingConfig`` (any
        object with ``model_name``, ``max_length``, ``layer_indices``,
        ``layer_agg``, ``pooling``, ``normalize``, ``normalize_residues``,
        ``use_chunking``, ``chunk_size`` and ``chunk_overlap``).
        """
        return embed_chunks_with_mode(
            model, tokenizer, sequences, config, device, _ANKH_MODE
        )

    def _compute_residue_tensors(
        self,
        model: Any,
        tokenizer: Any,
        sequences: list[str],
        *,
        layers: list[int] | None,
        layer_agg: str,
    ) -> list[Any]:
        """Run Ankh forward + return one residue-level torch tensor per sequence.

        Shared core of :meth:`embed_batch` and
        :meth:`embed_batch_per_residue`. Tokenises with
        ``is_split_into_words=True`` and strips the trailing EOS so
        residue indices align with amino-acid positions. Tensors are
        float32; callers cast to float16 on the way out.
        """
        import torch

        cleaned = [re.sub(r"[UZOB]", "X", s) for s in sequences]
        device_obj = next(model.parameters()).device

        # Ankh-specific: list-of-chars with is_split_into_words=True.
        inputs = tokenizer.batch_encode_plus(
            [list(c) for c in cleaned],
            padding="longest",
            truncation=True,
            add_special_tokens=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device_obj) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states
        del outputs

        layer_pooled = _aggregate_layers(hidden_states, layers, layer_agg)
        attn = inputs["attention_mask"]

        residue_tensors: list[Any] = []
        for i in range(layer_pooled.shape[0]):
            actual_len = int(attn[i].sum().item())
            # No CLS / no AA2fold; only strip trailing EOS.
            residues = layer_pooled[i, : actual_len - 1, :].float()
            residue_tensors.append(residues)
        return residue_tensors


def _pool_residues(residues: Any, pooling: str) -> Any:
    """Collapse a ``(L_i, D)`` residue tensor to a single ``(D,)`` vector."""
    if pooling == "max":
        return residues.max(dim=0).values
    return residues.mean(dim=0)


def _aggregate_layers(
    hidden_states: Any, layers: list[int] | None, layer_agg: str
) -> Any:
    """Stack the selected Ankh layers and aggregate across them.

    Operates on the batched tensors ``hidden_states[k]`` of shape
    ``(B, L, D)`` (T5-style padded batch) and returns a ``(B, L, D)``
    tensor after mean / sum aggregation across the selected layers.
    """
    import torch

    use_layers = layers if layers else [0]
    stack = torch.stack(
        [hidden_states[-(li + 1)] for li in use_layers], dim=0
    )
    if layer_agg == "sum":
        return stack.sum(dim=0)
    # "mean" and any unknown value fall back to mean.
    return stack.mean(dim=0)


#: Module-level plugin instance discovered via the
#: ``protea.backends`` entry_points group.
plugin = AnkhBackend()
