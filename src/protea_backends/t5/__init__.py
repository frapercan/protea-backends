"""T5-style embedding backend (ProtT5-XL, ProstT5).

Implements :class:`protea_contracts.EmbeddingBackend` for HuggingFace
``T5EncoderModel`` checkpoints. Supports two SentencePiece tokenisation
modes:

* **Space-joined** (``split_into_words=False``, default): the sequence is
  passed as ``"A C D E …"``. Matches ProtT5-XL and ProstT5 conventions.
* **List-of-chars** (``split_into_words=True``): the sequence is passed
  as ``list("ACDE…")`` with ``is_split_into_words=True``. Used by the
  Ankh family on top of T5 (separate plugin in ``protea_backends.ankh``).

ProstT5 mode is auto-detected from ``model_name`` (substring
``"prostt5"``); callers may force it via ``aa2fold=False``.

Heavy ML deps (torch, transformers) are imported lazily inside
``load_model`` / ``embed_batch`` so module import stays cheap.

Install runtime stack with ``pip install protea-backends[t5]``
(extras land in F2A.5 of master plan v3).

Two output paths:

* :meth:`T5Backend.embed_batch` returns the historical ``(B, D)``
  mean-pooled ``float16`` matrix.
* :meth:`T5Backend.embed_batch_per_residue` returns an
  :class:`protea_contracts.EmbeddingPayload` with ``B`` ragged
  ``(L_i, D)`` ``float16`` tensors and matching attention masks
  (AA2fold prefix and EOS already stripped). Wired in MIL.1b to
  match the ESM precedent from MIL.1a.

Example::

    from protea_backends.t5 import plugin

    model, tok = plugin.load_model(
        "Rostlab/prot_t5_xl_uniref50",
        "cpu",
        emit=lambda *a, **k: None,
    )
    embeddings = plugin.embed_batch(
        model, tok, ["MSEQ"], emit=lambda *a, **k: None
    )
    per_residue = plugin.embed_batch_per_residue(
        model, tok, ["MSEQ"], emit=lambda *a, **k: None
    )
"""

from __future__ import annotations

import re
from typing import Any, cast

import numpy as np
from protea_contracts import EmbeddingBackend, EmbeddingPayload


class T5Backend(EmbeddingBackend):
    """ProtT5 / ProstT5 backend (HuggingFace T5EncoderModel)."""

    name = "t5"

    def load_model(
        self,
        model_name: str,
        device: str,
        emit: Any,
    ) -> tuple[Any, Any]:
        """Load a T5 encoder + SentencePiece tokenizer, move to ``device``.

        ``model_name`` is a HuggingFace identifier (e.g.
        ``"Rostlab/prot_t5_xl_uniref50"`` or ``"Rostlab/ProstT5"``).
        Returns ``(model, tokenizer)``. fp16 on CUDA, fp32 on CPU.
        """
        import torch
        from transformers import T5EncoderModel, T5Tokenizer

        emit("backend.t5.load_start", None, {"model_name": model_name}, "info")
        device_obj = torch.device(device)
        dtype = torch.float16 if device_obj.type == "cuda" else torch.float32
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype=dtype,
        )
        model.eval()
        model.to(device)
        emit("backend.t5.load_done", None, {"model_name": model_name}, "info")
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

        Replaces ambiguous amino acids (U/Z/O/B → X) before tokenisation.
        Auto-detects ProstT5 from the model's name attribute on the
        config and prepends the ``<AA2fold>`` prefix when applicable.
        Strips the AA2fold prefix and trailing EOS so residue indices
        line up with amino-acid positions across backends.

        Returns ``(batch_size, hidden_dim)`` float16 ndarray.
        """
        import torch

        if not sequences:
            return np.zeros((0, 0), dtype=np.float16)

        residue_tensors, use_aa2fold = self._compute_residue_tensors(
            model, tokenizer, sequences, layers=layers, layer_agg=layer_agg
        )

        out: list[np.ndarray[Any, Any]] = []
        for residues in residue_tensors:
            pooled = _pool_residues(residues, pooling)
            out.append(pooled.cpu().numpy().astype(np.float16))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        emit(
            "backend.t5.embed_done",
            None,
            {"n_sequences": len(sequences), "use_aa2fold": use_aa2fold},
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
        the AA2fold prefix (ProstT5 only) and the trailing EOS already
        stripped so ``residues[i][j]`` is the embedding of the ``j``-th
        amino acid of ``sequences[i]``. The matching ``attention_mask[i]``
        is all-ones (kept for forward compat with padded-batch consumers).
        """
        if not sequences:
            return EmbeddingPayload(
                granularity="per_residue",
                residues=[],
                attention_mask=[],
            )

        residue_tensors, _use_aa2fold = self._compute_residue_tensors(
            model, tokenizer, sequences, layers=layers, layer_agg="mean"
        )

        residues_np = [r.cpu().numpy().astype(np.float16) for r in residue_tensors]
        masks = [np.ones((r.shape[0],), dtype=bool) for r in residues_np]

        emit(
            "backend.t5.embed_per_residue_done",
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

    def _compute_residue_tensors(
        self,
        model: Any,
        tokenizer: Any,
        sequences: list[str],
        *,
        layers: list[int] | None,
        layer_agg: str,
    ) -> tuple[list[Any], bool]:
        """Run T5 forward + return one residue-level torch tensor per sequence.

        Shared core of :meth:`embed_batch` and
        :meth:`embed_batch_per_residue`. Strips the AA2fold prefix (1
        token, ProstT5 only) and the trailing EOS so residue indices
        align with amino-acid positions. Tensors are float32; callers
        cast to float16 on the way out. Returns ``(tensors, use_aa2fold)``
        so the caller can emit the flag in its event payload.
        """
        import torch

        config_name = getattr(model.config, "name_or_path", "") or ""
        use_aa2fold = "prostt5" in str(config_name).lower()

        cleaned = [re.sub(r"[UZOB]", "X", s) for s in sequences]
        processed = [("<AA2fold> " if use_aa2fold else "") + " ".join(c) for c in cleaned]

        device_obj = next(model.parameters()).device
        inputs = tokenizer.batch_encode_plus(
            processed,
            padding="longest",
            truncation=True,
            add_special_tokens=True,
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
        start_idx = 1 if use_aa2fold else 0
        attn = inputs["attention_mask"]

        residue_tensors: list[Any] = []
        for i in range(layer_pooled.shape[0]):
            actual_len = int(attn[i].sum().item())
            residues = layer_pooled[i, start_idx : actual_len - 1, :].float()
            residue_tensors.append(residues)
        return residue_tensors, use_aa2fold


def _pool_residues(residues: Any, pooling: str) -> Any:
    """Collapse a ``(L_i, D)`` residue tensor to a single ``(D,)`` vector."""
    if pooling == "max":
        return residues.max(dim=0).values
    return residues.mean(dim=0)


def _aggregate_layers(
    hidden_states: Any, layers: list[int] | None, layer_agg: str
) -> Any:
    """Stack the selected T5 layers and aggregate across them.

    Operates on the batched tensors ``hidden_states[k]`` of shape
    ``(B, L, D)`` (T5 is padded-batch) and returns a ``(B, L, D)``
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
plugin = T5Backend()
