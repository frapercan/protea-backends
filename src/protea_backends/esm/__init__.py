"""ESM family embedding backend (ESM-1b, ESM-2 sizes 8M to 15B).

Implements the :class:`protea_contracts.EmbeddingBackend` ABC for
HuggingFace ``EsmModel`` checkpoints. Heavy ML deps (torch,
transformers) are imported lazily inside ``load_model`` /
``embed_batch`` so the plugin module itself stays cheap to import:
``protea-core`` discovers the entry_point at startup without paying
for a torch import unless the backend is actually used.

Install the runtime stack with ``pip install protea-backends[esm]``
(extras land in F2A.5 of master plan v3).

Two output paths:

* :meth:`EsmBackend.embed_batch` returns the historical ``(B, D)``
  mean-pooled ``float16`` matrix.
* :meth:`EsmBackend.embed_batch_per_residue` returns an
  :class:`protea_contracts.EmbeddingPayload` with ``B`` ragged
  ``(L_i, D)`` ``float16`` tensors and matching attention masks
  (CLS / EOS stripped). Wired in MIL.1a so MIL pooling heads and
  patch-level features in PROTEA can consume residue-level output
  without re-tokenising.

Example::

    from protea_backends.esm import plugin

    model, tokenizer = plugin.load_model(
        "facebook/esm2_t12_35M_UR50D", "cpu", emit=lambda *a, **k: None
    )
    embeddings = plugin.embed_batch(
        model, tokenizer, ["MSEQ"], emit=lambda *a, **k: None
    )
    per_residue = plugin.embed_batch_per_residue(
        model, tokenizer, ["MSEQ"], emit=lambda *a, **k: None
    )
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from protea_contracts import EmbeddingBackend, EmbeddingPayload


class EsmBackend(EmbeddingBackend):
    """ESM family backend (HuggingFace EsmModel + AutoTokenizer)."""

    name = "esm"

    def load_model(
        self,
        model_name: str,
        device: str,
        emit: Any,
    ) -> tuple[Any, Any]:
        """Load an ESM checkpoint + tokenizer, move to ``device``.

        ``model_name`` is a HuggingFace identifier (e.g.
        ``"facebook/esm2_t12_35M_UR50D"``). ``device`` is a torch
        device string (``"cpu"``, ``"cuda:0"``, etc.).

        Returns ``(model, tokenizer)``. The model is in eval mode and
        cast to fp16 on CUDA / fp32 on CPU.
        """
        import torch
        from transformers import AutoTokenizer, EsmModel

        emit("backend.esm.load_start", None, {"model_name": model_name}, "info")
        device_obj = torch.device(device)
        dtype = torch.float16 if device_obj.type == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name, output_hidden_states=True)
        model.eval()
        model.to(device)
        model.to(dtype)
        emit("backend.esm.load_done", None, {"model_name": model_name}, "info")
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

        Per-sequence processing (variable lengths) with the ``CLS`` and
        ``EOS`` tokens excluded from residue-level operations. The
        return shape is ``(batch_size, hidden_dim)`` as float16.

        ``layers`` selects transformer layers (None = last only).
        ``layer_agg`` aggregates across selected layers; ``pooling``
        aggregates residues to a single vector. Currently only
        ``"mean"`` pooling is implemented; other values fall back to
        mean for now (full parity with PROTEA's chunked output is
        scheduled for F2A.5 of master plan v3).
        """
        if not sequences:
            return np.zeros((0, 0), dtype=np.float16)

        residue_tensors = self._compute_residue_tensors(
            model, tokenizer, sequences, layers=layers, layer_agg=layer_agg
        )

        out: list[np.ndarray[Any, Any]] = []
        for residues in residue_tensors:
            pooled = _pool_residues(residues, pooling)
            out.append(pooled.cpu().numpy().astype(np.float16))

        emit(
            "backend.esm.embed_done",
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
        ``(L_i, hidden_dim)`` ``float16`` tensor for one sequence, CLS
        and EOS already stripped so ``residues[i][j]`` is the embedding
        of the ``j``-th amino acid of ``sequences[i]``. The matching
        ``attention_mask[i]`` is all-ones in MIL.1a (kept for forward
        compat with padded-batch consumers).

        ``layers`` selects transformer layers (None = last only).
        Layer aggregation across selected layers is fixed at ``mean``
        for this method; multi-layer concat for per-residue output is a
        MIL.2 concern.
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
            "backend.esm.embed_per_residue_done",
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
    ) -> list[Any]:
        """Run ESM forward + return one residue-level torch tensor per sequence.

        Shared core of :meth:`embed_batch` and
        :meth:`embed_batch_per_residue`. Strips CLS (pos 0) + EOS
        (last valid) so residue indices align with amino-acid
        positions. Tensors are float32 (caller casts to float16 on
        the way out).
        """
        import torch

        device_obj = next(model.parameters()).device
        residue_tensors: list[Any] = []

        with torch.no_grad():
            for seq in sequences:
                tokens = tokenizer(
                    seq,
                    return_tensors="pt",
                    truncation=True,
                    add_special_tokens=True,
                )
                tokens = {k: v.to(device_obj) for k, v in tokens.items()}
                outputs = model(**tokens, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # tuple of (1, L, D)

                layer_pooled = _aggregate_layers(hidden_states, layers, layer_agg)

                actual_len = int(tokens["attention_mask"].sum().item())
                residues = layer_pooled[1 : actual_len - 1, :].float()
                residue_tensors.append(residues)

                del outputs, hidden_states
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return residue_tensors


def _pool_residues(residues: Any, pooling: str) -> Any:
    """Collapse a ``(L_i, D)`` residue tensor to a single ``(D,)`` vector."""
    if pooling == "max":
        return residues.max(dim=0).values
    return residues.mean(dim=0)


def _aggregate_layers(
    hidden_states: Any, layers: list[int] | None, layer_agg: str
) -> Any:
    """Stack the selected ESM layers and aggregate across them."""
    import torch

    use_layers = layers if layers else [0]
    stack = torch.stack(
        [hidden_states[-(li + 1)][0] for li in use_layers], dim=0
    )
    if layer_agg == "sum":
        return stack.sum(dim=0)
    # "mean" and any unknown value fall back to mean.
    return stack.mean(dim=0)


#: Module-level plugin instance discovered via the
#: ``protea.backends`` entry_points group.
plugin = EsmBackend()
