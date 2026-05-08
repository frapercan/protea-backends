"""ESM family embedding backend (ESM-1b, ESM-2 sizes 8M to 15B).

Implements the :class:`protea_contracts.EmbeddingBackend` ABC for
HuggingFace ``EsmModel`` checkpoints. Heavy ML deps (torch,
transformers) are imported lazily inside ``load_model`` /
``embed_batch`` so the plugin module itself stays cheap to import:
``protea-core`` discovers the entry_point at startup without paying
for a torch import unless the backend is actually used.

Install the runtime stack with ``pip install protea-backends[esm]``
(extras land in F2A.5 of master plan v3).

Example::

    from protea_backends.esm import plugin

    model, tokenizer = plugin.load_model(
        "facebook/esm2_t12_35M_UR50D", "cpu", emit=lambda *a, **k: None
    )
    embeddings = plugin.embed_batch(
        model, tokenizer, ["MSEQ"], emit=lambda *a, **k: None
    )
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from protea_contracts import EmbeddingBackend


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
        import torch

        if not sequences:
            return np.zeros((0, 0), dtype=np.float16)

        out: list[np.ndarray[Any, Any]] = []
        device_obj = next(model.parameters()).device
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
                # Last layer if no selection.
                use_layers = layers if layers else [0]
                # Stack selected layers and average across them.
                stack = torch.stack(
                    [hidden_states[-(li + 1)][0] for li in use_layers], dim=0
                )
                if layer_agg == "mean":
                    layer_pooled = stack.mean(dim=0)
                elif layer_agg == "sum":
                    layer_pooled = stack.sum(dim=0)
                else:
                    layer_pooled = stack.mean(dim=0)

                # Strip CLS (pos 0) + EOS (last valid).
                actual_len = int(tokens["attention_mask"].sum().item())
                residues = layer_pooled[1 : actual_len - 1, :].float()
                if pooling in ("mean", "max"):
                    pooled = residues.mean(dim=0) if pooling == "mean" else residues.max(dim=0).values
                else:
                    pooled = residues.mean(dim=0)
                out.append(pooled.cpu().numpy().astype(np.float16))

                del outputs, hidden_states
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        emit(
            "backend.esm.embed_done",
            None,
            {"n_sequences": len(sequences)},
            "info",
        )
        return cast("np.ndarray[Any, Any]", np.stack(out).astype(np.float16))


#: Module-level plugin instance discovered via the
#: ``protea.backends`` entry_points group.
plugin = EsmBackend()
