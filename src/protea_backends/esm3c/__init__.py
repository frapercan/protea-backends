"""ESM-C embedding backend (esm3c-300m, esm3c-600m, esm3c-6b).

ESM-C ships through the standalone ``esm`` package (not HuggingFace
``transformers``). The model takes raw sequence strings via
``ESMProtein`` — there is no separate tokenizer object — and exposes
hidden states through ``LogitsConfig(return_hidden_states=True)``.

Two API differences from the HF backends:

* **No tokenizer.** ``load_model`` returns ``(model, None)``. Callers
  must tolerate ``tokenizer is None``; ``embed_batch`` ignores the
  ``tokenizer`` argument.
* **Per-sequence inference.** ``model.encode(ESMProtein(sequence=...))``
  is single-protein; the batch is iterated. FP16 autocast on CUDA.

BOS (position 0) and EOS (position -1) tokens are stripped before
residue-level pooling, matching PROTEA's ``_embed_esm3c`` reference
behaviour and the other backends in this package.

Heavy ML deps (``torch``, ``esm``) are imported lazily; install the
runtime stack with ``pip install protea-backends[esm3c]`` (extras
land in F2A.5 of master plan v3).

Example::

    from protea_backends.esm3c import plugin

    model, tok = plugin.load_model(
        "esmc_300m", "cpu", emit=lambda *a, **k: None,
    )
    assert tok is None
    embeddings = plugin.embed_batch(
        model, tok, ["MSEQ"], emit=lambda *a, **k: None,
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np
from protea_contracts import EmbeddingBackend


class EsmcBackend(EmbeddingBackend):
    """ESM-C family backend (standalone ``esm`` SDK, no tokenizer)."""

    name = "esm3c"

    def load_model(
        self,
        model_name: str,
        device: str,
        emit: Any,
    ) -> tuple[Any, Any]:
        """Load an ESM-C checkpoint, move to ``device``, return ``(model, None)``.

        ``model_name`` is an ESM SDK identifier (e.g. ``"esmc_300m"``,
        ``"esmc_600m"``). ``device`` is a torch device string. The
        second return slot is ``None`` because ESM-C does not expose a
        tokenizer object — sequences are passed as raw strings via
        ``ESMProtein(sequence=...)``.
        """
        import torch
        from esm.models.esmc import ESMC

        emit("backend.esm3c.load_start", None, {"model_name": model_name}, "info")
        device_obj = torch.device(device)
        model = ESMC.from_pretrained(model_name)
        model.eval()
        model.to(device)
        if device_obj.type == "cuda":
            model.to(torch.float16)
        emit("backend.esm3c.load_done", None, {"model_name": model_name}, "info")
        return model, None

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

        ``tokenizer`` is ignored (always ``None`` for ESM-C). Each
        sequence is fed individually through ``model.encode`` +
        ``model.logits`` with ``return_hidden_states=True``. BOS / EOS
        are stripped before residue pooling.

        Returns ``(batch_size, hidden_dim)`` float16 ndarray. ``layers``
        selects layer indices (None → last only); ``layer_agg`` is the
        cross-layer aggregator; ``pooling`` reduces residues to a single
        vector. Currently only ``mean`` is fully wired; other pooling
        falls back to mean (full parity with PROTEA's chunked output is
        scheduled for F2A.5 of master plan v3).
        """
        del tokenizer  # ESM-C has no tokenizer.

        import torch
        from esm.sdk.api import ESMProtein, LogitsConfig

        if not sequences:
            return np.zeros((0, 0), dtype=np.float16)

        device_obj = next(model.parameters()).device
        out: list[np.ndarray[Any, Any]] = []

        with torch.no_grad():
            for seq in sequences:
                protein = ESMProtein(sequence=seq)
                with torch.autocast(
                    device_type=device_obj.type,
                    dtype=torch.float16,
                    enabled=(device_obj.type == "cuda"),
                ):
                    protein_tensor = model.encode(protein)
                    logits_output = model.logits(
                        protein_tensor,
                        LogitsConfig(sequence=True, return_hidden_states=True),
                    )

                hs = logits_output.hidden_states
                if hs is None:
                    raise RuntimeError(
                        f"ESM-C returned no hidden_states for sequence {seq[:20]!r}"
                    )

                if isinstance(hs, torch.Tensor):
                    hs = [hs[i] for i in range(hs.shape[0])]

                use_layers = layers if layers else [0]
                stack = torch.stack(
                    [hs[-(li + 1)][0] for li in use_layers], dim=0
                )
                if layer_agg == "mean":
                    layer_pooled = stack.mean(dim=0)
                elif layer_agg == "sum":
                    layer_pooled = stack.sum(dim=0)
                else:
                    layer_pooled = stack.mean(dim=0)

                # Strip BOS (0) and EOS (-1).
                residues = layer_pooled[1:-1, :].float()
                if pooling in ("mean", "max"):
                    pooled = (
                        residues.mean(dim=0)
                        if pooling == "mean"
                        else residues.max(dim=0).values
                    )
                else:
                    pooled = residues.mean(dim=0)
                out.append(pooled.cpu().numpy().astype(np.float16))

                del logits_output, hs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        emit(
            "backend.esm3c.embed_done",
            None,
            {"n_sequences": len(sequences)},
            "info",
        )
        return np.stack(out).astype(np.float16)  # type: ignore[no-any-return]


#: Module-level plugin instance discovered via the
#: ``protea.backends`` entry_points group.
plugin = EsmcBackend()
