"""ESM-C embedding backend (esm3c-300m, esm3c-600m, esm3c-6b).

ESM-C ships through the standalone ``esm`` package (not HuggingFace
``transformers``). The model takes raw sequence strings via
``ESMProtein`` (there is no separate tokenizer object) and exposes
hidden states through ``LogitsConfig(return_hidden_states=True)``.

Two API differences from the HF backends:

* **No tokenizer.** ``load_model`` returns ``(model, None)``. Callers
  must tolerate ``tokenizer is None``; the embed methods ignore the
  ``tokenizer`` argument.
* **Per-sequence inference.** ``model.encode(ESMProtein(sequence=...))``
  is single-protein; the batch is iterated. FP16 autocast on CUDA.

BOS (position 0) and EOS (position -1) tokens are stripped before
residue-level pooling, matching PROTEA's ``_embed_esm3c`` reference
behaviour and the other backends in this package.

Heavy ML deps (``torch``, ``esm``) are imported lazily; install the
runtime stack with ``pip install protea-backends[esm3c]`` (extras
land in F2A.5 of master plan v3).

Three output paths:

* :meth:`EsmcBackend.embed_batch` returns the historical ``(B, D)``
  mean-pooled ``float16`` matrix.
* :meth:`EsmcBackend.embed_batch_per_residue` returns an
  :class:`protea_contracts.EmbeddingPayload` with ``B`` ragged
  ``(L_i, D)`` ``float16`` tensors and matching attention masks
  (BOS / EOS already stripped). Wired in MIL.1b alongside T5 and
  Ankh.
* :meth:`EsmcBackend.embed_chunks` returns one
  ``list[ChunkEmbedding]`` per sequence and is the bit-exact home of
  PROTEA's legacy ``_embed_esm3c`` pipeline (T2A.4): multi-layer
  selection / aggregation, per-residue normalisation, chunking, and
  ``mean`` / ``max`` / ``mean_max`` / ``cls`` pooling. ``protea-core``
  dispatches to it from ``compute_embeddings._dispatch_embed``. The
  ``tokenizer`` argument is accepted for signature parity with the
  other plugins and is ignored (ESM-C has no tokenizer object).

Example::

    from protea_backends.esm3c import plugin

    model, tok = plugin.load_model(
        "esmc_300m", "cpu", emit=lambda *a, **k: None,
    )
    assert tok is None
    embeddings = plugin.embed_batch(
        model, tok, ["MSEQ"], emit=lambda *a, **k: None,
    )
    per_residue = plugin.embed_batch_per_residue(
        model, tok, ["MSEQ"], emit=lambda *a, **k: None,
    )
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from protea_contracts import EmbeddingBackend, EmbeddingPayload

from protea_backends._chunk_helpers import (
    ChunkEmbedding,
    aggregate_1d,
    aggregate_residue_layers,
    chunk_and_pool,
    validate_layers,
)


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
        tokenizer object (sequences are passed as raw strings via
        ``ESMProtein(sequence=...)``).
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
        selects layer indices (None = last only); ``layer_agg`` is the
        cross-layer aggregator; ``pooling`` reduces residues to a single
        vector. Currently only ``mean`` is fully wired; other pooling
        falls back to mean (full parity with PROTEA's chunked output is
        scheduled for F2A.5 of master plan v3).
        """
        del tokenizer  # ESM-C has no tokenizer.
        import torch

        if not sequences:
            return np.zeros((0, 0), dtype=np.float16)

        residue_tensors = self._compute_residue_tensors(
            model, sequences, layers=layers, layer_agg=layer_agg
        )

        out: list[np.ndarray[Any, Any]] = []
        for residues in residue_tensors:
            pooled = _pool_residues(residues, pooling)
            out.append(pooled.cpu().numpy().astype(np.float16))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        emit(
            "backend.esm3c.embed_done",
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

        ``tokenizer`` is ignored (always ``None`` for ESM-C). Returns an
        :class:`EmbeddingPayload` with ``granularity="per_residue"``.
        Each entry in ``residues`` is the ``(L_i, hidden_dim)``
        ``float16`` tensor for one sequence, BOS and EOS already
        stripped so ``residues[i][j]`` is the embedding of the ``j``-th
        amino acid of ``sequences[i]``. The matching ``attention_mask[i]``
        is all-ones (kept for forward compat).
        """
        del tokenizer  # ESM-C has no tokenizer.

        if not sequences:
            return EmbeddingPayload(
                granularity="per_residue",
                residues=[],
                attention_mask=[],
            )

        residue_tensors = self._compute_residue_tensors(
            model, sequences, layers=layers, layer_agg="mean"
        )

        residues_np = [r.cpu().numpy().astype(np.float16) for r in residue_tensors]
        masks = [np.ones((r.shape[0],), dtype=bool) for r in residues_np]

        emit(
            "backend.esm3c.embed_per_residue_done",
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
        """Embed sequences with ESM-C and return PROTEA's chunked output.

        Bit-exact port of PROTEA's pre-plugin ``_embed_esm3c`` pipeline
        (T2A.4 of master plan v3.2). Iterates sequence by sequence
        through ``model.encode`` + ``model.logits`` with
        ``return_hidden_states=True`` (the ESM SDK does not support
        batched inference), truncates each sequence to
        ``config.max_length`` before encoding, and routes the resulting
        hidden states through the shared ``_chunk_helpers`` pipeline.

        ``tokenizer`` is accepted for signature parity with the other
        plugins and is ignored (ESM-C has no tokenizer object).

        ``config`` is duck-typed to PROTEA's ``EmbeddingConfig`` (any
        object with ``model_name``, ``max_length``, ``layer_indices``,
        ``layer_agg``, ``pooling``, ``normalize``, ``normalize_residues``,
        ``use_chunking``, ``chunk_size`` and ``chunk_overlap``).
        """
        del tokenizer  # ESM-C has no tokenizer.
        return embed_chunks_esm3c(model, sequences, config, device)

    def _compute_residue_tensors(
        self,
        model: Any,
        sequences: list[str],
        *,
        layers: list[int] | None,
        layer_agg: str,
    ) -> list[Any]:
        """Run ESM-C forward + return one residue-level torch tensor per sequence.

        Shared core of :meth:`embed_batch` and
        :meth:`embed_batch_per_residue`. Iterates sequence by sequence
        through ``model.encode`` + ``model.logits`` with
        ``return_hidden_states=True``; strips BOS (pos 0) and EOS
        (pos -1) so residue indices align with amino-acid positions.
        Tensors are float32; callers cast to float16 on the way out.
        """
        import torch
        from esm.sdk.api import ESMProtein, LogitsConfig

        device_obj = next(model.parameters()).device
        residue_tensors: list[Any] = []

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
                layer_pooled = _aggregate_layers(hs, layers, layer_agg)
                # Strip BOS (0) and EOS (-1).
                residues = layer_pooled[1:-1, :].float()
                residue_tensors.append(residues)

                del logits_output, hs
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
    """Stack the selected ESM-C layers and aggregate across them.

    ESM-C returns per-sequence ``hidden_states`` either as a tuple/list
    of layer tensors (each ``(1, L, D)``) or as a single tensor of
    shape ``(num_layers, 1, L, D)`` or ``(num_layers, L, D)``. The
    function normalises both forms to a list of ``(L, D)`` tensors, then
    stacks across the selected layers and aggregates.
    """
    import torch

    use_layers = layers if layers else [0]
    layer_views = [_squeeze_layer(hidden_states[-(li + 1)]) for li in use_layers]
    stack = torch.stack(layer_views, dim=0)
    if layer_agg == "sum":
        return stack.sum(dim=0)
    # "mean" and any unknown value fall back to mean.
    return stack.mean(dim=0)


def _squeeze_layer(layer: Any) -> Any:
    """Normalise one ESM-C layer's hidden state to a ``(L, D)`` tensor.

    The SDK is permissive about batch dimensions; per-sequence
    inference can emit either ``(L, D)`` or ``(1, L, D)``. Peel the
    leading batch axis when it is unit-sized so the residue extractor
    can always index ``layer[1:-1, :]``.
    """
    if layer.dim() == 3 and layer.shape[0] == 1:
        return layer[0]
    return layer


def embed_chunks_esm3c(
    model: Any,
    sequences: list[str],
    config: Any,
    device: str,
) -> list[list[ChunkEmbedding]]:
    """Module-level chunked pipeline for ESM-C.

    Iterates ``sequences`` and runs the SDK forward pass per sequence,
    then dispatches to :func:`_embed_chunks_one` for layer selection,
    residue pooling and chunking. Wrapped in :class:`torch.no_grad` for
    parity with the legacy PROTEA shim. ``device`` is accepted for
    signature parity with the other backends; the actual device is read
    from the loaded model's parameters.
    """
    import torch

    device_obj = torch.device(device) if isinstance(device, str) else device
    results: list[list[ChunkEmbedding]] = []
    with torch.no_grad():
        for seq_str in sequences:
            results.append(_embed_chunks_one(model, seq_str, config, device_obj))
    return results


def _embed_chunks_one(
    model: Any,
    seq_str: str,
    config: Any,
    device_obj: Any,
) -> list[ChunkEmbedding]:
    """Forward pass + pooling for one ESM-C sequence.

    Truncates ``seq_str`` to ``config.max_length`` before encoding,
    strips BOS (position 0) and EOS (position -1) before residue-level
    pooling, and applies the same chunk / pool / normalise logic as the
    legacy ``_embed_esm3c`` shim in PROTEA. CLS pooling reads position
    0 of each selected layer (before stripping); residue pooling reads
    positions ``1:-1``.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812  PyTorch convention
    from esm.sdk.api import ESMProtein, LogitsConfig

    protein = ESMProtein(sequence=seq_str[: config.max_length])
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
            f"ESM3c returned no hidden_states for sequence {seq_str[:20]!r}"
        )
    if isinstance(hs, torch.Tensor):
        hs = [hs[i] for i in range(hs.shape[0])]

    valid_layers = validate_layers(config.layer_indices, hs, "ESM3c", seq_str[:20])

    if config.pooling == "cls":
        layer_tensors_1d = [hs[-(li + 1)][0, 0, :].float() for li in valid_layers]
        pooled = aggregate_1d(layer_tensors_1d, config.layer_agg)
        if config.normalize:
            pooled = F.normalize(pooled.unsqueeze(0), p=2, dim=1).squeeze(0)
        chunks = [ChunkEmbedding(0, None, pooled.cpu().numpy())]
    else:
        layer_tensors_2d = [hs[-(li + 1)][0, 1:-1, :].float() for li in valid_layers]
        residues = aggregate_residue_layers(layer_tensors_2d, config.layer_agg)
        if config.normalize_residues:
            residues = F.normalize(residues, p=2, dim=1)
        chunks = chunk_and_pool(residues, config)

    del logits_output, hs
    torch.cuda.empty_cache()
    return chunks


#: Module-level plugin instance discovered via the
#: ``protea.backends`` entry_points group.
plugin = EsmcBackend()
