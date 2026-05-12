"""Chunked-embedding helpers shared by the per-backend plugins.

These are the layer-validation, layer-aggregation, chunk-span and
chunk-pooling primitives that PROTEA's pre-plugin ``compute_embeddings``
pipeline used for ESM / T5 / Ankh / ESM-C. They live in a private module
under ``protea_backends`` so each per-backend plugin can build the full
``list[list[ChunkEmbedding]]`` output expected by PROTEA without
duplicating the (delicate) tensor slicing logic.

``torch`` is imported lazily inside every helper so the module itself
stays free of heavy dependencies; importing ``protea_backends._chunk_helpers``
remains as cheap as importing the rest of the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ChunkEmbedding:
    """One pooled embedding for a contiguous residue span of a sequence.

    ``chunk_index_s`` and ``chunk_index_e`` use the same convention as the
    PROTEA DB columns: start is 0-based inclusive, end is exclusive. When
    chunking is disabled, ``chunk_index_s=0`` and ``chunk_index_e=None``
    (full sequence).
    """

    chunk_index_s: int
    chunk_index_e: int | None
    vector: np.ndarray[Any, Any]  # 1-D float32


def validate_layers(
    layer_indices: list[int],
    hidden_states: Any,
    model_tag: str,
    seq_id: str,
) -> list[int]:
    """Validate reverse-indexed layer indices against the model's hidden states.

    ``layer_indices = [0]`` is the last layer; ``[1]`` the penultimate; etc.
    Raises ``ValueError`` if any index is out of range. Returns a sorted,
    deduplicated list of valid indices.
    """
    import torch

    if isinstance(hidden_states, torch.Tensor):
        total = int(hidden_states.shape[0])
    else:
        total = len(hidden_states)

    req = sorted({int(li) for li in layer_indices})
    invalid = [li for li in req if not 0 <= li < total]
    if invalid:
        raise ValueError(
            f"[{model_tag}] seq={seq_id!r}: invalid layer_indices {invalid}. "
            f"Valid range: 0..{total - 1}  (0 = last layer)."
        )
    return req


def aggregate_residue_layers(layer_tensors: list[Any], layer_agg: str) -> Any:
    """Combine [L, D] tensors from multiple layers into one [L, D] tensor."""
    import torch

    if layer_agg == "last":
        return layer_tensors[-1]
    if layer_agg == "mean":
        return torch.stack(layer_tensors, dim=0).mean(dim=0)
    if layer_agg == "concat":
        return torch.cat(layer_tensors, dim=-1)
    raise ValueError(f"Unknown layer_agg: {layer_agg!r}. Choose: last, mean, concat")


def aggregate_1d(layer_tensors: list[Any], layer_agg: str) -> Any:
    """Combine [D] tensors from multiple layers into one [D] tensor (CLS path)."""
    import torch

    if layer_agg == "last":
        return layer_tensors[-1]
    if layer_agg == "mean":
        return torch.stack(layer_tensors, dim=0).mean(dim=0)
    if layer_agg == "concat":
        return torch.cat(layer_tensors, dim=-1)
    raise ValueError(f"Unknown layer_agg: {layer_agg!r}. Choose: last, mean, concat")


def compute_chunk_spans(length: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Compute (start, end) spans for overlapping chunks over ``length`` residues.

    Raises ``ValueError`` if ``overlap >= chunk_size``; such a configuration
    would produce O(L) single-residue chunks or an infinite loop.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
        )
    step = chunk_size - overlap
    spans: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + chunk_size, length)
        spans.append((start, end))
        start += step
    return spans


def chunk_and_pool(residues: Any, config: Any) -> list[ChunkEmbedding]:
    """Apply chunking (optional) and pooling to a residue tensor [L, D].

    Returns one ``ChunkEmbedding`` per chunk. Without chunking, returns a
    single element covering the full sequence. ``config`` is duck-typed:
    it only needs the ``use_chunking``, ``chunk_size``, ``chunk_overlap``,
    ``pooling`` and ``normalize`` attributes that PROTEA's
    ``EmbeddingConfig`` exposes.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812  PyTorch convention

    if config.use_chunking:
        spans = compute_chunk_spans(residues.shape[0], config.chunk_size, config.chunk_overlap)
    else:
        spans = [(0, residues.shape[0])]

    results: list[ChunkEmbedding] = []
    for start, end in spans:
        chunk = residues[start:end]  # [chunk_L, D]

        if config.pooling == "mean":
            pooled = chunk.mean(dim=0)
        elif config.pooling == "max":
            pooled = chunk.max(dim=0).values
        elif config.pooling == "mean_max":
            pooled = torch.cat([chunk.mean(dim=0), chunk.max(dim=0).values])
        else:
            raise ValueError(
                f"Pooling {config.pooling!r} is not supported in residue-level mode. "
                f"Use 'cls' for CLS token pooling."
            )

        if config.normalize:
            pooled = F.normalize(pooled.unsqueeze(0), p=2, dim=1).squeeze(0)

        chunk_index_e = end if config.use_chunking else None
        results.append(
            ChunkEmbedding(
                chunk_index_s=start,
                chunk_index_e=chunk_index_e,
                vector=pooled.float().cpu().numpy(),
            )
        )

    return results
