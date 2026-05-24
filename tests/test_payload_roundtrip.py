"""Round-trip stability for the payloads protea-backends consumes.

Every backend plugin in this package returns
:class:`protea_contracts.EmbeddingPayload` from
``embed_batch_per_residue`` and consumes only call-site kwargs (no
plugin-local pydantic payloads at the contract boundary). The payload
shape is the contract; this module exercises it from the consumer
vantage so a regression in :mod:`protea_contracts.embedding_payload`
also lights up the backends CI.

``ChunkEmbedding`` is the plugin-local dataclass returned by the
``embed_chunks`` path; the test below also pins its field set against
silent removal, which would corrupt PROTEA's chunked-embedding
ingestion (``chunk_index_s``/``chunk_index_e`` are DB-column-aligned
so a field rename breaks the persistence adapter).

GPU tests are intentionally skipped here: the contract-roundtrip is
backend-agnostic and runs without torch installed (matches the
``--cov-fail-under=15`` lightweight CI lane).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from protea_contracts import EmbeddingPayload, Granularity

from protea_backends._chunk_helpers import ChunkEmbedding


def test_embedding_payload_mean_pool_construct_and_dump() -> None:
    """Mean-pool :class:`EmbeddingPayload` must construct + dump without raising."""
    payload = EmbeddingPayload(
        granularity="mean_pool",
        embeddings=np.zeros((3, 8), dtype=np.float16),
    )
    assert payload.granularity == "mean_pool"
    matrix = payload.as_matrix()
    assert matrix.shape == (3, 8)
    # The dump path is exercised by the contract; the consumer side
    # only needs to confirm no validator regression breaks the typical
    # backend-output shape.
    assert "embeddings" in payload.model_dump()


def test_embedding_payload_per_residue_construct_and_dump() -> None:
    """Per-residue :class:`EmbeddingPayload` must accept ragged tensors."""
    residues = [
        np.zeros((4, 8), dtype=np.float16),
        np.zeros((7, 8), dtype=np.float16),
    ]
    masks = [
        np.ones((4,), dtype=bool),
        np.ones((7,), dtype=bool),
    ]
    payload = EmbeddingPayload(
        granularity="per_residue",
        residues=residues,
        attention_mask=masks,
    )
    matrix = payload.as_matrix()
    # ``as_matrix`` mean-pools each ragged tensor to (D,) and stacks.
    assert matrix.shape == (2, 8)


def test_embedding_payload_rejects_mismatched_shapes() -> None:
    """Per-residue shape mismatch must raise at construction.

    Pin-test that the shape invariant the backends rely on is still
    enforced upstream; a silently-relaxed validator would let a
    backend ship payloads the KNN consumer cannot consume.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EmbeddingPayload(
            granularity="per_residue",
            residues=[np.zeros((4, 8), dtype=np.float16)],
            attention_mask=[np.ones((5,), dtype=bool)],  # mismatched length
        )


def test_granularity_literal_values() -> None:
    """The :data:`Granularity` literal must keep its two values intact.

    Adding a new granularity is a contract change that requires every
    backend's ``embed_batch_per_residue`` to grow a branch; widening
    the set without coordinated rollout would silently down-cast on
    the consumer side. This pin-test forces the change to be visible.
    """
    from typing import get_args

    args = set(get_args(Granularity))
    assert args == {"mean_pool", "per_residue"}, (
        f"Granularity literal drifted: got {sorted(args)!r}; coordinate "
        "the change with every plugin's embed_batch_per_residue branch."
    )


def test_chunk_embedding_field_set_pinned() -> None:
    """``ChunkEmbedding`` field set must remain stable.

    The dataclass's ``chunk_index_s`` / ``chunk_index_e`` / ``vector``
    field names are aligned with the PROTEA DB columns of the same
    names; a rename here would break the persistence adapter in
    PROTEA without that repo's CI noticing.
    """
    fields = {f.name for f in dataclasses.fields(ChunkEmbedding)}
    assert fields == {"chunk_index_s", "chunk_index_e", "vector"}, (
        f"ChunkEmbedding field set drifted: {sorted(fields)!r}; coordinate "
        "the rename with the PROTEA persistence adapter (DB columns of "
        "the same names) before changing this set."
    )


def test_chunk_embedding_full_sequence_default_shape() -> None:
    """Full-sequence convention: ``chunk_index_s=0``, ``chunk_index_e=None``.

    Pin-test against silent removal of the ``None`` sentinel for the
    full-sequence case; PROTEA branches on it when emitting chunk rows.
    """
    ce = ChunkEmbedding(
        chunk_index_s=0,
        chunk_index_e=None,
        vector=np.zeros(8, dtype=np.float32),
    )
    assert ce.chunk_index_s == 0
    assert ce.chunk_index_e is None
    assert ce.vector.shape == (8,)
