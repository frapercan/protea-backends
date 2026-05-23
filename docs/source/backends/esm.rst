ESM (``esm``)
=============

The ``esm`` plugin wraps HuggingFace ``EsmModel`` checkpoints from
the Meta AI ESM family.

:Models supported: ESM-1b, ESM-2 at all published scales (8M, 35M,
                   150M, 650M, 3B, 15B).
:Extra: ``protea-backends[esm]``
:Heavy deps: ``torch``, ``transformers``
:Numerical type: fp16 on CUDA, fp32 on CPU.
:Pooling: mean over residues; ``CLS`` and ``EOS`` tokens excluded.

Canonical PROTEA checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three ESM-2 scales are part of the canonical 8-PLM research pipeline
(see ``project_canonical_8plm_embedding_configs.md`` in the PROTEA
memory store and ADR D35, PROTEA PR #418):

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - PLM key
     - HuggingFace checkpoint
     - ``embedding_config_id``
   * - ``esm2_150m``
     - ``facebook/esm2_t30_150M_UR50D``
     - ``500a0c59-be09-424d-9d51-b7997629c95a``
   * - ``esm2_650m``
     - ``facebook/esm2_t33_650M_UR50D``
     - ``c2e9dda3-e505-4170-b50d-435a451761ac``
   * - ``esm2_3b``
     - ``facebook/esm2_t36_3B_UR50D``
     - ``55e43f1c-1a3b-4b1d-88c0-26b433f5f673``

All three use the ``esm`` plugin with default pooling (mean over residues,
fp16 on CUDA).

Quirks and operational notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The 3B variant does not fit on a single 24 GB GPU at fp16 with
  reasonable batch sizes. Use the ``OperationTuning.batch_size``
  configuration knob (see ``docs/CONFIG_INVENTORY`` in ``protea-core``)
  to tune batch size per deployment target.
- ``output_hidden_states=True`` is required for layer-aware aggregation
  (selecting non-final layers via the ``layers`` argument of
  :meth:`~protea_backends.esm.EsmBackend.embed_batch`).
- The CUDA cache is cleared between sequences to keep memory steady on
  long batches.
- :meth:`~protea_backends.esm.EsmBackend.embed_batch_per_residue` returns
  per-residue ``(L_i, D)`` tensors for MIL-style pooling heads.
- :meth:`~protea_backends.esm.EsmBackend.embed_chunks` is the bit-exact
  port of PROTEA's legacy ``_embed_esm`` pipeline (multi-layer selection,
  chunk-and-pool, CLS pooling path).

API reference
~~~~~~~~~~~~~

.. automodule:: protea_backends.esm
   :members:
   :show-inheritance:
   :member-order: bysource
