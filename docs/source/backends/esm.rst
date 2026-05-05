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

Quirks and operational notes
----------------------------

- The 3B and 15B variants do not fit on a single 24 GB GPU at fp16
  with reasonable batch sizes. Use the ``OperationTuning.batch_size``
  configuration knob (see ``docs/CONFIG_INVENTORY`` in
  ``protea-core``) to tune batch size per deployment target.
- ``output_hidden_states=True`` is required for layer-aware aggregation
  (selecting non-final layers via the ``layers`` argument of
  :meth:`embed_batch`).
- The CUDA cache is cleared between sequences to keep memory steady on
  long batches.

API reference
-------------

.. automodule:: protea_backends.esm
   :members:
   :show-inheritance:
   :member-order: bysource
