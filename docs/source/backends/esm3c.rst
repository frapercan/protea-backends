ESM-C (``esm3c``)
=================

The ``esm3c`` plugin wraps the EvolutionaryScale ``esm`` package
(distinct from HuggingFace ``transformers``). ESM-C is a family of
efficient sequence-only encoders distilled from the larger ESM-3
multimodal models.

:Models supported: ESM-C 300M (``esmc_300m``),
                   ESM-C 600M (``esmc_600m``).
:Extra: ``protea-backends[esm3c]``
:Heavy deps: ``torch``, ``esm``
:Numerical type: fp16 on CUDA (via ``torch.autocast``), fp32 on CPU.
:Pooling: mean over residues; ``BOS`` (position 0) and trailing
          ``EOS`` excluded.

Canonical PROTEA checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ESM-C 600M is part of the canonical 8-PLM research pipeline
(see ``project_canonical_8plm_embedding_configs.md`` in the PROTEA
memory store and ADR D35, PROTEA PR #418):

.. list-table::
   :header-rows: 1
   :widths: 20 38 42

   * - PLM key
     - SDK checkpoint name
     - ``embedding_config_id``
   * - ``esmc_600m``
     - ``esmc_600m``
     - ``2bf1e753-022f-44b8-a131-9a90acb4024e``

ESM-C 300M (``esmc_300m``, config ``c85d1afe``) is classified as a
single-PLM baseline reference and is not part of the 8-PLM ensemble.

Quirks and operational notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **No tokenizer.** The ESM-C API consumes raw protein sequences and
  drives its own tokenisation internally. ``load_model`` therefore
  returns ``(model, None)``; consumers must tolerate ``tokenizer is
  None`` (the contract allows it).
- Inference uses two ESM API objects: ``ESMProtein`` to wrap the
  input sequence, and ``LogitsConfig(return_hidden_states=True)`` to
  request hidden states from ``model.logits``.
- On CUDA the plugin enters ``torch.autocast(dtype=torch.float16)``
  for the forward pass; ESM-C's autocast support is robust at fp16,
  unlike Ankh.
- The shape of ``hidden_states`` returned by the SDK has varied
  across ``esm`` versions; the plugin tolerates both tensor and list
  forms and normalises to a tensor of shape ``(L, D)``.

API reference
~~~~~~~~~~~~~

.. automodule:: protea_backends.esm3c
   :members:
   :show-inheritance:
   :member-order: bysource
