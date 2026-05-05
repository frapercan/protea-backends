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

Quirks and operational notes
----------------------------

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
-------------

.. automodule:: protea_backends.esm3c
   :members:
   :show-inheritance:
   :member-order: bysource
