Ankh (``ankh``)
===============

The ``ankh`` plugin wraps the Ankh family of encoder--decoder
checkpoints (Elnaggar et al., 2023). PROTEA uses the encoder side
only, via ``T5EncoderModel``.

:Models supported: Ankh-base, Ankh-large
                   (``ElnaggarLab/ankh-{base,large}``).
:Extra: ``protea-backends[ankh]``
:Heavy deps: ``torch``, ``transformers``, ``sentencepiece``
:Numerical type: bfloat16 on CUDA, fp32 on CPU.
:Pooling: mean over residues; trailing ``EOS`` excluded.

Quirks and operational notes
----------------------------

- **bfloat16, not fp16, on CUDA.** Ankh's LayerNorm overflows in
  fp16, producing NaNs in the hidden states. The plugin therefore
  loads weights as ``torch.bfloat16`` whenever the device is CUDA,
  and falls back to fp32 on CPU since most CPUs lack bf16 hardware.
- ``is_split_into_words=True`` is set when calling the tokenizer with
  a list of single-character residues. This avoids SentencePiece
  collapsing whitespace between residues into ``<unk>`` for short
  sequences.
- Hidden-state aggregation supports both per-layer mean and per-layer
  sum via ``layer_agg``; the default is ``mean``.

API reference
-------------

.. automodule:: protea_backends.ankh
   :members:
   :show-inheritance:
   :member-order: bysource
