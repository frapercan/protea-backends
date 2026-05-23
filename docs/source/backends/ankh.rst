Ankh (``ankh``)
===============

The ``ankh`` plugin wraps the Ankh family of encoder-decoder
checkpoints (Elnaggar et al., 2023). PROTEA uses the encoder side
only, via ``T5EncoderModel``.

:Models supported: Ankh-base, Ankh-large
                   (``ElnaggarLab/ankh-{base,large}``).
:Extra: ``protea-backends[ankh]``
:Heavy deps: ``torch``, ``transformers``, ``sentencepiece``
:Numerical type: bfloat16 on CUDA, fp32 on CPU.
:Pooling: mean over residues; trailing ``EOS`` excluded.

Canonical PROTEA checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both Ankh variants are part of the canonical 8-PLM research pipeline
(see ``project_canonical_8plm_embedding_configs.md`` in the PROTEA
memory store and ADR D35, PROTEA PR #418):

.. list-table::
   :header-rows: 1
   :widths: 20 38 42

   * - PLM key
     - HuggingFace checkpoint
     - ``embedding_config_id``
   * - ``ankh_base``
     - ``ElnaggarLab/ankh-base``
     - ``08234f06-ba76-4d7d-aaec-ae601096b4fa``
   * - ``ankh_large``
     - ``ElnaggarLab/ankh-large``
     - ``238f79b1-3068-4c6f-9013-5cc52b4f662b``

Quirks and operational notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~

.. automodule:: protea_backends.ankh
   :members:
   :show-inheritance:
   :member-order: bysource
