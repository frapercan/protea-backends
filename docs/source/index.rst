protea-backends
===============

Protein language model (PLM) embedding backends for the PROTEA stack.
Each backend is a thin adapter that implements the
:class:`protea_contracts.EmbeddingBackend` ABC and is discovered by
``protea-core`` via the ``protea.backends`` entry-points group, so a
deployment can ship only the backends it actually needs.

.. note::

   This package is a *runtime* dependency of ``protea-core`` only when
   the corresponding extra is installed. Heavy ML dependencies
   (``torch``, ``transformers``, ``sentencepiece``, ``esm``) live
   behind per-backend Poetry extras and are imported lazily inside
   ``load_model`` / ``embed_batch``. The plugin modules themselves are
   import-cheap, so discovery during ``protea-core`` startup is free
   even on machines without the heavy stack.

At a glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 14 28 28 30

   * - Plugin name
     - Models supported
     - Extra to install
     - Notes
   * - :doc:`esm <backends/esm>`
     - ESM-1b, ESM-2 (8M to 15B)
     - ``protea-backends[esm]``
     - HuggingFace ``EsmModel``; mean-pool last hidden state.
   * - :doc:`t5 <backends/t5>`
     - ProtT5-XL, ProstT5
     - ``protea-backends[t5]``
     - Encoder-only T5; ProstT5 is auto-detected and prefixed with
       ``<AA2fold>``.
   * - :doc:`ankh <backends/ankh>`
     - Ankh-base, Ankh-large
     - ``protea-backends[ankh]``
     - bfloat16 on CUDA (FP16 LayerNorm overflows);
       ``is_split_into_words=True`` to keep SentencePiece honest.
   * - :doc:`esm3c <backends/esm3c>`
     - ESM-C 300M, ESM-C 600M
     - ``protea-backends[esm3c]``
     - Standalone ``esm`` package; no tokenizer; ``model.encode`` +
       ``LogitsConfig(return_hidden_states=True)``.

Install
-------

.. code-block:: bash

   # one backend at a time
   pip install "protea-backends[esm]"
   pip install "protea-backends[t5]"
   pip install "protea-backends[ankh]"
   pip install "protea-backends[esm3c]"

   # everything at once
   pip install "protea-backends[all]"

Discovery
---------

``protea-core`` resolves a backend by name through
``importlib.metadata.entry_points``::

    from importlib.metadata import entry_points

    eps = entry_points(group="protea.backends")
    plugin = eps["esm"].load()
    model, tokenizer = plugin.load_model(
        "facebook/esm2_t12_35M_UR50D", "cpu", emit=lambda *a, **k: None
    )

The ``plugin`` symbol is a module-level instance; importing it does
not pay the cost of the heavy ML stack until ``load_model`` is
called.

Contracts
---------

Each backend implements :class:`protea_contracts.EmbeddingBackend`,
which carries three obligations:

- a class attribute ``name`` matching the entry-point name;
- ``load_model(model_name, device, emit)`` returning ``(model,
  tokenizer)`` (``tokenizer`` may be ``None`` for backends that do not
  expose one, such as ESM-C);
- ``embed_batch(model, tokenizer, sequences, *, emit, layers,
  layer_agg, pooling)`` returning a float16 ``ndarray`` of shape
  ``(batch_size, hidden_dim)``.

The ``emit`` callable is provided by ``protea-core`` and writes
structured ``JobEvent`` rows to the database in real time. Backends
should emit ``backend.<name>.load_start``, ``…load_done``, and
``…embed_done`` at minimum.

Plugin reference
----------------

.. toctree::
   :maxdepth: 2

   backends/index
   contributing
