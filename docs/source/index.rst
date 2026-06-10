protea-backends
===============

``protea-backends`` is the protein language model (PLM) embedding layer
of the PROTEA stack. It turns amino-acid sequences into the dense
vectors that everything downstream (the KNN index, the reranker
features, the evaluation harness) is built on.

Each PLM lives behind a small plugin that implements one contract,
:class:`protea_contracts.EmbeddingBackend`. ``protea-core`` finds these
plugins at startup through the ``protea.backends`` entry-points group
and dispatches embedding work to them by name. A deployment ships only
the plugins it needs, and discovery is cheap because no plugin imports
``torch`` until it is actually asked to load a model.

The problem this solves
-----------------------

Protein language models are heavy. ``torch``, ``transformers`` and the
EvolutionaryScale ``esm`` SDK together weigh several gigabytes, and a
given deployment usually runs one or two PLMs, not all of them. Folding
that weight directly into ``protea-core`` would make the platform slow
to import and awkward to install. ``protea-backends`` keeps the
embedding code in its own package and applies two disciplines:

- **Per-backend dependencies.** The heavy ML libraries live behind
  Poetry extras (``[esm]``, ``[t5]``, ``[ankh]``, ``[esm3c]``). A box
  that only runs ESM-2 never pulls the ``esm`` SDK wheel.
- **Lazy imports.** Plugin modules import nothing heavy at the top
  level. ``torch`` and friends are imported inside ``load_model`` and
  ``embed_batch``, so ``protea-core`` pays zero import cost for backends
  it never invokes.

Adding a PLM is then a one-file change here plus one line in
``pyproject.toml``. ``protea-core`` does not change.

What lives here
---------------

::

   src/protea_backends/
   ├── __init__.py          # package version, nothing heavy
   ├── _chunk_helpers.py    # shared layer / chunk / pool primitives + ChunkEmbedding
   ├── esm/__init__.py      # ESM-1b, ESM-2 (plugin name "esm")
   ├── t5/__init__.py       # ProtT5, ProstT5 (plugin name "t5")
   ├── ankh/__init__.py     # Ankh-base, Ankh-large (plugin name "ankh")
   └── esm3c/__init__.py    # ESM-C (plugin name "esm3c")

Each backend directory is one plugin: a single class that subclasses
:class:`protea_contracts.EmbeddingBackend` and a module-level ``plugin``
instance registered as an entry point. The shared tensor logic (layer
selection, chunk spans, pooling) lives once in ``_chunk_helpers`` so the
per-backend files stay thin.

PROTEA stack
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 18 12 50

   * - Repo
     - Role
     - Status
     - Summary
   * - `PROTEA <https://github.com/frapercan/PROTEA>`_
     - Platform
     - active
     - Backend platform: ORM, job queue, FastAPI surface, frontend, orchestration.
   * - `protea-contracts <https://github.com/frapercan/protea-contracts>`_
     - Contracts
     - active
     - Shared ABCs, pydantic payloads, feature schema, schema_sha. Imported by every repo.
   * - `protea-method <https://github.com/frapercan/protea-method>`_
     - Inference
     - active
     - Pure inference path (KNN, feature compute, reranker apply). Bind-mounted by LAFA containers.
   * - `protea-sources <https://github.com/frapercan/protea-sources>`_
     - Source plugin
     - active
     - Annotation source plugins (GOA, QuickGO, UniProt, InterPro). Discovered via entry_points.
   * - `protea-runners <https://github.com/frapercan/protea-runners>`_
     - Runner plugin
     - active
     - Experiment runner plugins (LightGBM, KNN, baseline). Discovered via entry_points.
   * - **protea-backends** (this repo)
     - Backend plugin
     - active
     - PLM embedding backends (ESM family, T5/ProstT5, Ankh, ESM-C). Discovered via entry_points.
   * - `protea-reranker-lab <https://github.com/frapercan/protea-reranker-lab>`_
     - Lab
     - active
     - LightGBM reranker training lab. Publishes boosters back via /reranker-models/import-by-reference.
   * - `cafaeval-protea <https://github.com/frapercan/cafaeval-protea>`_
     - Evaluator
     - active
     - Fork of cafaeval (CAFA-evaluator-PK) with PK-coverage fix and bit-exact parity harness.

The backends at a glance
------------------------

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

Install one extra at a time, or everything at once:

.. code-block:: bash

   pip install "protea-backends[esm]"     # one backend
   pip install "protea-backends[all]"     # everything

Where to go next
----------------

- :doc:`quickstart`: install one extra and embed a batch in five
  minutes.
- :doc:`concepts`: the :class:`protea_contracts.EmbeddingBackend`
  contract, the pooling and chunking model, entry-point discovery, and
  the lazy-import discipline.
- :doc:`backends/index`: one page per backend, with supported models,
  canonical ``embedding_config_id`` UUIDs and backend-specific quirks.
- :doc:`contributing`: add a new backend in one file plus one
  ``pyproject.toml`` line.
- :doc:`api`: the full autodoc reference.

.. toctree::
   :maxdepth: 2
   :caption: Guide
   :hidden:

   quickstart
   concepts
   backends/index
   contributing
   api
