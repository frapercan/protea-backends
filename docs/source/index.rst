protea-backends
===============

Protein language model (PLM) embedding backends for the PROTEA stack.
Each backend is a thin adapter that implements the
:class:`protea_contracts.EmbeddingBackend` ABC and is discovered by
``protea-core`` via the ``protea.backends`` entry-points group, so a
deployment can ship only the backends it actually needs.

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

The :doc:`quickstart` walks the full path from install to a first
embedding; :doc:`contract` covers the method surface and the ``emit``
event protocol.

Where to go next
----------------

- :doc:`quickstart`: install one extra and embed a batch in five
  minutes.
- :doc:`contract`: the :class:`protea_contracts.EmbeddingBackend`
  surface every backend implements, plus the ``emit`` event protocol.
- :doc:`backends/index`: one page per backend with supported models,
  canonical ``embedding_config_id`` UUIDs and backend-specific quirks.
- :doc:`contributing`: add a new backend in one file plus one
  ``pyproject.toml`` line.
- :doc:`api`: full autodoc reference.

.. toctree::
   :maxdepth: 2
   :caption: Guide
   :hidden:

   quickstart
   contract
   backends/index
   contributing
   api
