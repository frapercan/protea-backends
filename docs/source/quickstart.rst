Quickstart
==========

Five minutes from a fresh environment to your first embedding.

Install
-------

Install the package with the extra for the backend you want. The ``esm``
backend is used here; ``t5``, ``ankh`` and ``esm3c`` follow the same
pattern.

.. code-block:: bash

   pip install "protea-backends[esm]"

Each extra pulls only the heavy ML libraries that backend needs, so a
deployment that only runs ESM-2 never downloads the ``esm`` SDK wheel.
Install everything at once with ``pip install "protea-backends[all]"``.

Embed a batch
-------------

Resolve the plugin through the ``protea.backends`` entry-points group,
load a checkpoint, and embed a list of sequences:

.. code-block:: python

   from importlib.metadata import entry_points

   # Resolve the plugin by its entry-point name.
   plugin = entry_points(group="protea.backends")["esm"].load()
   assert plugin.name == "esm"

   # protea-core passes an emit callback that writes structured JobEvent
   # rows to the database. A no-op is fine for standalone usage.
   emit = lambda *a, **k: None

   # torch is imported lazily inside load_model, never at module import.
   model, tokenizer = plugin.load_model(
       "facebook/esm2_t30_150M_UR50D", "cpu", emit
   )

   # Returns a (B, D) float16 ndarray.
   embeddings = plugin.embed_batch(
       model, tokenizer, ["MSEQ", "MKTYV"], emit=emit
   )
   print(embeddings.shape, embeddings.dtype)  # (2, 640) float16

``protea-core`` performs the same discovery internally: it resolves every
registered backend at startup and dispatches ``compute_embeddings`` jobs
by plugin name (``esm``, ``t5``, ``ankh``, ``esm3c``).

Beyond the pooled vector
------------------------

Two further output paths are available on every backend:

- :meth:`embed_batch_per_residue` returns an
  :class:`protea_contracts.EmbeddingPayload` with one ragged
  ``(L_i, D)`` tensor per sequence (special tokens stripped), for
  MIL-style pooling heads and patch-level features.
- :meth:`embed_chunks` returns one
  :class:`~protea_backends._chunk_helpers.ChunkEmbedding` list per
  sequence and is the bit-exact home of PROTEA's legacy embedding
  pipeline: multi-layer selection and aggregation, per-residue
  normalisation, overlapping chunks, and ``mean`` / ``max`` /
  ``mean_max`` / ``cls`` pooling.

See :doc:`concepts` for the full method surface, the pooling model and
the discovery mechanism, and :doc:`backends/index` for per-backend
quirks.
