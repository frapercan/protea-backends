The backend contract
====================

Every backend in this package implements
:class:`protea_contracts.EmbeddingBackend`. The contract is deliberately
small: a backend is a thin adapter between a protein language model and
PROTEA's embedding pipeline, nothing more.

Obligations
-----------

A conforming backend provides:

``name``
    A class attribute matching the entry-point name (``esm``, ``t5``,
    ``ankh``, ``esm3c``). ``protea-core`` dispatches jobs by this name.

``load_model(model_name, device, emit)``
    Loads the checkpoint and returns ``(model, tokenizer)``. The model is
    in eval mode and on ``device``. ``tokenizer`` may be ``None`` for
    backends that do not expose one, such as ESM-C; downstream code must
    tolerate that.

``embed_batch(model, tokenizer, sequences, *, emit, layers, layer_agg, pooling)``
    Runs inference and returns a ``float16`` ``ndarray`` of shape
    ``(batch_size, hidden_dim)``. Residues are pooled to a single vector
    per sequence inside the method; ragged shapes are not exposed at this
    level.

Output paths
------------

Backends in this package extend the minimal contract with two further
methods, both consumed by ``protea-core``:

``embed_batch_per_residue(model, tokenizer, sequences, *, emit, layers)``
    Returns a :class:`protea_contracts.EmbeddingPayload` with
    ``granularity="per_residue"``: one ragged ``(L_i, hidden_dim)``
    ``float16`` tensor per sequence plus a matching attention mask, with
    the model's special tokens (CLS, EOS, BOS, AA2fold prefix) already
    stripped so ``residues[i][j]`` is the embedding of amino acid ``j``.

``embed_chunks(model, tokenizer, sequences, config, device)``
    Returns one
    :class:`~protea_backends._chunk_helpers.ChunkEmbedding` list per
    sequence. This is the bit-exact port of PROTEA's pre-plugin embedding
    pipeline: multi-layer selection and aggregation, per-residue
    normalisation, overlapping chunking and ``mean`` / ``max`` /
    ``mean_max`` / ``cls`` pooling. ``config`` is duck-typed to PROTEA's
    ``EmbeddingConfig``.

The ``emit`` callback
---------------------

``protea-core`` provides ``emit`` and uses it to write structured
``JobEvent`` rows to the database in real time. Its signature is
``emit(event_type, payload_or_none, context_dict, level)``. Backends
emit at minimum:

- ``backend.<name>.load_start`` at the start of ``load_model``;
- ``backend.<name>.load_done`` at the end of ``load_model``;
- ``backend.<name>.embed_done`` at the end of ``embed_batch``.

The per-residue and chunked paths emit their own ``…embed_per_residue_done``
events. For standalone use, pass a no-op such as ``lambda *a, **k: None``.

Shared helpers
--------------

The layer-validation, layer-aggregation, chunk-span and chunk-pooling
primitives shared by all backends live in
:mod:`protea_backends._chunk_helpers`. ``torch`` is imported lazily
inside each helper, so importing the module stays cheap.

.. seealso::

   :doc:`contributing` walks through adding a new backend end to end, and
   :doc:`api` lists the full autodoc surface.
