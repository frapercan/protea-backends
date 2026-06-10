Concepts
========

This page explains how a backend is shaped, how it is found, and the two
disciplines (lazy imports and per-backend extras) that keep the package
cheap to live with. Read it once and the per-backend pages become short.

The ``EmbeddingBackend`` contract
---------------------------------

Every backend in this package implements
:class:`protea_contracts.EmbeddingBackend`. The contract is deliberately
small: a backend is a thin adapter between a protein language model and
PROTEA's embedding pipeline, nothing more.

A conforming backend provides three things:

``name``
    A class attribute, a stable string such as ``"esm"`` or ``"t5"``.
    ``protea-core`` matches it against ``EmbeddingConfig.model_backend``
    to pick the backend for a job, and it must equal the entry-point
    name (see :ref:`discovery` below).

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

The contract types ``model`` and ``tokenizer`` as ``Any`` on purpose:
``protea-contracts`` does not depend on ``torch``, so it cannot name a
concrete model type. The backend is the only place that knows what the
objects really are.

The three output paths
----------------------

The base contract is the pooled ``(B, D)`` path. The backends in this
package extend it with two richer paths, both consumed by
``protea-core``:

``embed_batch`` (pooled)
    The historical path. One mean-pooled ``float16`` vector per
    sequence, returned as a ``(batch_size, hidden_dim)`` matrix. This is
    what the KNN index and the protein-level features are built on.

``embed_batch_per_residue(model, tokenizer, sequences, *, emit, layers)``
    Returns a :class:`protea_contracts.EmbeddingPayload` with
    ``granularity="per_residue"``: one ragged ``(L_i, hidden_dim)``
    ``float16`` tensor per sequence plus a matching boolean attention
    mask. The model's special tokens (CLS, EOS, BOS, the ProstT5
    ``<AA2fold>`` prefix) are stripped by the backend, so
    ``residues[i][j]`` is the embedding of amino acid ``j`` of sequence
    ``i``. This feeds MIL-style pooling heads and patch-level features.

``embed_chunks(model, tokenizer, sequences, config, device)``
    Returns one
    :class:`~protea_backends._chunk_helpers.ChunkEmbedding` list per
    sequence. This is the bit-exact port of PROTEA's pre-plugin
    embedding pipeline and is what ``compute_embeddings`` dispatches to
    in production. ``config`` is duck-typed to PROTEA's
    ``EmbeddingConfig`` (see the pooling model below).

The pooling and chunking model
------------------------------

The ``embed_chunks`` path is configurable through PROTEA's
``EmbeddingConfig``. A backend reads the following attributes off
``config`` (it never imports the type, only duck-types against it) and
applies them in the same order for every PLM, which is what makes the
output bit-exact across backends:

``layer_indices`` and ``layer_agg``
    ``layer_indices`` selects transformer layers by reverse index:
    ``0`` is the last layer, ``1`` the penultimate, and so on. The
    selected layers are combined with ``layer_agg``, one of ``last``,
    ``mean`` or ``concat`` (``concat`` joins along the feature axis).
    An out-of-range index raises ``ValueError``.

``pooling``
    Collapses the residue axis to one vector. ``mean`` and ``max`` work
    feature-wise; ``mean_max`` concatenates the two (doubling the
    dimension); ``cls`` reads the model's CLS / first-position token
    instead of pooling residues.

``use_chunking``, ``chunk_size``, ``chunk_overlap``
    With chunking on, the sequence is cut into overlapping spans of
    ``chunk_size`` residues with ``chunk_overlap`` between them, and each
    span is pooled independently into its own ``ChunkEmbedding``.
    ``chunk_overlap`` must be strictly less than ``chunk_size``. With
    chunking off, one ``ChunkEmbedding`` covers the whole sequence.

``normalize`` and ``normalize_residues``
    ``normalize_residues`` L2-normalises each residue vector before
    pooling; ``normalize`` L2-normalises the final pooled vector.

A :class:`~protea_backends._chunk_helpers.ChunkEmbedding` records its
span as ``chunk_index_s`` (0-based, inclusive) and ``chunk_index_e``
(exclusive, or ``None`` for a full sequence) alongside the pooled
``float32`` vector, matching the PROTEA DB columns.

.. _discovery:

Discovery via entry points
--------------------------

Backends are not imported by name from ``protea-core``. Each plugin
registers itself in ``pyproject.toml`` under the ``protea.backends``
entry-points group:

.. code-block:: toml

   [tool.poetry.plugins."protea.backends"]
   esm = "protea_backends.esm:plugin"
   t5 = "protea_backends.t5:plugin"
   ankh = "protea_backends.ankh:plugin"
   esm3c = "protea_backends.esm3c:plugin"

The right-hand side points at a module-level ``plugin`` instance (for
example ``EsmBackend()``). At startup ``protea-core`` enumerates the
group and builds a name-to-plugin map; when a job arrives it looks the
backend up by ``EmbeddingConfig.model_backend`` and calls ``load_model``
/ ``embed_batch`` on it. The same resolution is available standalone:

.. code-block:: python

   from importlib.metadata import entry_points

   plugin = entry_points(group="protea.backends")["esm"].load()
   assert plugin.name == "esm"

Because discovery only reads entry-point metadata, ``protea-core`` can
list every installed backend without importing any model code.

The lazy-import discipline
--------------------------

The reason discovery stays cheap is a rule the whole package follows:
**no heavy import at module top level**. ``torch``, ``transformers`` and
``esm`` are imported inside the methods that use them (``load_model``,
``embed_batch`` and the shared helpers), never at the top of a plugin
module. Importing ``protea_backends.esm`` brings in only ``numpy`` and
the contracts package; the multi-gigabyte stack is touched only when a
model is actually loaded.

This is what lets the unit-test CI run with no extras installed and
keeps ``protea-core`` startup free for backends a deployment never
invokes. It is enforced in review and by the test suite, which imports
every plugin module on a venv without ``torch``.

The ``emit`` callback
---------------------

``protea-core`` provides ``emit`` and uses it to write structured
``JobEvent`` rows to the database in real time. Its signature is
``emit(event_type, payload_or_none, context_dict, level)``. Backends
emit at minimum:

- ``backend.<name>.load_start`` at the start of ``load_model``;
- ``backend.<name>.load_done`` at the end of ``load_model``;
- ``backend.<name>.embed_done`` at the end of ``embed_batch``.

The per-residue path emits its own ``…embed_per_residue_done`` event.
For standalone use, pass a no-op such as ``lambda *a, **k: None``.

Shared helpers
--------------

The layer-validation, layer-aggregation, chunk-span and chunk-pooling
primitives shared by all backends live in
:mod:`protea_backends._chunk_helpers`, so the per-backend files only
carry the tokenisation and forward-pass logic specific to their PLM.
``torch`` is imported lazily inside each helper, so importing the module
stays cheap.

.. seealso::

   :doc:`contributing` walks through adding a new backend end to end, and
   :doc:`api` lists the full autodoc surface.
