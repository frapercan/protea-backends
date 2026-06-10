API reference
=============

Each backend plugin is documented in full, with its quirks and canonical
checkpoints, on its own page under :doc:`backends/index`:

- :doc:`backends/esm` (:class:`protea_backends.esm.EsmBackend`)
- :doc:`backends/t5` (:class:`protea_backends.t5.T5Backend`)
- :doc:`backends/ankh` (:class:`protea_backends.ankh.AnkhBackend`)
- :doc:`backends/esm3c` (:class:`protea_backends.esm3c.EsmcBackend`)

This page collects the rest of the public surface: the package root and
the shared chunked-embedding helpers.

Package
-------

.. automodule:: protea_backends
   :members:
   :undoc-members:
   :show-inheritance:

Shared helpers
--------------

The chunked-embedding primitives shared by every backend: layer
validation and aggregation, chunk-span computation, residue pooling and
the :class:`~protea_backends._chunk_helpers.ChunkEmbedding` dataclass.
``torch`` is imported lazily inside each helper, so importing the module
stays cheap.

.. automodule:: protea_backends._chunk_helpers
   :members:
   :show-inheritance:
   :member-order: bysource
