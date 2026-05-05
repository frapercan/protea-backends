Adding a new backend
====================

The whole point of the plugin layout is that adding a backend is a
one-file change in this repository plus one line in
``pyproject.toml``. ``protea-core`` does not change.

Five steps
----------

1. **Create a sub-module** under ``src/protea_backends/<your_name>/``
   with an ``__init__.py``. The directory name is the canonical
   plugin name and must match the ``name`` class attribute below.

2. **Implement the contract.** Subclass
   :class:`protea_contracts.EmbeddingBackend` and provide the three
   obligations:

   .. code-block:: python

      from typing import Any
      import numpy as np
      from protea_contracts import EmbeddingBackend


      class MyBackend(EmbeddingBackend):
          name = "mybackend"

          def load_model(self, model_name, device, emit):
              # Lazy import any heavy dependency here, never at module top.
              import torch
              from transformers import AutoTokenizer, AutoModel

              emit("backend.mybackend.load_start", None,
                   {"model_name": model_name}, "info")
              tok = AutoTokenizer.from_pretrained(model_name)
              model = AutoModel.from_pretrained(model_name).eval().to(device)
              emit("backend.mybackend.load_done", None,
                   {"model_name": model_name}, "info")
              return model, tok

          def embed_batch(self, model, tokenizer, sequences, *, emit,
                          layers=None, layer_agg="mean", pooling="mean"):
              # Per-sequence forward pass, mean-pool residues,
              # return float16 ndarray of shape (B, D).
              ...

      plugin = MyBackend()

3. **Register the entry point.** In ``pyproject.toml`` add:

   .. code-block:: toml

      [tool.poetry.plugins."protea.backends"]
      mybackend = "protea_backends.mybackend:plugin"

4. **Declare extras** for any heavy dependency you brought in:

   .. code-block:: toml

      [tool.poetry.dependencies]
      my-heavy-lib = { version = ">=1.0", optional = true }

      [tool.poetry.extras]
      mybackend = ["torch", "my-heavy-lib"]

5. **Add a test file** under ``tests/test_mybackend.py`` covering at
   minimum: instance type, ABC compliance, ``name`` attribute,
   discoverability via ``entry_points(group="protea.backends")``, and
   the public method signatures. The existing
   ``tests/test_esm.py``, ``tests/test_t5.py`` etc. are a fine
   template.

Conventions
-----------

- Plugin module imports must be cheap. Heavy ML imports go *inside*
  the methods that need them; never at module top.
- Always emit ``backend.<name>.load_start`` and ``…load_done`` events
  in ``load_model``, and a ``…embed_done`` event in ``embed_batch``.
  ``protea-core`` rolls these up into the per-job structured event
  log.
- Default ``pooling`` is ``"mean"`` over residues. Strip the special
  tokens (``CLS``, ``EOS``, prefix tokens, ``BOS``, etc.) appropriate
  to your tokenizer before pooling.
- The return ``ndarray`` must be ``float16`` and shape
  ``(batch_size, hidden_dim)``. Ragged shapes are not supported at
  this contract level; pool to a per-sequence vector inside
  ``embed_batch``.
- If your backend has no tokenizer (like ESM-C), return
  ``(model, None)`` from ``load_model``. Downstream code must
  tolerate ``tokenizer is None``.

CI expectations
---------------

The ``protea-backends`` repository CI runs ``ruff``, ``mypy`` strict
and ``pytest`` without installing any extras. Your new test file
must therefore pass on a venv that does not have ``torch`` /
``transformers`` / ``esm`` available. Mock those imports inside the
tests if necessary; the plugin module itself must remain importable
without them.

Documentation
-------------

Add a sibling ``docs/source/backends/<your_name>.rst`` page following
the structure of the existing ones (Models supported, Extra, Heavy
deps, Numerical type, Pooling, Quirks, ``automodule`` directive).
The ``backends/index.rst`` ``toctree`` picks the page up
automatically once it is committed.
