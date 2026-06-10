The backends
============

Each page below documents one backend plugin: its supported models,
the extra to install, the canonical ``embedding_config_id`` UUIDs used
by the PROTEA research pipeline, any backend-specific quirks, and the
autodoc listing of the plugin class. They share the contract and pooling
model described in :doc:`../concepts`, so the per-backend pages stay
focused on what makes each PLM different (tokenisation, numerical type,
special-token handling).

To add a PLM that is not listed here, see :doc:`../contributing`: a new
backend is a single file in this repository plus one line in
``pyproject.toml``.

The 8 canonical PROTEA checkpoints span three plugins:

- **ESM** (``esm`` plugin): ESM-2 150M, 650M, 3B
- **T5** (``t5`` plugin): ProtT5-XL (half-precision), ProstT5
- **Ankh** (``ankh`` plugin): Ankh-base, Ankh-large
- **ESM-C** (``esm3c`` plugin): ESM-C 600M

All ``embedding_config_id`` UUIDs are tracked in
``project_canonical_8plm_embedding_configs.md`` in the PROTEA memory
store (see also ADR D35, PROTEA PR #418).

.. toctree::
   :maxdepth: 1

   esm
   t5
   ankh
   esm3c
