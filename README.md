# protea-backends

Protein language model embedding backends for the PROTEA stack.
Each sub-module implements the `EmbeddingBackend` ABC from
`protea-contracts` and registers via `entry_points` group
`protea.backends`.

## Sub-modules

| Sub-module | Models | Status |
|------------|--------|--------|
| `protea_backends.esm` | ESM-1b, ESM-2 (8M / 35M / 150M / 650M / 3B / 15B) | F2A.1 (placeholder) |
| `protea_backends.t5` | ProtT5-XL, ProstT5 | F2A.2 (placeholder) |
| `protea_backends.ankh` | Ankh-base, Ankh-large | F2A.3 (placeholder) |
| `protea_backends.esm3c` | ESM-C 300M, ESM-C 600M | F2A.4 (placeholder) |

## Heavy deps as extras

`torch`, `transformers`, `sentencepiece` will become poetry extras
keyed per backend (`pip install protea-backends[esm]`) so HPC
deployments can pull only what they need. Today the bootstrap
keeps them out for fast CI install.

## Adding a new backend

A new backend = a new sub-module + an entry under
`[tool.poetry.plugins."protea.backends"]` in `pyproject.toml`.
`protea-core` discovers it via `entry_points` and registers it
in `compute_embeddings`.

The masterplan demonstrates this with the *adding-a-PLM* test
in F2: a single-file commit must be enough to add a new backend.

## Roadmap

This is the F0 bootstrap (T0.14 of the PROTEA master plan v3).
The current `_embed_esm`, `_embed_t5`, `_embed_ankh`, `_embed_esm3c`
helpers in `protea-core/operations/compute_embeddings.py` migrate
here in F2A.1-F2A.4.

## Development

```bash
poetry install
poetry run pytest
poetry run ruff check .
poetry run mypy src tests
```
