# protea-backends

**Protein language model embedding backends for the PROTEA stack.** Each backend
is a thin adapter that implements the
[`EmbeddingBackend`](https://github.com/frapercan/protea-contracts) ABC from
`protea-contracts` and is discovered by `protea-core` via the `protea.backends`
entry-points group, so a deployment ships only the backends it actually needs.

[![CI](https://github.com/frapercan/protea-backends/actions/workflows/ci.yml/badge.svg)](https://github.com/frapercan/protea-backends/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/readthedocs/protea-backends.svg)](https://protea-backends.readthedocs.io)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](https://unlicense.org/)
[![PyPI](https://img.shields.io/pypi/v/protea-backends.svg)](https://pypi.org/project/protea-backends/)

**Status:** in use. All four backends (the ESM family, T5 and ProstT5, Ankh, and
ESM-C) are deployed and have hydrated the canonical embedding pools. The plugin
API may still change.
Every internal dependency in this stack now names a commit rather than a
branch, checked on each pull request. Updating one is a pull request here that
moves the commit, gated by this repository's own checks.

A definitive campaign run is being prepared for the doctoral thesis, and
earlier experimental results are being recomputed rather than carried forward.
No headline number is quoted in this file as current.

---

<!-- protea-stack:start -->

## Repositories in the PROTEA stack

Single source of truth: [`docs/source/_data/stack.yaml`](https://github.com/frapercan/PROTEA/blob/develop/docs/source/_data/stack.yaml) in PROTEA. Run `python scripts/sync_stack.py` to regenerate this block.

| Repo | Role | Status | Summary |
|------|------|--------|---------|
| [PROTEA](https://github.com/frapercan/PROTEA) | Platform | `active` | Backend platform. Hosts the ORM, job queue, FastAPI surface, frontend, and orchestration. |
| [protea-contracts](https://github.com/frapercan/protea-contracts) | Contracts | `active` | Shared contract surface. ABCs, pydantic payloads, feature schema, schema_sha. Imported by every other repo. |
| [protea-method](https://github.com/frapercan/protea-method) | Inference | `active` | LAFA submission layer. Pure inference path (KNN, feature compute, reranker apply). Published to DockerHub; bind-mounted by LAFA containers. |
| [protea-sources](https://github.com/frapercan/protea-sources) | Source plugin | `active` | Annotation source plugins (GOA, QuickGO, UniProt, InterPro). Discovered via Python entry_points. |
| [protea-runners](https://github.com/frapercan/protea-runners) | Runner plugin | `active` | Experiment runner plugins (LightGBM, KNN, baseline). Discovered via Python entry_points. |
| **protea-backends** (this repo) | Backend plugin | `active` | Protein language model embedding backends (ESM family, T5/ProstT5, Ankh, ESM-C). Discovered via Python entry_points. |
| [protea-reranker-lab](https://github.com/frapercan/protea-reranker-lab) | Lab | `active` | LightGBM reranker training lab. Pulls datasets from PROTEA, trains boosters, publishes them back via /reranker-models/import-by-reference. |
| [cafaeval-protea](https://github.com/frapercan/cafaeval-protea) | Evaluator | `active` | Standalone fork of cafaeval (CAFA-evaluator-PK) with the PK-coverage fix and a bit-exact parity guarantee against upstream. |

<!-- protea-stack:end -->

---

## Why a separate package

Three reasons (see [ADR D1](https://github.com/frapercan/PROTEA/blob/develop/docs/source/adr/D01-project-structure.rst)):

1. **Plugin extensibility.** New backends are added without touching
   `protea-core`. A single-file commit in this repository plus one line
   in `pyproject.toml` is enough.
2. **Per-backend deps.** Heavy ML libraries (`torch`, `transformers`,
   `sentencepiece`, `esm`) live behind Poetry extras. A deployment that
   only needs ESM-2 does not pull the `esm` SDK's 1.3 GB wheel.
3. **Import-cheap discovery.** Plugin modules import nothing heavy at the
   top level. `torch` and friends are imported lazily inside `load_model`
   and `embed_batch`. `protea-core` startup pays zero cost for backends it
   never invokes.

---

## Backends shipped today

| Plugin | Models | Extra | Notes |
|--------|--------|-------|-------|
| `esm` | ESM-1b, ESM-2 (8M to 15B) | `[esm]` | HuggingFace `EsmModel`; mean-pool last hidden state. |
| `t5` | ProtT5-XL, ProstT5 | `[t5]` | Encoder-only T5; ProstT5 auto-detected, prefixed with `<AA2fold>`. |
| `ankh` | Ankh-base, Ankh-large | `[ankh]` | bfloat16 on CUDA (FP16 LayerNorm overflows); `is_split_into_words=True` for SentencePiece. |
| `esm3c` | ESM-C 300M, ESM-C 600M | `[esm3c]` | Standalone `esm` package; no tokenizer; `model.encode` + `LogitsConfig(return_hidden_states=True)`. |

The 8 canonical checkpoints used in the PROTEA research pipeline are tracked with
their `embedding_config_id` UUIDs in the PROTEA memory note
`project_canonical_8plm_embedding_configs.md`. Every checkpoint maps to exactly
one of the four plugins above (ESM-2 150M/650M/3B to `esm`, ProtT5/ProstT5 to
`t5`, Ankh-base/large to `ankh`, ESM-C 600M to `esm3c`).

---

## 5 minutes to your first embedding

Install the package with the extra for the backend you want (`esm` here;
`t5`, `ankh`, `esm3c` follow the same pattern):

```bash
pip install "protea-backends[esm]"
```

Discover and call the plugin:

```python
from importlib.metadata import entry_points
import numpy as np

# Resolve the plugin via its entry-point name.
plugin = entry_points(group="protea.backends")["esm"].load()
assert plugin.name == "esm"

# The emit callback writes structured JobEvent rows to the DB in real
# operation; use a no-op here for standalone usage.
emit = lambda *a, **k: None

# Load the model. torch is imported lazily inside load_model only.
model, tokenizer = plugin.load_model(
    "facebook/esm2_t30_150M_UR50D", "cpu", emit
)

# Embed a batch. Returns a (B, D) float16 ndarray.
embeddings = plugin.embed_batch(
    model, tokenizer, ["MSEQ", "MKTYV"], emit=emit
)
print(embeddings.shape, embeddings.dtype)  # (2, 640) float16
```

`protea-core` performs the same discovery internally: it resolves all
registered backends at startup and dispatches `compute_embeddings` jobs
by plugin name (`esm`, `t5`, `ankh`, `esm3c`).

---

## Install

```bash
# One backend at a time:
pip install "protea-backends[esm]"
pip install "protea-backends[t5]"
pip install "protea-backends[ankh]"
pip install "protea-backends[esm3c]"

# Everything at once:
pip install "protea-backends[all]"
```

---

## The `EmbeddingBackend` contract

Each backend implements `protea_contracts.EmbeddingBackend`, which carries
three obligations:

- A class attribute `name` matching the entry-point name.
- `load_model(model_name, device, emit)` returning `(model, tokenizer)`.
  `tokenizer` may be `None` for backends without one (ESM-C).
- `embed_batch(model, tokenizer, sequences, *, emit, layers, layer_agg, pooling)`
  returning a `float16` ndarray of shape `(batch_size, hidden_dim)`.

The `emit` callable is provided by `protea-core` and writes structured
`JobEvent` rows to the database in real time. Backends must emit at minimum:

- `backend.<name>.load_start` at the start of `load_model`
- `backend.<name>.load_done` at the end of `load_model`
- `backend.<name>.embed_done` at the end of `embed_batch`

---

## Adding a new backend

The full guide lives in the Sphinx docs at
[`docs/source/contributing.rst`](docs/source/contributing.rst). Five-step summary:

1. Create `src/protea_backends/<your_name>/__init__.py`.
2. Subclass `EmbeddingBackend`; implement `load_model` and `embed_batch`.
   Set `name = "<your_name>"`.
3. Add `<your_name> = "protea_backends.<your_name>:plugin"` under
   `[tool.poetry.plugins."protea.backends"]` in `pyproject.toml`.
4. Declare any heavy dependency as `optional = true` and wire an extras
   group `[<your_name>]`.
5. Add `tests/test_<your_name>.py` covering instance type, ABC compliance,
   `name` attribute, `entry_points` discoverability, and method signatures.

Key constraints enforced by CI:

- **Import-cheap:** top-level module imports must not trigger `torch` or any
  heavy ML library. Lazy imports inside `load_model` and `embed_batch` only.
- **Typed output:** `embed_batch` must return a `(B, D) float16 ndarray`.
  Cast inside the plugin if the upstream model produces another dtype.
- **No runtime dep on protea-core.** This package must remain installable
  independently of the PROTEA platform layer.

---

## Development

```bash
git clone https://github.com/frapercan/protea-backends.git
cd protea-backends
git checkout develop
git checkout -b feature/my-change

poetry install

poetry run pytest          # ~0.2 s, no extras needed
poetry run ruff check .
poetry run mypy src tests

# Build the docs:
poetry install --with docs
cd docs && make html
open docs/build/html/index.html
```

Branch strategy: all changes target `develop`; `main` tracks stable releases.
Open a pull request targeting `develop`.

---

## Documentation

The full Sphinx documentation (quickstart, the `EmbeddingBackend`
contract, one page per backend, contributing guide and API reference)
is published at [protea-backends.readthedocs.io](https://protea-backends.readthedocs.io)
and lives under [`docs/source/`](docs/source). Build it locally with:

```bash
poetry install --with docs
cd docs && make html
open docs/build/html/index.html
```

The `docs` workflow builds the same HTML on every push with warnings
treated as errors, acting as a quality gate. ReadTheDocs builds and
hosts the published site (see `.readthedocs.yaml`). Release notes are
tracked in [`CHANGELOG.md`](CHANGELOG.md).

---

## License

Released into the public domain under [The Unlicense](https://unlicense.org/).
See [`LICENSE`](LICENSE). Author: Francisco Miguel Pérez Canales.
