# protea-backends

Protein language model embedding backends for the
[PROTEA](https://github.com/frapercan/protea) stack. Each sub-module
implements the `EmbeddingBackend` ABC from
[`protea-contracts`](https://github.com/frapercan/protea-contracts)
and registers itself via the `protea.backends` `entry_points` group,
so a deployment can ship only the backend it actually needs.

## 5 minutes to your first embedding

Install the package with the extra for the backend you want
(`esm` here, but `t5`, `ankh`, `esm3c` follow the same pattern):

```bash
pip install "protea-backends[esm]"
```

Discover and call the plugin:

```python
from importlib.metadata import entry_points
import numpy as np

# Resolve the plugin instance via its entry point name.
plugin = entry_points(group="protea.backends")["esm"].load()
assert plugin.name == "esm"

# Ignore the structured-event callback for this example.
emit = lambda *a, **k: None

# Load the model on CPU. The plugin is import-cheap; torch is
# imported lazily inside load_model only.
model, tokenizer = plugin.load_model(
    "facebook/esm2_t6_8M_UR50D", "cpu", emit
)

# Embed a batch. Returns a (B, D) float16 ndarray.
embeddings = plugin.embed_batch(
    model, tokenizer, ["MSEQ", "MKTYV"], emit=emit
)
print(embeddings.shape, embeddings.dtype)  # (2, 320) float16
```

`protea-core` does the same dance internally: it discovers all four
backends at startup and dispatches `compute_embeddings` jobs by
plugin name (`esm`, `t5`, `ankh`, `esm3c`).

## Backends shipped today

| Plugin | Models | Extra | Notes |
|--------|--------|-------|-------|
| `esm` | ESM-1b, ESM-2 (8M to 15B) | `[esm]` | HuggingFace `EsmModel`; mean-pool last hidden state |
| `t5` | ProtT5-XL, ProstT5 | `[t5]` | Encoder-only T5; ProstT5 auto-detected, prefixed with `<AA2fold>` |
| `ankh` | Ankh-base, Ankh-large | `[ankh]` | bfloat16 on CUDA (FP16 LayerNorm overflows); `is_split_into_words=True` for SentencePiece |
| `esm3c` | ESM-C 300M, ESM-C 600M | `[esm3c]` | Standalone `esm` package; no tokenizer; `model.encode` + `LogitsConfig(return_hidden_states=True)` |

Install everything at once with `pip install "protea-backends[all]"`.

## Why a separate package

Three reasons (master plan v3, [ADR D1](../PROTEA/docs/source/adr/D01-project-structure.rst)):

1. **Plugin extensibility.** New backends are added without touching
   `protea-core`. A single-file commit in this repository plus one
   line in `pyproject.toml` is enough.
2. **Per-backend deps.** Heavy ML libraries (`torch`,
   `transformers`, `sentencepiece`, `esm`) live behind Poetry extras.
   A deployment that only needs ESM-2 does not pull `esm`'s 1.3 GB
   wheel.
3. **Discovery is import-cheap.** Each plugin module imports nothing
   heavy at top level; `torch` and friends are imported lazily inside
   `load_model` and `embed_batch`. `protea-core` startup pays no cost
   for backends it never invokes.

## Adding a new backend

The full guide lives in the Sphinx docs under
`docs/source/contributing.rst`. The five-step summary:

1. Create `src/protea_backends/<your_name>/__init__.py`.
2. Subclass `EmbeddingBackend` and implement `load_model` +
   `embed_batch`. Set `name = "<your_name>"`.
3. Add `<your_name> = "protea_backends.<your_name>:plugin"` under
   `[tool.poetry.plugins."protea.backends"]` in `pyproject.toml`.
4. Declare any new heavy dependency as `optional = true` and add it
   to a new extras group `[your_name]`.
5. Mirror the existing test files (`tests/test_<your_name>.py`)
   covering instance type, ABC compliance, name attribute,
   discoverability, and method signatures.

## Development

```bash
poetry install
poetry run pytest             # 24 tests, ~0.2s
poetry run ruff check .
poetry run mypy src tests

# Build the docs (optional group, opt in):
poetry install --with docs
cd docs && make html
open build/html/index.html
```

## Contributing

Contributions are welcome from research institutions and individual developers.

**Branch strategy:** all changes target `develop`; `main` tracks stable
releases only.

```bash
git clone https://github.com/frapercan/protea-backends.git
cd protea-backends
git checkout develop
git checkout -b feature/my-backend

poetry install

# Implement your backend (see "Adding a new backend" above)
# Verify locally before opening a PR:
poetry run pytest
poetry run ruff check .
poetry run mypy src tests

# Open a pull request targeting develop
```

Key constraints:
- **Import-cheap:** top-level module imports must not trigger `torch` or
  any heavy ML library. Lazy imports inside `load_model` and `embed_batch`
  only.
- **Typed outputs:** `embed_batch` must return a `(B, D) float16 ndarray`.
  If the upstream model returns another dtype, cast inside the plugin.
- **No runtime deps on protea-core.** This package must stay installable
  independently of PROTEA's platform layer.

## Documentation

Full Sphinx documentation in `docs/source/`. Build locally with the
commands above. Each backend has its own page documenting models
supported, the extra to install, heavy dependencies, numerical type,
pooling rule, and plugin-specific quirks (Ankh's bfloat16 requirement,
ProstT5's prefix detection, ESM-C's tokenizer-less API).

## License

MIT. See `LICENSE`.

<!-- protea-stack:start -->

## Repositories in the PROTEA stack

Single source of truth: [`docs/source/_data/stack.yaml`](https://github.com/frapercan/PROTEA/blob/develop/docs/source/_data/stack.yaml) in PROTEA. Run `python scripts/sync_stack.py` to regenerate this block.

| Repo | Role | Status | Summary |
|------|------|--------|---------|
| [PROTEA](https://github.com/frapercan/PROTEA) | Platform | `active` | Backend platform. Hosts the ORM, job queue, FastAPI surface, frontend, and orchestration. |
| [protea-contracts](https://github.com/frapercan/protea-contracts) | Contracts | `beta` | Shared contract surface. ABCs, pydantic payloads, feature schema, schema_sha. Imported by every other repo. |
| [protea-method](https://github.com/frapercan/protea-method) | Inference | `skeleton` | Pure inference path (KNN, feature compute, reranker apply). Target of the F2C extraction. Bind-mounted by the LAFA containers. |
| [protea-sources](https://github.com/frapercan/protea-sources) | Source plugin | `skeleton` | Annotation source plugins (GOA, QuickGO, UniProt). Discovered via Python entry_points. |
| [protea-runners](https://github.com/frapercan/protea-runners) | Runner plugin | `skeleton` | Experiment runner plugins (LightGBM lab, KNN baseline, future GNN). Discovered via Python entry_points. |
| **protea-backends** (this repo) | Backend plugin | `skeleton` | Protein language model embedding backends (ESM family, T5/ProstT5, Ankh, ESM3-C). Discovered via Python entry_points. |
| [protea-reranker-lab](https://github.com/frapercan/protea-reranker-lab) | Lab | `active` | LightGBM reranker training lab. Pulls datasets from PROTEA, trains boosters, publishes them back via /reranker-models/import-by-reference. |
| [cafaeval-protea](https://github.com/frapercan/cafaeval-protea) | Evaluator | `active` | Standalone fork of cafaeval (CAFA-evaluator-PK) with the PK-coverage fix and a bit-exact parity guarantee against the upstream. |

<!-- protea-stack:end -->
