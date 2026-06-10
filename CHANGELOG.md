# Changelog

All notable changes to `protea-backends` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Refactored the per-backend residue-pooling and layer-aggregation code
  into shared helpers in `protea_backends._chunk_helpers`
  (`pool_residues`, `stack_and_aggregate`, `aggregate_layers`), removing
  duplicated logic across the four backends. Behavior-preserving: plugin
  entry points and the `EmbeddingBackend` interface are unchanged.
- Aligned the packaged version with the `v0.1.0` release tag.

### Added

- Sphinx documentation pages: a quickstart, a backend-contract reference,
  and a consolidated API page. The toctree now reads quickstart, contract,
  backends, contributing, api.
- `docs` CI workflow (`.github/workflows/docs.yml`) that builds the HTML
  docs with warnings treated as errors and uploads the result as an
  artifact.
- This changelog.

## [0.1.0] - 2026-05-13

### Added

- Initial release. Four protein language model embedding backends
  discovered by `protea-core` through the `protea.backends`
  entry-points group:
  - `esm`: HuggingFace `EsmModel` (ESM-1b, ESM-2 8M to 15B).
  - `t5`: encoder-only T5 (ProtT5-XL, ProstT5 with `<AA2fold>` prefix).
  - `ankh`: Ankh-base and Ankh-large (bfloat16 on CUDA,
    `is_split_into_words=True` tokenisation).
  - `esm3c`: ESM-C via the standalone `esm` SDK (no tokenizer).
- Three output paths per backend: pooled `embed_batch`, per-residue
  `embed_batch_per_residue` (returns an `EmbeddingPayload`), and the
  bit-exact `embed_chunks` chunked pipeline.
- Heavy ML dependencies behind per-backend Poetry extras
  (`[esm]`, `[t5]`, `[ankh]`, `[esm3c]`, `[all]`) with import-cheap
  plugin modules (lazy `torch` / `transformers` / `esm` imports).

[Unreleased]: https://github.com/frapercan/protea-backends/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/frapercan/protea-backends/releases/tag/v0.1.0
