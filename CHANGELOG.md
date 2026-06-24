# Changelog

All notable changes to GridTrace are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of GridTrace
- Grid-quantized anchor algorithm (`round(v/ε)·ε` + SHA256 dedup)
- Two-phase retrieval (L1 anchor routing → L2 full-precision rerank)
- Exact unlearning with automatic orphan-anchor pruning
- Joint semantic representation `(q_vec + s_vec) / 2` with `match_score` self-check
- Three pluggable storage backends: `pgvector` (production), `SQLite` (dev), `Memory` (eval)
- FastAPI server with `/api/chat`, `/api/knowledge/*`, `/api/health`
- Eval engine + scripts (`run_eval.py`, `hyperparam_search.py`)
- Docker Compose for local development
- CI/CD: lint, unit, integration, publish, docs
- Pre-commit hooks (ruff, mypy, basic checks)
- Dependabot for pip/docker/github-actions

### Changed
- _None_

### Deprecated
- _None_

### Removed
- _None_

### Fixed
- _None_

### Security
- _None_

## [0.1.0] - 2026-06-24

### Added
- First public release.

[Unreleased]: https://github.com/<org>/GridTrace/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/<org>/GridTrace/releases/tag/v0.1.0
