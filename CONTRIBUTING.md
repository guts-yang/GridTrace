# Contributing to GridTrace

Thank you for your interest in contributing. GridTrace is a research-grade RAG
retrieval engine and we welcome bug reports, performance improvements, doc
fixes, and algorithm extensions.

## Quick Start

```bash
# 1. Fork and clone
git clone https://github.com/<your-name>/GridTrace.git
cd GridTrace

# 2. Install dev dependencies
uv sync --extra dev --extra embed

# 3. Install pre-commit hooks
uv run pre-commit install

# 4. Start infrastructure
docker compose up -d postgres
psql -U postgres -d gridtrace -f docker/init.sql

# 5. Run tests
make test
```

## Workflow

1. **Branch** — create a topic branch from `develop`:
   - `feature/<scope>-<desc>` — new functionality
   - `fix/<scope>-<desc>` — bug fix
   - `docs/<desc>` — documentation only
   - `refactor/<scope>-<desc>` — internal restructuring

2. **Commit** — follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat(retriever): add HNSW index option`
   - `fix(quantizer): handle zero-vector edge case`
   - `docs(api): add /knowledge/stats example`

3. **Lint & Test** — `make lint && make test-unit` must pass locally.

4. **Pull Request** — open PR against `develop`:
   - Fill in the PR template (test plan, screenshots, breaking-changes)
   - Ensure CI is green
   - Request at least one reviewer

## Code Style

- **Python 3.11+** — use modern syntax (`match`, `|` unions, `dict[str, int]`)
- **Type hints** — `mypy --strict` must pass; prefer explicit types over `Any`
- **Ruff** — line length 100, double quotes, sort imports
- **Docstrings** — Google style for public APIs, single-line for helpers

## Testing

- All new features must include unit tests (target: `tests/unit/`)
- Storage-layer changes must extend `tests/integration/` (PG required)
- Algorithm changes must include an entry in `data/eval/` and a
  `tests/eval/test_*.py` regression test

## Architecture Rules

Please review [`docs/architecture.md`](docs/architecture.md) before touching
the storage layer or pipeline. The key invariants:

- `gridtrace/storage/*` must implement the `StorageBackend` ABC — no direct SQL
  outside this layer
- `gridtrace/pipeline.py` is the only public entry point that mutates state
- Algorithms (`quantizer`, `retriever`, `embedder`) must be **pure** — no
  I/O, no global state

## Release Process

1. Merge `develop` → `main` via PR
2. Tag `vX.Y.Z` on `main`
3. `publish.yml` builds Docker image → GHCR
4. Update `CHANGELOG.md`

## Code of Conduct

Be respectful. Harassment of any kind is not tolerated. See
[GitHub's Community Guidelines](https://docs.github.com/en/site-policy/github-terms/github-community-guidelines).
