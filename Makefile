.PHONY: help dev test test-unit test-integration test-eval lint format typecheck eval clean reset-kb ingest docs-build

PYTHON ?= python
UV    ?= uv

help:
	@echo "GridTrace make targets"
	@echo "  make dev              Run API server (uvicorn --reload)"
	@echo "  make test             Run unit tests with coverage"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests (requires PG)"
	@echo "  make test-eval        Run evaluation tests"
	@echo "  make lint             ruff check + format check"
	@echo "  make format           ruff auto-fix + format"
	@echo "  make typecheck        mypy strict"
	@echo "  make eval             Run evaluation script"
	@echo "  make ingest           Ingest FAQ from data/faq/*.md"
	@echo "  make reset-kb         Reset knowledge base (destructive)"
	@echo "  make docs-build       Build documentation site (mkdocs)"
	@echo "  make clean            Remove caches and build artifacts"

dev:
	$(UV) run uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

test:
	$(UV) run pytest tests/ -v --cov=gridtrace --cov-report=term-missing

test-unit:
	$(UV) run pytest tests/unit/ -v

test-integration:
	$(UV) run pytest tests/integration/ -v

test-eval:
	$(UV) run pytest tests/eval/ -v

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck:
	$(UV) run mypy gridtrace/ api/

eval:
	$(UV) run python scripts/run_eval.py

ingest:
	$(UV) run python scripts/ingest_docs.py --source md --path data/faq/OpsWarden_FAQ.md

reset-kb:
	$(UV) run python scripts/reset_kb.py --confirm

docs-build:
	$(UV) run mkdocs build --strict

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist *.egg-info .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
