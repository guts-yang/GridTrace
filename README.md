# GridTrace — Quantized Unlearning-Aware Retrieval & Knowledge

> *Retrieval at the fundamental particle scale.*

[![CI](https://img.shields.io/github/actions/workflow/status/<org>/GridTrace/ci.yml?branch=main)](https://github.com/<org>/GridTrace/actions)
[![Coverage](https://img.shields.io/codecov/c/github/<org>/GridTrace)](https://codecov.io/gh/<org>/GridTrace)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

## Overview

GridTrace is a RAG retrieval algorithm built on **grid-quantized anchors**. It
compresses high-dimensional semantic vectors into compact anchors via the
deterministic map `round(v / ε) × ε`, uses those anchors as routing nodes for a
two-phase retrieval pipeline, and natively supports **exact unlearning** —
deleting a document automatically prunes its orphan anchors with no index
rebuild.

### Core Features

| Feature | Description |
|---|---|
| Grid-quantized anchors | O(1) deterministic vector compression — no clustering, no PQ |
| Two-phase retrieval | L1 anchor Top-K routing → L2 full-precision cosine rerank |
| Exact unlearning | Delete by `doc_id` / `page_index`; orphan anchors auto-pruned |
| Joint semantic representation | `(q_vec + s_vec) / 2` joint vector + `match_score` self-check |
| Self-learning loop | Tickets resolved → writeback to KB → auto-vectorize |
| Pluggable storage | `pgvector` (prod) / `SQLite` (dev) / `In-memory` (eval) backends |

## Architecture

```mermaid
graph TB
    subgraph Ingest ["Ingest"]
        Q[Question] --> EMB[Embedder]
        S[Solution] --> EMB
        EMB --> JOINT["Joint vector<br/>(qv+sv)/2 + L2"]
        JOINT --> QUANT[Quantizer<br/>round(v/ε)·ε]
        QUANT --> HASH[SHA256 quant_key]
        HASH --> ANCHOR{Anchor exists?}
        ANCHOR -->|no| CREATE[Create new anchor]
        ANCHOR -->|yes| REUSE[Reuse anchor ID]
        CREATE --> STORE[(kb_anchors)]
        REUSE --> STORE
        JOINT --> ENTRY[(kb_entries)]
    end

    subgraph Search ["Two-Phase Retrieval"]
        QUERY[User query] --> QEMB[Query Embedding]
        QEMB --> L1["L1: anchor cosine Top-K=8"]
        STORE --> L1
        L1 --> ROUTE[Route to candidate entries]
        ROUTE --> L2["L2: full-precision cosine<br/>+ threshold filter"]
        ENTRY --> L2
        L2 --> RESULT[Top-3 results]
    end

    subgraph Unlearn ["Exact Unlearning"]
        DEL[Delete doc_id] --> DEL_ENTRY[Delete entries]
        DEL_ENTRY --> PRUNE{Check ref count}
        PRUNE -->|count=0| DEL_ANCHOR[Delete orphan]
        PRUNE -->|count>0| KEEP[Keep anchor]
    end
```

See [`docs/architecture.md`](docs/architecture.md) for the full design.

## Algorithm

### 1. Grid Quantization

Unlike K-means or Product Quantization, GridTrace uses **deterministic grid
snapping** for vector compression:

```
anchor_vec = round(v / ε) × ε
```

`ε` (epsilon) controls quantization granularity (default `0.02`). After
quantization, a stable `quant_key` is generated via SHA256; entries sharing
the same key share the same anchor.

**Advantages**: no training, `O(d)` time complexity, fully deterministic.

### 2. Two-Phase Retrieval

| Phase | Operation | Scope | Complexity |
|---|---|---|---|
| L1 routing | Anchor-table cosine, Top-K | Anchor table (~N/10) | `O(A × d)` |
| L2 rerank | Full-precision cosine + threshold | Candidates (~Top-K × avg_bucket) | `O(C × d)` |

### 3. Exact Unlearning

After deleting all entries for a document, we check the reference count of
each associated anchor. If the count drops to zero, the anchor is removed.
The entire flow runs in `O(1)` per document — no index rebuild required.

### 4. Joint Semantic Representation

At ingest time, the question and solution are encoded separately; their mean
is L2-normalized to produce the joint vector:

```
joint = normalize((q_vec + s_vec) / 2)
```

Simultaneously `match_score = cosine(q_vec, s_vec)` is recorded as a
self-check quality score, reflecting the intrinsic semantic consistency of
the question-solution pair.

## Quick Start

### Requirements

- Python ≥ 3.11
- PostgreSQL ≥ 16 with the `pgvector` extension (or use `SQLite` / `Memory`
  backends for development)

### 1. Clone

```bash
git clone https://github.com/<org>/GridTrace.git
cd GridTrace
```

### 2. Install

```bash
# Recommended: uv (faster)
uv sync --extra dev --extra embed

# Traditional pip
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev,embed]"
```

### 3. Start Infrastructure

```bash
docker compose up -d postgres          # PostgreSQL + pgvector
psql -U postgres -d gridtrace -f docker/init.sql
```

### 4. Configure

```bash
cp .env.example .env
# Edit .env: set DATABASE_URL and LLM config
```

### 5. Run

```bash
make dev   # uvicorn --reload on :8000
```

### 6. Verify

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset a frozen account?"}'
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ANCHOR_QUANT_EPSILON` | `0.02` | Quantization grid granularity; smaller = more anchors |
| `RAG_ANCHOR_TOP_K` | `8` | L1 anchor routing Top-K |
| `RAG_SCORE_THRESHOLD` | `0.65` | L2 cosine similarity threshold |
| `RAG_TOP_K` | `3` | Final returned entry count |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | Semantic encoding model |
| `EMBEDDING_DEVICE` | `cpu` | Encoding device (`cpu` / `cuda`) |
| `GRIDTRACE_STORAGE_BACKEND` | `postgres` | `postgres` / `sqlite` / `memory` |
| `LLM_PROVIDER` | `echo` | `echo` / `deepseek` / `openai` / `ollama` |

## Evaluation

```bash
make eval
cat docs/eval_report.md
```

Key metrics:
- **Hit@1 / Hit@3** — Top-K hit rate
- **MRR** — Mean reciprocal rank
- **L1 Anchor Hit Rate** — L1 anchor routing hit rate
- **Avg Match Score** — self-check quality score

## Project Structure

```
GridTrace/
├── gridtrace/         # Core algorithm package
├── api/           # FastAPI REST layer
├── data/          # FAQ + evaluation datasets
├── scripts/       # Operational scripts
├── tests/         # unit / integration / eval
├── docs/          # architecture / algorithm / API / eval reports
├── docker/        # Dockerfiles + init.sql
└── .github/       # CI/CD workflows
```

See [`docs/architecture.md`](docs/architecture.md) for full module boundaries
and data flow.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). PRs are welcome — please run
`make lint test` before opening.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
