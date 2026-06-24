# GridTrace Architecture

> Module boundaries, data flow, and storage model.

## 1. Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       API Layer (api/)                      │
│   FastAPI routes → dependency injection → pipeline           │
├─────────────────────────────────────────────────────────────┤
│                   Pipeline Layer (gridtrace/pipeline.py)         │
│   ingest() / search() / generate() / unlearn() orchestration │
├──────────┬──────────┬────────────┬───────────────────────────┤
│ Quantizer│ Retriever│  Embedder  │     Generator            │
│ 网格量化  │ 双阶段检索│  语义编码   │     LLM 答案生成         │
├──────────┴──────────┴────────────┴───────────────────────────┤
│                 Storage Layer (gridtrace/storage/)              │
│   postgres.py │ sqlite.py │ memory.py — StorageBackend ABC  │
├─────────────────────────────────────────────────────────────┤
│                      Infrastructure                          │
│   PostgreSQL + pgvector │ sentence-transformers │ LLM API    │
└─────────────────────────────────────────────────────────────┘
```

### Module responsibilities

| Module | Responsibility | May import from | May NOT import from |
|---|---|---|---|
| `gridtrace/quantizer` | Grid snap + batch quantization | `gridtrace.utils.hashing` | storage, embedder, network |
| `gridtrace/embedder` | Text → vector | numpy | storage, network |
| `gridtrace/retriever` | L1 + L2 search | embedder, storage, utils | generator, pipeline, network |
| `gridtrace/generator` | Hits → answer text | storage (types), httpx | storage (impl), embedder |
| `gridtrace/storage/*` | Persistence | numpy, utils | embedder, retriever, pipeline |
| `gridtrace/pipeline` | End-to-end orchestration | everything in gridtrace | api, scripts |

The dependency graph is a DAG — circular imports are blocked by
construction.

## 2. Data Flow

### 2.1 Ingest

```
question ─┐
          ├─→ Embedder.encode → q_vec, s_vec ─→ (q+s)/2 → L2 → joint
solution ─┘                                                     │
                                                                ▼
                                       Quantizer.round(v/ε)·ε → anchor_vec
                                                                │
                                                                ▼
                                       SHA256(anchor_vec) → quant_key
                                                                │
                                                ┌───────────────┴────────────────┐
                                                ▼                                ▼
                                  upsert_anchor (id by quant_key)     ingest_entry (joint embedding)
                                                │                                │
                                                └──────────► kb_anchors ◄───────┘
                                                                  kb_entries
```

**`ref_count` invariant**: every `kb_anchors.ref_count == COUNT(kb_entries WHERE anchor_id = id)`.
PostgreSQL enforces this via the `trg_kb_anchors_refcount` trigger; the
SQLite and Memory backends maintain it manually.

### 2.2 Two-Phase Search

```
query ─→ Embedder.encode(kind="query") → q_vec
                │
                ├── L1: storage.search_anchors(q_vec, top_k=A)
                │       returns A AnchorHits ranked by cosine
                │
                ├── fetch_entries_by_anchors(anchor_ids)
                │       returns all entries pointing at those anchors
                │
                └── L2: cosine(q_vec, entry.embedding) on candidates
                        filter by score >= threshold
                        take top-K → SearchHit[]
```

### 2.3 Unlearn

```
delete_by_doc(doc_id):
    1. SELECT id, anchor_id FROM kb_entries WHERE doc_id = :d
    2. DELETE kb_entries WHERE doc_id = :d         (trigger decrements ref_count)
    3. DELETE FROM kb_anchors WHERE ref_count = 0   (prune orphans)
    4. INSERT INTO kb_unlearn_log (audit)
```

## 3. Storage Model

### `kb_anchors` (L1 routing table)

| Column        | Type        | Notes                                |
|---------------|-------------|--------------------------------------|
| `id`          | BIGSERIAL   | PK                                   |
| `quant_key`   | TEXT UNIQUE | SHA256 of quantized vec              |
| `anchor_vec`  | vector(512) | pgvector type (cosine distance)      |
| `ref_count`   | INTEGER     | Trigger-maintained                   |
| `created_at`  | TIMESTAMPTZ |                                      |

### `kb_entries` (L2 rerank pool)

| Column        | Type         | Notes                                   |
|---------------|--------------|-----------------------------------------|
| `id`          | BIGSERIAL    | PK                                      |
| `anchor_id`   | BIGINT       | FK → `kb_anchors(id)` ON DELETE CASCADE |
| `question`    | TEXT         | Original question                       |
| `solution`    | TEXT         | Original solution / answer              |
| `category`    | TEXT NULL    | Optional category                       |
| `doc_id`      | TEXT         | Document origin (for unlearn)           |
| `page_index`  | INTEGER      | Page / position within doc              |
| `match_score` | REAL         | `cosine(q_vec, s_vec)` self-check       |
| `embedding`   | vector(512)  | Joint vector `(q+s)/2` L2-normalized    |
| `created_at`  | TIMESTAMPTZ  |                                         |
| `updated_at`  | TIMESTAMPTZ  | Trigger-maintained                      |

### `kb_unlearn_log` (audit)

| Column            | Type        | Notes                       |
|-------------------|-------------|-----------------------------|
| `id`              | BIGSERIAL   | PK                          |
| `doc_id`          | TEXT        |                             |
| `entries_deleted` | INTEGER     |                             |
| `anchors_pruned`  | INTEGER     |                             |
| `triggered_at`    | TIMESTAMPTZ |                             |

## 4. Pluggable Backends

The `StorageBackend` ABC has exactly **9 abstract methods**; each
backend implements them with the same observable behavior. Tests in
`tests/unit/test_storage.py` validate the in-memory backend; PG and
SQLite share the same test suite via parametrization (see
`tests/integration/`).

| Backend   | Use case                        | Vector search       |
|-----------|---------------------------------|---------------------|
| Memory    | Tests, evaluation, single-proc  | numpy cosine        |
| SQLite    | Local dev, no Docker            | numpy cosine (load  |
|           |                                 | all anchors in mem) |
| PostgreSQL| Production, large-scale         | pgvector `<=>`      |

## 5. Failure Modes & Mitigations

See [`plan.md` § 六](../plan.md#六风险与缓解) for the full table.

| Failure | Mitigation |
|---|---|
| pgvector version incompatibility | `StorageBackend` ABC isolation + integration tests |
| BGE model download failure | `EMBEDDING_LOCAL_PATH` env + `MemoryBackend` + `HashingEmbedder` fallback in `pipeline._default_embedder` |
| LLM API unstable | `EchoGenerator` fallback (returns top hit solution) |
| Concurrent unlearn races | `prune_anchor_if_unused` runs inside DB transaction (PG: row-level lock) |
| Large KB anchor-table growth | Future: switch `kb_anchors.anchor_vec` index to HNSW (`CREATE INDEX … USING hnsw (anchor_vec vector_cosine_ops)`) |
