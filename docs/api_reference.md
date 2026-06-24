# GridTrace API Reference

> All endpoints are served by the FastAPI app in `api/server.py`.
> Base URL in dev: `http://localhost:8000`. OpenAPI docs at `/docs`.

## 1. Health

### `GET /api/health`

Service health check.

**Response 200**

```json
{
  "status": "ok",
  "version": "0.1.0",
  "storage_ok": true,
  "embedder": "SentenceTransformerEmbedder",
  "generator": "EchoGenerator",
  "kb": {
    "anchors": 10,
    "entries": 25,
    "docs": 5,
    "avg_match_score": 0.78
  }
}
```

## 2. Chat

### `POST /api/chat`

RAG question answering. Returns the LLM-generated answer plus the
retrieved KB references.

**Request**

```json
{
  "query": "How do I reset a frozen account?",
  "top_k": 3,
  "anchor_top_k": 8,
  "score_threshold": 0.65,
  "stream": false
}
```

| Field             | Type     | Default | Notes                                |
|-------------------|----------|---------|--------------------------------------|
| `query`           | string   | —       | required, 1–2048 chars               |
| `top_k`           | int      | 3       | final returned entries (1–20)        |
| `anchor_top_k`    | int      | 8       | L1 anchor pool size (1–100)          |
| `score_threshold` | float    | 0.65    | L2 cosine cutoff (-1.0 to 1.0)       |
| `stream`          | bool     | false   | reserved; not yet implemented        |

**Response 200**

```json
{
  "answer": "Run: UPDATE users SET status='active' WHERE id=:uid;",
  "query": "How do I reset a frozen account?",
  "refs": [
    {
      "entry_id": 1,
      "doc_id": "OpsWarden_FAQ",
      "page_index": 4,
      "score": 0.92,
      "match_score": 0.81,
      "category": "account"
    }
  ],
  "latency_ms": 12.34
}
```

## 3. Knowledge

### `POST /api/knowledge/entries` — ingest one entry

**Request**

```json
{
  "question": "How to reset a frozen account?",
  "solution": "UPDATE users SET status='active' WHERE id=:uid;",
  "category": "account",
  "doc_id": "OpsWarden_FAQ",
  "page_index": 4
}
```

**Response 201**

```json
{
  "entry_id": 42,
  "anchor_id": 7,
  "quant_key": "5a8e1c9b…",
  "match_score": 0.81,
  "reused_anchor": false
}
```

### `GET /api/knowledge/entries` — list entries

Query params: `limit` (1–500, default 50), `offset` (default 0).

**Response 200**

```json
{
  "total": 25,
  "items": [
    {
      "entry_id": 1,
      "doc_id": "OpsWarden_FAQ",
      "page_index": 1,
      "question": "How to handle 5xx spike?",
      "category": "incident",
      "match_score": 0.76
    }
  ]
}
```

### `GET /api/knowledge/entries/{entry_id}` — fetch one

**Response 200**

```json
{
  "entry_id": 1,
  "anchor_id": 1,
  "doc_id": "OpsWarden_FAQ",
  "page_index": 1,
  "question": "How to handle 5xx spike?",
  "solution": "Step 1: …",
  "category": "incident",
  "match_score": 0.76
}
```

**Response 404** when `entry_id` does not exist.

### `PUT /api/knowledge/entries/{entry_id}` — partial update

```json
{ "question": "New question text", "category": "ops" }
```

**Response 200**

```json
{ "entry_id": 1, "updated": true }
```

### `POST /api/knowledge/unlearn` — exact unlearn by doc_id

```json
{ "doc_id": "OpsWarden_FAQ" }
```

**Response 200**

```json
{
  "doc_id": "OpsWarden_FAQ",
  "entries_deleted": 10,
  "anchors_pruned": 3
}
```

### `GET /api/knowledge/stats` — KB statistics

**Response 200**

```json
{
  "total_anchors": 12,
  "total_entries": 25,
  "unique_docs": 5,
  "avg_match_score": 0.78,
  "orphan_anchors": 0
}
```

## 4. Error format

All errors return:

```json
{
  "code": 500,
  "message": "Internal Server Error",
  "data": null
}
```

Validation errors return 422 with a Pydantic detail array.
