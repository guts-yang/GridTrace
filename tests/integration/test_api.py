"""API integration test using FastAPI TestClient.

Uses the in-memory backend (default in tests) so this can run without
PostgreSQL.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("GRIDTRACE_STORAGE_BACKEND", "memory")
os.environ.setdefault("EMBEDDING_DIM", "64")
os.environ.setdefault("LLM_PROVIDER", "echo")

from fastapi.testclient import TestClient  # noqa: E402

from api.server import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert "version" in body


def test_kb_stats_empty(client: TestClient) -> None:
    r = client.get("/api/knowledge/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_entries"] >= 0


def test_ingest_and_list(client: TestClient) -> None:
    r = client.post(
        "/api/knowledge/entries",
        json={
            "question": "How to reset a frozen account?",
            "solution": "UPDATE users SET status='active' WHERE id=:uid;",
            "category": "account",
            "doc_id": "test-doc",
            "page_index": 1,
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["entry_id"] > 0
    assert body["match_score"] >= 0.0


def test_unlearn_endpoint(client: TestClient) -> None:
    r = client.post("/api/knowledge/unlearn", json={"doc_id": "test-doc"})
    assert r.status_code == 200
    body = r.json()
    assert body["entries_deleted"] >= 1


def test_chat_endpoint(client: TestClient) -> None:
    client.post(
        "/api/knowledge/entries",
        json={
            "question": "How to reset a frozen account?",
            "solution": "Use SQL: UPDATE users SET status='active' WHERE id=:uid;",
            "doc_id": "chat-doc",
            "page_index": 1,
        },
    )
    r = client.post("/api/chat", json={"query": "frozen account reset", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "refs" in body
    assert "latency_ms" in body
