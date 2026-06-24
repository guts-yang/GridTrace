"""Chat request / response models."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048, description="User question")
    top_k: int | None = Field(default=None, ge=1, le=20)
    anchor_top_k: int | None = Field(default=None, ge=1, le=100)
    score_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    stream: bool = Field(default=False, description="Reserved; not yet implemented")


class KBRef(BaseModel):
    entry_id: int
    doc_id: str
    page_index: int
    score: float
    match_score: float
    category: str | None = None


class ChatResponse(BaseModel):
    answer: str
    query: str
    refs: list[KBRef] = Field(default_factory=list)
    latency_ms: float
