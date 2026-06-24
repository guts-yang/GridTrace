"""Knowledge-base CRUD models."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    question: str = Field(..., min_length=1)
    solution: str = Field(..., min_length=1)
    category: str | None = None
    doc_id: str = "default"
    page_index: int = Field(default=0, ge=0)


class IngestResponse(BaseModel):
    entry_id: int
    anchor_id: int
    quant_key: str
    match_score: float
    reused_anchor: bool


class EntrySummary(BaseModel):
    entry_id: int
    doc_id: str
    page_index: int
    question: str
    category: str | None
    match_score: float


class EntryList(BaseModel):
    total: int
    items: list[EntrySummary]


class UpdateRequest(BaseModel):
    question: str | None = None
    solution: str | None = None
    category: str | None = None


class UnlearnRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)


class UnlearnResponse(BaseModel):
    doc_id: str
    entries_deleted: int
    anchors_pruned: int


class KBStatsResponse(BaseModel):
    total_anchors: int
    total_entries: int
    unique_docs: int
    avg_match_score: float
    orphan_anchors: int
