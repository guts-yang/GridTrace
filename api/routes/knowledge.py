"""CRUD endpoints for the knowledge base."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_pipeline
from api.schemas.knowledge import (
    EntryList,
    EntrySummary,
    IngestRequest,
    IngestResponse,
    KBStatsResponse,
    UnlearnRequest,
    UnlearnResponse,
    UpdateRequest,
)
from gridtrace.pipeline import GridTracePipeline

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.post("/entries", response_model=IngestResponse, status_code=201)
def add_entry(req: IngestRequest, pipeline: GridTracePipeline = Depends(get_pipeline)) -> IngestResponse:
    res = pipeline.ingest(
        question=req.question,
        solution=req.solution,
        category=req.category,
        doc_id=req.doc_id,
        page_index=req.page_index,
    )
    return IngestResponse(
        entry_id=res.entry_id,
        anchor_id=res.anchor_id,
        quant_key=res.quant_key,
        match_score=round(res.match_score, 4),
        reused_anchor=res.reused_anchor,
    )


@router.get("/entries", response_model=EntryList)
def list_entries(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    pipeline: GridTracePipeline = Depends(get_pipeline),
) -> EntryList:
    raw = pipeline.storage.list_entries(limit=limit, offset=offset)
    stats = pipeline.stats()
    return EntryList(
        total=stats.total_entries,
        items=[
            EntrySummary(
                entry_id=e.entry_id,
                doc_id=e.doc_id,
                page_index=e.page_index,
                question=e.question,
                category=e.category,
                match_score=round(e.match_score, 4),
            )
            for e in raw
        ],
    )


@router.get("/entries/{entry_id}")
def get_entry(entry_id: int, pipeline: GridTracePipeline = Depends(get_pipeline)):
    e = pipeline.storage.fetch_entry(entry_id)
    if e is None:
        raise HTTPException(status_code=404, detail=f"entry {entry_id} not found")
    return {
        "entry_id": e.entry_id,
        "anchor_id": e.anchor_id,
        "doc_id": e.doc_id,
        "page_index": e.page_index,
        "question": e.question,
        "solution": e.solution,
        "category": e.category,
        "match_score": round(e.match_score, 4),
    }


@router.put("/entries/{entry_id}")
def update_entry(
    entry_id: int,
    req: UpdateRequest,
    pipeline: GridTracePipeline = Depends(get_pipeline),
):
    e = pipeline.storage.update_entry(
        entry_id,
        question=req.question,
        solution=req.solution,
        category=req.category,
    )
    if e is None:
        raise HTTPException(status_code=404, detail=f"entry {entry_id} not found")
    return {"entry_id": e.entry_id, "updated": True}


@router.post("/unlearn", response_model=UnlearnResponse)
def unlearn(req: UnlearnRequest, pipeline: GridTracePipeline = Depends(get_pipeline)) -> UnlearnResponse:
    res = pipeline.unlearn(req.doc_id)
    return UnlearnResponse(
        doc_id=res.doc_id,
        entries_deleted=res.entries_deleted,
        anchors_pruned=res.anchors_pruned,
    )


@router.get("/stats", response_model=KBStatsResponse)
def kb_stats(pipeline: GridTracePipeline = Depends(get_pipeline)) -> KBStatsResponse:
    s = pipeline.stats()
    return KBStatsResponse(
        total_anchors=s.total_anchors,
        total_entries=s.total_entries,
        unique_docs=s.unique_docs,
        avg_match_score=round(s.avg_match_score, 4),
        orphan_anchors=s.orphan_anchors,
    )
