"""POST /api/chat — RAG question answering."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from api.deps import get_pipeline
from api.schemas.chat import ChatRequest, ChatResponse, KBRef
from gridtrace.pipeline import GridTracePipeline

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, pipeline: GridTracePipeline = Depends(get_pipeline)) -> ChatResponse:
    started = time.perf_counter()
    answer, hits = await pipeline.generate(req.query, top_k=req.top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return ChatResponse(
        answer=answer,
        query=req.query,
        refs=[
            KBRef(
                entry_id=h.entry_id,
                doc_id=h.doc_id,
                page_index=h.page_index,
                score=round(h.score, 4),
                match_score=round(h.match_score, 4),
                category=h.category,
            )
            for h in hits
        ],
        latency_ms=round(elapsed_ms, 2),
    )
