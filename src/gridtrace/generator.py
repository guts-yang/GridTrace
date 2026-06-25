"""Answer generators.

A :class:`Generator` turns a list of search hits + user query into a
final natural-language answer. The default :class:`EchoGenerator`
returns the top hit's solution verbatim (deterministic, no network);
:class:`DeepSeekGenerator` and :class:`OpenAIGenerator` use HTTP APIs
(OpenAI-compatible schema).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Sequence

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from gridtrace.storage._types import SearchHit
from gridtrace.utils.config import GridTraceConfig
from gridtrace.utils.logging import get_logger

logger = get_logger(__name__)

_PROMPT_TEMPLATE_ZH = """你是一名资深运维助理。请基于下面提供的【知识库参考】回答用户问题。
- 如果参考信息不足，请明确说明"知识库暂无相关条目"，不要编造答案。
- 回答要简洁、步骤清晰；保留命令、路径、参数原文。

【用户问题】
{query}

【知识库参考】(共 {n} 条)
{context}

【回答】"""


def _format_context(hits: Sequence[SearchHit]) -> str:
    blocks: list[str] = []
    for i, h in enumerate(hits, 1):
        cat = f" [{h.category}]" if h.category else ""
        blocks.append(
            f"### 参考 {i}{cat} (score={h.score:.3f}, doc={h.doc_id}#{h.page_index})\n"
            f"Q: {h.question}\n"
            f"A: {h.solution}"
        )
    return "\n\n".join(blocks) if blocks else "（无）"


class Generator(ABC):
    """Abstract answer generator."""

    @abstractmethod
    async def generate(self, query: str, hits: Sequence[SearchHit]) -> str: ...


class EchoGenerator(Generator):
    """Returns the top hit's solution verbatim (default for offline / tests)."""

    def __init__(self, *, fallback: str = "知识库暂无相关条目。") -> None:
        self._fallback = fallback

    async def generate(self, query: str, hits: Sequence[SearchHit]) -> str:
        if not hits:
            return self._fallback
        return hits[0].solution


class DeepSeekGenerator(Generator):
    """DeepSeek Chat (OpenAI-compatible)."""

    def __init__(self, config: GridTraceConfig) -> None:
        if not config.llm_api_key:
            raise ValueError("LLM_API_KEY is required for DeepSeekGenerator")
        self._cfg = config
        self._client = httpx.AsyncClient(
            base_url=config.llm_base_url,
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={"Authorization": f"Bearer {config.llm_api_key}"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def generate(self, query: str, hits: Sequence[SearchHit]) -> str:
        prompt = _PROMPT_TEMPLATE_ZH.format(
            query=query, n=len(hits), context=_format_context(hits)
        )
        payload = {
            "model": self._cfg.llm_model,
            "messages": [
                {"role": "system", "content": "你是一个严谨的运维助理。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._cfg.llm_temperature,
            "max_tokens": self._cfg.llm_max_tokens,
            "stream": False,
        }
        r = await self._client.post("/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        return str(data["choices"][0]["message"]["content"]).strip()


class OpenAIGenerator(Generator):
    """OpenAI Chat Completions (works with any OpenAI-compatible endpoint)."""

    def __init__(self, config: GridTraceConfig) -> None:
        if not config.llm_api_key:
            raise ValueError("LLM_API_KEY is required for OpenAIGenerator")
        self._cfg = config
        base = config.llm_base_url or "https://api.openai.com/v1"
        self._client = httpx.AsyncClient(
            base_url=base,
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={"Authorization": f"Bearer {config.llm_api_key}"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def generate(self, query: str, hits: Sequence[SearchHit]) -> str:
        prompt = _PROMPT_TEMPLATE_ZH.format(
            query=query, n=len(hits), context=_format_context(hits)
        )
        payload = {
            "model": self._cfg.llm_model,
            "messages": [
                {"role": "system", "content": "You are a precise operations assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._cfg.llm_temperature,
            "max_tokens": self._cfg.llm_max_tokens,
        }
        r = await self._client.post("/chat/completions", json=payload)
        r.raise_for_status()
        return str(r.json()["choices"][0]["message"]["content"]).strip()
