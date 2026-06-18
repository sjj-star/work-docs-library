"""Passage reranker mechanism layer."""

import json
import logging
from abc import ABC, abstractmethod

from .config import Config
from .llm_chat_client import BaseLLMClient

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rank(
        self, query: str, passages: list[tuple[int, str]]
    ) -> list[tuple[int, float]]:
        """Return list of (block_db_id, score) sorted by descending relevance."""


class LLMReranker(Reranker):
    """使用 LLM 判断 query-passage 相关度（0-10），适合 Agent 环境."""

    def __init__(self, client: BaseLLMClient | None = None) -> None:
        """Initialize with an optional LLM client; create a default client if None."""
        self.client = client or BaseLLMClient()

    def _parse_scores(self, raw: str, num_passages: int) -> list[float]:
        """Parse judge response into a list of scores."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Reranker returned non-JSON: {raw[:200]}")
            parsed = {}

        if isinstance(parsed, list) and len(parsed) == num_passages:
            return [float(s) for s in parsed]

        if isinstance(parsed, dict) and "scores" in parsed:
            scores = parsed["scores"]
            if isinstance(scores, list) and len(scores) == num_passages:
                return [float(s) for s in scores]

        logger.warning(f"Reranker returned unexpected format: {raw[:200]}")
        return [0.0] * num_passages

    def rank(
        self, query: str, passages: list[tuple[int, str]]
    ) -> list[tuple[int, float]]:
        """Score and sort passages by relevance to the query."""
        if not passages:
            return []

        system = (Config.PROMPT_DIR / "rerank_passage_system.txt").read_text(
            encoding="utf-8"
        )
        user_template = (Config.PROMPT_DIR / "rerank_passage_user.txt").read_text(
            encoding="utf-8"
        )

        numbered = "\n\n".join(
            f"[{i+1}] {text}" for i, (_id, text) in enumerate(passages)
        )
        user = user_template.format(
            query=query, passages=numbered, num_passages=len(passages)
        )

        raw = self.client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        scores = self._parse_scores(raw, len(passages))
        ranked = sorted(
            zip((pid for pid, _ in passages), scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked
