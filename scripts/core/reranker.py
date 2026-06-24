"""Passage reranker mechanism layer."""

import json
import logging
from abc import ABC, abstractmethod
from string import Template
from typing import Any

from .config import Config
from .llm_chat_client import BaseLLMClient

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rank(self, query: str, passages: list[tuple[int, str]]) -> list[tuple[int, float]]:
        """Return list of (block_db_id, score) sorted by descending relevance."""


class CrossEncoderReranker(Reranker):
    """使用本地 cross-encoder 模型对 query-passage 对进行相关性打分."""

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize with an optional cross-encoder model name.

        Args:
            model_name: Hugging Face model name or local path. Uses a small default
                open-source model if not provided.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model: Any = None

    def _load_model(self) -> Any:
        """Lazy-load the cross-encoder model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "CrossEncoderReranker requires the 'sentence-transformers' package. "
                    "Install it with: pip install 'sentence-transformers>=3.0.0'"
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rank(self, query: str, passages: list[tuple[int, str]]) -> list[tuple[int, float]]:
        """Score and sort passages by relevance to the query."""
        if not passages:
            return []

        model = self._load_model()
        pairs = [(query, passage) for _, passage in passages]
        raw_scores = model.predict(pairs)

        # CrossEncoder.predict may return a NumPy array or a plain list.
        scores: list[float]
        if hasattr(raw_scores, "tolist"):
            scores = [float(s) for s in raw_scores.tolist()]
        else:
            scores = [float(s) for s in raw_scores]

        if len(scores) != len(passages):
            raise RuntimeError(
                f"CrossEncoder returned {len(scores)} scores for {len(passages)} passages"
            )

        ranked = sorted(
            zip((pid for pid, _ in passages), scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked


class LLMReranker(Reranker):
    """使用 LLM 判断 query-passage 相关度（0-10），适合 Agent 环境."""

    def __init__(self, client: BaseLLMClient | None = None) -> None:
        """Initialize with an optional LLM client; create a default client if None."""
        self.client = client or BaseLLMClient()

    @staticmethod
    def _load_prompt_template(name: str) -> Template:
        """Load a prompt file and return it as a string.Template for safe substitution."""
        path = Config.PROMPT_DIR / f"{name}.txt"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        return Template(text)

    @staticmethod
    def _coerce_score(value: Any) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _parse_scores(self, raw: str, num_passages: int) -> list[float]:
        """Parse judge response into a list of scores."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Reranker returned non-JSON: {raw[:200]}")
            parsed = {}

        if isinstance(parsed, list) and len(parsed) == num_passages:
            return [self._coerce_score(s) for s in parsed]

        if isinstance(parsed, dict) and "scores" in parsed:
            scores = parsed["scores"]
            if isinstance(scores, list) and len(scores) == num_passages:
                return [self._coerce_score(s) for s in scores]

        logger.warning(f"Reranker returned unexpected format: {raw[:200]}")
        return [0.0] * num_passages

    def rank(self, query: str, passages: list[tuple[int, str]]) -> list[tuple[int, float]]:
        """Score and sort passages by relevance to the query."""
        if not passages:
            return []

        system = self._load_prompt_template("rerank_passage_system").template
        user_template = self._load_prompt_template("rerank_passage_user")

        numbered = "\n\n".join(f"[{i + 1}] {text}" for i, (_id, text) in enumerate(passages))
        user = user_template.safe_substitute(
            query=query,
            passages=numbered,
            num_passages=str(len(passages)),
        )

        try:
            raw = self.client.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
        except Exception:
            logger.exception("LLMReranker failed to score passages; returning neutral scores")
            return [(pid, 0.0) for pid, _ in passages]

        scores = self._parse_scores(raw, len(passages))
        ranked = sorted(
            zip((pid for pid, _ in passages), scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked
