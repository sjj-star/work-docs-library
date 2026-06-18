"""Evaluation framework mechanism layer."""

import json
import logging
from typing import Any

from .config import Config
from .llm_chat_client import BaseLLMClient

logger = logging.getLogger(__name__)


def _load_prompt(name: str) -> str:
    path = Config.PROMPT_DIR / f"{name}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _parse_json_response(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Judge returned non-JSON: {raw[:200]}")
        return {}


class BaseJudgeMetric:
    """Base class for LLM-as-judge metrics."""

    def __init__(self, client: BaseLLMClient | None = None) -> None:
        """Initialize the metric with an optional LLM client."""
        self.client = client or BaseLLMClient()

    def _format_prompt(self, name: str, **kwargs: Any) -> str:
        return _load_prompt(name).format(**kwargs)

    def _judge(
        self,
        system_prompt_name: str,
        user_prompt_name: str,
        **user_kwargs: Any,
    ) -> dict[str, Any]:
        system = _load_prompt(system_prompt_name)
        user = self._format_prompt(user_prompt_name, **user_kwargs)
        raw = self.client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
        )
        return _parse_json_response(raw)

    @staticmethod
    def _ensure_list(value: Any, key: str) -> list[Any]:
        if not isinstance(value, list):
            logger.warning(f"Judge response key '{key}' is not a list: {value!r}")
            return []
        return value


class FaithfulnessMetric(BaseJudgeMetric):
    """Evaluate whether claims in an answer are supported by retrieved contexts."""

    def score(self, question: str, answer: str, contexts: list[str]) -> dict[str, Any]:
        """Compute the faithfulness score for the given answer and contexts."""
        parsed = self._judge(
            "eval_faithfulness_system",
            "eval_faithfulness_user",
            question=question,
            answer=answer,
            contexts="\n\n---\n\n".join(contexts),
        )
        supported = self._ensure_list(parsed.get("supported", []), "supported")
        unsupported = self._ensure_list(parsed.get("unsupported", []), "unsupported")
        not_found = self._ensure_list(parsed.get("not_found", []), "not_found")
        total = len(supported) + len(unsupported) + len(not_found)
        score = len(supported) / total if total > 0 else 0.0
        return {
            "score": round(score, 4),
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "not_found_claims": not_found,
        }


class ContextPrecisionMetric(BaseJudgeMetric):
    """Evaluate whether retrieved contexts are relevant to the question."""

    def score(self, question: str, contexts: list[str]) -> dict[str, Any]:
        """Compute the precision score for the retrieved contexts."""
        parsed = self._judge(
            "eval_context_precision_system",
            "eval_context_precision_user",
            question=question,
            contexts="\n\n---\n\n".join(f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)),
        )
        relevance = self._ensure_list(parsed.get("relevance"), "relevance")
        if relevance and len(relevance) != len(contexts):
            logger.warning(
                f"Relevance length mismatch: expected {len(contexts)}, got {len(relevance)}. "
                "Falling back to False padding to match context count."
            )
            relevance = [False] * len(contexts)
        total = len(relevance)
        relevant_count = sum(1 for r in relevance if r)
        score = relevant_count / total if total > 0 else 0.0
        return {
            "score": round(score, 4),
            "relevance": relevance,
        }


class ContextRecallMetric(BaseJudgeMetric):
    """Evaluate whether ground-truth claims are present in retrieved contexts."""

    def score(self, ground_truth_answer: str, contexts: list[str]) -> dict[str, Any]:
        """Compute the recall score for the ground-truth answer against contexts."""
        parsed = self._judge(
            "eval_context_recall_system",
            "eval_context_recall_user",
            ground_truth_answer=ground_truth_answer,
            contexts="\n\n---\n\n".join(contexts),
        )
        attributable = self._ensure_list(parsed.get("attributable", []), "attributable")
        not_attributable = self._ensure_list(parsed.get("not_attributable", []), "not_attributable")
        total = len(attributable) + len(not_attributable)
        score = len(attributable) / total if total > 0 else 0.0
        return {
            "score": round(score, 4),
            "attributable_claims": attributable,
            "not_attributable_claims": not_attributable,
        }
