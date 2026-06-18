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


class FaithfulnessMetric:
    """Evaluate whether claims in an answer are supported by retrieved contexts."""

    def __init__(self, client: BaseLLMClient | None = None) -> None:
        """Initialize the metric with an optional LLM client."""
        self.client = client or BaseLLMClient()

    def score(self, question: str, answer: str, contexts: list[str]) -> dict[str, Any]:
        """Compute the faithfulness score for the given answer and contexts."""
        system = _load_prompt("eval_faithfulness_system")
        user = _load_prompt("eval_faithfulness_user").format(
            question=question,
            answer=answer,
            contexts="\n\n---\n\n".join(contexts),
        )
        raw = self.client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
        )
        parsed = _parse_json_response(raw)
        supported = parsed.get("supported", [])
        unsupported = parsed.get("unsupported", [])
        not_found = parsed.get("not_found", [])
        total = len(supported) + len(unsupported) + len(not_found)
        score = len(supported) / total if total > 0 else 0.0
        return {
            "score": round(score, 4),
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "not_found_claims": not_found,
        }


class ContextPrecisionMetric:
    """Evaluate whether retrieved contexts are relevant to the question."""

    def __init__(self, client: BaseLLMClient | None = None) -> None:
        """Initialize the metric with an optional LLM client."""
        self.client = client or BaseLLMClient()

    def score(self, question: str, contexts: list[str]) -> dict[str, Any]:
        """Compute the precision score for the retrieved contexts."""
        system = _load_prompt("eval_context_precision_system")
        user = _load_prompt("eval_context_precision_user").format(
            question=question,
            contexts="\n\n---\n\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)),
        )
        raw = self.client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
        )
        parsed = _parse_json_response(raw)
        relevance = parsed.get("relevance", [])
        if not isinstance(relevance, list):
            relevance = []
        total = len(relevance)
        relevant_count = sum(1 for r in relevance if r)
        score = relevant_count / total if total > 0 else 0.0
        return {
            "score": round(score, 4),
            "relevance": relevance,
        }


class ContextRecallMetric:
    """Evaluate whether ground-truth claims are present in retrieved contexts."""

    def __init__(self, client: BaseLLMClient | None = None) -> None:
        """Initialize the metric with an optional LLM client."""
        self.client = client or BaseLLMClient()

    def score(self, ground_truth_answer: str, contexts: list[str]) -> dict[str, Any]:
        """Compute the recall score for the ground-truth answer against contexts."""
        system = _load_prompt("eval_context_recall_system")
        user = _load_prompt("eval_context_recall_user").format(
            ground_truth_answer=ground_truth_answer,
            contexts="\n\n---\n\n".join(contexts),
        )
        raw = self.client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
        )
        parsed = _parse_json_response(raw)
        attributable = parsed.get("attributable", [])
        not_attributable = parsed.get("not_attributable", [])
        total = len(attributable) + len(not_attributable)
        score = len(attributable) / total if total > 0 else 0.0
        return {
            "score": round(score, 4),
            "attributable_claims": attributable,
            "not_attributable_claims": not_attributable,
        }
