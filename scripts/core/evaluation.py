"""Evaluation framework mechanism layer."""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Any

from .config import Config
from .llm_chat_client import BaseLLMClient
from .models import EvalDataset

if TYPE_CHECKING:
    from .knowledge_base_service import KnowledgeBaseService

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


def hit_rate_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Return 1.0 if any relevant id appears in the top-k retrieved ids."""
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0


def mean_reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """Return 1/rank of the first relevant id, or 0.0 if none."""
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[int], relevance: dict[int, float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    Uses the standard graded gain formula ``(2**rel - 1) / log2(i + 1)``.
    """
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        rel = relevance.get(rid, 0.0)
        dcg += (2**rel - 1) / math.log2(i + 1)
    ideal_rels = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal_rels, start=1))
    return dcg / idcg if idcg > 0 else 0.0


class EvalHarness:
    """End-to-end evaluation harness for retrieval metrics."""

    def __init__(
        self,
        service: KnowledgeBaseService,
        faithfulness: FaithfulnessMetric | None = None,
        context_precision: ContextPrecisionMetric | None = None,
        context_recall: ContextRecallMetric | None = None,
    ) -> None:
        """Initialize harness with a service and optional judge metrics.

        Judge metrics are instantiated lazily so that retrieval-only evaluations
        do not require an LLM API key.
        """
        self.service = service
        self._faithfulness = faithfulness
        self._context_precision = context_precision
        self._context_recall = context_recall

    @property
    def faithfulness(self) -> FaithfulnessMetric:
        """Lazy judge metric for answer faithfulness."""
        if self._faithfulness is None:
            self._faithfulness = FaithfulnessMetric()
        return self._faithfulness

    @property
    def context_precision(self) -> ContextPrecisionMetric:
        """Lazy judge metric for retrieved-context precision."""
        if self._context_precision is None:
            self._context_precision = ContextPrecisionMetric()
        return self._context_precision

    @property
    def context_recall(self) -> ContextRecallMetric:
        """Lazy judge metric for ground-truth recall against contexts."""
        if self._context_recall is None:
            self._context_recall = ContextRecallMetric()
        return self._context_recall

    def run_retrieval_eval(
        self,
        dataset: EvalDataset,
        retriever: str = "semantic",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run retrieval-only evaluation for each question in the dataset."""
        results: list[dict[str, Any]] = []
        for question in dataset.questions:
            if retriever == "semantic":
                hits = self.service.search_semantic(question.question, top_k=top_k)
            else:
                raise ValueError(f"Unsupported retriever: {retriever}")

            retrieved_ids = [hit["chunk"].id for hit in hits]
            relevant_ids = set(question.ground_truth_context_ids)
            relevance = {cid: 1.0 for cid in relevant_ids}

            results.append(
                {
                    "question": question.question,
                    "retrieved_ids": retrieved_ids,
                    f"hit_rate@{top_k}": hit_rate_at_k(retrieved_ids, relevant_ids, top_k),
                    "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
                    f"ndcg@{top_k}": ndcg_at_k(retrieved_ids, relevance, top_k),
                }
            )

        def _avg(key: str) -> float:
            values = [r[key] for r in results]
            return sum(values) / len(values) if values else 0.0

        return {
            "dataset_name": dataset.name,
            "retriever": retriever,
            "top_k": top_k,
            "num_questions": len(results),
            f"avg_hit_rate@{top_k}": _avg(f"hit_rate@{top_k}"),
            "avg_mrr": _avg("mrr"),
            f"avg_ndcg@{top_k}": _avg(f"ndcg@{top_k}"),
            "per_question": results,
        }

    def run_rag_eval(
        self,
        dataset: EvalDataset,
        retriever: str = "semantic",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run full RAG evaluation including generation metrics.

        For now, this is a placeholder that calls run_retrieval_eval and returns
        retrieval metrics. Generation metrics will be added when answer generation
        is integrated.
        """
        return self.run_retrieval_eval(dataset, retriever=retriever, top_k=top_k)
