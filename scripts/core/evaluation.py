"""Evaluation framework mechanism layer."""

from __future__ import annotations

import logging
import math
from string import Template
from typing import TYPE_CHECKING, Any

from .config import Config
from .llm_chat_client import BaseLLMClient
from .models import EvalDataset, EvalQuestion

if TYPE_CHECKING:
    from .knowledge_base_service import KnowledgeBaseService

logger = logging.getLogger(__name__)

#: Retriever strategies supported by :class:`EvalHarness`.
ALLOWED_RETRIEVERS: set[str] = {"semantic", "hybrid", "reranked"}


class BaseJudgeMetric:
    """Base class for LLM-as-judge metrics."""

    def __init__(self, client: BaseLLMClient | None = None) -> None:
        """Initialize the metric with an optional LLM client."""
        self.client = client or BaseLLMClient()

    @staticmethod
    def _format_prompt(name: str, **kwargs: Any) -> str:
        template = Config.load_prompt(name)
        return Template(template).safe_substitute(**kwargs)

    def _judge(
        self,
        system_prompt_name: str,
        user_prompt_name: str,
        **user_kwargs: Any,
    ) -> dict[str, Any]:
        system = Config.load_prompt(system_prompt_name)
        user = self._format_prompt(user_prompt_name, **user_kwargs)
        try:
            raw = self.client.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Judge LLM call failed for {user_prompt_name}: {exc}")
            raw = ""
        return Config.parse_llm_json(raw) or {}

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


class AnswerRelevancyMetric(BaseJudgeMetric):
    """Evaluate how relevant an answer is to the asked question."""

    def score(self, question: str, answer: str) -> float:
        """Compute the relevancy score for the given answer against the question.

        Returns a float in the range [0, 1]. If the judge response cannot be
        parsed or the score is missing/invalid, returns 0.0.
        """
        parsed = self._judge(
            "eval_answer_relevancy_system",
            "eval_answer_relevancy_user",
            question=question,
            answer=answer,
        )
        raw_score = parsed.get("score")
        try:
            score = float(raw_score)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            logger.warning(f"Judge returned invalid score: {raw_score!r}")
            return 0.0
        if not 0.0 <= score <= 1.0:
            logger.warning(f"Judge returned out-of-range score: {score}")
            return 0.0
        return round(score, 4)


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
    """End-to-end evaluation harness for retrieval and RAG metrics."""

    def __init__(
        self,
        service: KnowledgeBaseService,
        faithfulness: FaithfulnessMetric | None = None,
        context_precision: ContextPrecisionMetric | None = None,
        context_recall: ContextRecallMetric | None = None,
        answer_relevancy: AnswerRelevancyMetric | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        """Initialize harness with a service and optional judge metrics.

        Judge metrics and the generation client are instantiated lazily so that
        retrieval-only evaluations do not require an LLM API key.
        """
        self.service = service
        self._faithfulness = faithfulness
        self._context_precision = context_precision
        self._context_recall = context_recall
        self._answer_relevancy = answer_relevancy
        self._llm_client = llm_client

    @property
    def llm_client(self) -> BaseLLMClient:
        """Lazy LLM client shared by answer generation and judge metrics."""
        if self._llm_client is None:
            self._llm_client = BaseLLMClient()
        return self._llm_client

    @property
    def faithfulness(self) -> FaithfulnessMetric:
        """Lazy judge metric for answer faithfulness."""
        if self._faithfulness is None:
            self._faithfulness = FaithfulnessMetric(client=self.llm_client)
        return self._faithfulness

    @property
    def context_precision(self) -> ContextPrecisionMetric:
        """Lazy judge metric for retrieved-context precision."""
        if self._context_precision is None:
            self._context_precision = ContextPrecisionMetric(client=self.llm_client)
        return self._context_precision

    @property
    def context_recall(self) -> ContextRecallMetric:
        """Lazy judge metric for ground-truth recall against contexts."""
        if self._context_recall is None:
            self._context_recall = ContextRecallMetric(client=self.llm_client)
        return self._context_recall

    @property
    def answer_relevancy(self) -> AnswerRelevancyMetric:
        """Lazy judge metric for answer relevancy to the question."""
        if self._answer_relevancy is None:
            self._answer_relevancy = AnswerRelevancyMetric(client=self.llm_client)
        return self._answer_relevancy

    @staticmethod
    def _extract_hit_text(hit: dict[str, Any]) -> str:
        """Extract searchable text from a service search hit.

        Hits produced by :class:`KnowledgeBaseService` contain a ``chunk`` object
        with a ``content`` attribute. Tests may use stripped-down fake chunks, so
        fall back gracefully to avoid crashes.
        """
        chunk = hit.get("chunk")
        if chunk is not None:
            content = getattr(chunk, "content", None)
            if content:
                return str(content)
        return str(hit.get("text", ""))

    def _generate_answer(self, question: EvalQuestion, contexts: list[str]) -> str:
        """Generate an answer for the question using only the provided contexts."""
        system = Config.load_prompt("eval_generate_answer_system")
        user = Template(Config.load_prompt("eval_generate_answer_user")).safe_substitute(
            question=question.question,
            contexts="\n\n---\n\n".join(contexts),
        )
        try:
            return self.llm_client.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Answer generation failed: {exc}")
            return ""

    def run_retrieval_eval(
        self,
        dataset: EvalDataset,
        retriever: str = "semantic",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run retrieval-only evaluation for each question in the dataset."""
        if retriever not in ALLOWED_RETRIEVERS:
            raise ValueError(f"Unsupported retriever: {retriever}")

        search_method = getattr(self.service, f"search_{retriever}", None)
        if search_method is None:
            raise ValueError(f"Unsupported retriever: {retriever}")

        results: list[dict[str, Any]] = []
        for question in dataset.questions:
            hits = search_method(question.question, top_k=top_k)

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
