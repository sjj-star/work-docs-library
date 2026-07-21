"""Evaluation persistence tests."""

import pytest
from core.db import KnowledgeDB


@pytest.fixture(autouse=True)
def _dummy_llm_api_key(monkeypatch):
    """Provide a dummy LLM API key so metric client instantiation does not fail in tests."""
    monkeypatch.setattr("core.config.Config.LLM_API_KEY", "dummy-test-key")


def test_load_missing_eval_dataset_raises(tmp_path):
    db = KnowledgeDB(db_path=tmp_path / "eval.db")
    with pytest.raises(ValueError):
        db.load_eval_dataset("missing")


def test_faithfulness_metric(monkeypatch):
    from core.evaluation import FaithfulnessMetric

    def fake_chat(self, messages, **kwargs):
        return (
            '{"supported": ["The SPI reset starts with CS low."], '
            '"unsupported": [], "not_found": []}'
        )

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    metric = FaithfulnessMetric()
    result = metric.score(
        question="What is the SPI reset sequence?",
        answer="The SPI reset starts with CS low.",
        contexts=["To reset SPI, pull CS low first."],
    )
    assert result["score"] == 1.0
    assert result["supported_claims"] == ["The SPI reset starts with CS low."]
    assert result["unsupported_claims"] == []
    assert result["not_found_claims"] == []


def test_context_precision_metric(monkeypatch):
    from core.evaluation import ContextPrecisionMetric

    def fake_chat(self, messages, **kwargs):
        return '{"relevance": [true, false]}'

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    metric = ContextPrecisionMetric()
    result = metric.score(
        question="SPI reset sequence",
        contexts=["Pull CS low first.", "This is unrelated."],
    )
    assert result["score"] == 0.5
    assert result["relevance"] == [True, False]


def test_context_recall_metric(monkeypatch):
    from core.evaluation import ContextRecallMetric

    def fake_chat(self, messages, **kwargs):
        return '{"attributable": ["CS low first"], "not_attributable": ["Clock idle"]}'

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    metric = ContextRecallMetric()
    result = metric.score(
        ground_truth_answer="CS low first. Clock idle.",
        contexts=["To reset SPI, pull CS low first."],
    )
    assert result["score"] == 0.5
    assert result["attributable_claims"] == ["CS low first"]
    assert result["not_attributable_claims"] == ["Clock idle"]


def test_faithfulness_metric_exception_returns_zero(monkeypatch):
    from core.evaluation import FaithfulnessMetric

    monkeypatch.setattr(
        "core.llm_chat_client.BaseLLMClient.chat",
        lambda self, messages, **kwargs: (_ for _ in ()).throw(RuntimeError("LLM failed")),
    )
    metric = FaithfulnessMetric()
    result = metric.score(
        question="What is the SPI reset sequence?",
        answer="The SPI reset starts with CS low.",
        contexts=["To reset SPI, pull CS low first."],
    )
    assert result["score"] == 0.0
    assert result["supported_claims"] == []
    assert result["unsupported_claims"] == []
    assert result["not_found_claims"] == []


def test_context_precision_metric_exception_returns_zero(monkeypatch):
    from core.evaluation import ContextPrecisionMetric

    monkeypatch.setattr(
        "core.llm_chat_client.BaseLLMClient.chat",
        lambda self, messages, **kwargs: (_ for _ in ()).throw(RuntimeError("LLM failed")),
    )
    metric = ContextPrecisionMetric()
    result = metric.score(
        question="SPI reset sequence",
        contexts=["Pull CS low first.", "This is unrelated."],
    )
    assert result["score"] == 0.0
    assert result["relevance"] == []


def test_context_recall_metric_exception_returns_zero(monkeypatch):
    from core.evaluation import ContextRecallMetric

    monkeypatch.setattr(
        "core.llm_chat_client.BaseLLMClient.chat",
        lambda self, messages, **kwargs: (_ for _ in ()).throw(RuntimeError("LLM failed")),
    )
    metric = ContextRecallMetric()
    result = metric.score(
        ground_truth_answer="CS low first. Clock idle.",
        contexts=["To reset SPI, pull CS low first."],
    )
    assert result["score"] == 0.0
    assert result["attributable_claims"] == []
    assert result["not_attributable_claims"] == []


def test_answer_relevancy_metric_score(monkeypatch):
    from core.evaluation import AnswerRelevancyMetric

    def fake_chat(self, messages, **kwargs):
        return '{"score": 0.85, "reason": "Directly answers the question."}'

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    metric = AnswerRelevancyMetric()
    score = metric.score(
        question="What is the SPI reset sequence?",
        answer="Pull CS low first, then configure the clock.",
    )
    assert score == 0.85


def test_answer_relevancy_metric_malformed_response(monkeypatch):
    from core.evaluation import AnswerRelevancyMetric

    monkeypatch.setattr(
        "core.llm_chat_client.BaseLLMClient.chat",
        lambda self, messages, **kwargs: "not json",
    )
    metric = AnswerRelevancyMetric()
    score = metric.score(
        question="What is the SPI reset sequence?",
        answer="Pull CS low first.",
    )
    assert score == 0.0


def test_answer_relevancy_metric_exception(monkeypatch):
    from core.evaluation import AnswerRelevancyMetric

    monkeypatch.setattr(
        "core.llm_chat_client.BaseLLMClient.chat",
        lambda self, messages, **kwargs: (_ for _ in ()).throw(RuntimeError("LLM failed")),
    )
    metric = AnswerRelevancyMetric()
    score = metric.score(
        question="What is the SPI reset sequence?",
        answer="Pull CS low first.",
    )
    assert score == 0.0


def test_faithfulness_metric_handles_non_json(monkeypatch):
    from core.evaluation import FaithfulnessMetric

    monkeypatch.setattr(
        "core.llm_chat_client.BaseLLMClient.chat",
        lambda self, messages, **kwargs: "not json",
    )
    metric = FaithfulnessMetric()
    result = metric.score(
        question="?",
        answer="answer",
        contexts=["context"],
    )
    assert result["score"] == 0.0
    assert result["supported_claims"] == []


def test_faithfulness_metric_with_custom_client():
    from core.evaluation import FaithfulnessMetric
    from core.llm_chat_client import BaseLLMClient

    class FakeClient(BaseLLMClient):
        def chat(self, messages, **kwargs):
            return (
                '{"supported": ["The SPI reset starts with CS low."], '
                '"unsupported": [], "not_found": []}'
            )

    metric = FaithfulnessMetric(client=FakeClient())
    result = metric.score(
        question="What is the SPI reset sequence?",
        answer="The SPI reset starts with CS low.",
        contexts=["To reset SPI, pull CS low first."],
    )
    assert result["score"] == 1.0
    assert result["supported_claims"] == ["The SPI reset starts with CS low."]
    assert result["unsupported_claims"] == []
    assert result["not_found_claims"] == []


def test_context_precision_metric_length_mismatch(monkeypatch):
    from core.evaluation import ContextPrecisionMetric

    monkeypatch.setattr(
        "core.llm_chat_client.BaseLLMClient.chat",
        lambda self, messages, **kwargs: '{"relevance": [true]}',
    )
    metric = ContextPrecisionMetric()
    result = metric.score(
        question="SPI reset sequence",
        contexts=["Pull CS low first.", "This is unrelated."],
    )
    assert result["score"] == 0.0
    assert result["relevance"] == [False, False]


def test_context_precision_metric_non_list_relevance(monkeypatch):
    from core.evaluation import ContextPrecisionMetric

    monkeypatch.setattr(
        "core.llm_chat_client.BaseLLMClient.chat",
        lambda self, messages, **kwargs: '{"relevance": "yes"}',
    )
    metric = ContextPrecisionMetric()
    result = metric.score(
        question="SPI reset sequence",
        contexts=["Pull CS low first.", "This is unrelated."],
    )
    assert result["score"] == 0.0
    assert result["relevance"] == []


def test_metric_with_empty_contexts(monkeypatch):
    from core.evaluation import FaithfulnessMetric

    monkeypatch.setattr(
        "core.llm_chat_client.BaseLLMClient.chat",
        lambda self, messages, **kwargs: "not json",
    )
    metric = FaithfulnessMetric()
    result = metric.score(
        question="What is the SPI reset sequence?",
        answer="The SPI reset starts with CS low.",
        contexts=[],
    )
    assert result["score"] == 0.0
    assert result["supported_claims"] == []
    assert result["unsupported_claims"] == []
    assert result["not_found_claims"] == []


def test_hit_rate_at_k():
    from core.evaluation import hit_rate_at_k

    assert hit_rate_at_k([1, 2, 3], {3}, 5) == 1.0
    assert hit_rate_at_k([1, 2, 3], {4}, 5) == 0.0
    assert hit_rate_at_k([1, 2, 3], {3}, 2) == 0.0


def test_mean_reciprocal_rank():
    from core.evaluation import mean_reciprocal_rank

    assert mean_reciprocal_rank([1, 2, 3], {2}) == 0.5
    assert mean_reciprocal_rank([1, 2, 3], {4}) == 0.0
    assert mean_reciprocal_rank([1, 2, 3], {1}) == 1.0


def test_ndcg_at_k():
    from core.evaluation import ndcg_at_k

    relevance = {1: 1.0, 2: 1.0}
    assert ndcg_at_k([1, 2, 3], relevance, 2) == 1.0
    assert ndcg_at_k([3, 2, 1], relevance, 2) < 1.0


def test_ndcg_at_k_graded():
    from core.evaluation import ndcg_at_k

    relevance = {1: 3.0, 2: 2.0, 3: 1.0}
    # Ideal ordering at k=3 retrieves [1, 2, 3]
    assert ndcg_at_k([1, 2, 3], relevance, 3) == 1.0
    # Reversed ordering is worse than ideal
    assert ndcg_at_k([3, 2, 1], relevance, 3) < 1.0


def test_run_retrieval_eval_empty_dataset():
    from core.evaluation import EvalHarness
    from core.models import EvalDataset

    class FakeService:
        def search_semantic(self, query, top_k=5):
            return []

    harness = EvalHarness(FakeService())  # type: ignore[arg-type]
    ds = EvalDataset(name="empty", questions=[])
    result = harness.run_retrieval_eval(ds, retriever="semantic", top_k=5)
    assert result["num_questions"] == 0
    assert result["avg_hit_rate@5"] == 0.0
    assert result["avg_mrr"] == 0.0
    assert result["avg_ndcg@5"] == 0.0
    assert result["per_question"] == []


def test_run_retrieval_eval_unsupported_retriever():
    from core.evaluation import EvalHarness
    from core.models import EvalDataset, EvalQuestion

    class FakeService:
        pass

    harness = EvalHarness(FakeService())  # type: ignore[arg-type]
    ds = EvalDataset(
        name="bad", questions=[EvalQuestion(question="q", ground_truth_context_ids=[1])]
    )
    with pytest.raises(ValueError, match="Unsupported retriever"):
        harness.run_retrieval_eval(ds, retriever="unknown", top_k=5)


def test_run_retrieval_eval_hybrid():
    from core.evaluation import EvalHarness
    from core.models import EvalDataset, EvalQuestion

    class FakeChunk:
        def __init__(self, chunk_id):
            self.id = chunk_id

    class FakeService:
        def search_hybrid(self, query, top_k=5):
            return [{"chunk": FakeChunk(2)}, {"chunk": FakeChunk(1)}]

    harness = EvalHarness(FakeService())  # type: ignore[arg-type]
    ds = EvalDataset(
        name="hybrid", questions=[EvalQuestion(question="q", ground_truth_context_ids=[1, 2])]
    )
    result = harness.run_retrieval_eval(ds, retriever="hybrid", top_k=5)
    assert result["retriever"] == "hybrid"
    assert result["avg_hit_rate@5"] == 1.0
    assert result["avg_mrr"] > 0.0


def test_run_retrieval_eval_reranked():
    from core.evaluation import EvalHarness
    from core.models import EvalDataset, EvalQuestion

    class FakeChunk:
        def __init__(self, chunk_id):
            self.id = chunk_id

    class FakeService:
        def search_reranked(self, query, top_k=5):
            return [{"chunk": FakeChunk(1)}]

    harness = EvalHarness(FakeService())  # type: ignore[arg-type]
    ds = EvalDataset(
        name="reranked", questions=[EvalQuestion(question="q", ground_truth_context_ids=[1])]
    )
    result = harness.run_retrieval_eval(ds, retriever="reranked", top_k=5)
    assert result["retriever"] == "reranked"
    assert result["avg_hit_rate@5"] == 1.0
    assert result["avg_mrr"] == 1.0


def test_eval_harness_lazy_metrics():
    from core.evaluation import EvalHarness

    class FakeService:
        pass

    harness = EvalHarness(FakeService())  # type: ignore[arg-type]
    assert harness._faithfulness is None
    assert harness._context_precision is None
    assert harness._context_recall is None
    assert harness._answer_relevancy is None
    assert harness._llm_client is None


def test_faithfulness_metric_ignores_braces_and_dollar_literals(monkeypatch):
    """Regression for P0-1: raw user content with { } or $ must not break prompt formatting."""
    from core.evaluation import FaithfulnessMetric

    monkeypatch.setattr(
        "core.llm_chat_client.BaseLLMClient.chat",
        lambda self, messages, **kwargs: "not json",
    )
    metric = FaithfulnessMetric()
    result = metric.score(
        question="Question with {not_a_key}?",
        answer="Answer with $not_a_placeholder and {curly} braces.",
        contexts=["Context with {literal_braces} and $dollar_placeholder."],
    )
    assert result["score"] == 0.0
    assert result["supported_claims"] == []
    assert result["unsupported_claims"] == []
    assert result["not_found_claims"] == []


def test_load_missing_prompt_raises_file_not_found():
    """Missing prompt files must raise FileNotFoundError instead of returning empty string."""
    from core.config import Config

    with pytest.raises(FileNotFoundError):
        Config.load_prompt("definitely_missing_prompt")
