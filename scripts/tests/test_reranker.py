import pytest
from core.reranker import LLMReranker


@pytest.fixture(autouse=True)
def _dummy_llm_api_key(monkeypatch):
    """Provide a dummy LLM API key so LLMReranker instantiation does not fail in tests."""
    monkeypatch.setattr("core.config.Config.LLM_API_KEY", "dummy-test-key")


def test_llm_reranker_ranking(monkeypatch):
    def fake_chat(self, messages, **kwargs):
        return "[10, 5, 0]"

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    reranker = LLMReranker()
    passages = [(1, "SPI reset sequence"), (2, "GPIO config"), (3, "Timers")]
    result = reranker.rank("SPI reset", passages)
    assert len(result) == 3
    assert result[0] == (1, 10.0)
    assert result[1] == (2, 5.0)
    assert result[2] == (3, 0.0)


def test_llm_reranker_dict_scores(monkeypatch):
    def fake_chat(self, messages, **kwargs):
        return '{"scores": [3, 8]}'

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    reranker = LLMReranker()
    passages = [(1, "A"), (2, "B")]
    result = reranker.rank("query", passages)
    assert result[0] == (2, 8.0)
    assert result[1] == (1, 3.0)


def test_llm_reranker_empty():
    reranker = LLMReranker()
    assert reranker.rank("query", []) == []


def test_llm_reranker_malformed_response(monkeypatch):
    def fake_chat(self, messages, **kwargs):
        return "not json"

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    reranker = LLMReranker()
    passages = [(1, "A"), (2, "B")]
    result = reranker.rank("query", passages)
    assert len(result) == 2
    assert all(score == 0.0 for _, score in result)


def test_knowledge_base_service_search_reranked(tmp_path, monkeypatch):
    from core.db import KnowledgeDB
    from core.knowledge_base_service import KnowledgeBaseService
    from core.models import Document

    db = KnowledgeDB(db_path=tmp_path / "test.db")
    db.upsert_document(
        Document(
            doc_id="doc-a",
            title="Test",
            source_path="/tmp/test.pdf",
            file_type="pdf",
            total_pages=1,
            chapters=[],
            extracted_at="2026-01-01",
            file_hash="abc",
            status="done",
        )
    )
    db.insert_block("doc-a", "b1", "SPI reset sequence", 0, {})
    db.insert_block("doc-a", "b2", "GPIO config", 1, {})

    class FakeVectorIndex:
        def search(self, query_vector, top_k):
            return [(1, 0.9), (2, 0.8)]

    class FakeSparseIndex:
        def search(self, query, top_k):
            return [(2, 1.0), (1, 0.9)]

    class FakeEmbedder:
        def embed(self, texts):
            return [[0.0] * 10]

    svc = KnowledgeBaseService(db=db, vec=None, graph_store=None)
    svc.vec = FakeVectorIndex()  # type: ignore[assignment]
    monkeypatch.setattr(svc, "_get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(svc, "_get_sparse_index", lambda: FakeSparseIndex())

    def fake_rank(self, query, passages):
        return [(pid, float(10 - i)) for i, (pid, _) in enumerate(passages)]

    monkeypatch.setattr("core.reranker.LLMReranker.rank", fake_rank)

    results = svc.search_reranked("SPI", top_k=1)
    assert len(results) == 1
    assert all(r["score"] > 0 for r in results)


def test_llm_reranker_passage_with_braces(monkeypatch):
    def fake_chat(self, messages, **kwargs):
        return "[10, 0]"

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    reranker = LLMReranker()
    passages = [(1, "Register {ADDR} is used."), (2, "Other")]
    result = reranker.rank("register", passages)
    assert len(result) == 2
    assert result[0][0] == 1


def test_llm_reranker_placeholder_collision(monkeypatch):
    captured: list[str] = []

    def fake_chat(self, messages, **kwargs):
        captured.append(messages[1]["content"])
        return "[10]"

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    reranker = LLMReranker()
    passages = [(1, "Text with $passages substring")]
    result = reranker.rank("query containing $num_passages", passages)
    assert len(result) == 1
    assert result[0][0] == 1
    # The literal $passages from the passage should still be present in the prompt,
    # not substituted by Template.safe_substitute.
    assert "$passages" in captured[0]
    assert "$num_passages" in captured[0]


def test_llm_reranker_non_numeric_scores(monkeypatch):
    def fake_chat(self, messages, **kwargs):
        return '["N/A", 5]'

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    reranker = LLMReranker()
    passages = [(1, "A"), (2, "B")]
    result = reranker.rank("query", passages)
    scores_by_id = {pid: score for pid, score in result}
    assert scores_by_id[1] == 0.0
    assert scores_by_id[2] == 5.0


def test_search_reranked_falls_back_on_llm_error(tmp_path, monkeypatch):
    from core.db import KnowledgeDB
    from core.knowledge_base_service import KnowledgeBaseService
    from core.models import Document

    db = KnowledgeDB(db_path=tmp_path / "test.db")
    db.upsert_document(
        Document(
            doc_id="doc-a",
            title="Test",
            source_path="/tmp/test.pdf",
            file_type="pdf",
            total_pages=1,
            chapters=[],
            extracted_at="2026-01-01",
            file_hash="abc",
            status="done",
        )
    )
    db.insert_block("doc-a", "b1", "SPI reset sequence", 0, {})
    db.insert_block("doc-a", "b2", "GPIO config", 1, {})

    class FakeVectorIndex:
        def search(self, query_vector, top_k):
            return [(1, 0.9), (2, 0.8)]

    class FakeSparseIndex:
        def search(self, query, top_k):
            return [(2, 1.0), (1, 0.9)]

    class FakeEmbedder:
        def embed(self, texts):
            return [[0.0] * 10]

    svc = KnowledgeBaseService(db=db, vec=None, graph_store=None)
    svc.vec = FakeVectorIndex()  # type: ignore[assignment]
    monkeypatch.setattr(svc, "_get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(svc, "_get_sparse_index", lambda: FakeSparseIndex())

    def fake_rank_raise(self, query, passages):
        raise RuntimeError("LLM timeout")

    monkeypatch.setattr("core.reranker.LLMReranker.rank", fake_rank_raise)

    results = svc.search_reranked("SPI", top_k=2)
    assert len(results) == 2
    assert all(r["score"] > 0 for r in results)
