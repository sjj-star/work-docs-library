from core.config import Config
from core.hybrid_retriever import RRFFusionRetriever


def test_rrf_fusion_ranking():
    """Test that RRF correctly fuses overlapping and non-overlapping results."""

    class FakeVectorIndex:
        def search(self, query_vector, top_k):
            return [(1, 0.9), (2, 0.8), (3, 0.7)]

    class FakeSparseIndex:
        def search(self, query, top_k):
            return [(2, 1.0), (4, 0.9), (5, 0.8)]

    class FakeEmbedder:
        def embed(self, texts):
            return [[0.0] * 10]

    retriever = RRFFusionRetriever(FakeVectorIndex(), FakeSparseIndex(), k=60.0)  # type: ignore[reportArgumentType]
    hits = retriever.search("test", FakeEmbedder(), top_k=5)

    ids = [h[0] for h in hits]
    assert 2 in ids  # appears in both lists, should rank high
    assert set(ids) == {1, 2, 3, 4, 5}
    # Block 2 should have the highest RRF score because it appears in both lists
    assert hits[0][0] == 2


def test_rrf_empty_results():
    class FakeVectorIndex:
        def search(self, query_vector, top_k):
            return []

    class FakeSparseIndex:
        def search(self, query, top_k):
            return []

    class FakeEmbedder:
        def embed(self, texts):
            return [[0.0] * 10]

    retriever = RRFFusionRetriever(FakeVectorIndex(), FakeSparseIndex())  # type: ignore[reportArgumentType]
    assert retriever.search("test", FakeEmbedder()) == []


def test_rrf_default_k_uses_config():
    """RRFFusionRetriever without an explicit k should adopt Config.PLUGIN_HYBRID_RRF_K."""

    class FakeVectorIndex:
        def search(self, query_vector, top_k):
            return []

    class FakeSparseIndex:
        def search(self, query, top_k):
            return []

    retriever = RRFFusionRetriever(FakeVectorIndex(), FakeSparseIndex())  # type: ignore[reportArgumentType]
    assert retriever.k == float(Config.PLUGIN_HYBRID_RRF_K)


def test_rrf_tie_breaking_prefers_higher_block_id():
    """When two blocks have identical RRF scores, the higher block_db_id wins."""

    class FakeVectorIndex:
        def search(self, query_vector, top_k):
            return [(1, 0.5)]

    class FakeSparseIndex:
        def search(self, query, top_k):
            return [(2, 0.5)]

    class FakeEmbedder:
        def embed(self, texts):
            return [[0.0] * 10]

    retriever = RRFFusionRetriever(FakeVectorIndex(), FakeSparseIndex(), k=60.0)  # type: ignore[reportArgumentType]
    hits = retriever.search("test", FakeEmbedder(), top_k=2)
    assert [h[0] for h in hits] == [2, 1]


def test_knowledge_base_service_search_hybrid(tmp_path, monkeypatch):
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
    db.insert_block("doc-a", "b3", "Timer interrupt", 2, {})

    class FakeVectorIndex:
        def search(self, query_vector, top_k):
            return [(1, 0.9), (2, 0.8)]

    class FakeSparseIndex:
        def search(self, query, top_k):
            return [(2, 1.0), (3, 0.9)]

    class FakeEmbedder:
        def embed(self, texts):
            return [[0.0] * 10]

    svc = KnowledgeBaseService(db=db, vec=FakeVectorIndex(), graph_store=None)  # type: ignore[reportArgumentType]
    monkeypatch.setattr(svc, "_get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(svc, "_get_sparse_index", lambda: FakeSparseIndex())

    results = svc.search_hybrid("SPI", top_k=5)
    assert len(results) == 3
    # Block 2 appears in both dense and sparse, so it should rank first
    assert results[0]["chunk"].chunk_id == "b2"
    # All scores should be positive
    assert all(r["score"] > 0 for r in results)
