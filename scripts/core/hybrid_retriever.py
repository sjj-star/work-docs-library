"""Hybrid retriever combining dense vector search and BM25 sparse search via RRF."""

import logging

from .config import Config
from .sparse_index import BM25SparseIndex
from .vector_index import VectorIndex

logger = logging.getLogger(__name__)


class RRFFusionRetriever:
    """Reciprocal Rank Fusion 混合检索器：合并 BM25 与稠密向量排序."""

    def __init__(
        self,
        vector_index: VectorIndex,
        sparse_index: BM25SparseIndex,
        k: float | None = None,
    ) -> None:
        """Initialize the RRF fusion retriever.

        Args:
            vector_index: FAISS vector index for dense retrieval.
            sparse_index: BM25 sparse index for lexical retrieval.
            k: RRF constant controlling the score decay over rank.
                Defaults to Config.PLUGIN_HYBRID_RRF_K.
        """
        self.vector_index = vector_index
        self.sparse_index = sparse_index
        self.k = k if k is not None else float(Config.PLUGIN_HYBRID_RRF_K)

    def search(
        self,
        query_text: str,
        embedder,
        top_k: int = 10,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Run dense + sparse retrieval and fuse with RRF.

        Args:
            query_text: The raw query string.
            embedder: Object with an `embed(list[str]) -> list[list[float]]` method.
            top_k: Number of fused results to return.
            dense_top_k: Number of dense candidates.
                Defaults to Config.PLUGIN_SEARCH_TOP_K * 10.
            sparse_top_k: Number of sparse candidates. Defaults to Config.PLUGIN_BM25_TOP_K.

        Returns:
            List of (block_db_id, rrf_score) sorted by descending score.
        """
        if dense_top_k is None:
            dense_top_k = Config.PLUGIN_SEARCH_TOP_K * 10
        if sparse_top_k is None:
            sparse_top_k = Config.PLUGIN_BM25_TOP_K

        emb = embedder.embed([query_text])[0]
        dense_hits = self.vector_index.search(emb, top_k=dense_top_k)
        sparse_hits = self.sparse_index.search(query_text, top_k=sparse_top_k)

        scores: dict[int, float] = {}
        for rank, (db_id, _score) in enumerate(dense_hits, start=1):
            scores[db_id] = scores.get(db_id, 0.0) + 1.0 / (self.k + rank)
        for rank, (db_id, _score) in enumerate(sparse_hits, start=1):
            scores[db_id] = scores.get(db_id, 0.0) + 1.0 / (self.k + rank)

        sorted_hits = sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
        return sorted_hits[:top_k]
