"""vector_index 模块."""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)


class VectorIndex:
    """VectorIndex 类."""

    def __init__(
        self, dim: int, index_path: Path | None = None, id_map_path: Path | None = None
    ) -> None:
        # dim 为必需参数，由调用方传入（通常为 EmbeddingClient 探测到的实际维度）
        """初始化 VectorIndex."""
        self.dim = dim
        self.index_path = index_path or Config.FAISS_INDEX_PATH
        self.id_map_path = id_map_path or Config.ID_MAP_PATH
        self._index: faiss.IndexFlatIP | None = None
        self._id_map: list[int] = []  # faiss internal id -> chunk db id
        self._load()

    def _load(self) -> None:
        if Path(self.index_path).exists():
            index = faiss.read_index(str(self.index_path))
            self._index = index
            actual_dim = index.d
            if actual_dim != self.dim:
                # 严格错误：维度不匹配时直接失败，不允许自动调整
                raise RuntimeError(
                    f"VectorIndex dimension mismatch: existing index has {actual_dim} "
                    f"dimensions, but the embedding model produces {self.dim} dimensions. "
                    f"FAISS index dimensions cannot be changed after creation.\n"
                    f"You must either:\n"
                    f"  1. Update WORKDOCS_EMBEDDING_DIMENSION to {actual_dim} "
                    f"(if your model supports it)\n"
                    f"  2. Delete the existing index at {self.index_path} "
                    f"and re-ingest all documents\n"
                    f"  3. Switch back to a model that produces {actual_dim} dimensions"
                )
        else:
            self._index = faiss.IndexFlatIP(self.dim)
        if Path(self.id_map_path).exists():
            with open(self.id_map_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                # Backward-compatible: old format was {chunk_db_id: faiss_id}
                max_fid = max(loaded.values()) if loaded else -1
                self._id_map = [0] * (max_fid + 1)
                for db_id, fid in loaded.items():
                    self._id_map[int(fid)] = int(db_id)
                logger.info("VectorIndex migrated old dict-style id_map to list format")
            else:
                self._id_map = loaded
        else:
            self._id_map = []

    def _save(self) -> None:
        faiss.write_index(self._index, str(self.index_path))
        with open(self.id_map_path, "w", encoding="utf-8") as f:
            json.dump(self._id_map, f, ensure_ascii=False)

    def add(self, chunk_db_id: int, vector: list[float]) -> None:
        """Add 函数."""
        self.add_batch([(chunk_db_id, vector)])

    def add_batch(self, items: list[tuple]) -> None:
        """批量添加向量，只在最后统一持久化."""
        if self._index is None:
            raise RuntimeError("Index not initialized")
        if not items:
            return
        ids = []
        vectors = []
        for chunk_db_id, vector in items:
            vec = np.array([vector], dtype=np.float32)
            actual_dim = vec.shape[1]
            if self._index.d != actual_dim:
                raise RuntimeError(
                    f"Cannot add vector with {actual_dim} dimensions "
                    f"to index with {self._index.d} dimensions."
                )
            faiss.normalize_L2(vec)
            vectors.append(vec)
            ids.append(chunk_db_id)
        if vectors:
            all_vecs = np.vstack(vectors)
            self._index.add(all_vecs)  # type: ignore[reportCallIssue]
            self._id_map.extend(ids)
            self._save()

    def remove_doc(self, chunk_db_ids: list[int]) -> None:
        """remove_doc 函数."""
        if self._index is None:
            raise RuntimeError("Index not initialized")
        if not chunk_db_ids:
            return
        ids_to_remove = set(chunk_db_ids)
        new_map = []
        vectors = []
        for fid, db_id in enumerate(self._id_map):
            if db_id not in ids_to_remove:
                new_map.append(db_id)
                vectors.append(self._index.reconstruct(fid))  # type: ignore[reportCallIssue]
        self._index = faiss.IndexFlatIP(self.dim)
        if vectors:
            mat = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(mat)
            self._index.add(mat)  # type: ignore[reportCallIssue]
        self._id_map = new_map
        self._save()

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        """Search 函数."""
        if self._index is None:
            raise RuntimeError("Index not initialized")
        if self._index.ntotal == 0:
            return []
        vec = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, top_k)  # type: ignore[reportCallIssue]
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            results.append((self._id_map[int(idx)], float(score)))
        return results
