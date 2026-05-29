"""vector_index 模块."""

import fcntl
import json
import logging
import os
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
        self._lock_path = self.index_path.with_suffix(".lock")
        self._lock_fd: int | None = None
        self._index: faiss.Index | None = None
        self._id_map: list[int] = []  # faiss internal id -> chunk db id
        self._db_ids: set[int] = set()  # 已索引的 chunk_db_id 集合（防重复）
        self._load()

    def _acquire_lock(self) -> None:
        """获取进程级排他文件锁（防止并发写覆盖）."""
        if self._lock_fd is None:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._lock_path.touch(exist_ok=True)
            self._lock_fd = os.open(str(self._lock_path), os.O_RDWR | os.O_CREAT)
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)

    def _release_lock(self) -> None:
        """释放文件锁并关闭文件描述符."""
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    def close(self) -> None:
        """关闭 VectorIndex，释放文件锁和文件描述符."""
        self._release_lock()

    def _reload(self) -> None:
        """重新加载磁盘上的最新状态（调用方必须已持有锁）."""
        if Path(self.index_path).exists():
            self._index = faiss.read_index(str(self.index_path))
        else:
            self._index = faiss.IndexFlatIP(self.dim)
        if Path(self.id_map_path).exists():
            with open(self.id_map_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                max_fid = max(loaded.values()) if loaded else -1
                self._id_map = [0] * (max_fid + 1)
                for db_id, fid in loaded.items():
                    self._id_map[int(fid)] = int(db_id)
            else:
                self._id_map = list(loaded)
        else:
            self._id_map = []
        self._db_ids = set(self._id_map)

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
        self._db_ids = set(self._id_map)

    def _save(self) -> None:
        """原子保存 FAISS 索引和 id_map（临时文件 + rename）。."""
        assert self._index is not None
        # 原子写入索引
        tmp_index = self.index_path.with_suffix(".faiss.tmp")
        faiss.write_index(self._index, str(tmp_index))
        os.replace(str(tmp_index), str(self.index_path))

        # 原子写入 id_map
        tmp_map = self.id_map_path.with_suffix(".json.tmp")
        with open(tmp_map, "w", encoding="utf-8") as f:
            json.dump(self._id_map, f, ensure_ascii=False)
        os.replace(str(tmp_map), str(self.id_map_path))

    def add(self, chunk_db_id: int, vector: list[float]) -> None:
        """Add 函数."""
        self.add_batch([(chunk_db_id, vector)])

    def add_batch(self, items: list[tuple]) -> None:
        """批量添加向量，只在最后统一持久化."""
        if self._index is None:
            raise RuntimeError("Index not initialized")
        if not items:
            return
        self._acquire_lock()
        try:
            self._reload()  # 确保基于最新状态操作
            ids = []
            vectors = []
            for chunk_db_id, vector in items:
                if chunk_db_id in self._db_ids:
                    logger.warning(f"跳过重复向量 | db_id={chunk_db_id}")
                    continue
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
                self._db_ids.update(ids)
                self._save()
        finally:
            self._release_lock()

    def remove_doc(self, chunk_db_ids: list[int]) -> None:
        """remove_doc 函数."""
        if self._index is None:
            raise RuntimeError("Index not initialized")
        if not chunk_db_ids:
            return
        self._acquire_lock()
        try:
            self._reload()  # 确保基于最新状态操作
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
            self._db_ids = set(new_map)
            self._save()
        finally:
            self._release_lock()

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
