"""vector_index 模块."""

import fcntl
import logging
import os
from pathlib import Path
from types import TracebackType

import faiss
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)


def _index_id_map_to_set(index: faiss.Index) -> set[int]:
    """从 IndexIDMap2 的 id_map 中读取所有存储 ID."""
    raw_id_map = getattr(index, "id_map", None)
    if raw_id_map is None:
        return set()
    size = raw_id_map.size()
    return {int(raw_id_map.at(i)) for i in range(size)}


class _VectorIndexTransaction:
    """VectorIndex 事务上下文管理器."""

    def __init__(self, vec: "VectorIndex") -> None:
        self._vec = vec

    def __enter__(self) -> "VectorIndex":
        self._vec.begin_transaction()
        return self._vec

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self._vec.commit()
        else:
            self._vec.rollback()


class VectorIndex:
    """VectorIndex 类.

    底层使用 faiss.IndexIDMap2，直接以 block_db_id 作为存储 ID。
    """

    def __init__(self, dim: int, index_path: Path | None = None) -> None:
        """初始化 VectorIndex."""
        self.dim = dim
        self.index_path = index_path or Config.FAISS_INDEX_PATH
        self._lock_path = self.index_path.with_suffix(".lock")
        self._lock_fd: int | None = None
        self._index: faiss.Index | None = None
        self._db_ids: set[int] = set()  # 已索引的 block_db_id 集合（防重复）
        self._in_transaction: bool = False
        self._txn_added_ids: list[int] = []
        self._load()

    def _new_index(self) -> faiss.Index:
        """创建新的空 IndexIDMap2 索引."""
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

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

    def _load(self) -> None:
        """加载索引."""
        if Path(self.index_path).exists():
            index = faiss.read_index(str(self.index_path))
            actual_dim = index.d
            if actual_dim != self.dim:
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
            if not hasattr(index, "id_map"):
                raise RuntimeError(
                    f"Existing FAISS index at {self.index_path} is not IndexIDMap2. "
                    "Delete it and re-ingest all documents."
                )
        else:
            index = self._new_index()

        self._db_ids = _index_id_map_to_set(index)
        self._index = index

    def _reload(self) -> None:
        """重新加载磁盘上的最新状态（调用方必须已持有锁）."""
        # 事务期间不应 reload，否则可能丢失事务内的未保存变更
        if self._in_transaction:
            return
        self._load()

    def _save(self) -> None:
        """原子保存 FAISS 索引（临时文件 + rename）."""
        assert self._index is not None
        tmp_index = self.index_path.with_suffix(".faiss.tmp")
        faiss.write_index(self._index, str(tmp_index))
        os.replace(str(tmp_index), str(self.index_path))

    def add(self, chunk_db_id: int, vector: list[float]) -> None:
        """Add 函数."""
        self.add_batch([(chunk_db_id, vector)])

    def _validate_vectors(self, vectors: list[np.ndarray]) -> np.ndarray:
        """校验向量维度并归一化."""
        assert self._index is not None
        all_vecs = np.vstack(vectors)
        actual_dim = all_vecs.shape[1]
        if self._index.d != actual_dim:
            raise RuntimeError(
                f"Cannot add vector with {actual_dim} dimensions "
                f"to index with {self._index.d} dimensions."
            )
        faiss.normalize_L2(all_vecs)
        return all_vecs

    def _add_batch_locked(self, items: list[tuple[int, list[float]]]) -> None:
        """在已持有锁的情况下执行批量添加."""
        if not items:
            return
        assert self._index is not None

        ids = []
        vectors = []
        for chunk_db_id, vector in items:
            if chunk_db_id in self._db_ids:
                logger.warning(f"跳过重复向量 | db_id={chunk_db_id}")
                continue
            vec = np.array([vector], dtype=np.float32)
            vectors.append(vec)
            ids.append(chunk_db_id)

        if not vectors:
            return

        all_vecs = self._validate_vectors(vectors)
        all_ids = np.array(ids, dtype=np.int64)
        self._index.add_with_ids(all_vecs, all_ids)  # type: ignore[reportCallIssue]
        self._db_ids.update(ids)
        if self._in_transaction:
            self._txn_added_ids.extend(ids)
        else:
            self._save()

    def add_batch(self, items: list[tuple[int, list[float]]]) -> None:
        """批量添加向量，只在最后统一持久化."""
        if self._index is None:
            raise RuntimeError("Index not initialized")
        if not items:
            return

        if self._in_transaction:
            self._add_batch_locked(items)
        else:
            self._acquire_lock()
            try:
                self._reload()
                self._add_batch_locked(items)
            finally:
                self._release_lock()

    def remove_doc(self, chunk_db_ids: list[int]) -> None:
        """根据 block_db_id 列表删除向量."""
        if self._index is None:
            raise RuntimeError("Index not initialized")
        if not chunk_db_ids:
            return

        self._acquire_lock()
        try:
            self._reload()
            assert self._index is not None
            ids_to_remove = np.array(list(set(chunk_db_ids)), dtype=np.int64)
            if ids_to_remove.size == 0:
                return
            self._index.remove_ids(ids_to_remove)  # type: ignore[reportCallIssue]
            self._db_ids.difference_update(ids_to_remove.tolist())
            self._save()
        finally:
            self._release_lock()

    def begin_transaction(self) -> None:
        """开始一个 FAISS 写入事务."""
        if self._in_transaction:
            raise RuntimeError("VectorIndex 事务不支持嵌套")
        self._acquire_lock()
        self._reload()
        self._in_transaction = True
        self._txn_added_ids = []

    def commit(self) -> None:
        """提交事务并释放锁."""
        if not self._in_transaction:
            raise RuntimeError("当前未在事务中")
        self._save()
        self._in_transaction = False
        self._txn_added_ids = []
        self._release_lock()

    def rollback(self) -> None:
        """回滚事务：删除本次新增的向量并释放锁."""
        if not self._in_transaction:
            raise RuntimeError("当前未在事务中")
        assert self._index is not None
        if self._txn_added_ids:
            ids = np.array(list(set(self._txn_added_ids)), dtype=np.int64)
            self._index.remove_ids(ids)  # type: ignore[reportCallIssue]
            self._db_ids.difference_update(self._txn_added_ids)
            self._save()
        self._in_transaction = False
        self._txn_added_ids = []
        self._release_lock()

    def transaction(self) -> _VectorIndexTransaction:
        """返回事务上下文管理器."""
        return _VectorIndexTransaction(self)

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
            if idx < 0:
                continue
            results.append((int(idx), float(score)))
        return results
