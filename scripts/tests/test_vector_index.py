"""test_vector_index 模块."""

import faiss
import numpy as np
import pytest
from core.vector_index import VectorIndex


@pytest.fixture
def make_index(tmp_path):
    """make_index 函数."""

    def _make(dim=4):
        ip = tmp_path / f"index_{dim}.faiss"
        return VectorIndex(dim=dim, index_path=ip)

    return _make


def _norm_vec(vec):
    arr = np.array([vec], dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr[0].tolist()


def test_add_and_search(make_index):
    """Test add and search with block_db_id as stored id."""
    vi = make_index(dim=4)
    vec = _norm_vec([1.0, 0.0, 0.0, 0.0])
    vi.add(100, vec)
    results = vi.search(vec, top_k=1)
    assert len(results) == 1
    assert results[0][0] == 100
    assert abs(results[0][1] - 1.0) < 1e-4


def test_search_empty_index(make_index):
    """Test search empty index."""
    vi = make_index(dim=4)
    assert vi.search([1, 0, 0, 0], top_k=5) == []


def test_add_batch_and_search(make_index):
    """批量添加后搜索应返回正确 block_db_id."""
    vi = make_index(dim=4)
    vi.add_batch(
        [
            (10, _norm_vec([1, 0, 0, 0])),
            (20, _norm_vec([0, 1, 0, 0])),
            (30, _norm_vec([0, 0, 1, 0])),
        ]
    )
    results = vi.search(_norm_vec([0, 1, 0, 0]), top_k=1)
    assert results[0][0] == 20


def test_remove_doc(make_index):
    """Test remove doc by block_db_id."""
    vi = make_index(dim=4)
    vi.add(1, _norm_vec([1, 0, 0, 0]))
    vi.add(2, _norm_vec([0, 1, 0, 0]))
    vi.remove_doc([1])
    results = vi.search(_norm_vec([1, 0, 0, 0]), top_k=2)
    ids = [r[0] for r in results]
    assert 1 not in ids
    assert 2 in ids


def test_remove_doc_nonexistent(make_index):
    """删除不存在的 ID 不报错."""
    vi = make_index(dim=4)
    vi.add(1, _norm_vec([1, 0, 0, 0]))
    vi.remove_doc([999])
    results = vi.search(_norm_vec([1, 0, 0, 0]), top_k=1)
    assert results[0][0] == 1


def test_dimension_mismatch_error(make_index):
    """维度不匹配时应抛出错误，不允许自动重建."""
    vi = make_index(dim=4)
    vi.add(1, _norm_vec([1, 0, 0, 0]))
    with pytest.raises(RuntimeError, match="Cannot add vector with 6 dimensions"):
        vi.add(2, [1, 0, 0, 0, 0, 0])


def test_persistence(make_index, tmp_path):
    """Test persistence of IndexIDMap2."""
    vi = make_index(dim=4)
    vi.add(42, _norm_vec([0, 0, 0, 1]))
    # Create new instance pointing to same files
    vi2 = VectorIndex(dim=4, index_path=vi.index_path)
    results = vi2.search(_norm_vec([0, 0, 0, 1]), top_k=1)
    assert len(results) == 1
    assert results[0][0] == 42


def test_transaction_commit(make_index):
    """事务提交后数据应持久化."""
    vi = make_index(dim=4)
    vi.begin_transaction()
    vi.add_batch(
        [
            (10, _norm_vec([1, 0, 0, 0])),
            (20, _norm_vec([0, 1, 0, 0])),
        ]
    )
    vi.commit()

    results = vi.search(_norm_vec([1, 0, 0, 0]), top_k=1)
    assert results[0][0] == 10

    # 重启后仍应存在
    vi2 = VectorIndex(dim=4, index_path=vi.index_path)
    results = vi2.search(_norm_vec([0, 1, 0, 0]), top_k=1)
    assert results[0][0] == 20


def test_transaction_rollback(make_index):
    """事务回滚后数据应恢复到事务前状态."""
    vi = make_index(dim=4)
    vi.add(10, _norm_vec([1, 0, 0, 0]))

    vi.begin_transaction()
    vi.add_batch(
        [
            (20, _norm_vec([0, 1, 0, 0])),
            (30, _norm_vec([0, 0, 1, 0])),
        ]
    )
    vi.rollback()

    assert vi._db_ids == {10}
    results = vi.search(_norm_vec([0, 1, 0, 0]), top_k=5)
    ids = {r[0] for r in results}
    assert 20 not in ids
    assert 30 not in ids

    # 持久化后也一致
    vi2 = VectorIndex(dim=4, index_path=vi.index_path)
    assert vi2._db_ids == {10}


def test_transaction_rollback_partial_duplicates(make_index):
    """事务中部分 ID 已存在，回滚只删除实际新增的向量."""
    vi = make_index(dim=4)
    vi.add(10, _norm_vec([1, 0, 0, 0]))

    vi.begin_transaction()
    vi.add_batch(
        [
            (10, _norm_vec([1, 0, 0, 0])),  # 重复
            (20, _norm_vec([0, 1, 0, 0])),
            (30, _norm_vec([0, 0, 1, 0])),
        ]
    )
    vi.rollback()

    assert vi._db_ids == {10}
    results = vi.search(_norm_vec([1, 0, 0, 0]), top_k=1)
    assert results[0][0] == 10


def test_transaction_add_batch_exception(make_index):
    """add_batch 异常后 rollback 不应破坏已有索引."""
    vi = make_index(dim=4)
    vi.add(10, _norm_vec([1, 0, 0, 0]))

    def bad_normalize(*args, **kwargs):
        raise RuntimeError("normalize 失败")

    with pytest.raises(RuntimeError, match="normalize 失败"):
        vi.begin_transaction()
        try:
            import faiss as _faiss

            original = _faiss.normalize_L2
            _faiss.normalize_L2 = bad_normalize
            try:
                vi.add_batch([(20, _norm_vec([0, 1, 0, 0]))])
            finally:
                _faiss.normalize_L2 = original
        except Exception:
            vi.rollback()
            raise

    assert vi._db_ids == {10}


def test_close_releases_lock_fd(make_index):
    """close() 应释放文件锁并关闭文件描述符."""
    vi = make_index(dim=4)
    vi.add(1, _norm_vec([1, 0, 0, 0]))
    assert vi._lock_fd is None
    vi._acquire_lock()
    assert vi._lock_fd is not None
    vi.close()
    assert vi._lock_fd is None


def test_repeated_create_close_no_fd_leak(make_index):
    """重复创建和关闭 VectorIndex 不应导致 fd 泄漏."""
    for _ in range(5):
        vi = make_index(dim=4)
        vi.add(1, _norm_vec([1, 0, 0, 0]))
        vi.close()
        assert vi._lock_fd is None


def test_close_idempotent(make_index):
    """多次调用 close() 不应报错."""
    vi = make_index(dim=4)
    vi.close()
    vi.close()
    assert vi._lock_fd is None
