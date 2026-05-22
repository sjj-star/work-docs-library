"""test_vector_index 模块."""

import numpy as np
import pytest
from core.vector_index import VectorIndex


@pytest.fixture
def make_index(tmp_path):
    """make_index 函数."""

    def _make(dim=4):
        ip = tmp_path / f"index_{dim}.faiss"
        mp = tmp_path / f"map_{dim}.json"
        return VectorIndex(dim=dim, index_path=ip, id_map_path=mp)

    return _make


def test_add_and_search(make_index):
    """Test add and search."""
    vi = make_index(dim=4)
    vec = [1.0, 0.0, 0.0, 0.0]
    vi.add(100, vec)
    results = vi.search(vec, top_k=1)
    assert len(results) == 1
    assert results[0][0] == 100
    assert abs(results[0][1] - 1.0) < 1e-4


def test_search_empty_index(make_index):
    """Test search empty index."""
    vi = make_index(dim=4)
    assert vi.search([1, 0, 0, 0], top_k=5) == []


def test_remove_doc(make_index):
    """Test remove doc."""
    vi = make_index(dim=4)
    vi.add(1, [1, 0, 0, 0])
    vi.add(2, [0, 1, 0, 0])
    vi.remove_doc([1])
    results = vi.search([1, 0, 0, 0], top_k=2)
    ids = [r[0] for r in results]
    assert 1 not in ids
    assert 2 in ids


def test_dimension_mismatch_error(make_index):
    """维度不匹配时应抛出错误，不允许自动重建."""
    vi = make_index(dim=4)
    vi.add(1, [1, 0, 0, 0])
    # 添加不同维度的向量应抛出 RuntimeError
    with pytest.raises(RuntimeError, match="Cannot add vector with 6 dimensions"):
        vi.add(2, [1, 0, 0, 0, 0, 0])


def test_persistence(make_index, tmp_path):
    """Test persistence."""
    vi = make_index(dim=4)
    vi.add(42, [0, 0, 0, 1])
    # Create new instance pointing to same files
    vi2 = VectorIndex(dim=4, index_path=vi.index_path, id_map_path=vi.id_map_path)
    results = vi2.search([0, 0, 0, 1], top_k=1)
    assert len(results) == 1
    assert results[0][0] == 42


def test_load_migrates_old_dict_id_map(make_index, tmp_path):
    """Old id_map format was a dict {chunk_db_id: faiss_id}; ensure it migrates to list."""
    import json

    import faiss

    vi = make_index(dim=4)
    # Build a small faiss index manually
    index = faiss.IndexFlatIP(4)
    vec = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    faiss.normalize_L2(vec)
    index.add(vec)  # type: ignore[arg-type]
    faiss.write_index(index, str(vi.index_path))
    # Write old-style dict id_map: {db_id -> faiss_internal_id}
    old_map = {100: 0, 200: 1}
    vi.id_map_path.write_text(json.dumps(old_map), encoding="utf-8")
    # Reload
    vi2 = VectorIndex(dim=4, index_path=vi.index_path, id_map_path=vi.id_map_path)
    assert isinstance(vi2._id_map, list)
    assert vi2._id_map[0] == 100
    assert vi2._id_map[1] == 200
    # Search should return correct db ids
    results = vi2.search([1, 0, 0, 0], top_k=2)
    ids = [r[0] for r in results]
    assert 100 in ids
    assert 200 in ids


def test_close_releases_lock_fd(make_index):
    """close() 应释放文件锁并关闭文件描述符."""
    vi = make_index(dim=4)
    # add() 内部 try/finally 会自动释放锁，所以 add() 后 _lock_fd 应为 None
    vi.add(1, [1, 0, 0, 0])
    assert vi._lock_fd is None
    # 手动获取锁后验证 close() 能正确释放
    vi._acquire_lock()
    assert vi._lock_fd is not None
    vi.close()
    assert vi._lock_fd is None


def test_repeated_create_close_no_fd_leak(make_index):
    """重复创建和关闭 VectorIndex 不应导致 fd 泄漏."""
    for _ in range(5):
        vi = make_index(dim=4)
        vi.add(1, [1, 0, 0, 0])
        vi.close()
        assert vi._lock_fd is None


def test_close_idempotent(make_index):
    """多次调用 close() 不应报错."""
    vi = make_index(dim=4)
    vi.close()
    vi.close()  # 第二次调用不应抛异常
    assert vi._lock_fd is None
