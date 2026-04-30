"""Tests for scripts/core/batch_clients.py."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import pytest
from core.batch_clients import (
    BaseBatchClient,
    BatchClient,
    _build_jsonl,
    _parse_jsonl,
)
from core.config import Config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_config_db_path(monkeypatch, tmp_path):
    """将 Config.DB_PATH 重定向到 tmp_path，避免污染真实知识库."""
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "workdocs.db")
    monkeypatch.setattr(Config, "LLM_BATCH_TIMEOUT", 3600)
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "embedding-3")


class FakeBatchClient(BaseBatchClient):
    """用于测试 BaseBatchClient 通用流程的 Fake 子类."""

    def __init__(self, api_key: str = "test", base_url: str = "https://test.com") -> None:
        """初始化 FakeBatchClient."""
        super().__init__(api_key, base_url)
        self.uploaded: list[Path] = []
        self.created: list[str] = []
        self.statuses: list[dict[str, Any]] = []
        self.download_data: dict[str, str] = {}
        self.downloaded: list[str] = []
        self.deleted: list[str] = []

    def _upload_jsonl(self, file_path: Path) -> str:
        self.uploaded.append(file_path)
        return "file_id_123"

    def _create_batch(self, file_id: str) -> str:
        self.created.append(file_id)
        return "batch_id_123"

    def _get_batch_status(self, batch_id: str) -> dict[str, Any]:
        if self.statuses:
            return self.statuses.pop(0)
        return {"status": "completed", "output_file_id": "output_123"}

    def _download_file(self, file_id: str) -> str:
        self.downloaded.append(file_id)
        return self.download_data.get(file_id, "")

    def _delete_file(self, file_id: str) -> None:
        self.deleted.append(file_id)


# ---------------------------------------------------------------------------
# _build_jsonl
# ---------------------------------------------------------------------------


def test_build_jsonl_writes_correct_format(tmp_path):
    """测试 _build_jsonl 写入正确的 JSONL 格式."""
    output_path = tmp_path / "out.jsonl"
    requests = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
    result = _build_jsonl(requests, output_path)

    assert result == output_path
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"id": 1, "text": "hello"}
    assert json.loads(lines[1]) == {"id": 2, "text": "world"}


def test_build_jsonl_creates_parent_dirs(tmp_path):
    """测试 _build_jsonl 自动创建父目录."""
    output_path = tmp_path / "deep" / "nested" / "dir" / "out.jsonl"
    requests = [{"a": 1}]
    _build_jsonl(requests, output_path)
    assert output_path.exists()


def test_build_jsonl_unicode(tmp_path):
    """测试 _build_jsonl 正确处理 Unicode 内容."""
    output_path = tmp_path / "unicode.jsonl"
    requests = [{"text": "你好世界 🌍"}]
    _build_jsonl(requests, output_path)
    content = output_path.read_text(encoding="utf-8")
    assert "你好世界 🌍" in content
    assert json.loads(content.strip()) == {"text": "你好世界 🌍"}


# ---------------------------------------------------------------------------
# _parse_jsonl
# ---------------------------------------------------------------------------


def test_parse_jsonl_valid():
    """测试 _parse_jsonl 解析有效 JSONL."""
    text = '{"a": 1}\n{"b": 2}\n{"c": 3}'
    results = _parse_jsonl(text)
    assert results == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_parse_jsonl_skips_empty_lines():
    """测试 _parse_jsonl 跳过空行."""
    text = '{"a": 1}\n\n  \n{"b": 2}\n'
    results = _parse_jsonl(text)
    assert results == [{"a": 1}, {"b": 2}]


def test_parse_jsonl_json_decode_error(caplog):
    """测试 _parse_jsonl 遇到 JSON 解析错误时跳过并记录警告."""
    text = '{"a": 1}\nnot_json\n{"b": 2}'
    with caplog.at_level(logging.WARNING):
        results = _parse_jsonl(text)
    assert results == [{"a": 1}, {"b": 2}]
    assert any("JSONL 解析失败" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# BaseBatchClient.submit_and_wait
# ---------------------------------------------------------------------------


def test_submit_and_wait_success(monkeypatch, tmp_path):
    """测试 submit_and_wait 成功完成."""
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "workdocs.db")
    client = FakeBatchClient()
    client.statuses = [
        {"status": "in_progress", "request_counts": {"completed": 0, "total": 2}},
        {
            "status": "completed",
            "output_file_id": "output_123",
            "request_counts": {"completed": 2, "total": 2},
        },
    ]
    client.download_data = {
        "output_123": '{"custom_id":"req_1","result":"ok"}\n{"custom_id":"req_2","result":"ok2"}\n'
    }

    monkeypatch.setattr(time, "sleep", lambda s: None)

    requests = [{"body": "r1"}, {"body": "r2"}]
    results = client.submit_and_wait(requests, poll_interval=1)

    assert len(results) == 2
    assert results[0]["custom_id"] == "req_1"
    assert results[1]["custom_id"] == "req_2"
    assert len(client.uploaded) == 1
    assert len(client.created) == 1
    assert "output_123" in client.downloaded
    assert "file_id_123" in client.deleted


def test_submit_and_wait_failed(monkeypatch, tmp_path):
    """测试 submit_and_wait 在 batch 失败时抛出 RuntimeError."""
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "workdocs.db")
    client = FakeBatchClient()
    client.statuses = [
        {"status": "failed", "request_counts": {"completed": 0, "total": 1}},
    ]
    monkeypatch.setattr(time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="Batch 任务异常终止"):
        client.submit_and_wait([{"body": "r1"}], poll_interval=1)


def test_submit_and_wait_timeout(monkeypatch, tmp_path):
    """测试 submit_and_wait 在超时时抛出 RuntimeError."""
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "workdocs.db")
    client = FakeBatchClient()
    # 始终返回 in_progress，永远不会 completed
    client.statuses = [
        {"status": "in_progress", "request_counts": {"completed": 0, "total": 1}},
    ]

    call_idx = [0]

    def fake_time():
        call_idx[0] += 1
        if call_idx[0] == 1:
            return 0.0  # start_time
        return 99999.0  # 远超 timeout

    monkeypatch.setattr(time, "time", fake_time)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="Batch 任务轮询超时"):
        client.submit_and_wait([{"body": "r1"}], timeout=10, poll_interval=1)


def test_submit_and_wait_empty():
    """测试 submit_and_wait 传入空请求时直接返回空列表."""
    client = FakeBatchClient()
    results = client.submit_and_wait([])
    assert results == []
    assert client.uploaded == []


def test_submit_and_wait_file_size_check(monkeypatch, tmp_path):
    """测试 submit_and_wait 对超大 JSONL 文件抛出 RuntimeError."""
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "workdocs.db")
    client = FakeBatchClient()
    # 将文件大小限制临时设为 0 MB，使任何文件都超限
    monkeypatch.setattr(client, "MAX_FILE_SIZE_MB", 0)

    with pytest.raises(RuntimeError, match="JSONL 文件过大"):
        client.submit_and_wait([{"body": "x"}])


# ---------------------------------------------------------------------------
# BaseBatchClient.submit_parallel_batches
# ---------------------------------------------------------------------------


def test_submit_parallel_batches_single_chunk(monkeypatch, tmp_path):
    """测试 submit_parallel_batches 单 chunk 时直接调用 submit_and_wait."""
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "workdocs.db")
    client = FakeBatchClient()
    client.statuses = [
        {
            "status": "completed",
            "output_file_id": "out",
            "request_counts": {"completed": 2, "total": 2},
        },
    ]
    client.download_data = {"out": '{"id":1}\n{"id":2}\n'}
    monkeypatch.setattr(time, "sleep", lambda s: None)

    requests = [{"body": "r1"}, {"body": "r2"}]
    results = client.submit_parallel_batches(requests, max_file_size_mb=100)

    assert len(results) == 2
    assert client.created == ["file_id_123"]  # 只创建了一个 batch


def test_submit_parallel_batches_multiple_chunks(monkeypatch, tmp_path):
    """测试 submit_parallel_batches 多 chunk 时通过 ThreadPoolExecutor 并行提交."""
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "workdocs.db")
    client = FakeBatchClient()
    client.statuses = [
        {
            "status": "completed",
            "output_file_id": "out1",
            "request_counts": {"completed": 1, "total": 1},
        },
        {
            "status": "completed",
            "output_file_id": "out2",
            "request_counts": {"completed": 1, "total": 1},
        },
    ]
    client.download_data = {
        "out1": '{"chunk":0}\n',
        "out2": '{"chunk":1}\n',
    }
    monkeypatch.setattr(time, "sleep", lambda s: None)

    # max_file_size_mb=0 保证每个请求都会触发新 chunk（除了第一个）
    requests = [{"body": "a" * 100}, {"body": "b" * 100}]
    results = client.submit_parallel_batches(requests, max_file_size_mb=0)

    assert len(results) == 2
    assert len(client.created) == 2  # 两个 chunk，两个 batch


def test_submit_parallel_batches_empty():
    """测试 submit_parallel_batches 传入空请求时直接返回空列表."""
    client = FakeBatchClient()
    results = client.submit_parallel_batches([])
    assert results == []


# ---------------------------------------------------------------------------
# BatchClient.submit_embedding_batch
# ---------------------------------------------------------------------------


def test_submit_embedding_batch_success(monkeypatch, tmp_path):
    """测试 submit_embedding_batch 构造正确请求并返回排序后的 embeddings."""
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "workdocs.db")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "embedding-3")

    client = BatchClient(api_key="test-key", base_url="https://test.bigmodel.cn")

    captured_requests = []

    def _fake_submit_and_wait(reqs, timeout=None):
        captured_requests.extend(reqs)
        # 故意乱序返回，验证是否按 custom_id 排序
        return [
            {
                "custom_id": "embed_1",
                "response": {"body": {"data": [{"embedding": [0.1, 0.2]}]}},
            },
            {
                "custom_id": "embed_0",
                "response": {"body": {"data": [{"embedding": [0.3, 0.4]}]}},
            },
        ]

    monkeypatch.setattr(client, "submit_and_wait", _fake_submit_and_wait)

    texts = ["hello", "world"]
    embeddings = client.submit_embedding_batch(texts, model="embedding-3")

    assert len(embeddings) == 2
    # 验证按索引排序
    assert embeddings[0] == [0.3, 0.4]
    assert embeddings[1] == [0.1, 0.2]

    # 验证请求格式
    assert len(captured_requests) == 2
    assert captured_requests[0]["custom_id"] == "embed_0"
    assert captured_requests[0]["method"] == "POST"
    assert captured_requests[0]["url"] == Config.EMBEDDING_BATCH_ENDPOINT
    assert captured_requests[0]["body"]["model"] == "embedding-3"
    assert captured_requests[0]["body"]["input"] == "hello"

    assert captured_requests[1]["custom_id"] == "embed_1"
    assert captured_requests[1]["body"]["input"] == "world"


def test_submit_embedding_batch_empty(monkeypatch, tmp_path):
    """测试 submit_embedding_batch 传入空文本列表时返回空列表."""
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "workdocs.db")
    client = BatchClient(api_key="test-key", base_url="https://test.bigmodel.cn")

    called = False

    def _fake_submit_and_wait(reqs, timeout=None):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(client, "submit_and_wait", _fake_submit_and_wait)

    embeddings = client.submit_embedding_batch([])
    assert embeddings == []


# ---------------------------------------------------------------------------
# BatchClient 初始化
# ---------------------------------------------------------------------------


def test_batch_client_init_missing_key(monkeypatch):
    """测试 BatchClient 缺少 API key 时抛出 RuntimeError."""
    monkeypatch.setattr(Config, "LLM_API_KEY", "")
    with pytest.raises(RuntimeError, match="API key not configured"):
        BatchClient()


def test_batch_client_init_with_custom_params(monkeypatch, tmp_path):
    """测试 BatchClient 支持自定义参数."""
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "workdocs.db")
    client = BatchClient(
        api_key="test-key",
        base_url="https://test.example.com",
        batch_endpoint="/v2/chat/completions",
        completion_window="48h",
        files_url_path="v2/files",
        batches_url_path="v2/batches",
        download_url_template="{base_url}/v2/files/{file_id}",
        auto_delete_input_file=True,
        upload_mime_type="application/json",
    )
    assert client.api_key == "test-key"
    assert client.base_url == "https://test.example.com"
    assert client.batch_endpoint == "/v2/chat/completions"
    assert client.completion_window == "48h"
    assert client.files_url == "https://test.example.com/v2/files"
    assert client.batches_url == "https://test.example.com/v2/batches"
    assert client.download_url_template == "{base_url}/v2/files/{file_id}"
    assert client.auto_delete_input_file is True
    assert client.upload_mime_type == "application/json"


def test_batch_client_create_batch_payload(monkeypatch, tmp_path):
    """测试 BatchClient._create_batch 构造正确的 payload."""
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "workdocs.db")
    client = BatchClient(
        api_key="test-key",
        base_url="https://test.example.com",
        batch_endpoint="/v2/chat/completions",
        completion_window="48h",
        auto_delete_input_file=True,
    )

    posted = []

    def _fake_post(url, payload=None, files=None, timeout=120):
        posted.append((url, payload))
        return {"id": "batch_456"}

    monkeypatch.setattr(client, "_post", _fake_post)

    batch_id = client._create_batch("file_789")
    assert batch_id == "batch_456"
    assert posted[0][1]["endpoint"] == "/v2/chat/completions"
    assert posted[0][1]["completion_window"] == "48h"
    assert posted[0][1]["auto_delete_input_file"] is True


def test_batch_client_download_file(monkeypatch, tmp_path):
    """测试 BatchClient._download_file 使用配置的 URL 模板."""
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "workdocs.db")
    client = BatchClient(
        api_key="test-key",
        base_url="https://test.example.com",
        download_url_template="{base_url}/custom/files/{file_id}/download",
    )

    fetched_urls = []

    def _fake_get(url, **kwargs):
        class FakeResp:
            text = "downloaded"

            def raise_for_status(self):
                pass

        fetched_urls.append(url)
        return FakeResp()

    monkeypatch.setattr(client._session, "get", _fake_get)

    result = client._download_file("file_123")
    assert result == "downloaded"
    assert fetched_urls[0] == "https://test.example.com/custom/files/file_123/download"
