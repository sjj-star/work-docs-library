"""BigModelParserClient 单元测试."""

import io
import zipfile
from typing import Any

import pytest
from core.bigmodel_parser_client import BigModelParserClient


class FakeResponse:
    """Mock requests.Response."""

    def __init__(
        self,
        json_data: dict[str, Any] | None = None,
        content: bytes = b"",
        text: str = "",
        status_code: int = 200,
    ) -> None:
        self._json = json_data
        self.content = content
        self.text = text
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        if self._json is None:
            return {}
        return self._json


@pytest.fixture(autouse=True)
def mock_config(monkeypatch):
    """提供 fake API key."""
    monkeypatch.setattr("core.bigmodel_parser_client.Config.PARSER_API_KEY", "fake-key")


def test_create_task_success(monkeypatch, tmp_path):
    """create_task 成功返回 task_id."""
    client = BigModelParserClient(api_key="test-key")

    def fake_request(method, path, *, headers=None, **kwargs):
        return FakeResponse(json_data={"task_id": "task-123"})

    monkeypatch.setattr(client._client, "request", fake_request)

    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"pdf content")
    task_id = client.create_task(pdf)
    assert task_id == "task-123"


def test_create_task_no_task_id(monkeypatch, tmp_path):
    """create_task 无 task_id 时应抛出 RuntimeError."""
    client = BigModelParserClient(api_key="test-key")

    def fake_request(method, path, *, headers=None, **kwargs):
        return FakeResponse(json_data={"error": "bad request"})

    monkeypatch.setattr(client._client, "request", fake_request)

    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"pdf content")
    with pytest.raises(RuntimeError, match="Failed to create task"):
        client.create_task(pdf)


def test_poll_result_success(monkeypatch):
    """poll_result 成功返回结果."""
    client = BigModelParserClient(api_key="test-key")
    call_count = [0]

    def fake_request(method, path, *, headers=None, **kwargs):
        call_count[0] += 1
        return FakeResponse(json_data={"status": "succeeded", "data": {"url": "http://x"}})

    monkeypatch.setattr(client._client, "request", fake_request)

    result = client.poll_result("task-123")
    assert result["status"] == "succeeded"
    assert call_count[0] == 1


def test_poll_result_failed(monkeypatch):
    """poll_result 收到 failed 状态应抛出 RuntimeError."""
    client = BigModelParserClient(api_key="test-key")

    def fake_request(method, path, *, headers=None, **kwargs):
        return FakeResponse(json_data={"status": "failed"})

    monkeypatch.setattr(client._client, "request", fake_request)

    with pytest.raises(RuntimeError, match="BigModel parsing failed"):
        client.poll_result("task-123")


def test_poll_result_timeout(monkeypatch):
    """poll_result 轮询耗尽应抛出 RuntimeError."""
    client = BigModelParserClient(api_key="test-key")
    monkeypatch.setattr(client, "MAX_POLL_RETRIES", 2)
    monkeypatch.setattr(client, "POLL_INTERVAL", 0)

    def fake_request(method, path, *, headers=None, **kwargs):
        return FakeResponse(json_data={"status": "processing"})

    monkeypatch.setattr(client._client, "request", fake_request)

    with pytest.raises(RuntimeError, match="polling timeout"):
        client.poll_result("task-123")


def test_download_result(monkeypatch):
    """download_result 返回 bytes."""
    client = BigModelParserClient(api_key="test-key")

    def fake_request(method, path, *, headers=None, **kwargs):
        return FakeResponse(content=b"zip content")

    monkeypatch.setattr(client._client, "request", fake_request)

    data = client.download_result("http://example.com/result.zip")
    assert data == b"zip content"


def test_parse_pdf_full_flow(monkeypatch, tmp_path):
    """parse_pdf 完整流程."""
    client = BigModelParserClient(api_key="test-key")

    # 构造一个包含 markdown 的 zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("result.md", "# parsed")
    zip_bytes = zip_buffer.getvalue()

    call_idx = [0]

    def fake_request(method, path, *, headers=None, **kwargs):
        call_idx[0] += 1
        if call_idx[0] == 1:
            return FakeResponse(json_data={"task_id": "task-abc"})
        if call_idx[0] == 2:
            return FakeResponse(
                json_data={
                    "status": "succeeded",
                    "parsing_result_url": "http://example.com/result.zip",
                }
            )
        return FakeResponse(content=zip_bytes)

    monkeypatch.setattr(client._client, "request", fake_request)

    (tmp_path / "dummy.pdf").write_bytes(b"pdf")
    md_text, images = client.parse_pdf(tmp_path / "dummy.pdf", output_dir=tmp_path / "out")
    assert "# parsed" in md_text
    assert images == []


def test_parse_pdf_rejects_zip_path_traversal(monkeypatch, tmp_path):
    """parse_pdf 拒绝 ZIP 路径遍历."""
    client = BigModelParserClient(api_key="test-key")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("../evil.md", "bad")
    zip_bytes = zip_buffer.getvalue()

    call_idx = [0]

    def fake_request(method, path, *, headers=None, **kwargs):
        call_idx[0] += 1
        if call_idx[0] == 1:
            return FakeResponse(json_data={"task_id": "task-abc"})
        if call_idx[0] == 2:
            return FakeResponse(
                json_data={
                    "status": "succeeded",
                    "parsing_result_url": "http://example.com/result.zip",
                }
            )
        return FakeResponse(content=zip_bytes)

    monkeypatch.setattr(client._client, "request", fake_request)

    (tmp_path / "dummy.pdf").write_bytes(b"pdf")
    with pytest.raises(ValueError, match="Malformed ZIP entry path"):
        client.parse_pdf(tmp_path / "dummy.pdf", output_dir=tmp_path / "out")
