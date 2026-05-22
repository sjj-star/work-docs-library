"""BigModelParserClient 单元测试."""

import io
import zipfile
from pathlib import Path

import pytest
from core.bigmodel_parser_client import BigModelParserClient


class FakeResponse:
    """Mock requests.Response."""

    def __init__(self, json_data=None, content=b"", status_code=200, exc=None):
        self._json = json_data
        self.content = content
        self.status_code = status_code
        self._exc = exc

    def json(self):
        if self._exc:
            raise self._exc
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def mock_config(monkeypatch):
    """提供 fake API key."""
    monkeypatch.setattr("core.bigmodel_parser_client.Config.PARSER_API_KEY", "fake-key")


def test_create_task_success(monkeypatch, tmp_path):
    """create_task 成功返回 task_id."""
    client = BigModelParserClient(api_key="test-key")

    def fake_post(url, **kwargs):
        return FakeResponse(json_data={"task_id": "task-123"})

    monkeypatch.setattr("core.bigmodel_parser_client.requests.post", fake_post)

    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"pdf content")
    task_id = client.create_task(pdf)
    assert task_id == "task-123"


def test_create_task_no_task_id(monkeypatch, tmp_path):
    """create_task 无 task_id 时应抛出 RuntimeError."""
    client = BigModelParserClient(api_key="test-key")

    def fake_post(url, **kwargs):
        return FakeResponse(json_data={"error": "bad request"})

    monkeypatch.setattr("core.bigmodel_parser_client.requests.post", fake_post)

    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"pdf content")
    with pytest.raises(RuntimeError, match="Failed to create task"):
        client.create_task(pdf)


def test_poll_result_success(monkeypatch):
    """poll_result 成功返回结果."""
    client = BigModelParserClient(api_key="test-key")
    call_count = [0]

    def fake_get(url, **kwargs):
        call_count[0] += 1
        return FakeResponse(json_data={"status": "succeeded", "data": {"url": "http://x"}})

    monkeypatch.setattr("core.bigmodel_parser_client.requests.get", fake_get)

    result = client.poll_result("task-123")
    assert result["status"] == "succeeded"
    assert call_count[0] == 1


def test_poll_result_failed(monkeypatch):
    """poll_result 收到 failed 状态应抛出 RuntimeError."""
    client = BigModelParserClient(api_key="test-key")

    def fake_get(url, **kwargs):
        return FakeResponse(json_data={"status": "failed"})

    monkeypatch.setattr("core.bigmodel_parser_client.requests.get", fake_get)

    with pytest.raises(RuntimeError, match="BigModel parsing failed"):
        client.poll_result("task-123")


def test_poll_result_timeout(monkeypatch):
    """poll_result 轮询耗尽应抛出 RuntimeError."""
    client = BigModelParserClient(api_key="test-key")
    monkeypatch.setattr(client, "MAX_POLL_RETRIES", 2)
    monkeypatch.setattr(client, "POLL_INTERVAL", 0)

    def fake_get(url, **kwargs):
        return FakeResponse(json_data={"status": "processing"})

    monkeypatch.setattr("core.bigmodel_parser_client.requests.get", fake_get)

    with pytest.raises(RuntimeError, match="polling timeout"):
        client.poll_result("task-123")


def test_download_result(monkeypatch):
    """download_result 返回 bytes."""
    client = BigModelParserClient(api_key="test-key")

    def fake_get(url, **kwargs):
        return FakeResponse(content=b"zip content")

    monkeypatch.setattr("core.bigmodel_parser_client.requests.get", fake_get)

    data = client.download_result("http://example.com/result.zip")
    assert data == b"zip content"


def test_parse_pdf_full_flow(monkeypatch, tmp_path):
    """parse_pdf 端到端流程."""
    client = BigModelParserClient(api_key="test-key")

    # 构建内存 ZIP：包含 result.md 和一张图片
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("result.md", "# Test\n\nHello world.")
        zf.writestr("image.png", b"png data")
    zip_bytes = zip_buf.getvalue()

    def fake_post(url, **kwargs):
        return FakeResponse(json_data={"task_id": "task-abc"})

    def fake_get(url, **kwargs):
        if "result" in url:
            return FakeResponse(
                json_data={
                    "status": "succeeded",
                    "parsing_result_url": "http://x.zip",
                    "data": {"parsingResultUrl": "http://x.zip"},
                }
            )
        return FakeResponse(content=zip_bytes)

    monkeypatch.setattr("core.bigmodel_parser_client.requests.post", fake_post)
    monkeypatch.setattr("core.bigmodel_parser_client.requests.get", fake_get)

    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"pdf")

    md_text, images = client.parse_pdf(pdf)
    assert "Hello world." in md_text
    assert len(images) == 1
    assert images[0].name == "image.png"
