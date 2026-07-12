"""EmbeddingClient 单元测试."""

from __future__ import annotations

from typing import Any

import pytest
from core.config import Config
from core.embedding_client import EmbeddingClient


class FakeResponse:
    """模拟 requests.Response."""

    def __init__(self, embeddings: list[list[float]] | None = None, status_code: int = 200) -> None:
        self.status_code = status_code
        self._embeddings = embeddings or []

    def json(self) -> dict[str, Any]:
        return {"data": [{"embedding": e} for e in self._embeddings]}


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch, tmp_path):
    """避免测试污染真实路径."""
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "workdocs.db")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(Config, "EMBEDDING_BASE_URL", "https://test.com")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "embedding-3")
    monkeypatch.setattr(Config, "EMBEDDING_DIMENSION", 4)
    monkeypatch.setattr(Config, "EMBED_MAX_CHARS_PER_TEXT", 20)
    monkeypatch.setattr(Config, "EMBED_SPLIT_OVERLONG", True)


def test_embed_single_text(monkeypatch):
    """正常单条文本 embed."""
    client = EmbeddingClient()

    def _fake_request(method, path, *, headers=None, **kwargs):
        json_body = kwargs.get("json", {})
        inputs = json_body.get("input", [])
        return FakeResponse(embeddings=[[0.1, 0.2, 0.3, 0.4]] * len(inputs))

    monkeypatch.setattr(client._client, "request", _fake_request)
    result = client.embed_single("hello")
    assert len(result) == 4


def test_embed_overlong_text_splits(monkeypatch):
    """超长文本应自动拆分后平均."""
    client = EmbeddingClient()
    calls = []

    def _fake_request(method, path, *, headers=None, **kwargs):
        json_body = kwargs.get("json", {})
        inputs = json_body.get("input", [])
        calls.append(inputs)
        return FakeResponse(embeddings=[[1.0, 0.0, 0.0, 0.0]] * len(inputs))

    monkeypatch.setattr(client._client, "request", _fake_request)
    result = client.embed_single("a" * 100)
    # 100 字符按 20 字符拆分，应产生 5 个 chunk
    assert len(calls) == 1
    assert len(calls[0]) == 5
    assert len(result) == 4


def test_embed_single_failure_raises(monkeypatch, caplog):
    """单条失败时应抛出 APIError，而不是返回零向量."""
    from core.api_client import APIError

    client = EmbeddingClient()

    def _fake_request(method, path, *, headers=None, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(client._client, "request", _fake_request)
    with pytest.raises(APIError):
        client.embed_single("bad")


def test_embed_endpoint_default(monkeypatch):
    """默认 endpoint 与 base_url 拼接后不应重复版本路径."""
    client = EmbeddingClient()
    calls = []

    def _fake_request(method, path, *, headers=None, **kwargs):
        calls.append(path)
        return FakeResponse(embeddings=[[0.1, 0.2, 0.3, 0.4]])

    monkeypatch.setattr(client._client, "request", _fake_request)
    client.embed_single("hello")
    assert calls == ["/embeddings"]
