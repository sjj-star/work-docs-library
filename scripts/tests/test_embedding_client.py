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


def test_embed_single_failure_returns_zero_vector(monkeypatch, caplog):
    """单条失败时不影响其他 texts，并返回零向量."""
    client = EmbeddingClient()
    call_count = [0]

    def _fake_request(method, path, *, headers=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("boom")
        json_body = kwargs.get("json", {})
        inputs = json_body.get("input", [])
        return FakeResponse(embeddings=[[0.1, 0.2, 0.3, 0.4]] * len(inputs))

    monkeypatch.setattr(client._client, "request", _fake_request)
    with caplog.at_level("ERROR"):
        results = client.embed(["bad", "good"])
    assert results[0] == [0.0, 0.0, 0.0, 0.0]
    assert results[1] == [0.1, 0.2, 0.3, 0.4]
