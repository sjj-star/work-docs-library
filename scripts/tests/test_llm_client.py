"""test_llm_client 模块."""

import pytest
from core.embedding_client import EmbeddingClient
from core.llm_chat_client import BaseLLMClient as ChatClient


class FakeResponse:
    """FakeResponse 类."""

    def __init__(self, json_data, status_code=200):
        """初始化 FakeResponse."""
        self._json = json_data
        self.status_code = status_code

    def json(self):
        """Json 函数."""
        return self._json

    def raise_for_status(self):
        """raise_for_status 函数."""
        if self.status_code >= 400:
            raise Exception("HTTP Error")


class FakeSession:
    """FakeSession 类."""

    def __init__(self, response):
        """初始化 FakeSession."""
        self._response = response
        self.calls = []

    def post(self, url, **kwargs):
        """Post 函数."""
        self.calls.append((url, kwargs))
        return self._response

    def close(self):
        """Close 函数."""
        pass


@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    """mock_env 函数."""
    from core import config as config_module

    monkeypatch.setattr(config_module.Config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config_module.Config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config_module.Config, "LLM_BASE_URL", "")
    monkeypatch.setattr(config_module.Config, "LLM_MODEL", "gpt-4")
    monkeypatch.setattr(config_module.Config, "EMBEDDING_API_KEY", "test-emb-key")
    monkeypatch.setattr(config_module.Config, "EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(config_module.Config, "EMBEDDING_MODEL", "text-embedding-3-small")


def test_embedding_client_embed(monkeypatch):
    """Test embedding client embed."""
    fake_resp = FakeResponse(
        {
            "data": [
                {"index": 1, "embedding": [0.1, 0.2]},
                {"index": 0, "embedding": [0.3, 0.4]},
            ]
        }
    )
    session = FakeSession(fake_resp)
    monkeypatch.setattr("core.embedding_client.requests.Session", lambda: session)

    client = EmbeddingClient()
    result = client.embed(["hello", "world"])
    assert len(result) == 2
    assert result[0] == [0.3, 0.4]
    assert result[1] == [0.1, 0.2]
    assert "embeddings" in session.calls[0][0]


def test_chat_client_chat(monkeypatch):
    """Test chat client chat."""

    def _fake_post(self, url, payload, timeout=None):
        return {"choices": [{"message": {"content": "Summary: hello\nKeywords: a, b"}}]}

    monkeypatch.setattr(ChatClient, "_post", _fake_post)

    client = ChatClient()
    text = client.chat([{"role": "user", "content": "summarize"}])
    assert "Summary: hello" in text
