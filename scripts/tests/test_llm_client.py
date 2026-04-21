import pytest

from core.embedding_client import EmbeddingClient
from core.llm_chat_client import LLMChatClient as ChatClient


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP Error")


class FakeSession:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self._response

    def close(self):
        pass


@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    from core import config as config_module
    monkeypatch.setattr(config_module.Config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config_module.Config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config_module.Config, "LLM_BASE_URL", "")
    monkeypatch.setattr(config_module.Config, "LLM_MODEL", "gpt-4")
    monkeypatch.setattr(config_module.Config, "EMBEDDING_API_KEY", "test-emb-key")
    monkeypatch.setattr(config_module.Config, "EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(config_module.Config, "EMBEDDING_MODEL", "text-embedding-3-small")


def test_embedding_client_embed(monkeypatch):
    fake_resp = FakeResponse({
        "data": [
            {"index": 1, "embedding": [0.1, 0.2]},
            {"index": 0, "embedding": [0.3, 0.4]},
        ]
    })
    session = FakeSession(fake_resp)
    monkeypatch.setattr("core.embedding_client.requests.Session", lambda: session)

    client = EmbeddingClient()
    result = client.embed(["hello", "world"])
    assert len(result) == 2
    assert result[0] == [0.3, 0.4]
    assert result[1] == [0.1, 0.2]
    assert "embeddings" in session.calls[0][0]


def test_chat_client_chat(monkeypatch):
    def _fake_post(self, url, payload, timeout=None):
        return {"choices": [{"message": {"content": "Summary: hello\nKeywords: a, b"}}]}

    monkeypatch.setattr(ChatClient, "_post", _fake_post)

    client = ChatClient()
    text = client.chat([{"role": "user", "content": "summarize"}])
    assert "Summary: hello" in text


def test_chat_client_summarize(monkeypatch, tmp_path):
    def _fake_post(self, url, payload, timeout=None):
        return {"choices": [{"message": {"content": "Summary: test summary\nKeywords: x, y"}}]}

    monkeypatch.setattr(ChatClient, "_post", _fake_post)

    client = ChatClient()
    out = client.summarize("some long text")
    assert out["summary"] == "test summary"
    assert out["keywords"] == ["x", "y"]


def test_chat_client_vision_describe(monkeypatch, tmp_path):
    def _fake_post(self, url, payload, timeout=None):
        return {"choices": [{"message": {"content": "An image description"}}]}

    monkeypatch.setattr(ChatClient, "_post", _fake_post)

    img = tmp_path / "test.png"
    from PIL import Image
    Image.new("RGB", (10, 10), color="red").save(img, format="PNG")

    client = ChatClient()
    desc = client.vision_describe(str(img))
    assert desc == "An image description"
