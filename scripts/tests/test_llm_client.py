"""test_llm_client 模块."""

import pytest
from core.llm_chat_client import BaseLLMClient as ChatClient


@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    """提供测试用配置."""
    from core import config as config_module

    monkeypatch.setattr(config_module.Config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config_module.Config, "LLM_BASE_URL", "https://test.com")
    monkeypatch.setattr(config_module.Config, "LLM_MODEL", "gpt-4")


def test_chat_client_chat(monkeypatch):
    """Test chat client chat."""

    def _fake_post(self, url, payload, timeout=None):
        return {"choices": [{"message": {"content": "Summary: hello\nKeywords: a, b"}}]}

    monkeypatch.setattr(ChatClient, "_post", _fake_post)

    client = ChatClient()
    text = client.chat([{"role": "user", "content": "summarize"}])
    assert "Summary: hello" in text


def test_chat_client_user_agent_from_plugin_json(monkeypatch):
    """_load_user_agent 应正确解析 plugin.json 中的 runtime.host_version."""

    class MockPath:
        def __init__(self, *args, **kwargs):
            pass

        @property
        def parent(self):
            return self

        def resolve(self):
            return self

        def __truediv__(self, other):
            return self

        def exists(self):
            return True

        def read_text(self, **kwargs):
            return '{"runtime": {"host_version": "2.0.0"}}'

    monkeypatch.setattr("core.llm_chat_client.Path", MockPath)
    ua = ChatClient._load_user_agent()
    assert ua == "KimiCLI/2.0.0"


def test_chat_client_user_agent_empty_version_fallback(monkeypatch):
    """plugin.json 中 host_version 为空时应 fallback 到默认值."""

    class MockPath:
        def __init__(self, *args, **kwargs):
            pass

        @property
        def parent(self):
            return self

        def resolve(self):
            return self

        def __truediv__(self, other):
            return self

        def exists(self):
            return True

        def read_text(self, **kwargs):
            return '{"runtime": {}}'

    monkeypatch.setattr("core.llm_chat_client.Path", MockPath)
    ua = ChatClient._load_user_agent()
    assert ua == "KimiCLI/1.44.0"


def test_chat_client_user_agent_default():
    """plugin.json 不存在时 User-Agent 使用默认值."""
    ua = ChatClient._load_user_agent()
    assert ua.startswith("KimiCLI/")


def test_chat_client_post_uses_apiclient(monkeypatch):
    """_post 应调用底层 APIClient."""
    client = ChatClient()
    calls = []

    def _fake_request(method, path, *, headers=None, **kwargs):
        calls.append((method, path, kwargs))

        class Resp:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        return Resp()

    monkeypatch.setattr(client._client, "request", _fake_request)
    result = client._post("/chat/completions", {"model": "test"})
    assert result["choices"][0]["message"]["content"] == "ok"
    assert calls[0][0] == "POST"
    assert calls[0][1] == "/chat/completions"
    assert client._client.user_agent.startswith("KimiCLI/")


def test_chat_client_missing_api_key(monkeypatch):
    """未配置 API Key 时应抛出 RuntimeError."""
    from core import config as config_module

    monkeypatch.setattr(config_module.Config, "LLM_API_KEY", "")
    with pytest.raises(RuntimeError, match="LLM API key not configured"):
        ChatClient()
