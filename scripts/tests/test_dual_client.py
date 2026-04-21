"""
测试双客户端架构 - LLM 对话客户端和 Embedding 客户端的独立配置
所有测试使用 mock _post 方法，不调用真实 API。
"""
import logging
import pytest
from pathlib import Path

# 添加项目根目录到 Python 路径
_SKILL_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_SKILL_ROOT))

from core.llm_chat_client import LLMChatClient
from core.embedding_client import EmbeddingClient
from core.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLMChatClient 测试
# ---------------------------------------------------------------------------

def test_llm_chat_client_independence(monkeypatch):
    """测试 LLM 对话客户端的独立性（mock API）"""
    monkeypatch.setattr(Config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "LLM_API_KEY", "test-llm-key")
    monkeypatch.setattr(Config, "LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(Config, "LLM_THINKING_ENABLED", False)

    def _fake_post(self, url, payload, timeout=None):
        return {"choices": [{"message": {"content": "mocked response"}}]}

    monkeypatch.setattr(LLMChatClient, "_post", _fake_post)

    client = LLMChatClient()

    # 验证配置独立性
    assert client.provider == "openai"
    assert client.api_key == "test-llm-key"
    assert client.base_url == "https://api.openai.com/v1"
    assert client.model == "gpt-4o-mini"
    assert client.thinking_enabled is False

    # 测试基本对话功能
    result = client.chat([{"role": "user", "content": "Hello"}])
    assert isinstance(result, str)
    assert result == "mocked response"

    client.close()


def test_kimi_model_temperature_fix(monkeypatch):
    """测试 Kimi 模型的 temperature 强制为 1.0"""
    captured_payloads = []

    def _fake_post(self, url, payload, timeout=None):
        captured_payloads.append(payload)
        return {"choices": [{"message": {"content": "kimi response"}}]}

    monkeypatch.setattr(LLMChatClient, "_post", _fake_post)

    monkeypatch.setattr(Config, "LLM_PROVIDER", "kimi")
    monkeypatch.setattr(Config, "LLM_MODEL", "kimi-k2.5")
    monkeypatch.setattr(Config, "LLM_API_KEY", "test-kimi-key")
    monkeypatch.setattr(Config, "LLM_BASE_URL", "https://api.moonshot.cn/v1")
    monkeypatch.setattr(Config, "LLM_THINKING_ENABLED", False)

    client = LLMChatClient()

    # 传入 temperature=0.3，但 Kimi 模型应强制使用 1.0
    result = client.chat([{"role": "user", "content": "Hello"}], temperature=0.3)
    assert result == "kimi response"

    # 验证请求 payload 中的 temperature 被强制设为 1.0
    assert captured_payloads[-1]["temperature"] == 1.0

    # 测试思考模式回退（Kimi 不支持思考模式）
    result2 = client.chat_with_thinking([{"role": "user", "content": "Hello"}])
    assert result2 == "kimi response"
    # 验证没有发送 extra_body
    for payload in captured_payloads:
        assert "extra_body" not in payload

    client.close()


def test_thinking_mode_unsupported(monkeypatch):
    """测试不支持的思考模式处理（非 Kimi 模型）"""
    captured_payloads = []

    def _fake_post(self, url, payload, timeout=None):
        captured_payloads.append(payload)
        return {"choices": [{"message": {"content": "thinking response"}}]}

    monkeypatch.setattr(LLMChatClient, "_post", _fake_post)

    monkeypatch.setattr(Config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(Config, "LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(Config, "LLM_THINKING_ENABLED", True)

    client = LLMChatClient()

    # 非 Kimi 模型且 thinking_enabled=True，chat() 应该自动添加 extra_body
    result = client.chat([{"role": "user", "content": "Hello"}])
    assert result == "thinking response"

    assert "extra_body" in captured_payloads[-1]
    assert captured_payloads[-1]["extra_body"]["thinking"]["type"] == "enabled"

    client.close()


# ---------------------------------------------------------------------------
# EmbeddingClient 测试
# ---------------------------------------------------------------------------

def test_embedding_client_independence(monkeypatch):
    """测试 Embedding 客户端的独立性（mock API）"""
    monkeypatch.setattr(Config, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "test-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(Config, "EMBEDDING_DIMENSION", 4)

    def _fake_post(self, url, payload, timeout=None):
        texts = payload["input"]
        return {
            "data": [
                {"index": i, "embedding": [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1), 0.4 * (i + 1)]}
                for i in range(len(texts))
            ]
        }

    monkeypatch.setattr(EmbeddingClient, "_post", _fake_post)

    client = EmbeddingClient()

    # 验证配置独立性
    assert client.provider == "openai"
    assert client.api_key == "test-emb-key"
    assert client.base_url == "https://api.openai.com/v1"
    assert client.model == "text-embedding-3-small"

    # 测试嵌入功能
    test_texts = ["This is a test sentence.", "Another test sentence."]
    embeddings = client.embed(test_texts)

    # 验证维度
    assert client.get_embedding_dimension() == 4

    assert len(embeddings) == len(test_texts)
    dims = [len(emb) for emb in embeddings]
    assert len(set(dims)) == 1  # 所有维度相同
    assert dims[0] == 4

    client.close()


# ---------------------------------------------------------------------------
# 双客户端协作测试
# ---------------------------------------------------------------------------

def test_dual_client_different_providers(monkeypatch):
    """测试双客户端可以使用不同供应商（mock API）"""
    llm_payloads = []
    emb_payloads = []

    def _fake_llm_post(self, url, payload, timeout=None):
        llm_payloads.append(payload)
        return {"choices": [{"message": {"content": "llm result"}}]}

    def _fake_emb_post(self, url, payload, timeout=None):
        emb_payloads.append(payload)
        texts = payload["input"]
        return {"data": [{"index": i, "embedding": [0.1, 0.2, 0.3]} for i in range(len(texts))]}

    monkeypatch.setattr(LLMChatClient, "_post", _fake_llm_post)
    monkeypatch.setattr(EmbeddingClient, "_post", _fake_emb_post)

    monkeypatch.setattr(Config, "LLM_PROVIDER", "kimi")
    monkeypatch.setattr(Config, "LLM_MODEL", "kimi-k2.5")
    monkeypatch.setattr(Config, "LLM_API_KEY", "test-llm-key")
    monkeypatch.setattr(Config, "LLM_BASE_URL", "https://api.moonshot.cn/v1")
    monkeypatch.setattr(Config, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "test-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_BASE_URL", "https://api.openai.com/v1")

    llm_client = LLMChatClient()
    embed_client = EmbeddingClient()

    # 验证它们可以使用不同的配置
    assert llm_client.provider == "kimi"
    assert embed_client.provider == "openai"

    # 测试基本功能
    llm_result = llm_client.chat([{"role": "user", "content": "Test"}])
    embed_result = embed_client.embed(["Test sentence"])

    assert isinstance(llm_result, str)
    assert len(embed_result) == 1
    assert len(embed_result[0]) == 3

    llm_client.close()
    embed_client.close()
