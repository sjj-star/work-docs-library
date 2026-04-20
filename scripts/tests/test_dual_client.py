"""
测试双客户端架构 - LLM 对话客户端和 Embedding 客户端的独立配置
"""
import os
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


def test_llm_chat_client_independence():
    """测试 LLM 对话客户端的独立性"""
    # 确保 LLM 配置存在
    if not Config.llm_configured():
        pytest.skip("LLM 未配置")
    
    client = LLMChatClient()
    
    # 验证配置独立性
    assert client.provider == Config.LLM_PROVIDER
    assert client.api_key == Config.LLM_API_KEY
    assert client.base_url == Config.LLM_BASE_URL
    assert client.model == Config.LLM_MODEL
    assert client.thinking_enabled == Config.LLM_THINKING_ENABLED
    
    # 测试基本对话功能
    result = client.chat([{"role": "user", "content": "Hello"}])
    assert isinstance(result, str)
    assert len(result) > 0
    
    client.close()


def test_embedding_client_independence():
    """测试 Embedding 客户端的独立性"""
    # 确保 Embedding 配置存在
    if not Config.embedding_configured():
        pytest.skip("Embedding 未配置")
    
    client = EmbeddingClient()
    
    # 验证配置独立性
    assert client.provider == Config.EMBEDDING_PROVIDER.lower()  # provider 存储为小写
    assert client.api_key == Config.EMBEDDING_API_KEY
    assert client.base_url == Config.EMBEDDING_BASE_URL
    assert client.model == Config.EMBEDDING_MODEL
    
    # 测试嵌入功能
    test_texts = ["This is a test sentence.", "Another test sentence."]
    embeddings = client.embed(test_texts)
    
    # 验证维度（在 embed 后才能获取）
    assert client.get_embedding_dimension() == Config.EMBEDDING_DIMENSION
    
    assert len(embeddings) == len(test_texts)
    # 验证所有 embedding 维度一致（不一定等于配置的维度，取决于实际 API）
    dims = [len(emb) for emb in embeddings]
    assert len(set(dims)) == 1  # 所有维度相同
    
    client.close()


def test_dual_client_different_providers():
    """测试双客户端可以使用不同供应商"""
    # 这个测试假设 LLM 和 Embedding 可以配置为不同供应商
    if not (Config.llm_configured() and Config.embedding_configured()):
        pytest.skip("需要同时配置 LLM 和 Embedding")
    
    # 创建两个独立的客户端
    llm_client = LLMChatClient()
    embed_client = EmbeddingClient()
    
    # 验证它们可以使用不同的配置
    logger.info(f"LLM 客户端: {Config.LLM_PROVIDER} - {Config.LLM_MODEL}")
    logger.info(f"Embedding 客户端: {Config.EMBEDDING_PROVIDER} - {Config.EMBEDDING_MODEL}")
    
    # 测试基本功能
    llm_result = llm_client.chat([{"role": "user", "content": "Test"}])
    embed_result = embed_client.embed(["Test sentence"])
    
    assert isinstance(llm_result, str)
    assert len(embed_result) == 1
    assert len(embed_result[0]) > 0  # 验证返回了有效的 embedding
    
    llm_client.close()
    embed_client.close()


def test_kimi_model_temperature_fix():
    """测试 Kimi 模型的 temperature 限制修复"""
    if Config.LLM_PROVIDER != "kimi" or not Config.LLM_MODEL.startswith("kimi"):
        pytest.skip("需要 Kimi 模型")
    
    client = LLMChatClient()
    
    # 测试 Kimi 模型自动使用 temperature=1.0
    result = client.chat([{"role": "user", "content": "Hello"}], temperature=0.3)
    assert isinstance(result, str)
    assert len(result) > 0
    
    # 测试思考模式回退
    result2 = client.chat_with_thinking([{"role": "user", "content": "Hello"}])
    assert isinstance(result2, str)
    assert len(result2) > 0
    
    client.close()


def test_thinking_mode_unsupported():
    """测试不支持的思考模式处理"""
    if Config.LLM_PROVIDER == "openai" and "gpt-4" in Config.LLM_MODEL:
        pytest.skip("OpenAI GPT-4 支持思考模式")
    
    client = LLMChatClient()
    
    # 测试思考模式（应该回退到普通对话）
    result = client.chat_with_thinking([{"role": "user", "content": "Hello"}])
    assert isinstance(result, str)
    assert len(result) > 0
    
    client.close()