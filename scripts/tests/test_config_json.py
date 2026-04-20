"""
测试 config.json 配置优先级和凭证注入机制

注意：本测试文件不修改 Config 类属性，只测试 _resolve_config 的解析逻辑，
以避免 importlib.reload 导致其他测试文件中的模块引用旧 Config 类。
"""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

_SKILL_ROOT = Path(__file__).resolve().parent.parent.parent


class TestResolveConfig:
    """测试 _resolve_config 的三层优先级（不依赖模块重载）"""

    def test_json_path_env_variable_highest_priority(self, monkeypatch):
        """Kimi CLI 注入的环境变量（json_path）优先级最高"""
        import core.config as cfg

        monkeypatch.setenv("llm.api_key", "injected_oauth_token")
        monkeypatch.setenv("WORKDOCS_LLM_API_KEY", "env_api_key")

        with patch.object(cfg, "_CONFIG_JSON", {"llm": {"api_key": "json_token"}}):
            result = cfg._resolve_config("WORKDOCS_LLM_API_KEY", "llm.api_key", "default")

        assert result == "injected_oauth_token"

    def test_config_json_second_priority(self, monkeypatch):
        """config.json 优先级第二（当没有 Kimi CLI 注入时）"""
        import core.config as cfg

        monkeypatch.delenv("llm.api_key", raising=False)
        monkeypatch.setenv("WORKDOCS_LLM_API_KEY", "env_api_key")

        with patch.object(cfg, "_CONFIG_JSON", {"llm": {"api_key": "json_token"}}):
            result = cfg._resolve_config("WORKDOCS_LLM_API_KEY", "llm.api_key", "default")

        assert result == "json_token"

    def test_env_name_variable_third_priority(self, monkeypatch):
        """.env 环境变量优先级第三（当 config.json 字段为空时）"""
        import core.config as cfg

        monkeypatch.delenv("llm.api_key", raising=False)
        monkeypatch.setenv("WORKDOCS_LLM_API_KEY", "env_api_key")

        with patch.object(cfg, "_CONFIG_JSON", {"llm": {"api_key": ""}}):
            result = cfg._resolve_config("WORKDOCS_LLM_API_KEY", "llm.api_key", "default")

        assert result == "env_api_key"

    def test_default_fallback(self, monkeypatch):
        """当所有来源都为空时回退到默认值"""
        import core.config as cfg

        monkeypatch.delenv("llm.api_key", raising=False)
        monkeypatch.delenv("WORKDOCS_LLM_API_KEY", raising=False)

        with patch.object(cfg, "_CONFIG_JSON", {}):
            result = cfg._resolve_config("WORKDOCS_LLM_API_KEY", "llm.api_key", "default_value")

        assert result == "default_value"

    def test_nested_json_path(self, monkeypatch):
        """嵌套 JSON 路径解析"""
        import core.config as cfg

        monkeypatch.delenv("llm.model", raising=False)
        monkeypatch.delenv("WORKDOCS_LLM_MODEL", raising=False)

        with patch.object(cfg, "_CONFIG_JSON", {"llm": {"model": "kimi-k2.5"}}):
            result = cfg._resolve_config("WORKDOCS_LLM_MODEL", "llm.model", "gpt-4o-mini")

        assert result == "kimi-k2.5"

    def test_no_json_path_uses_env_only(self, monkeypatch):
        """当不提供 json_path 时，只检查 env_name 和默认值"""
        import core.config as cfg

        monkeypatch.setenv("WORKDOCS_CONTEXT_STRATEGY", "aggressive")

        with patch.object(cfg, "_CONFIG_JSON", {"context_strategy": "smart"}):
            result = cfg._resolve_config("WORKDOCS_CONTEXT_STRATEGY", "", "balanced")

        assert result == "aggressive"

    def test_boolean_value_from_json(self, monkeypatch):
        """config.json 中的布尔值会被转换为字符串"""
        import core.config as cfg

        monkeypatch.delenv("llm.thinking_enabled", raising=False)
        monkeypatch.delenv("WORKDOCS_LLM_THINKING_ENABLED", raising=False)

        with patch.object(cfg, "_CONFIG_JSON", {"llm": {"thinking_enabled": True}}):
            result = cfg._resolve_config("WORKDOCS_LLM_THINKING_ENABLED", "llm.thinking_enabled", "0")

        assert result == "True"

    def test_integer_value_from_json(self, monkeypatch):
        """config.json 中的整数值会被转换为字符串"""
        import core.config as cfg

        monkeypatch.delenv("embedding.dimension", raising=False)
        monkeypatch.delenv("WORKDOCS_EMBEDDING_DIMENSION", raising=False)

        with patch.object(cfg, "_CONFIG_JSON", {"embedding": {"dimension": 768}}):
            result = cfg._resolve_config("WORKDOCS_EMBEDDING_DIMENSION", "embedding.dimension", "1536")

        assert result == "768"

    def test_json_path_env_overrides_json_content(self, monkeypatch):
        """运行时注入的环境变量同时覆盖 config.json 和 .env"""
        import core.config as cfg

        monkeypatch.setenv("llm.api_key", "runtime_oauth_token")
        monkeypatch.setenv("WORKDOCS_LLM_API_KEY", "static_key")

        with patch.object(cfg, "_CONFIG_JSON", {"llm": {"api_key": "persistent_token"}}):
            result = cfg._resolve_config("WORKDOCS_LLM_API_KEY", "llm.api_key", "")

        assert result == "runtime_oauth_token"


class TestConfigJsonFileLoading:
    """测试 _load_config_json 文件加载逻辑"""

    def test_load_existing_config_json(self, tmp_path, monkeypatch):
        """加载存在的 config.json"""
        import core.config as cfg

        fake_config = {"llm": {"api_key": "test_key"}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(fake_config), encoding="utf-8")

        with patch.object(cfg, "_SKILL_ROOT", tmp_path):
            result = cfg._load_config_json()

        assert result == fake_config

    def test_missing_config_json_returns_empty_dict(self, tmp_path, monkeypatch):
        """config.json 不存在时返回空 dict"""
        import core.config as cfg

        with patch.object(cfg, "_SKILL_ROOT", tmp_path):
            result = cfg._load_config_json()

        assert result == {}

    def test_invalid_json_returns_empty_dict(self, tmp_path, monkeypatch):
        """config.json 内容非法时返回空 dict"""
        import core.config as cfg

        config_file = tmp_path / "config.json"
        config_file.write_text("not valid json{", encoding="utf-8")

        with patch.object(cfg, "_SKILL_ROOT", tmp_path):
            result = cfg._load_config_json()

        assert result == {}


class TestConfigIntegration:
    """通过直接读取 Config 类属性验证集成行为（避免 importlib.reload）"""

    def test_current_config_has_expected_defaults(self):
        """验证当前 Config 至少有合理的默认值（不假设具体值，避免环境差异）"""
        from core.config import Config

        assert Config.LLM_PROVIDER in ("openai", "kimi")
        assert Config.EMBEDDING_PROVIDER in ("openai", "kimi", "bigmodel")
        assert Config.EMBEDDING_DIMENSION > 0
        assert Config.BATCH_SIZE > 0
        assert isinstance(Config.LLM_THINKING_ENABLED, bool)
