"""测试环境变量配置优先级和默认值.

config.json 机制已移除，配置唯一来源为 .env/环境变量，其次为代码默认值。
"""

import os

from core.config import Config, _resolve_config


class TestResolveConfig:
    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("WORKDOCS_TEST_MODEL", "custom-model")
        assert _resolve_config("WORKDOCS_TEST_MODEL", "default-model") == "custom-model"

    def test_empty_env_uses_default(self, monkeypatch):
        monkeypatch.delenv("WORKDOCS_TEST_MODEL", raising=False)
        assert _resolve_config("WORKDOCS_TEST_MODEL", "default-model") == "default-model"

    def test_whitespace_env_treated_as_empty(self, monkeypatch):
        monkeypatch.setenv("WORKDOCS_TEST_MODEL", "")
        assert _resolve_config("WORKDOCS_TEST_MODEL", "default-model") == "default-model"


class TestConfigDefaults:
    def test_llm_defaults(self):
        assert Config.LLM_MODEL == "kimi-k2.5"
        assert Config.LLM_BASE_URL == "https://api.moonshot.cn/v1"
        assert Config.LLM_MODE == "batch"

    def test_embedding_defaults(self):
        assert Config.EMBEDDING_MODEL == "embedding-3"
        assert Config.EMBEDDING_BASE_URL == "https://open.bigmodel.cn/api/paas/v4"

    def test_numeric_defaults(self):
        assert Config.EMBEDDING_DIMENSION == 1024
        assert Config.BLOCK_MAX_CHARS == 6000
        assert Config.BATCH_POLL_INTERVAL == 10

    def test_parser_defaults(self):
        assert Config.PARSER_TABLE_DETECTION_ENABLED is True
        assert Config.PARSER_TABLE_MIN_ROWS == 2
        assert Config.PARSER_FIGURE_MIN_SCORE == 2.0


class TestSensitiveMasking:
    def test_to_dict_always_masks_api_keys(self, monkeypatch):
        monkeypatch.setattr(Config, "LLM_API_KEY", "secret-llm")
        monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "secret-emb")
        monkeypatch.setattr(Config, "PARSER_API_KEY", "secret-parser")
        result = Config.to_dict()
        assert result["LLM_API_KEY"] == "***"
        assert result["EMBEDDING_API_KEY"] == "***"
        assert result["PARSER_API_KEY"] == "***"

    def test_to_dict_ignores_mask_sensitive_false(self, monkeypatch):
        monkeypatch.setattr(Config, "LLM_API_KEY", "secret-llm")
        result = Config.to_dict(mask_sensitive=False)
        assert result["LLM_API_KEY"] == "***"


class TestBooleanParsing:
    def test_thinking_enabled_false_by_default(self):
        assert Config.LLM_THINKING_ENABLED is False

    def test_table_detection_enabled_true_by_default(self):
        assert Config.PARSER_TABLE_DETECTION_ENABLED is True

    def test_boolean_env_value_one(self, monkeypatch):
        monkeypatch.setenv("WORKDOCS_LLM_THINKING_ENABLED", "1")
        assert _resolve_config("WORKDOCS_LLM_THINKING_ENABLED", "0") == "1"


class TestEnvIsolation:
    def test_workdocs_env_cleared_at_session_start(self):
        # conftest 在测试会话开始时会清理 WORKDOCS_* 环境变量
        for key in os.environ:
            assert not key.startswith("WORKDOCS_"), f"Unexpected WORKDOCS_ env var: {key}"
