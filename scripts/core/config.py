"""config 模块."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_SKILL_ROOT = Path(__file__).resolve().parent.parent.parent

# 加载 .env（不覆盖已有环境变量，因为 Kimi CLI 可能已注入凭证）
load_dotenv(_SKILL_ROOT / ".env", override=False)
load_dotenv(_SKILL_ROOT / "scripts" / ".env", override=False)

logger = logging.getLogger(__name__)


def _load_config_json() -> dict:
    """加载项目根目录的 config.json（由 plugin.json 的 config_file 指定）."""
    config_path = _SKILL_ROOT / "config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"读取 config.json 失败: {e}")
    return {}


# 全局加载一次，供所有配置解析使用
_CONFIG_JSON = _load_config_json()


def _resolve_config(env_name: str, json_path: str = "", default: str = "") -> str:
    """按优先级解析配置值：.

    1. 环境变量 json_path（Kimi CLI 运行时注入，如 llm.api_key）
    2. config.json 路径（Kimi CLI 持久化 + 用户手动配置）
    3. 环境变量 env_name（.env 文件，如 WORKDOCS_LLM_API_KEY）
    4. 默认值.
    """
    # 1. Kimi CLI 注入的环境变量（运行时动态，最高优先级）
    if json_path:
        env_val = os.getenv(json_path, "")
        if env_val:
            return env_val

    # 2. config.json（统一配置入口，用户手动维护的参数）
    if json_path:
        keys = json_path.split(".")
        data = _CONFIG_JSON
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, {})
            else:
                break
        if data is None:
            pass  # null 视为未配置，继续回退
        elif isinstance(data, (str, int, float, bool)):
            val = str(data)
            if val:  # 空字符串视为未配置，继续回退
                return val

    # 3. .env 环境变量（开发/独立运行回退）
    env_val = os.getenv(env_name, "")
    if env_val:
        return env_val

    return default


class Config:
    """Config 类."""

    DB_PATH: Path = _SKILL_ROOT / "knowledge_base" / "workdocs.db"
    FAISS_INDEX_PATH: Path = _SKILL_ROOT / "knowledge_base" / "faiss.index"
    ID_MAP_PATH: Path = _SKILL_ROOT / "knowledge_base" / "id_map.json"
    PROMPT_DIR: Path = _SKILL_ROOT / "scripts" / "prompts"

    # LLM 对话模型配置（总结用）
    LLM_API_KEY: str = _resolve_config("WORKDOCS_LLM_API_KEY", "llm.api_key", "")
    LLM_BASE_URL: str = _resolve_config(
        "WORKDOCS_LLM_BASE_URL", "llm.endpoint", "https://api.moonshot.cn/v1"
    )
    LLM_MODEL: str = _resolve_config("WORKDOCS_LLM_MODEL", "llm.model", "kimi-k2.5")
    LLM_THINKING_ENABLED: bool = (
        _resolve_config("WORKDOCS_LLM_THINKING_ENABLED", "llm.thinking_enabled", "0") == "1"
    )

    # Embedding 模型配置（向量化用）- 完全独立
    EMBEDDING_API_KEY: str = _resolve_config("WORKDOCS_EMBEDDING_API_KEY", "embedding.api_key", "")
    EMBEDDING_BASE_URL: str = _resolve_config(
        "WORKDOCS_EMBEDDING_BASE_URL",
        "embedding.endpoint",
        "https://open.bigmodel.cn/api/paas/v4",
    )
    EMBEDDING_MODEL: str = _resolve_config(
        "WORKDOCS_EMBEDDING_MODEL", "embedding.model", "embedding-3"
    )

    # 文件解析配置
    PARSER_API_KEY: str = _resolve_config("WORKDOCS_PARSER_API_KEY", "parser.api_key", "")

    # 嵌入向量维度配置
    EMBEDDING_DIMENSION: int = 0  # 将在下方初始化

    # LLM Batch 处理配置
    LLM_BATCH_MAX_CHARS: int = 0  # 将在下方初始化
    LLM_BATCH_TIMEOUT: int = 0  # 将在下方初始化


    # --- API Endpoint 配置（服务商无感化） ---
    LLM_BATCH_ENDPOINT: str = _resolve_config(
        "WORKDOCS_LLM_BATCH_ENDPOINT", "llm.batch_endpoint", "/v1/chat/completions"
    )
    EMBEDDING_BATCH_ENDPOINT: str = _resolve_config(
        "WORKDOCS_EMBEDDING_BATCH_ENDPOINT",
        "embedding.batch_endpoint",
        "/v4/embeddings",
    )
    LLM_BATCH_COMPLETION_WINDOW: str = _resolve_config(
        "WORKDOCS_LLM_BATCH_COMPLETION_WINDOW",
        "llm.completion_window",
        "24h",
    )
    BATCH_FILE_DOWNLOAD_TEMPLATE: str = _resolve_config(
        "WORKDOCS_BATCH_FILE_DOWNLOAD_TEMPLATE",
        "batch.download_template",
        "{base_url}/files/{file_id}/content",
    )

    # --- Batch 轮询/超时参数 ---
    BATCH_POLL_INTERVAL: int = 0  # 将在下方初始化
    BATCH_MAX_POLL_RETRIES: int = 0  # 将在下方初始化
    BATCH_MAX_FILE_SIZE_MB: int = 0  # 将在下方初始化
    BATCH_PARALLEL_WORKERS: int = 0  # 将在下方初始化

    # --- Embedding 客户端参数 ---
    EMBED_MAX_RETRIES: int = 0  # 将在下方初始化
    EMBED_RETRY_BACKOFF: int = 0  # 将在下方初始化
    EMBED_TIMEOUT: int = 0  # 将在下方初始化
    EMBED_MAX_BATCH_SIZE: int = 0  # 将在下方初始化

    # --- LLM 客户端参数 ---
    LLM_MAX_RETRIES: int = 0  # 将在下方初始化
    LLM_RETRY_BACKOFF: int = 0  # 将在下方初始化
    LLM_TIMEOUT: int = 0  # 将在下方初始化
    # --- 文件解析参数 ---
    PARSER_TIMEOUT: int = 0  # 将在下方初始化
    PARSER_MAX_RETRIES: int = 0  # 将在下方初始化
    PARSER_POLL_INTERVAL: int = 0  # 将在下方初始化

    # --- Plugin 默认值 ---
    PLUGIN_SEARCH_TOP_K: int = 0  # 将在下方初始化
    PLUGIN_QUERY_TOP_K: int = 0  # 将在下方初始化
    PLUGIN_GRAPH_MAX_DEPTH: int = 0  # 将在下方初始化
    PLUGIN_SUBGRAPH_DEPTH: int = 0  # 将在下方初始化
    PLUGIN_DEFAULT_LIMIT: int = 0  # 将在下方初始化

    # --- Pipeline 业务常量 ---
    DEFAULT_SUMMARY_LENGTH: int = 0  # 将在下方初始化
    GRAPH_MAX_PATH_DEPTH: int = 0  # 将在下方初始化

    # --- 目录配置 ---
    BATCH_TEMP_DIR: str = _resolve_config("WORKDOCS_BATCH_TEMP_DIR", "batch.temp_dir", "batch_temp")
    GRAPH_OUTPUT_DIR: str = _resolve_config(
        "WORKDOCS_GRAPH_OUTPUT_DIR", "graph.output_dir", "graphs"
    )

    @classmethod
    def _initialize_numeric_configs(cls):
        """初始化数值类型的配置（在类定义后调用）."""
        cls.EMBEDDING_DIMENSION = int(
            _resolve_config("WORKDOCS_EMBEDDING_DIMENSION", "embedding.dimension", "1024")
        )
        cls.LLM_BATCH_MAX_CHARS = int(
            _resolve_config("WORKDOCS_LLM_BATCH_MAX_CHARS", "llm.batch_max_chars", "10000")
        )
        cls.LLM_BATCH_TIMEOUT = int(
            _resolve_config("WORKDOCS_LLM_BATCH_TIMEOUT", "llm.batch_timeout", "3600")
        )
        cls.BATCH_SIZE = int(_resolve_config("WORKDOCS_BATCH_SIZE", "batch_size", "4"))
        # Vision 图片配置（LLM multimodal batch）
        cls.LLM_VISION_MAX_EDGE = int(
            _resolve_config("WORKDOCS_LLM_VISION_MAX_EDGE", "llm.vision_max_edge", "1024")
        )
        cls.LLM_VISION_QUALITY = int(
            _resolve_config("WORKDOCS_LLM_VISION_QUALITY", "llm.vision_quality", "85")
        )

        # Batch 轮询/超时
        cls.BATCH_POLL_INTERVAL = int(
            _resolve_config("WORKDOCS_BATCH_POLL_INTERVAL", "batch.poll_interval", "10")
        )
        cls.BATCH_MAX_POLL_RETRIES = int(
            _resolve_config("WORKDOCS_BATCH_MAX_POLL_RETRIES", "batch.max_poll_retries", "360")
        )
        cls.BATCH_MAX_FILE_SIZE_MB = int(
            _resolve_config("WORKDOCS_BATCH_MAX_FILE_SIZE_MB", "batch.max_file_size_mb", "100")
        )
        cls.BATCH_PARALLEL_WORKERS = int(
            _resolve_config("WORKDOCS_BATCH_PARALLEL_WORKERS", "batch.parallel_workers", "4")
        )

        # Embedding 客户端
        cls.EMBED_MAX_RETRIES = int(
            _resolve_config("WORKDOCS_EMBED_MAX_RETRIES", "embedding.max_retries", "3")
        )
        cls.EMBED_RETRY_BACKOFF = int(
            _resolve_config("WORKDOCS_EMBED_RETRY_BACKOFF", "embedding.retry_backoff", "2")
        )
        cls.EMBED_TIMEOUT = int(
            _resolve_config("WORKDOCS_EMBED_TIMEOUT", "embedding.timeout", "120")
        )
        cls.EMBED_MAX_BATCH_SIZE = int(
            _resolve_config("WORKDOCS_EMBED_MAX_BATCH_SIZE", "embedding.max_batch_size", "100")
        )

        # LLM 客户端
        cls.LLM_MAX_RETRIES = int(
            _resolve_config("WORKDOCS_LLM_MAX_RETRIES", "llm.max_retries", "3")
        )
        cls.LLM_RETRY_BACKOFF = int(
            _resolve_config("WORKDOCS_LLM_RETRY_BACKOFF", "llm.retry_backoff", "2")
        )
        cls.LLM_TIMEOUT = int(_resolve_config("WORKDOCS_LLM_TIMEOUT", "llm.timeout", "120"))
        # 文件解析
        cls.PARSER_TIMEOUT = int(_resolve_config("WORKDOCS_PARSER_TIMEOUT", "parser.timeout", "60"))
        cls.PARSER_MAX_RETRIES = int(
            _resolve_config("WORKDOCS_PARSER_MAX_RETRIES", "parser.max_retries", "60")
        )
        cls.PARSER_POLL_INTERVAL = int(
            _resolve_config("WORKDOCS_PARSER_POLL_INTERVAL", "parser.poll_interval", "3")
        )

        # Plugin 默认值
        cls.PLUGIN_SEARCH_TOP_K = int(
            _resolve_config("WORKDOCS_PLUGIN_SEARCH_TOP_K", "plugin.search_top_k", "5")
        )
        cls.PLUGIN_QUERY_TOP_K = int(
            _resolve_config("WORKDOCS_PLUGIN_QUERY_TOP_K", "plugin.query_top_k", "10")
        )
        cls.PLUGIN_GRAPH_MAX_DEPTH = int(
            _resolve_config("WORKDOCS_PLUGIN_GRAPH_MAX_DEPTH", "plugin.graph_max_depth", "3")
        )
        cls.PLUGIN_SUBGRAPH_DEPTH = int(
            _resolve_config("WORKDOCS_PLUGIN_SUBGRAPH_DEPTH", "plugin.subgraph_depth", "1")
        )
        cls.PLUGIN_DEFAULT_LIMIT = int(
            _resolve_config("WORKDOCS_PLUGIN_DEFAULT_LIMIT", "plugin.default_limit", "100")
        )

        # Pipeline 业务常量
        cls.DEFAULT_SUMMARY_LENGTH = int(
            _resolve_config("WORKDOCS_DEFAULT_SUMMARY_LENGTH", "pipeline.summary_length", "200")
        )
        cls.GRAPH_MAX_PATH_DEPTH = int(
            _resolve_config("WORKDOCS_GRAPH_MAX_PATH_DEPTH", "graph.max_path_depth", "6")
        )

    @classmethod
    def to_dict(cls, mask_sensitive: bool = True) -> dict[str, Any]:
        """返回当前所有配置项的字典表示.

        Args:
            mask_sensitive: 若为 True，API Key 等敏感字段以 *** 脱敏显示

        """
        sensitive_keys = {
            "LLM_API_KEY",
            "EMBEDDING_API_KEY",
            "PARSER_API_KEY",
        }
        result: dict[str, Any] = {}
        for key in dir(cls):
            if key.startswith("_"):
                continue
            val = getattr(cls, key)
            if callable(val):
                continue
            if isinstance(val, Path):
                val = str(val)
            if key in sensitive_keys and mask_sensitive and isinstance(val, str) and val:
                result[key] = val[:4] + "***" if len(val) > 4 else "***"
            else:
                result[key] = val
        return result

    @classmethod
    def llm_configured(cls) -> bool:
        """检查 LLM 对话模型是否完整配置."""
        return bool(cls.LLM_MODEL and cls.LLM_API_KEY and cls.LLM_BASE_URL)

    @classmethod
    def embedding_configured(cls) -> bool:
        """检查 Embedding 模型是否完整配置."""
        return bool(cls.EMBEDDING_MODEL and cls.EMBEDDING_API_KEY and cls.EMBEDDING_BASE_URL)

    @classmethod
    def ensure_dirs(cls) -> None:
        """ensure_dirs 函数."""
        cls.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def setup_logging(cls, level: int = logging.INFO) -> None:
        """setup_logging 函数."""
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )


# 初始化数值配置
Config._initialize_numeric_configs()
