"""config 模块.

配置来源：
1. 环境变量（.env 文件或系统环境变量，如 WORKDOCS_LLM_API_KEY）
2. 代码硬编码默认值
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_SKILL_ROOT = Path(__file__).resolve().parent.parent.parent

# 加载 .env（不覆盖已有环境变量，因为外部可能已注入凭证）
load_dotenv(_SKILL_ROOT / ".env", override=False)
load_dotenv(_SKILL_ROOT / "scripts" / ".env", override=False)

logger = logging.getLogger(__name__)


def _resolve_config(env_name: str, default: str = "") -> str:
    """按优先级解析配置值.

    1. 环境变量 env_name（.env 文件或系统环境变量）
    2. 默认值
    """
    env_val = os.getenv(env_name, "")
    if env_val:
        return env_val
    return default


class Config:
    """Config 类."""

    DB_PATH: Path = _SKILL_ROOT / "knowledge_base" / "workdocs.db"
    FAISS_INDEX_PATH: Path = _SKILL_ROOT / "knowledge_base" / "faiss.index"
    PROMPT_DIR: Path = _SKILL_ROOT / "scripts" / "prompts"

    # LLM 对话模型配置（重排序 / Agentic 规划 / 评估 Judge / Chat 模式实体提取）
    LLM_API_KEY: str = _resolve_config("WORKDOCS_LLM_API_KEY", "")
    LLM_BASE_URL: str = _resolve_config("WORKDOCS_LLM_BASE_URL", "https://api.moonshot.cn/v1")
    LLM_MODEL: str = _resolve_config("WORKDOCS_LLM_MODEL", "kimi-k2.5")
    LLM_THINKING_ENABLED: bool = _resolve_config("WORKDOCS_LLM_THINKING_ENABLED", "0") == "1"
    LLM_MODE: str = _resolve_config("WORKDOCS_LLM_MODE", "batch")

    # Embedding 模型配置（向量化用）- 完全独立
    EMBEDDING_API_KEY: str = _resolve_config("WORKDOCS_EMBEDDING_API_KEY", "")
    EMBEDDING_BASE_URL: str = _resolve_config(
        "WORKDOCS_EMBEDDING_BASE_URL",
        "https://open.bigmodel.cn/api/paas/v4",
    )
    EMBEDDING_MODEL: str = _resolve_config("WORKDOCS_EMBEDDING_MODEL", "embedding-3")

    # 文件解析配置
    PARSER_API_KEY: str = _resolve_config("WORKDOCS_PARSER_API_KEY", "")
    PARSER_BASE_URL: str = _resolve_config(
        "WORKDOCS_PARSER_BASE_URL",
        "https://open.bigmodel.cn/api/paas/v4",
    )

    # 嵌入向量维度配置
    EMBEDDING_DIMENSION: int = 0  # 将在下方初始化

    # LLM Batch 处理配置
    LLM_BATCH_MAX_CHARS: int = 0  # 将在下方初始化
    LLM_BATCH_TIMEOUT: int = 0  # 将在下方初始化

    # Content Block 存储粒度配置（向量化粒度，字符数限制）
    BLOCK_MAX_CHARS: int = 0  # 将在下方初始化

    # --- API Endpoint 配置（服务商无感化） ---
    # LLM Batch 请求体中的 endpoint 由 LLM_BASE_URL 路径 + LLM_CHAT_ENDPOINT 自动推导，
    # 无需单独配置。保留 LLM_CHAT_ENDPOINT 即可同时驱动同步 Chat 与 Batch 两种模式。
    LLM_CHAT_ENDPOINT: str = _resolve_config("WORKDOCS_LLM_CHAT_ENDPOINT", "/chat/completions")
    EMBEDDING_ENDPOINT: str = _resolve_config(
        "WORKDOCS_EMBEDDING_ENDPOINT",
        "/embeddings",
    )
    LLM_BATCH_COMPLETION_WINDOW: str = _resolve_config(
        "WORKDOCS_LLM_BATCH_COMPLETION_WINDOW",
        "24h",
    )
    BATCH_FILE_DOWNLOAD_TEMPLATE: str = _resolve_config(
        "WORKDOCS_BATCH_FILE_DOWNLOAD_TEMPLATE",
        "{base_url}/files/{file_id}/content",
    )

    # --- Batch 轮询/超时参数 ---
    BATCH_POLL_INTERVAL: int = 0  # 将在下方初始化
    BATCH_MAX_POLL_RETRIES: int = 0  # 将在下方初始化
    BATCH_MAX_FILE_SIZE_MB: int = 0  # 将在下方初始化
    BATCH_PARALLEL_WORKERS: int = 0  # 将在下方初始化

    # --- LLM 客户端参数 ---
    # 注：LLM/Embedding 的重试与超时统一由下方 HTTP_* 配置驱动（见 core/api_client.py）。
    # LLM_TIMEOUT 单独用于 LLM 同步对话请求的超时。
    LLM_TIMEOUT: int = 0  # 将在下方初始化
    # Chat 模式动态超时参数（仅当 LLM_MODE=chat 时生效）
    LLM_TIMEOUT_PER_10K_CHARS: int = 0  # 将在下方初始化
    LLM_TIMEOUT_PER_IMAGE: int = 0  # 将在下方初始化
    LLM_TIMEOUT_MAX: int = 0  # 将在下方初始化
    # --- 统一 HTTP 客户端重试配置 ---
    HTTP_TIMEOUT: int = 0  # 将在下方初始化
    HTTP_RETRY_MAX_ATTEMPTS: int = 0  # 将在下方初始化
    HTTP_RETRY_BASE_DELAY: float = 0.0  # 将在下方初始化
    HTTP_RETRY_MAX_DELAY: float = 0.0  # 将在下方初始化
    HTTP_RETRY_JITTER: bool = False  # 将在下方初始化
    HTTP_RETRY_RESPECT_RETRY_AFTER: bool = False  # 将在下方初始化
    # --- Embedding 输入长度保护 ---
    EMBED_MAX_CHARS_PER_TEXT: int = 0  # 将在下方初始化
    EMBED_SPLIT_OVERLONG: bool = False  # 将在下方初始化
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

    # 使用日志清理策略
    USAGE_LOG_MAX_DAYS: int = 0  # 将在下方初始化
    USAGE_LOG_MAX_ROWS: int = 0  # 将在下方初始化

    # BM25 / hybrid retrieval config
    PLUGIN_BM25_TOP_K: int = 0  # 将在下方初始化
    PLUGIN_HYBRID_RRF_K: int = 0  # 将在下方初始化

    # --- Pipeline 业务常量 ---
    GRAPH_MAX_PATH_DEPTH: int = 0  # 将在下方初始化

    # --- 目录配置 ---
    PARSE_OUTPUT_DIR: str = _resolve_config("WORKDOCS_PARSE_OUTPUT_DIR", "parsed")
    BATCH_OUTPUT_DIR: str = _resolve_config("WORKDOCS_BATCH_OUTPUT_DIR", "batch")
    GRAPH_OUTPUT_DIR: str = _resolve_config("WORKDOCS_GRAPH_OUTPUT_DIR", "graphs")

    @classmethod
    def _initialize_numeric_configs(cls):
        """初始化数值类型的配置（在类定义后调用）."""
        cls.EMBEDDING_DIMENSION = int(_resolve_config("WORKDOCS_EMBEDDING_DIMENSION", "1024"))
        cls.LLM_BATCH_MAX_CHARS = int(_resolve_config("WORKDOCS_LLM_BATCH_MAX_CHARS", "10000"))
        cls.BLOCK_MAX_CHARS = int(_resolve_config("WORKDOCS_BLOCK_MAX_CHARS", "6000"))
        cls.LLM_BATCH_TIMEOUT = int(_resolve_config("WORKDOCS_LLM_BATCH_TIMEOUT", "3600"))
        # Image 图片处理配置
        cls.IMAGE_MAX_SIZE = int(_resolve_config("WORKDOCS_IMAGE_MAX_SIZE", "1024"))
        cls.IMAGE_QUALITY = int(_resolve_config("WORKDOCS_IMAGE_QUALITY", "80"))
        cls.IMAGE_GRAYSCALE_QUALITY = int(_resolve_config("WORKDOCS_IMAGE_GRAYSCALE_QUALITY", "75"))
        cls.IMAGE_GRAYSCALE_CHROMA_DIST = float(
            _resolve_config(
                "WORKDOCS_IMAGE_GRAYSCALE_CHROMA_DIST",
                "15.0",
            )
        )
        cls.IMAGE_GRAYSCALE_LOW_CHROMA_RATIO = float(
            _resolve_config(
                "WORKDOCS_IMAGE_GRAYSCALE_LOW_CHROMA_RATIO",
                "0.95",
            )
        )
        cls.IMAGE_BLACKWHITE_EDGE_RATIO = float(
            _resolve_config(
                "WORKDOCS_IMAGE_BLACKWHITE_EDGE_RATIO",
                "0.90",
            )
        )

        # Batch 轮询/超时
        cls.BATCH_POLL_INTERVAL = int(_resolve_config("WORKDOCS_BATCH_POLL_INTERVAL", "10"))
        cls.BATCH_MAX_POLL_RETRIES = int(_resolve_config("WORKDOCS_BATCH_MAX_POLL_RETRIES", "360"))
        cls.BATCH_MAX_FILE_SIZE_MB = int(_resolve_config("WORKDOCS_BATCH_MAX_FILE_SIZE_MB", "100"))
        cls.BATCH_PARALLEL_WORKERS = int(_resolve_config("WORKDOCS_BATCH_PARALLEL_WORKERS", "4"))

        # LLM 客户端
        cls.LLM_TIMEOUT = int(_resolve_config("WORKDOCS_LLM_TIMEOUT", "300"))
        cls.LLM_TIMEOUT_PER_10K_CHARS = int(
            _resolve_config("WORKDOCS_LLM_TIMEOUT_PER_10K_CHARS", "60")
        )
        cls.LLM_TIMEOUT_PER_IMAGE = int(_resolve_config("WORKDOCS_LLM_TIMEOUT_PER_IMAGE", "30"))
        cls.LLM_TIMEOUT_MAX = int(_resolve_config("WORKDOCS_LLM_TIMEOUT_MAX", "1800"))
        # 统一 HTTP 客户端重试配置
        cls.HTTP_TIMEOUT = int(_resolve_config("WORKDOCS_HTTP_TIMEOUT", "120"))
        cls.HTTP_RETRY_MAX_ATTEMPTS = int(_resolve_config("WORKDOCS_HTTP_RETRY_MAX_ATTEMPTS", "3"))
        cls.HTTP_RETRY_BASE_DELAY = float(_resolve_config("WORKDOCS_HTTP_RETRY_BASE_DELAY", "1.0"))
        cls.HTTP_RETRY_MAX_DELAY = float(_resolve_config("WORKDOCS_HTTP_RETRY_MAX_DELAY", "60.0"))
        cls.HTTP_RETRY_JITTER = (
            _resolve_config("WORKDOCS_HTTP_RETRY_JITTER", "true").lower() == "true"
        )
        cls.HTTP_RETRY_RESPECT_RETRY_AFTER = (
            _resolve_config("WORKDOCS_HTTP_RETRY_RESPECT_RETRY_AFTER", "true").lower() == "true"
        )
        cls.EMBED_MAX_CHARS_PER_TEXT = int(
            _resolve_config("WORKDOCS_EMBED_MAX_CHARS_PER_TEXT", "8192")
        )
        cls.EMBED_SPLIT_OVERLONG = (
            _resolve_config("WORKDOCS_EMBED_SPLIT_OVERLONG", "true").lower() == "true"
        )
        # 文件解析
        cls.PARSER_TIMEOUT = int(_resolve_config("WORKDOCS_PARSER_TIMEOUT", "60"))
        cls.PARSER_MAX_RETRIES = int(_resolve_config("WORKDOCS_PARSER_MAX_RETRIES", "60"))
        cls.PARSER_POLL_INTERVAL = int(_resolve_config("WORKDOCS_PARSER_POLL_INTERVAL", "3"))

        # PDF Parser 参数（Milestone 1-3 配置化）
        cls.PARSER_MIN_IMAGE_WIDTH = int(_resolve_config("WORKDOCS_PARSER_MIN_IMAGE_WIDTH", "100"))
        cls.PARSER_MIN_IMAGE_HEIGHT = int(
            _resolve_config("WORKDOCS_PARSER_MIN_IMAGE_HEIGHT", "100")
        )
        cls.PARSER_PAGE_RENDER_DPI = int(_resolve_config("WORKDOCS_PARSER_PAGE_RENDER_DPI", "200"))
        cls.PARSER_TABLE_DETECTION_ENABLED = (
            _resolve_config("WORKDOCS_PARSER_TABLE_DETECTION_ENABLED", "true").lower() == "true"
        )
        cls.PARSER_TABLE_OVERLAP_THRESHOLD = float(
            _resolve_config("WORKDOCS_PARSER_TABLE_OVERLAP_THRESHOLD", "0.5")
        )
        cls.PARSER_TABLE_MIN_ROWS = int(_resolve_config("WORKDOCS_PARSER_TABLE_MIN_ROWS", "2"))
        cls.PARSER_TABLE_MIN_COLS = int(_resolve_config("WORKDOCS_PARSER_TABLE_MIN_COLS", "2"))
        cls.PARSER_TABLE_MIN_HEIGHT_PT = float(
            _resolve_config("WORKDOCS_PARSER_TABLE_MIN_HEIGHT_PT", "20.0")
        )
        cls.PARSER_TABLE_MIN_WIDTH_RATIO = float(
            _resolve_config("WORKDOCS_PARSER_TABLE_MIN_WIDTH_RATIO", "0.15")
        )
        cls.PARSER_TAB_MERGE_THRESHOLD_PT = float(
            _resolve_config("WORKDOCS_PARSER_TAB_MERGE_THRESHOLD_PT", "4.0")
        )
        cls.PARSER_IMAGE_SIZE_LIMIT = float(
            _resolve_config("WORKDOCS_PARSER_IMAGE_SIZE_LIMIT", "0.05")
        )
        cls.PARSER_MAX_IMAGES_PER_PAGE = int(
            _resolve_config("WORKDOCS_PARSER_MAX_IMAGES_PER_PAGE", "30")
        )
        # GapsFirstScanner 参数（Caption-driven 提取器）
        cls.PARSER_FIGURE_MIN_SCORE = float(
            _resolve_config("WORKDOCS_PARSER_FIGURE_MIN_SCORE", "2.0")
        )
        cls.PARSER_EDGE_LABEL_MAX_LEN = int(
            _resolve_config("WORKDOCS_PARSER_EDGE_LABEL_MAX_LEN", "30")
        )
        # Plugin 默认值
        cls.PLUGIN_SEARCH_TOP_K = int(_resolve_config("WORKDOCS_PLUGIN_SEARCH_TOP_K", "5"))
        cls.PLUGIN_QUERY_TOP_K = int(_resolve_config("WORKDOCS_PLUGIN_QUERY_TOP_K", "10"))
        cls.PLUGIN_GRAPH_MAX_DEPTH = int(_resolve_config("WORKDOCS_PLUGIN_GRAPH_MAX_DEPTH", "3"))
        cls.PLUGIN_SUBGRAPH_DEPTH = int(_resolve_config("WORKDOCS_PLUGIN_SUBGRAPH_DEPTH", "1"))
        cls.PLUGIN_DEFAULT_LIMIT = int(_resolve_config("WORKDOCS_PLUGIN_DEFAULT_LIMIT", "100"))

        # 使用日志清理策略
        cls.USAGE_LOG_MAX_DAYS = int(_resolve_config("WORKDOCS_USAGE_LOG_MAX_DAYS", "30"))
        cls.USAGE_LOG_MAX_ROWS = int(_resolve_config("WORKDOCS_USAGE_LOG_MAX_ROWS", "10000"))

        # BM25 / hybrid retrieval config
        cls.PLUGIN_BM25_TOP_K = int(_resolve_config("WORKDOCS_PLUGIN_BM25_TOP_K", "50"))
        cls.PLUGIN_HYBRID_RRF_K = int(_resolve_config("WORKDOCS_PLUGIN_HYBRID_RRF_K", "60"))

        # Pipeline 业务常量
        cls.GRAPH_MAX_PATH_DEPTH = int(_resolve_config("WORKDOCS_GRAPH_MAX_PATH_DEPTH", "6"))

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
            # Sensitive keys are always masked regardless of the caller's request.
            if key in sensitive_keys and isinstance(val, str) and val:
                result[key] = "***"
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
            stream=sys.stderr,
        )


# 初始化数值配置
Config._initialize_numeric_configs()
