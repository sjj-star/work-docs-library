"""config 模块."""

import json
import logging
import os
import sys
from pathlib import Path

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
        if isinstance(data, (str, int, float, bool)):
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

    @staticmethod
    def _parse_int_env(name: str, default: int) -> int:
        """安全地解析整数环境变量."""
        try:
            value = os.getenv(name)
            if value is None:
                return default
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"环境变量 {name} 值无效 (使用默认值 {default}): {os.getenv(name)}")
            return default

    @staticmethod
    def _parse_float_env(name: str, default: float) -> float:
        """安全地解析浮点数环境变量."""
        try:
            value = os.getenv(name)
            if value is None:
                return default
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"环境变量 {name} 值无效 (使用默认值 {default}): {os.getenv(name)}")
            return default

    # LLM 对话模型配置（总结用）
    # LLM 对话模型配置（总结用）
    LLM_PROVIDER: str = _resolve_config("WORKDOCS_LLM_PROVIDER", "llm.provider", "kimi")
    LLM_API_KEY: str = _resolve_config("WORKDOCS_LLM_API_KEY", "llm.api_key", "")
    LLM_BASE_URL: str = _resolve_config(
        "WORKDOCS_LLM_BASE_URL", "llm.endpoint", "https://api.moonshot.cn/v1"
    )
    LLM_MODEL: str = _resolve_config("WORKDOCS_LLM_MODEL", "llm.model", "kimi-k2.5")
    LLM_THINKING_ENABLED: bool = (
        _resolve_config("WORKDOCS_LLM_THINKING_ENABLED", "llm.thinking_enabled", "0") == "1"
    )

    # Embedding 模型配置（向量化用）- 完全独立
    EMBEDDING_PROVIDER: str = _resolve_config(
        "WORKDOCS_EMBEDDING_PROVIDER", "embedding.provider", "bigmodel"
    )
    EMBEDDING_API_KEY: str = _resolve_config("WORKDOCS_EMBEDDING_API_KEY", "embedding.api_key", "")
    EMBEDDING_BASE_URL: str = _resolve_config(
        "WORKDOCS_EMBEDDING_BASE_URL", "embedding.endpoint", "https://open.bigmodel.cn/api/paas/v4"
    )
    EMBEDDING_MODEL: str = _resolve_config(
        "WORKDOCS_EMBEDDING_MODEL", "embedding.model", "embedding-3"
    )

    # BigModel (智谱) 文件解析配置
    BIGMODEL_API_KEY: str = _resolve_config("WORKDOCS_BIGMODEL_API_KEY", "bigmodel.api_key", "")

    # 嵌入向量维度配置
    EMBEDDING_DIMENSION: int = 0  # 将在下方初始化

    # LLM Batch 处理配置
    LLM_BATCH_MAX_CHARS: int = 0  # 将在下方初始化
    LLM_BATCH_TIMEOUT: int = 0  # 将在下方初始化
    # 向量化 chunk 最大字符数（独立配置，embedding 模型上下文通常较短）
    EMBED_BATCH_MAX_CHARS: int = 0  # 将在下方初始化

    @classmethod
    def _initialize_numeric_configs(cls):
        """初始化数值类型的配置（在类定义后调用）."""
        cls.EMBEDDING_DIMENSION = int(
            _resolve_config("WORKDOCS_EMBEDDING_DIMENSION", "embedding.dimension", "1024")
        )
        cls.LLM_BATCH_MAX_CHARS = int(_resolve_config("WORKDOCS_LLM_BATCH_MAX_CHARS", "", "10000"))
        cls.LLM_BATCH_TIMEOUT = int(_resolve_config("WORKDOCS_LLM_BATCH_TIMEOUT", "", "3600"))
        cls.EMBED_BATCH_MAX_CHARS = int(
            _resolve_config("WORKDOCS_EMBED_BATCH_MAX_CHARS", "", "4000")
        )
        cls.BATCH_SIZE = int(_resolve_config("WORKDOCS_BATCH_SIZE", "batch_size", "4"))
        # Vision 图片配置（LLM multimodal batch）
        cls.LLM_VISION_MAX_EDGE = int(_resolve_config("WORKDOCS_LLM_VISION_MAX_EDGE", "", "1024"))
        cls.LLM_VISION_QUALITY = int(_resolve_config("WORKDOCS_LLM_VISION_QUALITY", "", "85"))

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
