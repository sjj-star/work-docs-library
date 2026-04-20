import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_SKILL_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_SKILL_ROOT / ".env", override=False)
load_dotenv(_SKILL_ROOT / "scripts" / ".env", override=True)

logger = logging.getLogger(__name__)


class Config:
    DB_PATH: Path = _SKILL_ROOT / "knowledge_base" / "workdocs.db"
    FAISS_INDEX_PATH: Path = _SKILL_ROOT / "knowledge_base" / "faiss.index"
    ID_MAP_PATH: Path = _SKILL_ROOT / "knowledge_base" / "id_map.json"
    PROMPT_DIR: Path = _SKILL_ROOT / "scripts" / "prompts"
    
    @staticmethod
    def _parse_int_env(name: str, default: int) -> int:
        """安全地解析整数环境变量"""
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
        """安全地解析浮点数环境变量"""
        try:
            value = os.getenv(name)
            if value is None:
                return default
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"环境变量 {name} 值无效 (使用默认值 {default}): {os.getenv(name)}")
            return default

    # LLM 对话模型配置（总结用）
    LLM_PROVIDER: str = os.getenv("WORKDOCS_LLM_PROVIDER", "openai")
    LLM_API_KEY: str = os.getenv("WORKDOCS_LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("WORKDOCS_LLM_BASE_URL", "")
    LLM_MODEL: str = os.getenv("WORKDOCS_LLM_MODEL", "gpt-4o-mini")
    LLM_THINKING_ENABLED: bool = os.getenv("WORKDOCS_LLM_THINKING_ENABLED", "0") == "1"
    
    # Embedding 模型配置（向量化用）- 完全独立
    EMBEDDING_PROVIDER: str = os.getenv("WORKDOCS_EMBEDDING_PROVIDER", "openai")
    EMBEDDING_API_KEY: str = os.getenv("WORKDOCS_EMBEDDING_API_KEY", "")
    EMBEDDING_BASE_URL: str = os.getenv("WORKDOCS_EMBEDDING_BASE_URL", "")
    EMBEDDING_MODEL: str = os.getenv("WORKDOCS_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # 嵌入向量维度配置
    # 此值作为 dimensions 参数传递给 Embedding API
    # 支持的 API 会使用此值，不支持的 API 会忽略它
    # 实际返回的维度由模型决定，首次调用时自动验证
    EMBEDDING_DIMENSION: int = 0  # 将在下方初始化
    
    # 上下文窗口管理配置
    LLM_CONTEXT_MAX_TOKENS: int = 0  # 将在下方初始化
    LLM_CONTEXT_SAFETY_MARGIN: float = 0.0  # 将在下方初始化
    CHUNK_MAX_CHARS: int = 0  # 将在下方初始化
    CONTEXT_STRATEGY: str = os.getenv("WORKDOCS_CONTEXT_STRATEGY", "smart")
    CONTEXT_RECENT_COUNT: int = 0  # 将在下方初始化
    COST_OPTIMIZATION: str = os.getenv("WORKDOCS_COST_OPTIMIZATION", "balanced")
    
    @classmethod
    def _initialize_numeric_configs(cls):
        """初始化数值类型的配置（在类定义后调用）"""
        cls.EMBEDDING_DIMENSION = cls._parse_int_env("WORKDOCS_EMBEDDING_DIMENSION", 1536)
        cls.LLM_CONTEXT_MAX_TOKENS = cls._parse_int_env("WORKDOCS_LLM_CONTEXT_MAX_TOKENS", 6000)
        cls.LLM_CONTEXT_SAFETY_MARGIN = cls._parse_float_env("WORKDOCS_LLM_CONTEXT_SAFETY_MARGIN", 0.9)
        cls.CHUNK_MAX_CHARS = cls._parse_int_env("WORKDOCS_CHUNK_MAX_CHARS", 24000)
        cls.CONTEXT_RECENT_COUNT = cls._parse_int_env("WORKDOCS_CONTEXT_RECENT_COUNT", 3)
        cls.IMAGE_MAX_EDGE = cls._parse_int_env("WORKDOCS_IMAGE_MAX_EDGE", 1024)
        cls.IMAGE_QUALITY = cls._parse_int_env("WORKDOCS_IMAGE_QUALITY", 85)
        cls.BATCH_SIZE = cls._parse_int_env("WORKDOCS_BATCH_SIZE", 4)
        cls.AUTO_VISION = cls._parse_int_env("WORKDOCS_AUTO_VISION", 0)

    @classmethod
    def llm_configured(cls) -> bool:
        """检查 LLM 对话模型是否完整配置"""
        return bool(cls.LLM_MODEL and cls.LLM_API_KEY and cls.LLM_BASE_URL)
    
    @classmethod
    def embedding_configured(cls) -> bool:
        """检查 Embedding 模型是否完整配置"""
        return bool(cls.EMBEDDING_MODEL and cls.EMBEDDING_API_KEY and cls.EMBEDDING_BASE_URL)

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def setup_logging(cls, level: int = logging.INFO) -> None:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )


# 初始化数值配置
Config._initialize_numeric_configs()
