import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_SKILL_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_SKILL_ROOT / ".env", override=False)
load_dotenv(_SKILL_ROOT / "scripts" / ".env", override=True)


class Config:
    DB_PATH: Path = _SKILL_ROOT / "knowledge_base" / "workdocs.db"
    FAISS_INDEX_PATH: Path = _SKILL_ROOT / "knowledge_base" / "faiss.index"
    ID_MAP_PATH: Path = _SKILL_ROOT / "knowledge_base" / "id_map.json"
    PROMPT_DIR: Path = _SKILL_ROOT / "scripts" / "prompts"

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
    EMBEDDING_DIM: int = int(os.getenv("WORKDOCS_EMBEDDING_DIM", "1536"))

    @classmethod
    def llm_configured(cls) -> bool:
        """检查 LLM 对话模型是否完整配置"""
        return bool(cls.LLM_MODEL and cls.LLM_API_KEY and cls.LLM_BASE_URL)
    
    @classmethod
    def embedding_configured(cls) -> bool:
        """检查 Embedding 模型是否完整配置"""
        return bool(cls.EMBEDDING_MODEL and cls.EMBEDDING_API_KEY and cls.EMBEDDING_BASE_URL)

    IMAGE_MAX_EDGE: int = int(os.getenv("WORKDOCS_IMAGE_MAX_EDGE", "1024"))
    IMAGE_QUALITY: int = int(os.getenv("WORKDOCS_IMAGE_QUALITY", "85"))
    BATCH_SIZE: int = int(os.getenv("WORKDOCS_BATCH_SIZE", "4"))
    AUTO_VISION: int = int(os.getenv("WORKDOCS_AUTO_VISION", "0"))

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
