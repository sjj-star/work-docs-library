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

    LLM_PROVIDER: str = os.getenv("WORKDOCS_LLM_PROVIDER", "openai")
    LLM_API_KEY: str = os.getenv("WORKDOCS_LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("WORKDOCS_LLM_BASE_URL", "")
    LLM_MODEL: str = os.getenv("WORKDOCS_LLM_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("WORKDOCS_EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIM: int = int(os.getenv("WORKDOCS_EMBEDDING_DIM", "1536"))

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
