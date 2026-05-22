"""conftest 模块."""

import logging
import os
import sys
from pathlib import Path

import pytest

# Ensure scripts/ is on path for imports
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# 模块级别环境隔离：必须在任何测试模块（及被测代码）导入之前清理环境变量，
# 并阻止 dotenv 重新加载 .env 文件，否则 config.py 等模块会在 fixture 运行前
# 就被 .env 值固化。
# ---------------------------------------------------------------------------
_TEST_ENV_BACKUP: dict[str, str] = {}
for _key in list(os.environ.keys()):
    if _key.startswith("WORKDOCS_") or "." in _key:
        _TEST_ENV_BACKUP[_key] = os.environ[_key]
        del os.environ[_key]

# 阻止 load_dotenv 在测试中重新加载 .env 文件
# 必须在 core.config 等任何使用 dotenv 的模块导入之前执行
import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *args, **kwargs: None  # type: ignore[method-assign]

# 创建测试级临时目录，并重定向 Config 默认路径，防止未显式隔离的测试
# 意外加载生产环境数据（faiss.index / workdocs.db / global.json）
import tempfile  # noqa: E402

_TEST_TMP_DIR = Path(tempfile.mkdtemp(prefix="workdocs_test_"))

from core.config import Config  # noqa: E402

Config.DB_PATH = _TEST_TMP_DIR / "workdocs.db"
Config.FAISS_INDEX_PATH = _TEST_TMP_DIR / "faiss.index"
Config.ID_MAP_PATH = _TEST_TMP_DIR / "id_map.json"
Config.GRAPH_OUTPUT_DIR = str(_TEST_TMP_DIR / "graphs")


@pytest.fixture(autouse=True, scope="session")
def _restore_test_env():
    """测试会话结束后恢复原始环境变量."""
    yield
    for _key, _value in _TEST_ENV_BACKUP.items():
        os.environ[_key] = _value
    for _key in list(os.environ.keys()):
        if (_key.startswith("WORKDOCS_") or "." in _key) and _key not in _TEST_ENV_BACKUP:
            del os.environ[_key]


@pytest.fixture(autouse=True)
def _set_caplog_level(caplog):
    caplog.set_level(logging.INFO)
