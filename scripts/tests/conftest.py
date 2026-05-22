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
# 否则 config.py 等模块会在 fixture 运行前就被 .env 值固化。
# ---------------------------------------------------------------------------
_TEST_ENV_BACKUP: dict[str, str] = {}
for _key in list(os.environ.keys()):
    if _key.startswith("WORKDOCS_") or "." in _key:
        _TEST_ENV_BACKUP[_key] = os.environ[_key]
        del os.environ[_key]


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
