"""conftest 模块."""

import logging
import sys
from pathlib import Path

import pytest

# Ensure scripts/ is on path for imports
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))


@pytest.fixture(autouse=True)
def _set_caplog_level(caplog):
    caplog.set_level(logging.INFO)
