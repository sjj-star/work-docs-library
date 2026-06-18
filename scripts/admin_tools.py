#!/usr/bin/env python3
"""内部管理命令入口.

提供不暴露为 MCP 的数据改写/阶段调试功能，供高级用户手动维护：
- 六阶段 pipeline 单步执行
- 实体/关系增删改
- 文档重处理、全局图谱重建

stdout 输出 JSON 结果，日志输出到 stderr.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is importable when script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import Config
from plugin_router import TOOL_MAP

Config.setup_logging()
logger = logging.getLogger("admin_tools")

# 命令名到 tool 函数名的映射
_ADMIN_COMMANDS: dict[str, str] = {
    "config": "config",
    "status": "status",
    "stage1_parse": "doc_parse",
    "stage2_build_jsonl": "doc_build_batches",
    "stage2_build_batches": "doc_build_batches",  # 兼容别名
    "stage3_submit_batches": "doc_submit_batches",
    "stage4_ingest_results": "doc_ingest_results",
    "stage5_build_embed_jsonl": "doc_build_embed_jsonl",
    "stage6_submit_embed_batches": "doc_submit_embed_batches",
    "reprocess": "reprocess",
    "evaluate": "evaluate",
    "run_eval": "evaluate",  # 兼容别名
    "rebuild_global_graph": "rebuild_global_graph",
    "graph_upsert_entity": "graph_upsert_entity",
    "graph_delete_entity": "graph_delete_entity",
    "graph_upsert_relation": "graph_upsert_relation",
    "graph_delete_relation": "graph_delete_relation",
    "graph_feedback": "graph_feedback",
}


def _json_load(value: str) -> dict:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to the requested admin tool."""
    parser = argparse.ArgumentParser(
        description="Work Docs Library internal admin tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=list(_ADMIN_COMMANDS.keys()),
        help="Admin command to run",
    )
    parser.add_argument(
        "--params",
        type=_json_load,
        default="{}",
        help='JSON parameters for the command (default: "{}")',
    )
    args = parser.parse_args(argv)

    tool_name = _ADMIN_COMMANDS[args.command]
    func = TOOL_MAP[tool_name]

    try:
        result = func(args.params)
    except Exception as e:
        logger.exception("Admin command %s failed", args.command)
        result = {"success": False, "error": str(e)}

    sys.stdout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()
    return 0 if result.get("success", False) else 1


if __name__ == "__main__":
    sys.exit(main())
