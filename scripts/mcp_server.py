#!/usr/bin/env python3
"""轻量级 stdio MCP server，为 work-docs-library 暴露精简后的 Agent 工具.

stdout 仅用于 MCP JSON-RPC 消息；所有日志/调试信息必须写入 stderr.
"""

import json
import logging
import sys
from typing import Any

from core.config import Config
from plugin_router import (
    tool_explore,
    tool_ingest,
    tool_read,
    tool_search,
    tool_status,
)

Config.setup_logging()
logger = logging.getLogger("mcp_server")

# 仅暴露适合 Agent 自主调用的 5 个公共工具
MCP_TOOL_MAP: dict[str, Any] = {
    "ingest": tool_ingest,
    "search": tool_search,
    "explore": tool_explore,
    "read": tool_read,
    "status": tool_status,
}

# 与 MCP 新接口对齐的参数 schema
MCP_TOOL_SCHEMAS: dict[str, dict] = {
    "ingest": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件或目录路径"},
            "dry_run": {"type": "boolean", "description": "预览模式，不调用 API", "default": False},
        },
        "required": ["path"],
    },
    "search": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "搜索文本"},
            "top_k": {
                "type": "integer",
                "description": "返回结果数量",
                "default": Config.PLUGIN_SEARCH_TOP_K,
            },
            "mode": {
                "type": "string",
                "description": "搜索模式：semantic / hybrid / reranked",
                "enum": ["semantic", "hybrid", "reranked"],
                "default": "hybrid",
            },
            "include_graph": {
                "type": "boolean",
                "description": "是否扩展关联知识图谱",
                "default": True,
            },
            "graph_depth": {
                "type": "integer",
                "description": "图谱扩展深度",
                "default": Config.PLUGIN_SUBGRAPH_DEPTH,
            },
            "rerank_candidate_k": {
                "type": "integer",
                "description": "reranked 模式候选集大小（可选）",
            },
            "session_id": {
                "type": "string",
                "description": "可选会话跟踪 ID，用于 status scope=trace 回放路径",
            },
        },
        "required": ["text"],
    },
    "explore": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "description": "探索模式",
                "enum": ["entity", "neighbors", "subgraph", "path", "provenance", "conflicts"],
            },
            "entity_type": {"type": "string", "description": "实体类型"},
            "name": {"type": "string", "description": "实体名称"},
            "name_pattern": {"type": "string", "description": "名称模糊匹配"},
            "doc_id": {"type": "string", "description": "限定文档 ID"},
            "rel_type": {"type": "string", "description": "关系类型过滤"},
            "direction": {
                "type": "string",
                "description": "邻居方向",
                "enum": ["out", "in", "both"],
                "default": "out",
            },
            "depth": {
                "type": "integer",
                "description": "邻居/子图深度",
                "default": Config.PLUGIN_SUBGRAPH_DEPTH,
            },
            "rel_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "子图关系类型过滤列表",
            },
            "from_type": {"type": "string", "description": "路径起点实体类型"},
            "from_name": {"type": "string", "description": "路径起点实体名称"},
            "to_type": {"type": "string", "description": "路径终点实体类型"},
            "to_name": {"type": "string", "description": "路径终点实体名称"},
            "max_depth": {
                "type": "integer",
                "description": "路径最大搜索深度",
                "default": Config.PLUGIN_GRAPH_MAX_DEPTH,
            },
            "limit": {
                "type": "integer",
                "description": "最大返回数量",
                "default": Config.PLUGIN_DEFAULT_LIMIT,
            },
            "session_id": {
                "type": "string",
                "description": "可选会话跟踪 ID，用于 status scope=trace 回放路径",
            },
        },
        "required": ["mode"],
    },
    "read": {
        "type": "object",
        "description": (
            "读取内容。必须提供 chunk_db_id，或提供 doc_id 与 "
            "chapter/chapter_regex/concept 之一"
        ),
        "properties": {
            "chunk_db_id": {"type": "integer", "description": "Block 数据库 ID"},
            "doc_id": {
                "type": "string",
                "description": "文档 ID（与 chapter/chapter_regex/concept 组合使用）",
            },
            "chapter": {"type": "string", "description": "章节标题子串匹配"},
            "chapter_regex": {"type": "string", "description": "章节标题正则匹配"},
            "concept": {"type": "string", "description": "概念名匹配"},
            "with_entities": {
                "type": "boolean",
                "description": "是否同时返回关联图谱实体/关系",
                "default": True,
            },
            "session_id": {
                "type": "string",
                "description": "可选会话跟踪 ID，用于 status scope=trace 回放路径",
            },
        },
    },
    "status": {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "description": "文档 ID，提供时返回详情"},
            "scope": {
                "type": "string",
                "description": "状态仪表盘范围",
                "enum": [
                    "overview",
                    "documents",
                    "vectors",
                    "graph",
                    "blocks",
                    "headings",
                    "conflicts",
                    "feedback",
                    "config",
                    "quality",
                    "ingest_pipeline",
                    "toc",
                    "trace",
                    "usage",
                    "all",
                ],
                "default": "overview",
            },
            "top_n": {"type": "integer", "description": "最近处理文档数量", "default": 20},
            "session_id": {"type": "string", "description": "scope=trace 时过滤会话路径"},
        },
    },
}


def _send_response(response: dict) -> None:
    """将 JSON-RPC 响应写入 stdout 并立即刷新."""
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _handle_initialize(request_id: Any) -> None:
    _send_response(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "workdocs", "version": "1.2.0"},
            },
        }
    )


def _handle_tools_list(request_id: Any) -> None:
    tools = [
        {
            "name": name,
            "description": func.__doc__.strip().split("\n")[0] if func.__doc__ else name,
            "inputSchema": MCP_TOOL_SCHEMAS[name],
        }
        for name, func in MCP_TOOL_MAP.items()
    ]
    _send_response({"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools}})


def _handle_tool_call(request_id: Any, params: dict) -> None:
    name = params.get("name", "")
    arguments = params.get("arguments") or {}
    func = MCP_TOOL_MAP.get(name)
    if func is None:
        _send_response(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {"success": False, "error": f"Unknown tool: {name}"},
                                ensure_ascii=False,
                            ),
                        }
                    ],
                    "isError": True,
                },
            }
        )
        return

    try:
        result = func(arguments)
        text = json.dumps(result, ensure_ascii=False, default=str)
        _send_response(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": text}],
                    "isError": not bool(result.get("success", False)),
                },
            }
        )
    except Exception as e:  # pragma: no cover - catch-all safety net
        logger.exception("MCP tool call failed: %s", name)
        text = json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
        _send_response(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": text}], "isError": True},
            }
        )


def _process_request(request: dict) -> None:
    method = request.get("method")
    request_id = request.get("id")  # notifications have no id

    if method == "initialize":
        _handle_initialize(request_id)
        return

    if method == "notifications/initialized":
        # no response required
        return

    if method == "tools/list":
        _handle_tools_list(request_id)
        return

    if method == "tools/call":
        _handle_tool_call(request_id, request.get("params", {}))
        return

    # unknown method
    if request_id is not None:
        _send_response(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        )


def main() -> None:
    """从 stdin 读取 JSON-RPC 请求，stdout 返回响应."""
    logger.info("workdocs MCP server started")
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON-RPC request: %s", e)
                continue
            _process_request(request)
    except Exception:
        logger.exception("MCP server crashed")
        raise


if __name__ == "__main__":
    main()
