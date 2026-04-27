#!/usr/bin/env python3
"""Plugin router for work-docs-library.

Reads JSON parameters from stdin and returns structured JSON via stdout.
Each tool is dispatched by sys.argv[1].
"""

import json
import logging
import os
import sys
from pathlib import Path

from core.config import Config
from core.knowledge_base_service import KnowledgeBaseService

_SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SKILL_ROOT / "scripts"))

# --- Auto-switch to venv Python if available ---
_VENV_PYTHON = _SKILL_ROOT / "venv" / "bin" / "python3"
_VENV_PYTHON_ALT = _SKILL_ROOT / "venv" / "bin" / "python"
_venv_pythons = [p for p in (_VENV_PYTHON, _VENV_PYTHON_ALT) if p.exists()]
if _venv_pythons and sys.executable not in {str(p) for p in _venv_pythons}:
    os.execv(str(_venv_pythons[0]), [str(_venv_pythons[0])] + sys.argv)

# 延迟初始化 logging：只在实际运行时配置，避免测试导入时污染全局 logging 状态
if __name__ == "__main__":
    Config.setup_logging()
    # Redirect all root log handlers to stderr so stdout stays pure JSON
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.stream = sys.stderr

logger = logging.getLogger("plugin_router")


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------


def _chunk_to_dict(ck, preview_len: int = 500) -> dict:
    """将 Chunk 对象序列化为字典（用于 Plugin 返回）."""
    return {
        "doc_id": ck.doc_id,
        "chunk_id": ck.chunk_id,
        "chunk_type": ck.chunk_type,
        "chapter_title": ck.chapter_title,
        "content_preview": ck.content[:preview_len],
        "summary": ck.summary,
        "keywords": ck.keywords,
    }


def _entity_to_dict(e) -> dict:
    """将 GraphEntity 序列化为字典."""
    return {
        "type": e.entity_type,
        "name": e.name,
        "properties": e.properties,
        "source_doc_ids": sorted(list(e.source_doc_ids)),
        "source_chapter": e.source_chapter,
    }


def _relation_to_dict(r) -> dict:
    """将 GraphRelation 序列化为字典."""
    return {
        "type": r.rel_type,
        "from": r.from_name,
        "to": r.to_name,
        "from_type": r.from_type,
        "to_type": r.to_type,
        "properties": r.properties,
        "source_doc_ids": sorted(list(r.source_doc_ids)),
    }


# ---------------------------------------------------------------------------
# Tool 实现
# ---------------------------------------------------------------------------


def _get_service() -> KnowledgeBaseService:
    """获取 KnowledgeBaseService 实例（确保目录存在）."""
    Config.ensure_dirs()
    return KnowledgeBaseService()


def tool_ingest(params: dict) -> dict:
    """导入文档."""
    path = params.get("path")
    if not path:
        return {"success": False, "error": "Missing required parameter: path"}
    dry_run = params.get("dry_run", False)

    svc = _get_service()
    doc_ids = svc.ingest_document(str(path), dry_run=dry_run)
    if not doc_ids:
        return {"success": True, "doc_ids": [], "message": "No documents found or ingested."}
    return {
        "success": True,
        "doc_ids": doc_ids,
        "message": f"Ingested {len(doc_ids)} document(s).",
    }


def tool_search(params: dict) -> dict:
    """语义向量搜索."""
    text = params.get("text")
    if not text:
        return {"success": False, "error": "Missing required parameter: text"}
    top_k = params.get("top_k", 5)

    svc = _get_service()
    try:
        results = svc.search_semantic(str(text), top_k=top_k)
    except RuntimeError as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "results": [{"score": r["score"], **_chunk_to_dict(r["chunk"])} for r in results],
    }


def tool_query(params: dict) -> dict:
    """结构化 chunk 查询."""
    doc_id = params.get("doc_id")
    chapter = params.get("chapter")
    chapter_regex = params.get("chapter_regex")
    keyword = params.get("keyword")
    concept = params.get("concept")
    top_k = params.get("top_k", 10)

    svc = _get_service()
    try:
        chunks = svc.query_chunks(
            doc_id=doc_id,
            chapter=chapter,
            chapter_regex=chapter_regex,
            keyword=keyword,
            concept=concept,
            top_k=top_k,
        )
    except ValueError as e:
        return {"success": False, "error": str(e)}

    return {"success": True, "results": [_chunk_to_dict(ck) for ck in chunks]}


def tool_status(params: dict) -> dict:
    """列出所有文档."""
    svc = _get_service()
    docs = svc.list_documents()
    return {
        "success": True,
        "documents": [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "status": d.status,
                "total_pages": d.total_pages,
                "extracted_at": d.extracted_at,
            }
            for d in docs
        ],
    }


def tool_toc(params: dict) -> dict:
    """获取文档目录或按标题搜索."""
    svc = _get_service()
    doc_id = params.get("doc_id")
    match = params.get("match")

    if doc_id:
        doc = svc.get_document(doc_id)
        if not doc:
            return {"success": False, "error": f"Document {doc_id} not found."}
        chapters = svc.db.get_chapters(doc_id)
        return {
            "success": True,
            "doc_id": doc_id,
            "title": doc.title,
            "total_pages": doc.total_pages,
            "status": doc.status,
            "chapters": [
                {
                    "title": ch.title,
                    "start_page": ch.start_page,
                    "end_page": ch.end_page,
                    "level": ch.level,
                }
                for ch in chapters
            ],
        }
    elif match:
        docs = svc.db.search_documents_by_title(match)
        return {
            "success": True,
            "match": match,
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "status": d.status,
                    "total_pages": d.total_pages,
                }
                for d in docs
                if d
            ],
        }
    else:
        return {"success": False, "error": "Provide either doc_id or match."}


def tool_progress(params: dict) -> dict:
    """获取文档处理进度."""
    doc_id = params.get("doc_id")
    if not doc_id:
        return {"success": False, "error": "Missing required parameter: doc_id"}

    svc = _get_service()
    stats = svc.get_document_progress(doc_id)
    return {"success": True, "doc_id": doc_id, **stats}


def tool_reprocess(params: dict) -> dict:
    """强制重新处理文档."""
    doc_id = params.get("doc_id")
    if not doc_id:
        return {"success": False, "error": "Missing required parameter: doc_id"}

    svc = _get_service()
    try:
        new_id = svc.reprocess_document(doc_id)
        return {"success": True, "doc_id": new_id, "message": "Reprocessed."}
    except (ValueError, RuntimeError) as e:
        return {"success": False, "error": str(e)}


def tool_get_content(params: dict) -> dict:
    """获取 chunk 完整内容."""
    doc_id = params.get("doc_id")
    chapter = params.get("chapter")
    chunk_db_id = params.get("chunk_db_id")

    svc = _get_service()
    try:
        result = svc.get_chunk_content(
            chunk_db_id=chunk_db_id,
            doc_id=doc_id,
            chapter=chapter,
        )
    except ValueError as e:
        return {"success": False, "error": str(e)}

    chunks = result["chunks"]
    chunk_meta = [
        {
            "chunk_db_id": ck.id,
            "chunk_id": ck.chunk_id,
            "chapter_title": ck.chapter_title,
        }
        for ck in chunks
    ]

    first = chunks[0]
    return {
        "success": True,
        "query_type": result["query_type"],
        "doc_id": first.doc_id,
        "chapter_title": first.chapter_title,
        "content": result["content"],
        "total_chars": len(result["content"]),
        "chunks": chunk_meta,
    }


# ---------------------------------------------------------------------------
# 图谱查询工具
# ---------------------------------------------------------------------------


def tool_graph_query(params: dict) -> dict:
    """查询图谱实体.

    参数:
        entity_type: 实体类型（如 Module, Signal, Register）
        name: 精确名称匹配
        name_pattern: 名称模糊匹配（子串，大小写不敏感）
    """
    entity_type = params.get("entity_type")
    name = params.get("name")
    name_pattern = params.get("name_pattern")

    svc = _get_service()

    if name and entity_type:
        # 精确查询
        entity = svc.get_entity(entity_type, name)
        if not entity:
            return {"success": True, "count": 0, "entities": []}
        return {"success": True, "count": 1, "entities": [_entity_to_dict(entity)]}

    # 搜索查询
    entities = svc.find_entities(entity_type=entity_type, name_pattern=name_pattern)
    return {
        "success": True,
        "count": len(entities),
        "entities": [_entity_to_dict(e) for e in entities],
    }


def tool_graph_neighbors(params: dict) -> dict:
    """查询实体的邻居节点.

    参数:
        entity_type: 实体类型
        name: 实体名称
        rel_type: 关系类型过滤（可选）
        direction: 方向 out/in/both（默认 out）
    """
    entity_type = params.get("entity_type")
    name = params.get("name")
    if not entity_type or not name:
        return {"success": False, "error": "Missing entity_type or name"}

    rel_type = params.get("rel_type")
    direction = params.get("direction", "out")

    svc = _get_service()
    neighbors = svc.get_neighbors(entity_type, name, rel_type, direction)
    return {
        "success": True,
        "center": {"type": entity_type, "name": name},
        "direction": direction,
        "count": len(neighbors),
        "neighbors": [
            {
                "entity": _entity_to_dict(entity),
                "relation": rel,
            }
            for entity, rel in neighbors
        ],
    }


def tool_graph_path(params: dict) -> dict:
    """查找两实体间的路径.

    参数:
        from_type, from_name: 起点实体
        to_type, to_name: 终点实体
        max_depth: 最大搜索深度（默认 3，最大 6）
    """
    from_type = params.get("from_type")
    from_name = params.get("from_name")
    to_type = params.get("to_type")
    to_name = params.get("to_name")
    if not all([from_type, from_name, to_type, to_name]):
        return {"success": False, "error": "Missing from/to entity parameters"}

    max_depth = params.get("max_depth", 3)
    svc = _get_service()
    paths = svc.find_path(str(from_type), str(from_name), str(to_type), str(to_name), max_depth)

    # 将节点 ID 解析为 (type, name)
    def _parse_nid(nid: str) -> dict:
        parts = nid.split("::", 1)
        return {"type": parts[0], "name": parts[1] if len(parts) > 1 else nid}

    return {
        "success": True,
        "from": {"type": from_type, "name": from_name},
        "to": {"type": to_type, "name": to_name},
        "max_depth": max_depth,
        "path_count": len(paths),
        "paths": [[_parse_nid(nid) for nid in p] for p in paths],
    }


def tool_graph_subgraph(params: dict) -> dict:
    """提取以某实体为中心的子图.

    参数:
        center_type, center_name: 中心实体
        depth: 搜索深度（默认 1）
        rel_types: 关系类型过滤列表（可选）
    """
    center_type = params.get("center_type")
    center_name = params.get("center_name")
    if not center_type or not center_name:
        return {"success": False, "error": "Missing center_type or center_name"}

    depth = params.get("depth", 1)
    rel_types = set(params.get("rel_types", [])) if params.get("rel_types") else None

    svc = _get_service()
    subgraph = svc.get_subgraph(center_type, center_name, depth, rel_types)

    return {
        "success": True,
        "center": {"type": center_type, "name": center_name},
        "depth": depth,
        "node_count": subgraph.node_count,
        "edge_count": subgraph.edge_count,
        "entities": [_entity_to_dict(e) for e in subgraph.entities()],
        "relations": [_relation_to_dict(r) for r in subgraph.relations()],
    }


# ---------------------------------------------------------------------------
# Tool 映射
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "ingest": tool_ingest,
    "search": tool_search,
    "query": tool_query,
    "status": tool_status,
    "toc": tool_toc,
    "progress": tool_progress,
    "reprocess": tool_reprocess,
    "get_content": tool_get_content,
    "graph_query": tool_graph_query,
    "graph_neighbors": tool_graph_neighbors,
    "graph_path": tool_graph_path,
    "graph_subgraph": tool_graph_subgraph,
}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            json.dumps(
                {"success": False, "error": "Missing tool name argument"}, ensure_ascii=False
            )
        )
        sys.exit(1)

    tool_name = sys.argv[1]
    func = TOOL_MAP.get(tool_name)
    if not func:
        print(
            json.dumps(
                {"success": False, "error": f"Unknown tool: {tool_name}"}, ensure_ascii=False
            )
        )
        sys.exit(1)

    try:
        params = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except json.JSONDecodeError as e:
        print(
            json.dumps({"success": False, "error": f"Invalid JSON input: {e}"}, ensure_ascii=False)
        )
        sys.exit(1)

    try:
        result = func(params)
    except Exception as e:
        logger.exception("Tool %s failed", tool_name)
        result = {"success": False, "error": str(e)}

    print(json.dumps(result, ensure_ascii=False))
