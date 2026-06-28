#!/usr/bin/env python3
"""Plugin router for work-docs-library.

Reads JSON parameters from stdin and returns structured JSON via stdout.
Each tool is dispatched by sys.argv[1].
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.config import Config
from core.graph_store import GraphEntity, GraphRelation
from core.knowledge_base_service import KnowledgeBaseService
from core.query_service import QueryService
from core.status_collector import StatusCollector

_SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SKILL_ROOT / "scripts"))

# --- Auto-switch to venv Python if available ---
_VENV_PYTHON = _SKILL_ROOT / ".venv" / "bin" / "python3"
_VENV_PYTHON_ALT = _SKILL_ROOT / ".venv" / "bin" / "python"
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


#: 评估工具支持的检索器类型。
_EVALUATION_RETRIEVERS: set[str] = {"semantic", "hybrid", "reranked"}

#: MCP 搜索工具支持的模式。
_SEARCH_MODES: set[str] = {"semantic", "hybrid", "reranked"}

#: MCP 探索工具支持的模式。
_EXPLORE_MODES: set[str] = {
    "entity",
    "neighbors",
    "subgraph",
    "path",
    "provenance",
    "conflicts",
}


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
    }


def _entity_to_dict(e) -> dict:
    """将 GraphEntity 序列化为字典."""
    return {
        "entity_type": e.entity_type,
        "name": e.name,
        "properties": e.properties,
        "doc_properties": e.doc_properties,
        "source_doc_ids": sorted(list(e.source_doc_ids)),
        "source_chapter": e.source_chapter,
        "confidence": e.confidence,
        "verified": e.verified,
        "created_at": e.created_at,
        "updated_at": e.updated_at,
        "feedback_score": e.feedback_score,
    }


def _relation_to_dict(r) -> dict:
    """将 GraphRelation 序列化为字典."""
    return {
        "rel_type": r.rel_type,
        "from_name": r.from_name,
        "to_name": r.to_name,
        "from_type": r.from_type,
        "to_type": r.to_type,
        "properties": r.properties,
        "source_doc_ids": sorted(list(r.source_doc_ids)),
        "confidence": r.confidence,
        "verified": r.verified,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
        "feedback_score": r.feedback_score,
    }


# ---------------------------------------------------------------------------
# Tool 实现
# ---------------------------------------------------------------------------


def _get_service() -> KnowledgeBaseService:
    """获取 KnowledgeBaseService 实例（确保目录存在）."""
    Config.ensure_dirs()
    return KnowledgeBaseService()


def _get_query_service() -> QueryService:
    """获取 QueryService 实例（底层使用 KnowledgeBaseService）."""
    return QueryService(_get_service())


def _resolve_allowed_path(path: str | Path | None, base_dirs: list[Path] | None = None) -> Path:
    """Resolve a user-provided path and enforce a path sandbox.

    Relative paths are resolved against ``Path.cwd()``. The resolved path must
    be contained within at least one of ``base_dirs``.
    """
    if not path:
        raise ValueError("Path is empty or not provided")
    resolved = Path(path).expanduser().resolve()
    if base_dirs is None:
        base_dirs = [Path.cwd(), Config.DB_PATH.parent, Path(tempfile.gettempdir())]
    if not any(resolved.is_relative_to(b) for b in base_dirs):
        allowed = ", ".join(str(b) for b in base_dirs)
        raise ValueError(f"Path {resolved} is outside allowed directories: {allowed}")
    return resolved


# ---------------------------------------------------------------------------
# 文档导入工具
# ---------------------------------------------------------------------------


def tool_ingest(params: dict) -> dict:
    """导入文档."""
    try:
        path = _resolve_allowed_path(params.get("path"))
    except ValueError as e:
        return {"success": False, "error": f"Unsafe path: {e}"}
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


def tool_doc_parse(params: dict) -> dict:
    """阶段1: PDF → Markdown."""
    try:
        path = _resolve_allowed_path(params.get("path"))
        output_dir_param = params.get("output_dir")
        if output_dir_param:
            _resolve_allowed_path(output_dir_param)
    except ValueError as e:
        return {"success": False, "error": f"Unsafe path: {e}"}

    from core.doc_graph_pipeline import DocGraphPipeline

    pipe = DocGraphPipeline()
    try:
        doc_id, output_dir, text, images = pipe.stage1_parse(str(path))
        return {
            "success": True,
            "doc_id": doc_id,
            "output_dir": str(output_dir),
            "chars": len(text),
            "images": len(images),
            "message": f"解析完成，可手动编辑 {output_dir}/result.md 后执行 doc_build_batches",
        }
    except Exception as e:
        logger.exception("doc_parse failed")
        return {"success": False, "error": str(e)}


def tool_doc_build_batches(params: dict) -> dict:
    """阶段2: Markdown → Batch JSONL."""
    doc_id = params.get("doc_id")
    if not doc_id:
        return {"success": False, "error": "Missing required parameter: doc_id"}

    from core.doc_graph_pipeline import DocGraphPipeline

    pipe = DocGraphPipeline()
    try:
        jsonl_path, batches, requests, _content_blocks, _heading_maps = pipe.stage2_build_jsonl(
            doc_id, max_chars=params.get("max_chars")
        )
        return {
            "success": True,
            "doc_id": doc_id,
            "jsonl_path": str(jsonl_path),
            "batch_count": len(batches),
            "request_count": len(requests),
            "message": "JSONL 已生成，可审查后执行 doc_submit_batches",
        }
    except Exception as e:
        logger.exception("doc_build_batches failed")
        return {"success": False, "error": str(e)}


def tool_doc_submit_batches(params: dict) -> dict:
    """阶段3: 提交 Batch API 并保存结果."""
    doc_id = params.get("doc_id")
    if not doc_id:
        return {"success": False, "error": "Missing required parameter: doc_id"}

    from core.doc_graph_pipeline import DocGraphPipeline

    pipe = DocGraphPipeline()
    try:
        file_path = params.get("file_path")
        if file_path:
            file_path = _resolve_allowed_path(file_path)
        else:
            doc = pipe.db.get_document(doc_id)
            if doc:
                file_path = _resolve_allowed_path(doc.source_path)
            else:
                return {
                    "success": False,
                    "error": f"无法找到文档 {doc_id} 的源文件路径，请提供 file_path 参数",
                }

        jsonl_path_param = params.get("jsonl_path")
        jsonl_path = _resolve_allowed_path(jsonl_path_param) if jsonl_path_param else None

        results_path = pipe.stage3_submit_batches(
            doc_id=doc_id,
            file_path=str(file_path),
            jsonl_path=jsonl_path,
            force=params.get("force", False),
        )
        return {
            "success": True,
            "doc_id": doc_id,
            "results_path": str(results_path),
            "message": (
                f"Batch 已提交，结果已保存至 {results_path}，执行 doc_ingest_results 完成入库"
            ),
        }
    except Exception as e:
        logger.exception("doc_submit_batches failed")
        return {"success": False, "error": str(e)}


def tool_doc_ingest_results(params: dict) -> dict:
    """阶段4: 从 Batch 结果文件解析并入库."""
    doc_id = params.get("doc_id")
    if not doc_id:
        return {"success": False, "error": "Missing required parameter: doc_id"}

    from core.doc_graph_pipeline import DocGraphPipeline

    pipe = DocGraphPipeline()
    try:
        file_path = params.get("file_path")
        if file_path:
            file_path = _resolve_allowed_path(file_path)
        else:
            doc = pipe.db.get_document(doc_id)
            if doc:
                file_path = _resolve_allowed_path(doc.source_path)
            else:
                return {
                    "success": False,
                    "error": f"无法找到文档 {doc_id} 的源文件路径，请提供 file_path 参数",
                }

        results_path_param = params.get("results_path")
        if results_path_param:
            results_path = _resolve_allowed_path(results_path_param)
        else:
            results_path = (
                Path(Config.DB_PATH).parent / Config.BATCH_OUTPUT_DIR / f"{doc_id}_results.jsonl"
            )

        result_doc_id = pipe.stage4_ingest_results(
            doc_id=doc_id,
            file_path=str(file_path),
            results_path=results_path,
            force=params.get("force", False),
        )
        return {
            "success": True,
            "doc_id": result_doc_id,
            "message": "文档已入库完成",
        }
    except Exception as e:
        logger.exception("doc_ingest_results failed")
        return {"success": False, "error": str(e)}


def tool_doc_build_embed_jsonl(params: dict) -> dict:
    """阶段5: 从 SQLite content_blocks 构建 Embedding Batch JSONL（本地，不调用 API）."""
    doc_id = params.get("doc_id")
    if not doc_id:
        return {"success": False, "error": "Missing required parameter: doc_id"}

    from core.doc_graph_pipeline import DocGraphPipeline

    pipe = DocGraphPipeline()
    try:
        embed_jsonl_path = pipe.stage5_build_embed_jsonl(doc_id)
        return {
            "success": True,
            "doc_id": doc_id,
            "embed_jsonl_path": str(embed_jsonl_path),
            "message": (
                f"Embedding JSONL 已生成，可审查后执行 doc_submit_embed_batches | "
                f"path={embed_jsonl_path}"
            ),
        }
    except Exception as e:
        logger.exception("doc_build_embed_jsonl failed")
        return {"success": False, "error": str(e)}


def tool_doc_submit_embed_batches(params: dict) -> dict:
    """阶段6: 提交 Embedding Batch API 并解析结果入库."""
    doc_id = params.get("doc_id")
    if not doc_id:
        return {"success": False, "error": "Missing required parameter: doc_id"}

    from core.doc_graph_pipeline import DocGraphPipeline

    pipe = DocGraphPipeline()
    try:
        embed_jsonl_path_param = params.get("embed_jsonl_path")
        embed_jsonl_path = (
            _resolve_allowed_path(embed_jsonl_path_param) if embed_jsonl_path_param else None
        )

        result_doc_id = pipe.stage6_submit_embed_batches(doc_id, embed_jsonl_path)
        return {
            "success": True,
            "doc_id": result_doc_id,
            "message": "Embedding 向量化入库完成",
        }
    except Exception as e:
        logger.exception("doc_submit_embed_batches failed")
        return {"success": False, "error": str(e)}


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


def tool_evaluate(params: dict) -> dict:
    """运行评估数据集.

    参数:
        dataset_name: 评估数据集名称（required）
        retriever: 检索器类型，默认 "semantic"
        top_k: 返回结果数量，默认由 Config.PLUGIN_SEARCH_TOP_K 决定
    """
    dataset_name = params.get("dataset_name")
    if not dataset_name:
        return {"success": False, "error": "Missing required parameter: dataset_name"}

    retriever = params.get("retriever", "semantic")
    if retriever not in _EVALUATION_RETRIEVERS:
        return {"success": False, "error": f"Unsupported retriever: {retriever}"}

    svc = _get_service()
    try:
        result = svc.evaluate_dataset(
            dataset_name=dataset_name,
            retriever=retriever,
            top_k=params.get("top_k", Config.PLUGIN_SEARCH_TOP_K),
        )
        return {"success": True, "dataset_name": dataset_name, **result}
    except Exception as e:
        logger.exception("evaluate failed")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# 状态与调试工具
# ---------------------------------------------------------------------------


def tool_status(params: dict) -> dict:
    """列出文档、查看文档详情，或返回知识库各维度结构化状态.

    参数:
        doc_id: 可选，提供时返回该文档的详细状态和进度统计（与 scope=overview 兼容）
        scope: 状态维度，可选值：
            overview（默认）, documents, vectors, graph, blocks, headings,
            conflicts, feedback, config, quality, ingest_pipeline, toc, all
        top_n: 列表类数据默认返回条数（默认 20）
    """
    svc = _get_service()
    doc_id = params.get("doc_id")
    scope = params.get("scope", "overview")
    top_n = params.get("top_n", 20)
    if not isinstance(top_n, int) or top_n < 0:
        top_n = 20

    if scope == "overview":
        if doc_id:
            doc = svc.get_document(doc_id)
            if not doc:
                return {"success": False, "error": f"Document {doc_id} not found."}
            stats = svc.get_document_progress(doc_id)
            return {
                "success": True,
                "doc_id": doc_id,
                "title": doc.title,
                "status": doc.status,
                "total_pages": doc.total_pages,
                **stats,
            }

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

    if scope == "toc":
        if not doc_id:
            return {"success": False, "error": "Missing required parameter: doc_id for scope=toc"}
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

    if scope == "config":
        return tool_config({})

    collector = StatusCollector(svc)
    try:
        if scope == "documents":
            return collector.collect_documents_status(top_n)
        if scope == "vectors":
            return collector.collect_vectors_status()
        if scope == "graph":
            return collector.collect_graph_status()
        if scope == "blocks":
            return collector.collect_blocks_status(top_n)
        if scope == "headings":
            return collector.collect_headings_status()
        if scope == "conflicts":
            return collector.collect_conflicts_status(top_n)
        if scope == "feedback":
            return collector.collect_feedback_status(top_n)
        if scope == "quality":
            return collector.collect_quality_status()
        if scope == "ingest_pipeline":
            return collector.collect_ingest_pipeline_status()
        if scope == "all":
            return collector.collect_all(top_n)
    except Exception as e:
        logger.exception("status scope=%s failed", scope)
        return {"success": False, "error": f"Status collection failed for scope '{scope}': {e}"}

    return {"success": False, "error": f"Unknown scope: {scope}"}


# ---------------------------------------------------------------------------
# 查询工具（新 MCP 接口）
# ---------------------------------------------------------------------------


def tool_search(params: dict) -> dict:
    """统一搜索入口：语义 / 混合 / 重排序，可选联合知识图谱.

    参数:
        text: 搜索文本（required）
        top_k: 返回结果数量（默认 PLUGIN_SEARCH_TOP_K）
        mode: 搜索模式，semantic | hybrid | reranked（默认 hybrid）
        include_graph: 是否扩展关联图谱（默认 true）
        graph_depth: 图谱扩展深度（默认 PLUGIN_SUBGRAPH_DEPTH）
        rerank_candidate_k: reranked 模式候选集大小（可选）
    """
    text = params.get("text")
    if not text:
        return {"success": False, "error": "Missing required parameter: text"}

    mode = params.get("mode", "hybrid")
    if mode not in _SEARCH_MODES:
        return {"success": False, "error": f"Unsupported search mode: {mode}"}

    top_k = params.get("top_k", Config.PLUGIN_SEARCH_TOP_K)
    include_graph = params.get("include_graph", True)
    graph_depth = params.get("graph_depth", Config.PLUGIN_SUBGRAPH_DEPTH)
    rerank_candidate_k = params.get("rerank_candidate_k")

    qs = _get_query_service()
    try:
        result = qs.search(
            text=str(text),
            top_k=top_k,
            mode=mode,
            include_graph=include_graph,
            graph_depth=graph_depth,
            rerank_candidate_k=rerank_candidate_k,
        )
    except Exception as e:
        logger.exception("search failed")
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "text": text,
        "mode": mode,
        "chunks": [
            {"score": c["score"], **_chunk_to_dict(c["chunk"])} for c in result["chunks"]
        ],
        "entities": [_entity_to_dict(e) for e in result["entities"]],
        "relations": [_relation_to_dict(r) for r in result["relations"]],
        "source_documents": result["source_documents"],
    }


def tool_explore(params: dict) -> dict:
    """统一图谱探索入口.

    参数:
        mode: 探索模式（required）
            entity | neighbors | subgraph | path | provenance | conflicts
        其余参数按模式传递.
    """
    mode = params.get("mode")
    if not mode:
        return {"success": False, "error": "Missing required parameter: mode"}
    if mode not in _EXPLORE_MODES:
        return {"success": False, "error": f"Unsupported explore mode: {mode}"}

    qs = _get_query_service()
    try:
        return qs.explore(**params)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.exception("explore failed")
        return {"success": False, "error": str(e)}


def tool_read(params: dict) -> dict:
    """统一内容读取入口（按 chunk_db_id 或章节/概念查询）.

    参数:
        chunk_db_id: 直接按 block DB ID 查询
        doc_id: 文档 ID（chapter/chapter_regex/concept 查询必需）
        chapter: 章节标题子串匹配
        chapter_regex: 章节标题正则匹配
        concept: 概念名匹配
        with_entities: 是否同时返回关联实体/关系（默认 true）
    """
    chunk_db_id = params.get("chunk_db_id")
    doc_id = params.get("doc_id")
    chapter = params.get("chapter")
    chapter_regex = params.get("chapter_regex")
    concept = params.get("concept")
    with_entities = params.get("with_entities", True)

    if (
        chunk_db_id is None
        and chapter is None
        and chapter_regex is None
        and concept is None
    ):
        return {
            "success": False,
            "error": "Provide chunk_db_id, or doc_id with chapter/chapter_regex/concept",
        }

    qs = _get_query_service()
    try:
        result = qs.read(
            doc_id=doc_id,
            chapter=chapter,
            chapter_regex=chapter_regex,
            concept=concept,
            chunk_db_id=chunk_db_id,
            with_entities=with_entities,
        )
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.exception("read failed")
        return {"success": False, "error": str(e)}

    result["entities"] = [_entity_to_dict(e) for e in result.get("entities", [])]
    result["relations"] = [_relation_to_dict(r) for r in result.get("relations", [])]
    return {"success": True, **result}


# ---------------------------------------------------------------------------
# 图谱写操作工具
# ---------------------------------------------------------------------------


def tool_graph_upsert_entity(params: dict) -> dict:
    """添加或更新知识图谱实体.

    参数:
        entity_type: 实体类型
        name: 实体名称
        properties: 属性字典（可选）
        source_doc_ids: 来源文档 ID 列表（可选）
        confidence: 置信度 0.0-1.0（默认 1.0）
        verified: 是否已人工验证（可选，不提供时保持原有值）
    """
    entity_type = params.get("entity_type")
    name = params.get("name")
    if not entity_type or not name:
        return {"success": False, "error": "Missing entity_type or name"}

    svc = _get_service()
    existing = svc.get_entity(entity_type, name)

    if existing:
        # 更新模式（原 graph_update_entity + graph_verify_entity）
        ok = svc.update_entity(
            entity_type=entity_type,
            name=name,
            properties=params.get("properties"),
            confidence=params.get("confidence"),
            verified=params.get("verified"),
        )
        if not ok:
            return {"success": False, "error": f"Entity {entity_type}::{name} not found"}
        return {
            "success": True,
            "entity_type": entity_type,
            "name": name,
            "mode": "updated",
            "message": "Updated.",
        }

    # 创建模式（原 graph_add_entity）
    entity = GraphEntity(
        entity_type=entity_type,
        name=name,
        properties=params.get("properties") or {},
        source_doc_ids=set(params.get("source_doc_ids") or []),
        confidence=params.get("confidence", 1.0),
        verified=params.get("verified", False),
    )
    conflicts = svc.add_entity(entity)
    return {
        "success": True,
        "entity": _entity_to_dict(entity),
        "conflicts": conflicts,
        "mode": "created",
        "message": f"Entity {entity_type}::{name} created. Conflicts: {len(conflicts)}",
    }


def tool_graph_delete_entity(params: dict) -> dict:
    """删除知识图谱中的实体（级联删除关联边）.

    参数:
        entity_type: 实体类型
        name: 实体名称
    """
    entity_type = params.get("entity_type")
    name = params.get("name")
    if not entity_type or not name:
        return {"success": False, "error": "Missing entity_type or name"}

    svc = _get_service()
    ok = svc.delete_entity(entity_type, name)
    if not ok:
        return {"success": False, "error": f"Entity {entity_type}::{name} not found"}
    return {"success": True, "entity_type": entity_type, "name": name, "message": "Deleted."}


def tool_graph_upsert_relation(params: dict) -> dict:
    """添加或更新知识图谱中的关系.

    参数:
        rel_type: 关系类型
        from_type, from_name: 起点实体
        to_type, to_name: 终点实体
        properties: 关系属性字典（可选）
        confidence: 置信度（默认 1.0）
        verified: 是否已验证（默认 false）
    """
    rel_type = params.get("rel_type")
    from_type = params.get("from_type")
    from_name = params.get("from_name")
    to_type = params.get("to_type")
    to_name = params.get("to_name")
    if not all([rel_type, from_type, from_name, to_type, to_name]):
        return {"success": False, "error": "Missing required relation parameters"}

    relation = GraphRelation(
        rel_type=str(rel_type),
        from_type=str(from_type),
        from_name=str(from_name),
        to_type=str(to_type),
        to_name=str(to_name),
        properties=params.get("properties") or {},
        confidence=params.get("confidence", 1.0),
        verified=params.get("verified", False),
    )
    svc = _get_service()
    conflicts = svc.add_relation(relation)
    return {
        "success": True,
        "relation": _relation_to_dict(relation),
        "conflicts": conflicts,
        "message": f"Relation {rel_type} added/updated. Conflicts: {len(conflicts)}",
    }


def tool_graph_delete_relation(params: dict) -> dict:
    """删除知识图谱中的关系.

    参数:
        rel_type: 关系类型
        from_type, from_name: 起点实体
        to_type, to_name: 终点实体
    """
    rel_type = params.get("rel_type")
    from_type = params.get("from_type")
    from_name = params.get("from_name")
    to_type = params.get("to_type")
    to_name = params.get("to_name")
    if not all([rel_type, from_type, from_name, to_type, to_name]):
        return {"success": False, "error": "Missing required relation parameters"}

    svc = _get_service()
    ok = svc.delete_relation(
        str(from_type), str(from_name), str(to_type), str(to_name), str(rel_type)
    )
    if not ok:
        return {"success": False, "error": "Relation not found"}
    return {
        "success": True,
        "rel_type": rel_type,
        "from_name": f"{from_type}::{from_name}",
        "to_name": f"{to_type}::{to_name}",
        "message": "Deleted.",
    }


# ---------------------------------------------------------------------------
# 反馈与冲突工具
# ---------------------------------------------------------------------------


def tool_graph_feedback(params: dict) -> dict:
    """提交或查询对知识图谱实体/关系的反馈.

    参数:
        action: "submit" 或 "query"（默认 submit）
        -- submit 模式 --
        rating: 评分 +1（正确）或 -1（错误）
        entity_type, entity_name: 目标实体（可选）
        relation_type, from_type, from_name, to_type, to_name: 目标关系（可选）
        comment: 评论说明（可选）
        -- query 模式 --
        entity_type, entity_name: 过滤条件（可选）
        limit: 最大返回数量（默认 100）
    """
    action = params.get("action", "submit")
    svc = _get_service()

    if action == "submit":
        rating = params.get("rating")
        if rating not in (1, -1):
            return {"success": False, "error": "rating must be +1 or -1"}
        feedback_id = svc.submit_feedback(
            rating=rating,
            entity_type=params.get("entity_type"),
            entity_name=params.get("entity_name"),
            relation_type=params.get("relation_type"),
            relation_from_type=params.get("from_type"),
            relation_from_name=params.get("from_name"),
            relation_to_type=params.get("to_type"),
            relation_to_name=params.get("to_name"),
            comment=params.get("comment", ""),
        )
        return {
            "success": True,
            "feedback_id": feedback_id,
            "message": "Feedback submitted.",
        }
    elif action == "query":
        feedbacks = svc.get_feedback(
            entity_type=params.get("entity_type"),
            entity_name=params.get("entity_name"),
            limit=params.get("limit", Config.PLUGIN_DEFAULT_LIMIT),
        )
        return {"success": True, "count": len(feedbacks), "feedbacks": feedbacks}
    else:
        return {"success": False, "error": "action must be 'submit' or 'query'"}


def tool_graph_conflicts(params: dict) -> dict:
    """查询知识图谱冲突日志（同名实体属性覆盖记录）.

    参数:
        entity_type: 实体类型过滤
        name: 实体名称过滤
        limit: 最大返回数量（默认 100）
    """
    svc = _get_service()
    logs = svc.get_conflict_logs(
        entity_type=params.get("entity_type"),
        name=params.get("name"),
        limit=params.get("limit", Config.PLUGIN_DEFAULT_LIMIT),
    )
    return {"success": True, "count": len(logs), "logs": logs}


# ---------------------------------------------------------------------------
# 维护与溯源工具
# ---------------------------------------------------------------------------


def tool_graph_provenance(params: dict) -> dict:
    """查询知识图谱实体的来源溯源（实体 → 来源 block → 原始文档）.

    参数:
        entity_type: 实体类型（required）
        name: 实体名称（required）
        doc_id: 可选，限定只查某文档的来源
    """
    entity_type = params.get("entity_type")
    name = params.get("name")
    if not entity_type or not name:
        return {"success": False, "error": "Missing entity_type or name"}

    doc_id = params.get("doc_id")
    svc = _get_service()
    entity = svc.get_entity(entity_type, name)
    if not entity:
        return {"success": True, "count": 0, "sources": []}

    sources: list[dict[str, Any]] = []
    target_doc_ids = [doc_id] if doc_id else sorted(entity.source_doc_ids)

    for src_doc_id in target_doc_ids:
        doc = svc.get_document(src_doc_id)
        doc_title = doc.title if doc else src_doc_id

        # 通过桥接索引反向查找包含该实体的 blocks（O(1) 而非 O(N) 扫描）
        chunks = svc.find_chunks_by_entity(entity_type, name, doc_id=src_doc_id)
        for ck in chunks:
            for me in ck.metadata.get("extracted_entities", []):
                if me.get("type") == entity_type and me.get("name") == name:
                    props_snapshot: dict[str, Any] = {}
                    if entity.doc_properties and src_doc_id in entity.doc_properties:
                        props_snapshot = dict(entity.doc_properties[src_doc_id])
                    sources.append(
                        {
                            "doc_id": src_doc_id,
                            "doc_title": doc_title,
                            "chunk_db_id": ck.id,
                            "chapter_title": ck.chapter_title,
                            "confidence": me.get("confidence", entity.confidence),
                            "properties_snapshot": props_snapshot,
                        }
                    )
                    break

    return {
        "success": True,
        "entity_type": entity_type,
        "name": name,
        "count": len(sources),
        "sources": sources,
    }


def tool_rebuild_global_graph(params: dict) -> dict:
    """全量重建全局知识图谱（清空后从所有子图重新加载，用于修复不一致）."""
    svc = _get_service()
    stats = svc.rebuild_global_graph()
    return {
        "success": True,
        "message": f"全局图谱已重建 | nodes={stats['nodes']} | edges={stats['edges']}",
        "stats": stats,
    }


def tool_config(params: dict) -> dict:
    """打印当前生效的配置参数值（支持敏感字段脱敏）."""
    from core.config import Config

    # Always mask sensitive keys regardless of caller request (defense in depth).
    config_dict = Config.to_dict(mask_sensitive=True)

    groups: dict[str, dict[str, Any]] = {
        "LLM 配置": {},
        "图片配置": {},
        "Embedding 配置": {},
        "解析器配置": {},
        "Batch 配置": {},
        "Plugin 配置": {},
        "Pipeline 配置": {},
        "路径配置": {},
        "其他": {},
    }

    group_map: dict[str, str] = {
        "LLM_API_KEY": "LLM 配置",
        "LLM_BASE_URL": "LLM 配置",
        "LLM_MODEL": "LLM 配置",
        "LLM_THINKING_ENABLED": "LLM 配置",
        "LLM_MODE": "LLM 配置",
        "LLM_BATCH_ENDPOINT": "LLM 配置",
        "LLM_BATCH_MAX_CHARS": "LLM 配置",
        "LLM_BATCH_TIMEOUT": "LLM 配置",
        "LLM_BATCH_COMPLETION_WINDOW": "LLM 配置",
        "LLM_MAX_RETRIES": "LLM 配置",
        "LLM_RETRY_BACKOFF": "LLM 配置",
        "LLM_TIMEOUT": "LLM 配置",
        "IMAGE_MAX_SIZE": "图片配置",
        "IMAGE_QUALITY": "图片配置",
        "IMAGE_GRAYSCALE_QUALITY": "图片配置",
        "IMAGE_GRAYSCALE_CHROMA_DIST": "图片配置",
        "IMAGE_GRAYSCALE_LOW_CHROMA_RATIO": "图片配置",
        "IMAGE_BLACKWHITE_EDGE_RATIO": "图片配置",
        "EMBEDDING_API_KEY": "Embedding 配置",
        "EMBEDDING_BASE_URL": "Embedding 配置",
        "EMBEDDING_MODEL": "Embedding 配置",
        "EMBEDDING_ENDPOINT": "Embedding 配置",
        "EMBEDDING_DIMENSION": "Embedding 配置",
        "EMBED_MAX_RETRIES": "Embedding 配置",
        "EMBED_RETRY_BACKOFF": "Embedding 配置",
        "EMBED_TIMEOUT": "Embedding 配置",
        "PARSER_API_KEY": "解析器配置",
        "PARSER_BASE_URL": "解析器配置",
        "PARSER_TIMEOUT": "解析器配置",
        "PARSER_MAX_RETRIES": "解析器配置",
        "PARSER_POLL_INTERVAL": "解析器配置",
        "PARSER_MIN_IMAGE_WIDTH": "解析器配置",
        "PARSER_MIN_IMAGE_HEIGHT": "解析器配置",
        "PARSER_PAGE_RENDER_DPI": "解析器配置",
        "PARSER_TABLE_DETECTION_ENABLED": "解析器配置",
        "PARSER_TABLE_OVERLAP_THRESHOLD": "解析器配置",
        "PARSER_TABLE_MIN_ROWS": "解析器配置",
        "PARSER_TABLE_MIN_COLS": "解析器配置",
        "PARSER_TABLE_MIN_HEIGHT_PT": "解析器配置",
        "PARSER_TABLE_MIN_WIDTH_RATIO": "解析器配置",
        "PARSER_TAB_MERGE_THRESHOLD_PT": "解析器配置",
        "PARSER_IMAGE_SIZE_LIMIT": "解析器配置",
        "PARSER_MAX_IMAGES_PER_PAGE": "解析器配置",
        "PARSER_FIGURE_MIN_SCORE": "解析器配置",
        "PARSER_EDGE_LABEL_MAX_LEN": "解析器配置",
        "BATCH_POLL_INTERVAL": "Batch 配置",
        "BATCH_MAX_POLL_RETRIES": "Batch 配置",
        "BATCH_MAX_FILE_SIZE_MB": "Batch 配置",
        "BATCH_PARALLEL_WORKERS": "Batch 配置",
        "BATCH_FILE_DOWNLOAD_TEMPLATE": "Batch 配置",
        "PLUGIN_SEARCH_TOP_K": "Plugin 配置",
        "PLUGIN_QUERY_TOP_K": "Plugin 配置",
        "PLUGIN_GRAPH_MAX_DEPTH": "Plugin 配置",
        "PLUGIN_SUBGRAPH_DEPTH": "Plugin 配置",
        "PLUGIN_DEFAULT_LIMIT": "Plugin 配置",
        "PLUGIN_BM25_TOP_K": "Plugin 配置",
        "PLUGIN_HYBRID_RRF_K": "Plugin 配置",
        "RERANK_CROSS_ENCODER_MODEL": "Plugin 配置",
        "GRAPH_MAX_PATH_DEPTH": "Pipeline 配置",
        "BLOCK_MAX_CHARS": "Pipeline 配置",
        "DB_PATH": "路径配置",
        "FAISS_INDEX_PATH": "路径配置",
        "PROMPT_DIR": "路径配置",
        "PARSE_OUTPUT_DIR": "路径配置",
        "BATCH_OUTPUT_DIR": "路径配置",
        "GRAPH_OUTPUT_DIR": "路径配置",
    }

    for key, val in config_dict.items():
        group = group_map.get(key, "其他")
        groups[group][key] = val

    groups = {k: v for k, v in groups.items() if v}

    return {
        "success": True,
        "llm_configured": Config.llm_configured(),
        "embedding_configured": Config.embedding_configured(),
        "config_groups": groups,
    }


# ---------------------------------------------------------------------------
# Tool 映射
# ---------------------------------------------------------------------------

TOOL_MAP = {
    # MCP 公共工具（5 个）
    "ingest": tool_ingest,
    "search": tool_search,
    "explore": tool_explore,
    "read": tool_read,
    "status": tool_status,
    # 内部/管理工具（不暴露为 MCP）
    "config": tool_config,
    "doc_parse": tool_doc_parse,
    "doc_build_batches": tool_doc_build_batches,
    "doc_submit_batches": tool_doc_submit_batches,
    "doc_ingest_results": tool_doc_ingest_results,
    "doc_build_embed_jsonl": tool_doc_build_embed_jsonl,
    "doc_submit_embed_batches": tool_doc_submit_embed_batches,
    "reprocess": tool_reprocess,
    "evaluate": tool_evaluate,
    "graph_upsert_entity": tool_graph_upsert_entity,
    "graph_delete_entity": tool_graph_delete_entity,
    "graph_upsert_relation": tool_graph_upsert_relation,
    "graph_delete_relation": tool_graph_delete_relation,
    "graph_feedback": tool_graph_feedback,
    "graph_conflicts": tool_graph_conflicts,
    "graph_provenance": tool_graph_provenance,
    "rebuild_global_graph": tool_rebuild_global_graph,
}
