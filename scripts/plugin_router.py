#!/usr/bin/env python3
"""
Plugin router for work-docs-library.
Reads JSON parameters from stdin and returns structured JSON via stdout.
Each tool is dispatched by sys.argv[1].
"""
import json
import logging
import os
import sys
from pathlib import Path

_SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SKILL_ROOT / "scripts"))

from core.config import Config

Config.setup_logging()
# Redirect all root log handlers to stderr so stdout stays pure JSON
for handler in logging.root.handlers:
    handler.stream = sys.stderr

logger = logging.getLogger("plugin_router")

# Import agent_batch_helper helpers after path setup
import agent_batch_helper as abh
from core.db import KnowledgeDB
from core.llm_client import EmbeddingClient
from core.pipeline import IngestionPipeline
from core.vector_index import VectorIndex


def _chunk_to_dict(ck, preview_len: int = 500):
    return {
        "doc_id": ck.doc_id,
        "chunk_id": ck.chunk_id,
        "chunk_type": ck.chunk_type,
        "page_start": ck.page_start,
        "page_end": ck.page_end,
        "chapter_title": ck.chapter_title,
        "content_preview": ck.content[:preview_len],
        "summary": ck.summary,
        "keywords": ck.keywords,
    }


def tool_ingest(params: dict):
    path = params.get("path")
    dry_run = params.get("dry_run", False)
    auto_chapter = params.get("auto_chapter", False)

    Config.ensure_dirs()
    pipe = IngestionPipeline()
    try:
        doc_ids = pipe.ingest(path, dry_run=dry_run, auto_chapter=auto_chapter)
        if not doc_ids:
            return {"success": True, "doc_ids": [], "message": "No documents found or ingested."}
        return {"success": True, "doc_ids": doc_ids, "message": f"Ingested {len(doc_ids)} document(s)."}
    finally:
        pipe.close()


def tool_search(params: dict):
    text = params.get("text")
    top_k = params.get("top_k", 5)

    Config.ensure_dirs()
    db = KnowledgeDB()
    vec = VectorIndex()
    try:
        embedder = EmbeddingClient()
    except RuntimeError as e:
        return {"success": False, "error": str(e)}

    try:
        emb = embedder.embed([text])[0]
        hits = vec.search(emb, top_k=top_k)
        results = []
        for db_id, score in hits:
            chunk = db.get_chunk_by_db_id(db_id)
            if not chunk:
                continue
            results.append({
                "score": round(score, 4),
                "doc_id": chunk.doc_id,
                "chapter_title": chunk.chapter_title,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "content_preview": chunk.content[:500],
                "summary": chunk.summary,
                "keywords": chunk.keywords,
            })
        return {"success": True, "results": results}
    finally:
        embedder.close()


def _parse_page(page_str: str):
    if "-" in page_str:
        a, b = page_str.split("-", 1)
        return int(a), int(b)
    p = int(page_str)
    return p, p


def tool_query(params: dict):
    Config.ensure_dirs()
    doc_id = params.get("doc_id")
    page = params.get("page")
    chapter = params.get("chapter")
    chapter_regex = params.get("chapter_regex")
    keyword = params.get("keyword")
    top_k = params.get("top_k", 10)

    db = KnowledgeDB()
    results = []
    if page:
        ps, pe = _parse_page(page)
        results = db.query_by_page(doc_id, ps, pe)
    elif chapter:
        results = db.query_by_chapter(doc_id, chapter)
    elif chapter_regex:
        results = db.query_by_chapter_regex(doc_id, chapter_regex)
    elif keyword:
        results = db.query_by_keyword(keyword)
    else:
        return {"success": False, "error": "Provide page, chapter, chapter_regex, or keyword."}

    return {"success": True, "results": [_chunk_to_dict(ck) for ck in results[:top_k]]}


def tool_status(params: dict):
    Config.ensure_dirs()
    db = KnowledgeDB()
    docs = db.list_documents()
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


def tool_toc(params: dict):
    Config.ensure_dirs()
    db = KnowledgeDB()
    doc_id = params.get("doc_id")
    match = params.get("match")

    if doc_id:
        doc = db.get_document(doc_id)
        if not doc:
            return {"success": False, "error": f"Document {doc_id} not found."}
        chapters = db.get_chapters(doc_id)
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
        docs = db.search_documents_by_title(match)
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
                for d in docs if d
            ],
        }
    else:
        return {"success": False, "error": "Provide either doc_id or match."}


def tool_progress(params: dict):
    Config.ensure_dirs()
    doc_id = params.get("doc_id")
    db = KnowledgeDB()
    with db._connect() as conn:
        def count(status: str = None):
            if status:
                return conn.execute(
                    "SELECT COUNT(*) as c FROM chunks WHERE doc_id = ? AND status = ?",
                    (doc_id, status),
                ).fetchone()["c"]
            return conn.execute(
                "SELECT COUNT(*) as c FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchone()["c"]

        return {
            "success": True,
            "doc_id": doc_id,
            "total": count(),
            "done": count("done"),
            "embedded": count("embedded"),
            "skipped": count("skipped"),
            "pending": count("pending"),
            "failed": count("failed"),
        }


def tool_reprocess(params: dict):
    doc_id = params.get("doc_id")
    Config.ensure_dirs()
    pipe = IngestionPipeline()
    try:
        doc = pipe.db.get_document(doc_id)
        if not doc:
            return {"success": False, "error": f"Document {doc_id} not found."}
        pipe._process_one(doc.source_path, dry_run=False, auto_chapter=False, force=True)
        return {"success": True, "doc_id": doc_id, "message": "Reprocessed."}
    finally:
        pipe.close()


def tool_auto_summarize(params: dict):
    Config.ensure_dirs()
    doc_id = params["doc_id"]
    out_dir = Path(params.get("output_dir", "./auto_batches"))
    batch_size = params.get("batch_size", 10)
    target_chars = params.get("target_chars", 25000)
    do_filter = params.get("filter", False)

    out_dir = abh._resolve_output_dir(out_dir, doc_id)
    result = abh.run_auto_summarize(
        doc_id=doc_id,
        out_dir=out_dir,
        batch_size=batch_size,
        target_chars=target_chars,
        do_filter=do_filter,
    )
    return {"success": True, **result}


TOOL_MAP = {
    "ingest": tool_ingest,
    "search": tool_search,
    "query": tool_query,
    "status": tool_status,
    "toc": tool_toc,
    "auto_summarize": tool_auto_summarize,
    "progress": tool_progress,
    "reprocess": tool_reprocess,
}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Missing tool name argument"}, ensure_ascii=False))
        sys.exit(1)

    tool_name = sys.argv[1]
    func = TOOL_MAP.get(tool_name)
    if not func:
        print(json.dumps({"success": False, "error": f"Unknown tool: {tool_name}"}, ensure_ascii=False))
        sys.exit(1)

    try:
        params = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON input: {e}"}, ensure_ascii=False))
        sys.exit(1)

    try:
        result = func(params)
    except Exception as e:
        logger.exception("Tool %s failed", tool_name)
        result = {"success": False, "error": str(e)}

    print(json.dumps(result, ensure_ascii=False))
