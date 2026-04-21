#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_SKILL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_ROOT))

from core.config import Config
from core.db import KnowledgeDB

logger = logging.getLogger("doc_extractor")
from core.embedding_client import EmbeddingClient
from core.pipeline import IngestionPipeline
from core.vector_index import VectorIndex
from core.chapter_editor import ChapterEditor


def cmd_ingest(args):
    Config.ensure_dirs()
    pipe = IngestionPipeline()
    try:
        doc_ids = pipe.ingest(args.path, dry_run=args.dry_run, auto_chapter=args.auto_chapter)
        if not doc_ids:
            logger.info("No documents found or ingested | path=%s", args.path)
            return
        logger.info("Ingested document IDs: %s", ", ".join(doc_ids))
    finally:
        pipe.close()


def cmd_status(args):
    db = KnowledgeDB()
    docs = db.list_documents()
    print(f"{'Doc ID':<32} {'Status':<12} {'Title'}")
    for d in docs:
        print(f"{d.doc_id:<32} {d.status:<12} {d.title}")


def cmd_chapter_edit(args):
    editor = ChapterEditor()
    editor.interactive_edit(args.doc_id)


def cmd_query(args):
    db = KnowledgeDB()
    results = []
    if args.page:
        ps, pe = _parse_page(args.page)
        results = db.query_by_page(args.doc_id, ps, pe)
    elif args.chapter:
        results = db.query_by_chapter(args.doc_id, args.chapter)
    elif args.chapter_regex:
        results = db.query_by_chapter_regex(args.doc_id, args.chapter_regex)
    elif args.keyword:
        results = db.query_by_keyword(args.keyword)
    else:
        logger.warning("Query requires --page, --chapter, --chapter-regex, or --keyword")
        return
    _print_chunks(results, top_k=args.top_k)


def cmd_search(args):
    Config.ensure_dirs()
    db = KnowledgeDB()
    try:
        embedder = EmbeddingClient()
    except RuntimeError as e:
        logger.error("Failed to initialize embedder: %s", e)
        return
    try:
        # 先调用 embed 探测维度，再用探测到的维度创建 VectorIndex
        emb = embedder.embed([args.text])[0]
        vec = VectorIndex(dim=len(emb))
        hits = vec.search(emb, top_k=args.top_k)
        print(f"Vector search top-{args.top_k} results:\n")
        for db_id, score in hits:
            chunk = db.get_chunk_by_db_id(db_id)
            if not chunk:
                continue
            print(f"--- Score: {score:.4f} | {chunk.doc_id} | {chunk.chapter_title} P{chunk.page_start}-{chunk.page_end} ---")
            print(chunk.content[:500])
            if chunk.summary:
                print(f"[Summary]: {chunk.summary}")
            print()
    finally:
        embedder.close()


def cmd_list_pending(args):
    db = KnowledgeDB()
    if args.doc_id:
        rows = db.get_embedded_but_unsummarized_chunks(args.doc_id)
        print(f"Found {len(rows)} embedded-but-unsummarized chunk(s) for doc {args.doc_id}:")
    else:
        rows = db.get_embedded_but_unsummarized_chunks()
        print(f"Found {len(rows)} embedded-but-unsummarized chunk(s) in total:")
    for db_id, doc_id, chunk_id, content, ctype, ps, pe, ch_title in rows[:args.top_k]:
        preview = content[:120].replace("\n", " ")
        print(f"  chunk_db_id={db_id} | {doc_id} | {chunk_id} | {ch_title} P{ps}-{pe} | {preview}...")


def cmd_write_summary(args):
    db = KnowledgeDB()
    db.update_chunk_summary(args.chunk_db_id, args.summary)
    logger.info("Summary written | chunk_db_id=%s", args.chunk_db_id)


def cmd_write_keywords(args):
    db = KnowledgeDB()
    db.update_chunk_keywords(args.chunk_db_id, args.keywords)
    logger.info("Keywords written | chunk_db_id=%s", args.chunk_db_id)


def cmd_write_embedding(args):
    import json
    db = KnowledgeDB()

    file_path = Path(args.embedding_file).resolve()
    skill_root = Config.DB_PATH.parent.parent.resolve()
    try:
        file_path.relative_to(skill_root)
    except ValueError:
        logger.error("Embedding file outside skill directory | path=%s | skill_root=%s", file_path, skill_root)
        return

    with open(file_path, "r", encoding="utf-8") as f:
        emb = json.load(f)

    if not isinstance(emb, list) or not emb:
        logger.error("Embedding must be a non-empty JSON array of floats")
        return
    if not all(isinstance(x, (int, float)) for x in emb):
        logger.error("Embedding must contain only numeric values")
        return

    # 使用 embedding 的实际维度创建 VectorIndex
    vec = VectorIndex(dim=len(emb))

    db.update_chunk_embedding(args.chunk_db_id, emb)
    vec.add(args.chunk_db_id, emb)
    logger.info("Embedding written | chunk_db_id=%s", args.chunk_db_id)


def cmd_toc(args):
    db = KnowledgeDB()
    if args.doc_id:
        doc = db.get_document(args.doc_id)
        if not doc:
            logger.error("Document not found | doc_id=%s", args.doc_id)
            return
        print(f"Document: {doc.title}")
        print(f"Pages: {doc.total_pages} | Chapters: {len(doc.chapters)} | Status: {doc.status}")
        print("-" * 70)
        chapters = db.get_chapters(doc.doc_id)
        for i, ch in enumerate(chapters, 1):
            indent = "  " * (ch.level - 1)
            print(f"{i:3d}. {indent}[L{ch.level}] P{ch.start_page:3d}-{ch.end_page:<3d} {ch.title}")
    elif args.match:
        docs = db.search_documents_by_title(args.match)
        if not docs:
            logger.info("No documents found matching '%s'", args.match)
            return
        print(f"Found {len(docs)} document(s) matching '{args.match}':")
        print(f"{'Doc ID':<32} {'Status':<12} {'Pages':<6} {'Title'}")
        print("-" * 70)
        for d in docs:
            print(f"{d.doc_id:<32} {d.status:<12} {d.total_pages:<6} {d.title}")
    else:
        logger.warning("TOC requires either --doc-id or --match")


def cmd_reprocess(args):
    Config.ensure_dirs()
    pipe = IngestionPipeline()
    try:
        doc = pipe.db.get_document(args.doc_id)
        if not doc:
            logger.error("Document not found | doc_id=%s", args.doc_id)
            return
        pipe._process_one(doc.source_path, dry_run=False, auto_chapter=False, force=True)
        logger.info("Reprocessed | doc_id=%s", args.doc_id)
    finally:
        pipe.close()


def _parse_page(page_str: str):
    """解析页码范围字符串，支持格式: '5' 或 '10-20'"""
    import re
    
    if not page_str or not isinstance(page_str, str):
        raise ValueError(f"页码必须是字符串，当前类型: {type(page_str)}")
    
    page_str = page_str.strip()
    
    if "-" in page_str:
        # 验证格式
        if not re.match(r"^\d+-\d+$", page_str):
            raise ValueError(f"页码范围格式无效，应为 'start-end': {page_str}")
        
        a, b = page_str.split("-", 1)
        try:
            start, end = int(a), int(b)
        except ValueError:
            raise ValueError(f"页码必须是数字: {page_str}")
        
        if start > end:
            raise ValueError(f"起始页码不能大于结束页码: {start} > {end}")
        if start < 1:
            raise ValueError(f"页码必须大于等于 1: {start}")
        
        return start, end
    
    # 单个页码
    try:
        p = int(page_str)
    except ValueError:
        raise ValueError(f"页码必须是数字: {page_str}")
    
    if p < 1:
        raise ValueError(f"页码必须大于等于 1: {p}")
    
    return p, p


def _print_chunks(chunks, top_k: int = 10):
    for i, ck in enumerate(chunks[:top_k], 1):
        print(f"\n--- Result {i} | {ck.doc_id} | {ck.chapter_title} P{ck.page_start}-{ck.page_end} [{ck.chunk_type}] ---")
        content = ck.content
        marker = "\n\n[IMAGES ON THIS PAGE]"
        idx = content.find(marker)
        if idx != -1 and idx > 600:
            # Show beginning and the image block so users don't miss it
            preview = content[:400] + "\n...(truncated)..." + content[idx:]
            print(preview)
        else:
            print(content[:800])
        if ck.summary:
            print(f"[Summary]: {ck.summary}")
        if ck.keywords:
            print(f"[Keywords]: {', '.join(ck.keywords)}")


def main():
    Config.setup_logging()
    parser = argparse.ArgumentParser(description="work-docs-location knowledge extraction CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Extract and store documents")
    p_ingest.add_argument("--path", required=True, help="File or directory path")
    p_ingest.add_argument("--dry-run", action="store_true", help="Preview without API calls")
    p_ingest.add_argument("--auto-chapter", action="store_true", help="Use auto-detected chapters without override")
    p_ingest.set_defaults(func=cmd_ingest)

    p_status = sub.add_parser("status", help="List ingested documents")
    p_status.set_defaults(func=cmd_status)

    p_edit = sub.add_parser("chapter-edit", help="Interactively edit chapters")
    p_edit.add_argument("--doc-id", required=True)
    p_edit.set_defaults(func=cmd_chapter_edit)

    p_query = sub.add_parser("query", help="Query by page/chapter/keyword")
    p_query.add_argument("--doc-id", help="Required for --page, --chapter, or --chapter-regex")
    p_query.add_argument("--page", help="e.g., 50-60")
    p_query.add_argument("--chapter", help="Chapter title substring (LIKE match)")
    p_query.add_argument("--chapter-regex", help="Chapter title regex match")
    p_query.add_argument("--keyword", help="Keyword in chunk keywords")
    p_query.add_argument("--top-k", type=int, default=10)
    p_query.set_defaults(func=cmd_query)

    p_search = sub.add_parser("search", help="Semantic vector search")
    p_search.add_argument("--text", required=True)
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.set_defaults(func=cmd_search)

    p_toc = sub.add_parser("toc", help="Show table of contents or search documents by title")
    p_toc.add_argument("--doc-id", help="Show full TOC of a specific document")
    p_toc.add_argument("--match", help="Fuzzy match document titles (e.g., AHB, DSP, JTAG)")
    p_toc.set_defaults(func=cmd_toc)

    p_pending = sub.add_parser("list-pending", help="List chunks waiting for Agent summary")
    p_pending.add_argument("--doc-id", help="Filter by document ID")
    p_pending.add_argument("--top-k", type=int, default=50)
    p_pending.set_defaults(func=cmd_list_pending)

    p_ws = sub.add_parser("write-summary", help="Write a summary back to a chunk")
    p_ws.add_argument("--chunk-db-id", type=int, required=True)
    p_ws.add_argument("--summary", required=True)
    p_ws.set_defaults(func=cmd_write_summary)

    p_wk = sub.add_parser("write-keywords", help="Write keywords back to a chunk")
    p_wk.add_argument("--chunk-db-id", type=int, required=True)
    p_wk.add_argument("--keywords", required=True, help="Comma-separated keywords")
    p_wk.set_defaults(func=cmd_write_keywords)

    p_we = sub.add_parser("write-embedding", help="Write an embedding vector back to a chunk")
    p_we.add_argument("--chunk-db-id", type=int, required=True)
    p_we.add_argument("--embedding-file", required=True, help="Path to JSON file containing a single embedding array")
    p_we.set_defaults(func=cmd_write_embedding)

    p_re = sub.add_parser("reprocess", help="Force reprocess a document")
    p_re.add_argument("--doc-id", required=True)
    p_re.set_defaults(func=cmd_reprocess)

    args = parser.parse_args()
    args.func(args)
    sys.exit(0)


if __name__ == "__main__":
    main()
