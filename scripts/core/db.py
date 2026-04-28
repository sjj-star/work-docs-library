"""db 模块."""

import json
import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from .config import Config
from .models import Chapter, Chunk, Document

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2"


class KnowledgeDB:
    """KnowledgeDB 类."""

    def __init__(self, db_path: Path | None = None) -> None:
        """初始化 KnowledgeDB."""
        self.db_path = str(db_path or Config.DB_PATH)
        self._init_tables()
        self._check_schema_version()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_tables(self) -> None:
        sql = """
        CREATE TABLE IF NOT EXISTS _schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            source_path TEXT UNIQUE,
            file_type TEXT,
            total_pages INTEGER DEFAULT 0,
            chapters TEXT,
            extracted_at TEXT,
            file_hash TEXT,
            status TEXT DEFAULT 'pending'
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            chunk_id TEXT,
            content TEXT,
            chunk_type TEXT,
            chapter_title TEXT,
            keywords TEXT,
            summary TEXT,
            metadata TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
        CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON chunks(doc_id, chapter_title);
        CREATE TABLE IF NOT EXISTS conflict_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT,
            name TEXT,
            property_key TEXT,
            old_value TEXT,
            new_value TEXT,
            timestamp TEXT,
            doc_id TEXT
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT,
            entity_name TEXT,
            relation_from_type TEXT,
            relation_from_name TEXT,
            relation_to_type TEXT,
            relation_to_name TEXT,
            relation_type TEXT,
            rating INTEGER,
            comment TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_conflict_entity ON conflict_logs(entity_type, name);
        CREATE INDEX IF NOT EXISTS idx_feedback_entity ON feedback(entity_type, entity_name);
        """
        with self._connect() as conn:
            conn.executescript(sql)
            conn.execute(
                "INSERT OR REPLACE INTO _schema_meta (key, value) VALUES (?, ?)",
                ("version", SCHEMA_VERSION),
            )

    def _check_schema_version(self) -> None:
        """检测 schema 版本，旧版本提示重建."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value FROM _schema_meta WHERE key = ?", ("version",)
                ).fetchone()
            current = row["value"] if row else None
        except sqlite3.OperationalError:
            current = None

        if current != SCHEMA_VERSION:
            logger.warning(
                f"数据库 schema 版本过旧（v{current or 'unknown'}），"
                f"建议执行 reprocess 重建以清理无用列。当前仍可正常运行。"
            )

    def upsert_document(self, doc: Document) -> None:
        """upsert_document 函数."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents
                (
                    doc_id, title, source_path, file_type, total_pages,
                    chapters, extracted_at, file_hash, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    doc_id=excluded.doc_id,
                    title=excluded.title,
                    total_pages=excluded.total_pages,
                    chapters=excluded.chapters,
                    extracted_at=excluded.extracted_at,
                    file_hash=excluded.file_hash,
                    status=excluded.status
                """,
                (
                    doc.doc_id,
                    doc.title,
                    doc.source_path,
                    doc.file_type,
                    doc.total_pages,
                    json.dumps([c.to_dict() for c in doc.chapters], ensure_ascii=False),
                    doc.extracted_at,
                    doc.file_hash,
                    str(doc.status),
                ),
            )

    def get_document(self, doc_id: str) -> Document | None:
        """get_document 函数."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if not row:
            return None
        chapters = json.loads(row["chapters"] or "[]")
        return Document(
            doc_id=row["doc_id"],
            title=row["title"],
            source_path=row["source_path"],
            file_type=row["file_type"],
            total_pages=row["total_pages"],
            chapters=[Chapter(**c) for c in chapters],
            extracted_at=row["extracted_at"],
            file_hash=row["file_hash"],
            status=row["status"],
        )

    def search_documents_by_title(self, pattern: str) -> list[Document]:
        """search_documents_by_title 函数."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id FROM documents WHERE title LIKE ? ORDER BY extracted_at DESC",
                (f"%{pattern}%",),
            ).fetchall()
        docs = [self.get_document(r["doc_id"]) for r in rows if r["doc_id"]]
        return [d for d in docs if d is not None]

    def get_document_by_path(self, path: str) -> Document | None:
        """get_document_by_path 函数."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT doc_id FROM documents WHERE source_path = ?", (path,)
            ).fetchone()
        return self.get_document(row["doc_id"]) if row else None

    def list_documents(self) -> list[Document]:
        """list_documents 函数."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id FROM documents ORDER BY extracted_at DESC"
            ).fetchall()
        docs = [self.get_document(r["doc_id"]) for r in rows if r["doc_id"]]
        return [d for d in docs if d is not None]

    def update_document_status(self, doc_id: str, status: str) -> None:
        """update_document_status 函数."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = ? WHERE doc_id = ?",
                (status, doc_id),
            )

    def get_chapters(self, doc_id: str) -> list[Chapter]:
        """get_chapters 函数."""
        doc = self.get_document(doc_id)
        if not doc:
            return []
        return doc.chapters

    def insert_chunk(self, chunk: Chunk) -> int:
        """insert_chunk 函数."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO chunks
                (
                    doc_id, chunk_id, content, chunk_type, chapter_title,
                    keywords, summary, metadata, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.doc_id,
                    chunk.chunk_id,
                    chunk.content,
                    str(chunk.chunk_type),
                    chunk.chapter_title,
                    json.dumps(chunk.keywords, ensure_ascii=False),
                    chunk.summary,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                    str(chunk.status),
                ),
            )
            assert cur.lastrowid is not None
            return cur.lastrowid

    def update_chunk_status(self, chunk_db_id: int, status: str) -> None:
        """update_chunk_status 函数."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE chunks SET status = ? WHERE id = ?",
                (status, chunk_db_id),
            )

    def update_chunks_embedded_batch(self, items: list[tuple]) -> None:
        """批量更新 chunk embedding 和状态（在一个连接中完成）."""
        with self._connect() as conn:
            for chunk_db_id, embedding in items:
                row = conn.execute(
                    "SELECT metadata FROM chunks WHERE id = ?", (chunk_db_id,)
                ).fetchone()
                meta = json.loads(row["metadata"] or "{}") if row else {}
                meta["embedding"] = embedding
                conn.execute(
                    "UPDATE chunks SET metadata = ?, status = 'embedded' WHERE id = ?",
                    (json.dumps(meta, ensure_ascii=False), chunk_db_id),
                )

    def get_pending_chunks(self, doc_id: str | None = None) -> list[tuple]:
        """get_pending_chunks 函数."""
        sql = (
            "SELECT id, doc_id, chunk_id, content, chunk_type, "
            "chapter_title FROM chunks WHERE status = 'pending'"
        )
        params = ()
        if doc_id:
            sql += " AND doc_id = ?"
            params = (doc_id,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            (
                r["id"],
                r["doc_id"],
                r["chunk_id"],
                r["content"],
                r["chunk_type"],
                r["chapter_title"],
            )
            for r in rows
        ]

    def update_chunk_summary(self, db_id: int, summary: str) -> None:
        """update_chunk_summary 函数."""
        with self._connect() as conn:
            conn.execute("UPDATE chunks SET summary = ? WHERE id = ?", (summary, db_id))

    def query_by_doc(self, doc_id: str) -> list[Chunk]:
        """获取文档的所有 chunks."""
        sql = "SELECT * FROM chunks WHERE doc_id = ? ORDER BY created_at"
        with self._connect() as conn:
            rows = conn.execute(sql, (doc_id,)).fetchall()
        return self._rows_to_chunks(rows)

    def query_by_chapter(self, doc_id: str, chapter_title: str) -> list[Chunk]:
        """query_by_chapter 函数."""
        sql = "SELECT * FROM chunks WHERE doc_id = ? AND chapter_title LIKE ? ORDER BY created_at"
        with self._connect() as conn:
            rows = conn.execute(sql, (doc_id, f"%{chapter_title}%")).fetchall()
        return self._rows_to_chunks(rows)

    def query_by_chapter_regex(self, doc_id: str, pattern: str) -> list[Chunk]:
        """query_by_chapter_regex 函数."""
        import re

        compiled = re.compile(pattern)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY created_at", (doc_id,)
            ).fetchall()
        chunks = self._rows_to_chunks(rows)
        return [ck for ck in chunks if compiled.search(ck.chapter_title)]

    def query_by_keyword(self, keyword: str) -> list[Chunk]:
        """query_by_keyword 函数."""
        sql = "SELECT * FROM chunks WHERE keywords LIKE ? ORDER BY doc_id, created_at"
        with self._connect() as conn:
            rows = conn.execute(sql, (f"%{keyword}%",)).fetchall()
        return self._rows_to_chunks(rows)

    def query_by_concept(self, doc_id: str, concept_name: str) -> list[Chunk]:
        """Return chunks whose metadata.entities contain the given concept name."""
        sql = """
            SELECT * FROM chunks
            WHERE doc_id = ? AND (
                metadata LIKE ?
                OR summary LIKE ?
                OR keywords LIKE ?
            )
            ORDER BY created_at
        """
        pattern = f'"name": "{concept_name}"'
        with self._connect() as conn:
            rows = conn.execute(
                sql, (doc_id, pattern, f"%{concept_name}%", f"%{concept_name}%")
            ).fetchall()
        return self._rows_to_chunks(rows)

    def get_chunk_by_db_id(self, db_id: int) -> Chunk | None:
        """get_chunk_by_db_id 函数."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE id = ?", (db_id,)).fetchone()
        return self._rows_to_chunks([row])[0] if row else None

    def delete_chunks_by_doc(self, doc_id: str) -> None:
        """delete_chunks_by_doc 函数."""
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    def delete_chunk_by_id(self, db_id: int) -> None:
        """按 ID 删除单个 chunk."""
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE id = ?", (db_id,))

    # -- 冲突日志 --

    def insert_conflict_logs(self, logs: list[dict]) -> None:
        """批量插入冲突日志."""
        if not logs:
            return
        with self._connect() as conn:
            for log in logs:
                conn.execute(
                    """
                    INSERT INTO conflict_logs
                    (entity_type, name, property_key, old_value, new_value, timestamp, doc_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log.get("entity_type", ""),
                        log.get("name", ""),
                        log.get("property_key", ""),
                        str(log.get("old_value", "")),
                        str(log.get("new_value", "")),
                        log.get("timestamp", ""),
                        log.get("doc_id", ""),
                    ),
                )

    def query_conflict_logs(
        self,
        entity_type: str | None = None,
        name: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """查询冲突日志."""
        sql = "SELECT * FROM conflict_logs WHERE 1=1"
        params: list = []
        if entity_type:
            sql += " AND entity_type = ?"
            params.append(entity_type)
        if name:
            sql += " AND name = ?"
            params.append(name)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": r["id"],
                "entity_type": r["entity_type"],
                "name": r["name"],
                "property_key": r["property_key"],
                "old_value": r["old_value"],
                "new_value": r["new_value"],
                "timestamp": r["timestamp"],
                "doc_id": r["doc_id"],
            }
            for r in rows
        ]

    # -- 反馈 --

    def insert_feedback(
        self,
        rating: int,
        entity_type: str | None = None,
        entity_name: str | None = None,
        relation_type: str | None = None,
        relation_from_type: str | None = None,
        relation_from_name: str | None = None,
        relation_to_type: str | None = None,
        relation_to_name: str | None = None,
        comment: str = "",
    ) -> int:
        """插入反馈记录."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO feedback
                (entity_type, entity_name, relation_from_type, relation_from_name,
                 relation_to_type, relation_to_name, relation_type, rating, comment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_type or "",
                    entity_name or "",
                    relation_from_type or "",
                    relation_from_name or "",
                    relation_to_type or "",
                    relation_to_name or "",
                    relation_type or "",
                    rating,
                    comment,
                ),
            )
            assert cur.lastrowid is not None
            return cur.lastrowid

    def query_feedback(
        self,
        entity_type: str | None = None,
        entity_name: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """查询反馈记录."""
        sql = "SELECT * FROM feedback WHERE 1=1"
        params: list = []
        if entity_type:
            sql += " AND entity_type = ?"
            params.append(entity_type)
        if entity_name:
            sql += " AND entity_name = ?"
            params.append(entity_name)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": r["id"],
                "entity_type": r["entity_type"],
                "entity_name": r["entity_name"],
                "relation_type": r["relation_type"],
                "rating": r["rating"],
                "comment": r["comment"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def get_entity_feedback_score(self, entity_type: str, entity_name: str) -> int:
        """获取实体的累计反馈评分."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT SUM(rating) as score FROM feedback"
                " WHERE entity_type = ? AND entity_name = ?",
                (entity_type, entity_name),
            ).fetchone()
        return row["score"] or 0

    def _rows_to_chunks(self, rows: list[sqlite3.Row]) -> list[Chunk]:
        chunks = []
        for r in rows:
            kw_raw = r["keywords"] or ""
            try:
                keywords = json.loads(kw_raw)
            except json.JSONDecodeError:
                keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]
            chunks.append(
                Chunk(
                    id=r["id"],
                    doc_id=r["doc_id"],
                    chunk_id=r["chunk_id"],
                    content=r["content"],
                    chunk_type=r["chunk_type"],
                    chapter_title=r["chapter_title"],
                    keywords=keywords,
                    summary=r["summary"] or "",
                    status=r["status"] or "pending",
                    metadata=json.loads(r["metadata"] or "{}"),
                )
            )
        return chunks
