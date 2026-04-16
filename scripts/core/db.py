import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional

from .config import Config
from .models import Chunk, Document, Chapter


class KnowledgeDB:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = str(db_path or Config.DB_PATH)
        self._init_tables()

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
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            source_path TEXT UNIQUE,
            file_type TEXT,
            total_pages INTEGER DEFAULT 0,
            chapters TEXT,
            chapters_override TEXT,
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
            page_start INTEGER,
            page_end INTEGER,
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
        """
        with self._connect() as conn:
            conn.executescript(sql)

    def upsert_document(self, doc: Document) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents
                (doc_id, title, source_path, file_type, total_pages, chapters, extracted_at, file_hash, status)
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
                    doc.status,
                ),
            )

    def get_document(self, doc_id: str) -> Optional[Document]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
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

    def search_documents_by_title(self, pattern: str) -> List[Document]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id FROM documents WHERE title LIKE ? ORDER BY extracted_at DESC",
                (f"%{pattern}%",)
            ).fetchall()
        return [self.get_document(r["doc_id"]) for r in rows if r["doc_id"]]

    def get_document_by_path(self, path: str) -> Optional[Document]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT doc_id FROM documents WHERE source_path = ?", (path,)
            ).fetchone()
        return self.get_document(row["doc_id"]) if row else None

    def list_documents(self) -> List[Document]:
        with self._connect() as conn:
            rows = conn.execute("SELECT doc_id FROM documents ORDER BY extracted_at DESC").fetchall()
        return [self.get_document(r["doc_id"]) for r in rows if r["doc_id"]]

    def update_document_status(self, doc_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = ? WHERE doc_id = ?",
                (status, doc_id),
            )

    def set_chapters_override(self, doc_id: str, chapters_json: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET chapters_override = ? WHERE doc_id = ?",
                (chapters_json, doc_id),
            )

    def get_chapters(self, doc_id: str) -> List[Chapter]:
        doc = self.get_document(doc_id)
        if not doc:
            return []
        with self._connect() as conn:
            row = conn.execute(
                "SELECT chapters_override FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        if row and row["chapters_override"]:
            data = json.loads(row["chapters_override"])
        else:
            data = [c.to_dict() for c in doc.chapters]
        return [Chapter(**c) for c in data]

    def insert_chunk(self, chunk: Chunk) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO chunks
                (doc_id, chunk_id, content, chunk_type, page_start, page_end, chapter_title, keywords, summary, metadata, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.doc_id,
                    chunk.chunk_id,
                    chunk.content,
                    chunk.chunk_type,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.chapter_title,
                    json.dumps(chunk.keywords, ensure_ascii=False),
                    chunk.summary,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                    "pending",
                ),
            )
            return cur.lastrowid

    def update_chunk_status(self, chunk_db_id: int, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chunks SET status = ? WHERE id = ?",
                (status, chunk_db_id),
            )

    def update_chunk_embedding(self, chunk_db_id: int, embedding: List[float]) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metadata FROM chunks WHERE id = ?", (chunk_db_id,)
            ).fetchone()
            meta = json.loads(row["metadata"] or "{}")
            meta["embedding"] = embedding
            conn.execute(
                "UPDATE chunks SET metadata = ? WHERE id = ?",
                (json.dumps(meta, ensure_ascii=False), chunk_db_id),
            )

    def get_pending_chunks(self, doc_id: Optional[str] = None) -> List[tuple]:
        sql = "SELECT id, doc_id, chunk_id, content, chunk_type, page_start, page_end, chapter_title FROM chunks WHERE status = 'pending'"
        params = ()
        if doc_id:
            sql += " AND doc_id = ?"
            params = (doc_id,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [(r["id"], r["doc_id"], r["chunk_id"], r["content"], r["chunk_type"], r["page_start"], r["page_end"], r["chapter_title"]) for r in rows]

    def get_embedded_but_unsummarized_chunks(self, doc_id: Optional[str] = None) -> List[tuple]:
        sql = "SELECT id, doc_id, chunk_id, content, chunk_type, page_start, page_end, chapter_title FROM chunks WHERE status = 'embedded' AND (summary IS NULL OR summary = '')"
        params = ()
        if doc_id:
            sql += " AND doc_id = ?"
            params = (doc_id,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [(r["id"], r["doc_id"], r["chunk_id"], r["content"], r["chunk_type"], r["page_start"], r["page_end"], r["chapter_title"]) for r in rows]

    def update_chunk_summary(self, db_id: int, summary: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE chunks SET summary = ? WHERE id = ?", (summary, db_id))

    def update_chunk_keywords(self, db_id: int, keywords: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE chunks SET keywords = ? WHERE id = ?", (keywords, db_id))

    def set_chunk_done(self, db_id: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE chunks SET status = 'done' WHERE id = ?", (db_id,))

    def query_by_page(self, doc_id: str, page_start: int, page_end: int) -> List[Chunk]:
        sql = """
            SELECT * FROM chunks WHERE doc_id = ?
            AND page_start <= ? AND page_end >= ?
            ORDER BY page_start
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (doc_id, page_end, page_start)).fetchall()
        return self._rows_to_chunks(rows)

    def query_by_chapter(self, doc_id: str, chapter_title: str) -> List[Chunk]:
        sql = "SELECT * FROM chunks WHERE doc_id = ? AND chapter_title LIKE ? ORDER BY page_start"
        with self._connect() as conn:
            rows = conn.execute(sql, (doc_id, f"%{chapter_title}%")).fetchall()
        return self._rows_to_chunks(rows)

    def query_by_chapter_regex(self, doc_id: str, pattern: str) -> List[Chunk]:
        import re
        compiled = re.compile(pattern)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY page_start", (doc_id,)
            ).fetchall()
        chunks = self._rows_to_chunks(rows)
        return [ck for ck in chunks if compiled.search(ck.chapter_title)]

    def query_by_keyword(self, keyword: str) -> List[Chunk]:
        sql = "SELECT * FROM chunks WHERE keywords LIKE ? ORDER BY doc_id, page_start"
        with self._connect() as conn:
            rows = conn.execute(sql, (f"%{keyword}%",)).fetchall()
        return self._rows_to_chunks(rows)

    def get_chunk_by_db_id(self, db_id: int) -> Optional[Chunk]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE id = ?", (db_id,)).fetchone()
        return self._rows_to_chunks([row])[0] if row else None

    def delete_chunks_by_doc(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    def _rows_to_chunks(self, rows: List[sqlite3.Row]) -> List[Chunk]:
        chunks = []
        for r in rows:
            kw_raw = r["keywords"] or ""
            try:
                keywords = json.loads(kw_raw)
            except json.JSONDecodeError:
                keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]
            chunks.append(
                Chunk(
                    doc_id=r["doc_id"],
                    chunk_id=r["chunk_id"],
                    content=r["content"],
                    chunk_type=r["chunk_type"],
                    page_start=r["page_start"],
                    page_end=r["page_end"],
                    chapter_title=r["chapter_title"],
                    keywords=keywords,
                    summary=r["summary"] or "",
                    status=r["status"] or "pending",
                    metadata=json.loads(r["metadata"] or "{}"),
                )
            )
        return chunks
