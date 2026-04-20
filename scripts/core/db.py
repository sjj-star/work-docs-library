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
        CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON chunks(doc_id, chapter_title);

        CREATE TABLE IF NOT EXISTS chapter_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            chapter_title TEXT NOT NULL,
            start_page INTEGER,
            end_page INTEGER,
            summary TEXT,
            concepts TEXT,
            relationships TEXT,
            key_figures TEXT,
            key_tables TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        );
        CREATE INDEX IF NOT EXISTS idx_chapter_doc ON chapter_summaries(doc_id);
        CREATE INDEX IF NOT EXISTS idx_chapter_title ON chapter_summaries(doc_id, chapter_title);

        CREATE TABLE IF NOT EXISTS concept_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            concept_name TEXT NOT NULL,
            definition TEXT,
            first_mentioned_page INTEGER,
            related_concepts TEXT,
            UNIQUE(doc_id, concept_name)
        );
        CREATE INDEX IF NOT EXISTS idx_concept_doc ON concept_index(doc_id);
        CREATE INDEX IF NOT EXISTS idx_concept_name ON concept_index(concept_name);
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

    def update_chunk_metadata(self, db_id: int, metadata: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chunks SET metadata = ? WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False), db_id),
            )

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

    def query_by_concept(self, doc_id: str, concept_name: str) -> List[Chunk]:
        """Return chunks whose metadata.entities contain the given concept name."""
        sql = """
            SELECT * FROM chunks
            WHERE doc_id = ? AND (
                metadata LIKE ?
                OR summary LIKE ?
                OR keywords LIKE ?
            )
            ORDER BY page_start
        """
        pattern = f'%"name": "{concept_name}"%'
        with self._connect() as conn:
            rows = conn.execute(sql, (doc_id, pattern, f"%{concept_name}%", f"%{concept_name}%")).fetchall()
        return self._rows_to_chunks(rows)

    def get_chunk_by_db_id(self, db_id: int) -> Optional[Chunk]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE id = ?", (db_id,)).fetchone()
        return self._rows_to_chunks([row])[0] if row else None

    def delete_chunks_by_doc(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    # Chapter summaries ------------------------------------------------------
    def upsert_chapter_summary(self, doc_id: str, chapter_title: str, data: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chapter_summaries
                (doc_id, chapter_title, start_page, end_page, summary, concepts, relationships, key_figures, key_tables, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO UPDATE SET
                    start_page=excluded.start_page,
                    end_page=excluded.end_page,
                    summary=excluded.summary,
                    concepts=excluded.concepts,
                    relationships=excluded.relationships,
                    key_figures=excluded.key_figures,
                    key_tables=excluded.key_tables,
                    status=excluded.status
                """,
                (
                    doc_id,
                    chapter_title,
                    data.get("start_page"),
                    data.get("end_page"),
                    data.get("summary"),
                    json.dumps(data.get("concepts", []), ensure_ascii=False),
                    json.dumps(data.get("relationships", []), ensure_ascii=False),
                    json.dumps(data.get("key_figures", []), ensure_ascii=False),
                    json.dumps(data.get("key_tables", []), ensure_ascii=False),
                    data.get("status", "pending"),
                ),
            )

    def get_chapter_summary(self, doc_id: str, chapter_title: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chapter_summaries WHERE doc_id = ? AND chapter_title = ?",
                (doc_id, chapter_title),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "doc_id": row["doc_id"],
            "chapter_title": row["chapter_title"],
            "start_page": row["start_page"],
            "end_page": row["end_page"],
            "summary": row["summary"] or "",
            "concepts": json.loads(row["concepts"] or "[]"),
            "relationships": json.loads(row["relationships"] or "[]"),
            "key_figures": json.loads(row["key_figures"] or "[]"),
            "key_tables": json.loads(row["key_tables"] or "[]"),
            "status": row["status"],
        }

    def list_chapter_summaries(self, doc_id: str) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chapter_summaries WHERE doc_id = ? ORDER BY start_page",
                (doc_id,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "doc_id": r["doc_id"],
                "chapter_title": r["chapter_title"],
                "start_page": r["start_page"],
                "end_page": r["end_page"],
                "summary": r["summary"] or "",
                "concepts": json.loads(r["concepts"] or "[]"),
                "relationships": json.loads(r["relationships"] or "[]"),
                "key_figures": json.loads(r["key_figures"] or "[]"),
                "key_tables": json.loads(r["key_tables"] or "[]"),
                "status": r["status"],
            }
            for r in rows
        ]

    def update_chapter_summary_status(self, doc_id: str, chapter_title: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chapter_summaries SET status = ? WHERE doc_id = ? AND chapter_title = ?",
                (status, doc_id, chapter_title),
            )

    def delete_chapter_summaries_by_doc(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chapter_summaries WHERE doc_id = ?", (doc_id,))

    # Concept index ----------------------------------------------------------
    def upsert_concept(self, doc_id: str, concept_name: str, definition: str = "", first_mentioned_page: int = None, related_concepts: list = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO concept_index
                (doc_id, concept_name, definition, first_mentioned_page, related_concepts)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(doc_id, concept_name) DO UPDATE SET
                    definition=excluded.definition,
                    first_mentioned_page=excluded.first_mentioned_page,
                    related_concepts=excluded.related_concepts
                """,
                (
                    doc_id,
                    concept_name,
                    definition,
                    first_mentioned_page,
                    json.dumps(related_concepts or [], ensure_ascii=False),
                ),
            )

    def get_concept(self, doc_id: str, concept_name: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM concept_index WHERE doc_id = ? AND concept_name = ?",
                (doc_id, concept_name),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "doc_id": row["doc_id"],
            "concept_name": row["concept_name"],
            "definition": row["definition"] or "",
            "first_mentioned_page": row["first_mentioned_page"],
            "related_concepts": json.loads(row["related_concepts"] or "[]"),
        }

    def query_concepts(self, doc_id: str) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM concept_index WHERE doc_id = ? ORDER BY concept_name",
                (doc_id,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "doc_id": r["doc_id"],
                "concept_name": r["concept_name"],
                "definition": r["definition"] or "",
                "first_mentioned_page": r["first_mentioned_page"],
                "related_concepts": json.loads(r["related_concepts"] or "[]"),
            }
            for r in rows
        ]

    def query_concept_by_name(self, concept_name: str) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM concept_index WHERE concept_name = ? ORDER BY doc_id",
                (concept_name,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "doc_id": r["doc_id"],
                "concept_name": r["concept_name"],
                "definition": r["definition"] or "",
                "first_mentioned_page": r["first_mentioned_page"],
                "related_concepts": json.loads(r["related_concepts"] or "[]"),
            }
            for r in rows
        ]

    def delete_concepts_by_doc(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM concept_index WHERE doc_id = ?", (doc_id,))

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
