"""db 模块."""

import json
import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Config
from .models import Chapter, Document, EvalDataset, EvalQuestion

logger = logging.getLogger(__name__)

class KnowledgeDB:
    """KnowledgeDB 类."""

    def __init__(self, db_path: Path | None = None) -> None:
        """初始化 KnowledgeDB."""
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
            extracted_at TEXT,
            file_hash TEXT,
            status TEXT DEFAULT 'pending'
        );
        -- v3: content_blocks + heading_maps (方案C: 内容块-标题映射解耦架构)
        CREATE TABLE IF NOT EXISTS content_blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            block_id TEXT NOT NULL,
            content TEXT NOT NULL,
            seq_index INTEGER NOT NULL,
            metadata TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        );
        CREATE INDEX IF NOT EXISTS idx_blocks_doc ON content_blocks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_blocks_seq ON content_blocks(doc_id, seq_index);

        CREATE TABLE IF NOT EXISTS heading_maps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            heading_title TEXT NOT NULL,
            heading_level INTEGER NOT NULL,
            parent_heading TEXT,
            block_db_ids TEXT NOT NULL,
            content_summary TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        );
        CREATE INDEX IF NOT EXISTS idx_headings_doc ON heading_maps(doc_id);
        CREATE INDEX IF NOT EXISTS idx_headings_title ON heading_maps(doc_id, heading_title);
        CREATE INDEX IF NOT EXISTS idx_headings_level ON heading_maps(doc_id, heading_level);
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
        CREATE TABLE IF NOT EXISTS eval_datasets (
            name TEXT PRIMARY KEY,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS eval_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            question TEXT NOT NULL,
            ground_truth_answer TEXT,
            ground_truth_context_ids TEXT,
            ground_truth_doc_ids TEXT,
            tags TEXT,
            metadata TEXT,
            FOREIGN KEY (dataset_name) REFERENCES eval_datasets(name)
        );
        CREATE INDEX IF NOT EXISTS idx_eval_q_dataset ON eval_questions(dataset_name);
        CREATE INDEX IF NOT EXISTS idx_conflict_entity ON conflict_logs(entity_type, name);
        CREATE INDEX IF NOT EXISTS idx_feedback_entity ON feedback(entity_type, entity_name);
        """
        with self._connect() as conn:
            conn.executescript(sql)

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

    # -- v3: content_blocks + heading_maps (方案C) --

    def insert_block(
        self, doc_id: str, block_id: str, content: str, seq_index: int, metadata: dict | None = None
    ) -> int:
        """插入 content_block，返回 db_id."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO content_blocks (doc_id, block_id, content, seq_index, metadata, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                (
                    doc_id,
                    block_id,
                    content,
                    seq_index,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            assert cur.lastrowid is not None
            return cur.lastrowid

    def insert_heading_map(
        self,
        doc_id: str,
        heading_title: str,
        heading_level: int,
        parent_heading: str | None,
        block_db_ids: list[int],
        content_summary: str | None = None,
    ) -> int:
        """插入 heading_map，返回 db_id."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO heading_maps (
                    doc_id, heading_title, heading_level,
                    parent_heading, block_db_ids, content_summary
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    heading_title,
                    heading_level,
                    parent_heading or "",
                    json.dumps(block_db_ids, ensure_ascii=False),
                    content_summary or "",
                ),
            )
            assert cur.lastrowid is not None
            return cur.lastrowid

    def query_blocks_by_doc(self, doc_id: str) -> list[dict]:
        """获取文档的所有 content_blocks，按 seq_index 排序."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM content_blocks WHERE doc_id = ? ORDER BY seq_index",
                (doc_id,),
            ).fetchall()
        return self._rows_to_blocks(rows)

    def query_by_heading(self, doc_id: str, heading_title: str) -> list[dict]:
        """按标题子串匹配查询关联的 content_blocks（按 heading_level 排序）."""
        if not heading_title:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT block_db_ids FROM heading_maps "
                "WHERE doc_id = ? AND heading_title LIKE '%' || ? || '%' "
                "ORDER BY heading_level, heading_title",
                (doc_id, heading_title),
            ).fetchall()
            if not rows:
                return []
            block_ids: list[int] = []
            seen: set[int] = set()
            for row in rows:
                for bid in json.loads(row["block_db_ids"] or "[]"):
                    if bid not in seen:
                        seen.add(bid)
                        block_ids.append(bid)
            if not block_ids:
                return []
            placeholders = ",".join("?" * len(block_ids))
            cb_rows = conn.execute(
                f"SELECT * FROM content_blocks WHERE id IN ({placeholders}) ORDER BY seq_index",
                tuple(block_ids),
            ).fetchall()
        return self._rows_to_blocks(cb_rows)

    def query_by_heading_recursive(self, doc_id: str, heading_title: str) -> list[dict]:
        """递归查询标题及其所有子标题关联的 content_blocks.

        先按子串匹配找到目标标题，再递归收集 parent_heading 指向它的所有子标题.

        先按子串匹配找到目标标题，再递归收集 parent_heading 指向它的所有子标题.
        """
        with self._connect() as conn:
            # 1. 找到匹配的目标标题（按层级排序，取第一个最匹配的）
            target_rows = conn.execute(
                "SELECT heading_title, block_db_ids FROM heading_maps "
                "WHERE doc_id = ? AND heading_title LIKE '%' || ? || '%' "
                "ORDER BY heading_level, heading_title",
                (doc_id, heading_title),
            ).fetchall()
            if not target_rows:
                return []

            target_title = target_rows[0]["heading_title"]
            all_block_ids: list[int] = []
            seen: set[int] = set()

            def _collect(title: str) -> None:
                # 收集当前标题的 blocks
                rows = conn.execute(
                    "SELECT block_db_ids FROM heading_maps WHERE doc_id = ? AND heading_title = ?",
                    (doc_id, title),
                ).fetchall()
                for row in rows:
                    for bid in json.loads(row["block_db_ids"] or "[]"):
                        if bid not in seen:
                            seen.add(bid)
                            all_block_ids.append(bid)
                # 递归收集子标题
                child_rows = conn.execute(
                    "SELECT heading_title FROM heading_maps "
                    "WHERE doc_id = ? AND parent_heading = ?",
                    (doc_id, title),
                ).fetchall()
                for cr in child_rows:
                    _collect(cr["heading_title"])

            _collect(target_title)

            if not all_block_ids:
                return []
            placeholders = ",".join("?" * len(all_block_ids))
            cb_rows = conn.execute(
                f"SELECT * FROM content_blocks WHERE id IN ({placeholders}) ORDER BY seq_index",
                tuple(all_block_ids),
            ).fetchall()
        return self._rows_to_blocks(cb_rows)

    def delete_blocks_by_doc(self, doc_id: str) -> None:
        """删除文档的所有 content_blocks."""
        with self._connect() as conn:
            conn.execute("DELETE FROM content_blocks WHERE doc_id = ?", (doc_id,))

    def query_heading_maps_by_doc(self, doc_id: str) -> list[dict]:
        """获取文档的所有 heading_maps."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM heading_maps WHERE doc_id = ? ORDER BY heading_level, heading_title",
                (doc_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_heading_maps_by_doc(self, doc_id: str) -> None:
        """删除文档的所有 heading_maps."""
        with self._connect() as conn:
            conn.execute("DELETE FROM heading_maps WHERE doc_id = ?", (doc_id,))

    def update_block_status(self, block_db_id: int, status: str) -> None:
        """更新 content_block 状态."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE content_blocks SET status = ? WHERE id = ?",
                (status, block_db_id),
            )

    def update_blocks_embedded_batch(self, items: list[tuple]) -> None:
        """批量更新 content_block embedding 和状态（在一个连接中完成）."""
        with self._connect() as conn:
            for block_db_id, embedding in items:
                row = conn.execute(
                    "SELECT metadata FROM content_blocks WHERE id = ?", (block_db_id,)
                ).fetchone()
                meta = json.loads(row["metadata"] or "{}") if row else {}
                meta["embedding"] = embedding
                conn.execute(
                    "UPDATE content_blocks SET metadata = ?, status = 'embedded' WHERE id = ?",
                    (json.dumps(meta, ensure_ascii=False), block_db_id),
                )

    def get_pending_blocks(self, doc_id: str | None = None) -> list[tuple]:
        """获取待处理的 content_blocks."""
        sql = (
            "SELECT id, doc_id, block_id, content, seq_index, metadata "
            "FROM content_blocks WHERE status = 'pending'"
        )
        params: tuple = ()
        if doc_id:
            sql += " AND doc_id = ?"
            params = (doc_id,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            (
                r["id"],
                r["doc_id"],
                r["block_id"],
                r["content"],
                r["seq_index"],
                json.loads(r["metadata"] or "{}"),
            )
            for r in rows
        ]

    def get_block_by_db_id(self, db_id: int) -> dict | None:
        """按 ID 查询单个 content_block."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM content_blocks WHERE id = ?", (db_id,)).fetchone()
        return self._rows_to_blocks([row])[0] if row else None

    def _rows_to_blocks(self, rows: list[sqlite3.Row]) -> list[dict]:
        return [
            {
                "id": r["id"],
                "doc_id": r["doc_id"],
                "block_id": r["block_id"],
                "content": r["content"],
                "seq_index": r["seq_index"],
                "status": r["status"] or "pending",
                "metadata": json.loads(r["metadata"] or "{}"),
            }
            for r in rows
        ]

    # -- 冲突日志 --

    def insert_conflict_logs(self, logs: list[dict]) -> None:
        """批量插入冲突日志.

        支持实体冲突和关系冲突。关系冲突通过合成 entity_type/name
        保留完整上下文，无需修改数据库 schema。
        """
        if not logs:
            return
        with self._connect() as conn:
            for log in logs:
                entity_type = log.get("entity_type", "")
                name = log.get("name", "")
                # 关系冲突：合成可读的 entity_type / name
                if not entity_type and "from_type" in log:
                    rel_type = log.get("rel_type", "")
                    entity_type = f"Relation[{rel_type}]"
                    name = (
                        f"{log.get('from_type', '')}::"
                        f"{log.get('from_name', '')} -> "
                        f"{log.get('to_type', '')}::"
                        f"{log.get('to_name', '')}"
                    )
                conn.execute(
                    """
                    INSERT INTO conflict_logs
                    (entity_type, name, property_key, old_value, new_value, timestamp, doc_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entity_type,
                        name,
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

    def get_relation_feedback_score(
        self,
        relation_type: str,
        relation_from_type: str,
        relation_from_name: str,
        relation_to_type: str,
        relation_to_name: str,
    ) -> int:
        """获取关系的累计反馈评分."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT SUM(rating) as score FROM feedback"
                " WHERE relation_type = ? AND relation_from_type = ?"
                " AND relation_from_name = ? AND relation_to_type = ?"
                " AND relation_to_name = ?",
                (
                    relation_type,
                    relation_from_type,
                    relation_from_name,
                    relation_to_type,
                    relation_to_name,
                ),
            ).fetchone()
        return row["score"] or 0

    # -- 状态统计（用于 status 工具） --

    def count_documents_by_status(self) -> dict[str, int]:
        """按状态统计文档数量."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as c FROM documents GROUP BY status"
            ).fetchall()
        return {row["status"]: row["c"] for row in rows}

    def count_blocks_by_status(self, doc_id: str | None = None) -> dict[str, int]:
        """按状态统计 content_blocks 数量，可限定文档."""
        sql = "SELECT status, COUNT(*) as c FROM content_blocks WHERE 1=1"
        params: list = []
        if doc_id:
            sql += " AND doc_id = ?"
            params.append(doc_id)
        sql += " GROUP BY status"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        result: dict[str, int] = {"pending": 0, "embedded": 0, "done": 0, "skipped": 0, "failed": 0}
        for row in rows:
            status = row["status"] or "pending"
            result[status] = row["c"]
        return result

    def count_blocks_by_doc(self) -> list[dict]:
        """按文档统计 content_blocks 数量及各状态数量."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, status, COUNT(*) as c
                FROM content_blocks
                GROUP BY doc_id, status
                ORDER BY doc_id
                """
            ).fetchall()
        from collections import defaultdict
        docs: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "pending": 0, "embedded": 0, "done": 0, "skipped": 0, "failed": 0}
        )
        for row in rows:
            doc_id = row["doc_id"]
            status = row["status"] or "pending"
            count = row["c"]
            docs[doc_id]["total"] += count
            docs[doc_id][status] = count
        return [
            {"doc_id": doc_id, **counts}
            for doc_id, counts in sorted(docs.items())
        ]

    def count_headings(self) -> dict[str, Any]:
        """统计 heading_maps 总数及层级分布."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM heading_maps").fetchone()["c"]
            level_rows = conn.execute(
                "SELECT heading_level, COUNT(*) as c FROM heading_maps GROUP BY heading_level"
            ).fetchall()
        return {
            "total": total,
            "by_level": {str(row["heading_level"]): row["c"] for row in level_rows},
        }

    def count_headings_by_doc(self) -> list[dict]:
        """按文档统计 heading_maps 数量."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id, COUNT(*) as c FROM heading_maps GROUP BY doc_id ORDER BY doc_id"
            ).fetchall()
        return [{"doc_id": row["doc_id"], "count": row["c"]} for row in rows]

    def count_conflict_logs(self) -> int:
        """冲突日志总数."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) as c FROM conflict_logs").fetchone()["c"] or 0

    def get_recent_conflicts(self, limit: int = 20) -> list[dict]:
        """最近冲突日志."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conflict_logs ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_feedback(self) -> dict[str, Any]:
        """反馈统计：总数、平均评分、低分数量."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM feedback").fetchone()["c"] or 0
            avg_row = conn.execute("SELECT AVG(rating) as avg FROM feedback").fetchone()
            low = conn.execute(
                "SELECT COUNT(*) as c FROM feedback WHERE rating <= 0"
            ).fetchone()["c"] or 0
        return {
            "total": total,
            "average_rating": round(float(avg_row["avg"] or 0), 2),
            "low_rating_count": low,
        }

    def get_low_rating_feedback(self, limit: int = 20) -> list[dict]:
        """低分反馈记录."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE rating <= 0 ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_failed_documents(self, limit: int = 100) -> list[Document | None]:
        """获取失败的文档列表."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id FROM documents WHERE status = ? ORDER BY extracted_at DESC LIMIT ?",
                ("failed", limit),
            ).fetchall()
        return [self.get_document(row["doc_id"]) for row in rows if row["doc_id"]]

    def get_pending_blocks_summary(self, limit: int = 20) -> list[dict]:
        """获取 pending block 摘要."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, doc_id, block_id, seq_index FROM content_blocks "
                "WHERE status = 'pending' ORDER BY doc_id, seq_index LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- 评估数据集 --

    def save_eval_dataset(self, dataset: EvalDataset) -> None:
        """保存评估数据集及其问题列表."""
        created_at = dataset.created_at or datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_datasets (name, metadata, created_at)
                VALUES (?, ?, ?)
                """,
                (
                    dataset.name,
                    json.dumps(dataset.metadata, ensure_ascii=False),
                    created_at,
                ),
            )
            conn.execute("DELETE FROM eval_questions WHERE dataset_name = ?", (dataset.name,))
            for q in dataset.questions:
                conn.execute(
                    """
                    INSERT INTO eval_questions (
                        dataset_name, question, ground_truth_answer,
                        ground_truth_context_ids, ground_truth_doc_ids, tags, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset.name,
                        q.question,
                        q.ground_truth_answer,
                        json.dumps(q.ground_truth_context_ids, ensure_ascii=False),
                        json.dumps(q.ground_truth_doc_ids, ensure_ascii=False),
                        json.dumps(q.tags, ensure_ascii=False),
                        json.dumps(q.metadata, ensure_ascii=False),
                    ),
                )

    def load_eval_dataset(self, name: str) -> EvalDataset:
        """加载评估数据集及其问题列表."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM eval_datasets WHERE name = ?", (name,)
            ).fetchone()
            if not row:
                raise ValueError(f"Eval dataset not found: {name}")
            q_rows = conn.execute(
                "SELECT * FROM eval_questions WHERE dataset_name = ? ORDER BY id",
                (name,),
            ).fetchall()
        questions = [
            EvalQuestion(
                id=r["id"],
                question=r["question"],
                ground_truth_answer=r["ground_truth_answer"] or "",
                ground_truth_context_ids=json.loads(r["ground_truth_context_ids"] or "[]"),
                ground_truth_doc_ids=json.loads(r["ground_truth_doc_ids"] or "[]"),
                tags=json.loads(r["tags"] or "[]"),
                metadata=json.loads(r["metadata"] or "{}"),
            )
            for r in q_rows
        ]
        return EvalDataset(
            name=row["name"],
            questions=questions,
            created_at=row["created_at"] or "",
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def list_eval_datasets(self) -> list[str]:
        """列出所有评估数据集名称，按创建时间降序."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM eval_datasets ORDER BY created_at DESC"
            ).fetchall()
        return [r["name"] for r in rows]

    def delete_eval_dataset(self, name: str) -> bool:
        """删除评估数据集及其问题."""
        with self._connect() as conn:
            conn.execute("DELETE FROM eval_questions WHERE dataset_name = ?", (name,))
            cur = conn.execute("DELETE FROM eval_datasets WHERE name = ?", (name,))
            return cur.rowcount > 0
