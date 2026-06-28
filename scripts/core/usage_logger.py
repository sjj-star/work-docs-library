"""使用日志记录模块.

统一记录知识库查询调用、向量命中、实体/关系命中、用户标记.
不记录原始用户问题全文，只记录参数摘要和检索上下文引用.
"""

import logging
import time
from typing import Any

from .db import KnowledgeDB
from .models import Chunk

logger = logging.getLogger(__name__)


class UsageLogger:
    """知识库使用日志记录器."""

    def __init__(self, db: KnowledgeDB) -> None:
        """初始化.

        Args:
            db: KnowledgeDB 实例.
        """
        self._db = db

    def log_query(
        self,
        tool_name: str,
        mode: str | None,
        params: dict[str, Any],
        result: dict[str, Any],
        session_id: str | None = None,
        start_time: float | None = None,
    ) -> int:
        """记录一次查询调用.

        Args:
            tool_name: search / explore / read / status / ingest
            mode: 子模式，例如 hybrid / entity
            params: 调用参数摘要
            result: 查询返回结果字典
            session_id: 可选会话跟踪 ID
            start_time: 可选起始时间戳，用于计算耗时

        Returns:
            日志行 id
        """
        elapsed_ms = 0
        if start_time is not None:
            elapsed_ms = int((time.time() - start_time) * 1000)

        vector_hits = self._extract_vector_hits(result)
        entity_hits = self._extract_entity_hits(result)
        relation_hits = self._extract_relation_hits(result)

        # 摘要 params：移除过长的文本字段
        summary_params = self._summarize_params(params)

        try:
            log_id = self._db.log_usage(
                tool_name=tool_name,
                mode=mode,
                params=summary_params,
                vector_hits=vector_hits,
                entity_hits=entity_hits,
                relation_hits=relation_hits,
                elapsed_ms=elapsed_ms,
                session_id=session_id,
            )

            # 更新向量 block 命中计数
            block_ids = [h["block_db_id"] for h in vector_hits if h.get("block_db_id")]
            if block_ids:
                self._db.increment_block_activation(block_ids)

            return log_id
        except Exception:
            logger.exception("Failed to log usage")
            return -1

    @staticmethod
    def _extract_vector_hits(result: dict[str, Any]) -> list[dict[str, Any]]:
        """从结果中提取向量命中块."""
        hits: list[dict[str, Any]] = []
        for item in result.get("chunks", []):
            if isinstance(item, dict):
                chunk = item.get("chunk")
                score = item.get("score")
            else:
                chunk = item
                score = None
            if isinstance(chunk, Chunk):
                hits.append(
                    {
                        "block_db_id": chunk.id,
                        "doc_id": chunk.doc_id,
                        "chapter_title": chunk.chapter_title,
                        "score": round(float(score), 4) if score is not None else None,
                    }
                )
            elif isinstance(chunk, dict):
                hits.append(
                    {
                        "block_db_id": chunk.get("id") or chunk.get("chunk_db_id"),
                        "doc_id": chunk.get("doc_id"),
                        "chapter_title": chunk.get("chapter_title"),
                        "score": round(float(score), 4) if score is not None else None,
                    }
                )
        return hits

    @staticmethod
    def _extract_entity_hits(result: dict[str, Any]) -> list[dict[str, Any]]:
        """从结果中提取命中实体."""
        entities: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in result.get("entities", []):
            if hasattr(raw, "entity_type") and hasattr(raw, "name"):
                entity_type = raw.entity_type
                name = raw.name
            elif isinstance(raw, dict):
                entity_type = raw.get("entity_type")
                name = raw.get("name")
            else:
                continue
            if not entity_type or not name:
                continue
            key = f"{entity_type}::{name}"
            if key in seen:
                continue
            seen.add(key)
            entities.append({"type": entity_type, "name": name})
        return entities

    @staticmethod
    def _extract_relation_hits(result: dict[str, Any]) -> list[dict[str, Any]]:
        """从结果中提取命中关系."""
        relations: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in result.get("relations", []):
            if hasattr(raw, "rel_type"):
                rel_type = raw.rel_type
                from_type = raw.from_type
                from_name = raw.from_name
                to_type = raw.to_type
                to_name = raw.to_name
            elif isinstance(raw, dict):
                rel_type = raw.get("rel_type")
                from_type = raw.get("from_type")
                from_name = raw.get("from_name")
                to_type = raw.get("to_type")
                to_name = raw.get("to_name")
            else:
                continue
            if not all([rel_type, from_type, from_name, to_type, to_name]):
                continue
            key = f"{from_type}::{from_name}--{rel_type}--{to_type}::{to_name}"
            if key in seen:
                continue
            seen.add(key)
            relations.append(
                {
                    "rel_type": rel_type,
                    "from_type": from_type,
                    "from_name": from_name,
                    "to_type": to_type,
                    "to_name": to_name,
                }
            )
        return relations

    @staticmethod
    def _summarize_params(params: dict[str, Any]) -> dict[str, Any]:
        """对参数做摘要，避免记录过长文本."""
        summary: dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, str) and len(v) > 200:
                summary[k] = v[:200] + "..."
            else:
                summary[k] = v
        return summary
