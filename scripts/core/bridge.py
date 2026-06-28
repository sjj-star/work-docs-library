"""实体与 Chunk 的双向桥接索引.

纯机制层：无策略参数，只维护 chunk_db_id ↔ (entity_type, entity_name) 的多对多映射.
生命周期由 KnowledgeBaseService 管理，但本模块可独立测试和复用.
"""

import logging
from typing import Any, NamedTuple

from .models import Chunk

logger = logging.getLogger(__name__)


class _EntityRef(NamedTuple):
    """实体引用 — 不可变、可哈希，用作桥接索引的 key."""

    entity_type: str
    entity_name: str


class EntityChunkBridge:
    """实体与 Chunk 的双向桥接索引.

    纯机制层：无策略参数，只维护 chunk_db_id ↔ _EntityRef 的多对多映射.
    """

    def __init__(self) -> None:
        """初始化空的双向桥接索引."""
        self._forward: dict[int, set[_EntityRef]] = {}
        self._reverse: dict[_EntityRef, set[int]] = {}

    @staticmethod
    def extract_refs(chunk: Chunk | dict[str, Any]) -> set[_EntityRef]:
        """从 chunk 或 block metadata 提取实体引用."""
        refs: set[_EntityRef] = set()
        meta = chunk.metadata if isinstance(chunk, Chunk) else chunk.get("metadata", {})
        for me in meta.get("extracted_entities", []):
            et = me.get("type", "")
            en = me.get("name", "")
            if et and en:
                refs.add(_EntityRef(et, en))
        return refs

    def rebuild(self, db) -> None:
        """全量重建：遍历 SQLite 所有 content_blocks 的 metadata.

        Args:
            db: KnowledgeDB 实例或任何提供 list_documents / query_blocks_by_doc 的对象.
        """
        self._forward.clear()
        self._reverse.clear()
        for doc in db.list_documents():
            for block in db.query_blocks_by_doc(doc.doc_id):
                self.attach(block["id"], self.extract_refs(block))
        logger.info(
            f"Bridge 重建完成 | entities={len(self._reverse)} | blocks={len(self._forward)}"
        )

    def attach(self, chunk_db_id: int, entity_refs: set[_EntityRef]) -> None:
        """绑定 chunk 与实体引用（幂等：同一 chunk 重复 attach 不会累积）."""
        self.detach(chunk_db_id)
        if not entity_refs:
            return
        self._forward[chunk_db_id] = set(entity_refs)
        for ref in entity_refs:
            self._reverse.setdefault(ref, set()).add(chunk_db_id)

    def detach(self, chunk_db_id: int) -> None:
        """解绑 chunk 与所有实体引用."""
        old_refs = self._forward.pop(chunk_db_id, set())
        for ref in old_refs:
            if ref in self._reverse:
                self._reverse[ref].discard(chunk_db_id)
                if not self._reverse[ref]:
                    del self._reverse[ref]

    def get_entities(self, chunk_db_id: int) -> set[_EntityRef]:
        """正向查询：chunk 中提及的实体（O(1)）."""
        return set(self._forward.get(chunk_db_id, set()))

    def get_chunks(self, entity_ref: _EntityRef) -> set[int]:
        """反向查询：提及该实体的所有 blocks（O(1)）."""
        return set(self._reverse.get(entity_ref, set()))

    def get_chunk_ids(self) -> set[int]:
        """返回所有已索引的 chunk_db_id 集合."""
        return set(self._forward.keys())
