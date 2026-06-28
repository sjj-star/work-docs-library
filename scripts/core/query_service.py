"""QueryService — 查询组合与上下文聚合层.

职责：
- 为 MCP 新接口（search / explore / read）提供统一查询入口
- 组合 KnowledgeBaseService 的原子能力，返回丰富上下文
- 不直接操作 DB/Vector/Graph，只通过 service 读取

设计原则：
- 本地不执行 LLM 综合/智能路由，只返回结构化数据
- 所有智能化动态分析由外部 Agent/Skill 完成
"""

import copy
import logging
import time
from typing import Any

from .config import Config
from .knowledge_base_service import KnowledgeBaseService
from .models import Chunk
from .usage_logger import UsageLogger

logger = logging.getLogger(__name__)


class QueryService:
    """查询服务：聚合 chunks / entities / relations / sources 上下文."""

    def __init__(self, service: KnowledgeBaseService) -> None:
        """依赖注入 KnowledgeBaseService 实例.

        Args:
            service: 已初始化的 KnowledgeBaseService，提供底层原子方法.
        """
        self._svc = service
        self._usage_logger = UsageLogger(service.db)

    # ------------------------------------------------------------------
    # 公共查询接口
    # ------------------------------------------------------------------

    def search(
        self,
        text: str,
        top_k: int = Config.PLUGIN_SEARCH_TOP_K,
        mode: str = "hybrid",
        include_graph: bool = True,
        graph_depth: int = Config.PLUGIN_SUBGRAPH_DEPTH,
        rerank_candidate_k: int | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """统一搜索入口.

        Args:
            text: 搜索文本
            top_k: 返回 chunk 数量
            mode: semantic | hybrid | reranked
            include_graph: 是否通过桥接索引扩展相关实体/关系
            graph_depth: 子图扩展深度
            rerank_candidate_k: reranked 模式候选集大小
            session_id: 可选会话跟踪 ID

        Returns:
            {"chunks": [...], "entities": [...], "relations": [...], "source_documents": [...]}
        """
        start = time.time()
        if mode == "semantic":
            raw_results = self._svc.search_semantic(text, top_k=top_k)
        elif mode == "hybrid":
            raw_results = self._svc.search_hybrid(text, top_k=top_k)
        elif mode == "reranked":
            raw_results = self._svc.search_reranked(
                text, top_k=top_k, candidate_k=rerank_candidate_k
            )
        else:
            raise ValueError(f"Unsupported search mode: {mode}")

        chunks = [{"score": r["score"], "chunk": copy.deepcopy(r["chunk"])} for r in raw_results]

        if not include_graph:
            result = {
                "chunks": chunks,
                "entities": [],
                "relations": [],
                "source_documents": self._collect_source_documents(chunks),
            }
            self._usage_logger.log_query(
                tool_name="search",
                mode=mode,
                params={"text": text, "top_k": top_k, "include_graph": False},
                result=result,
                session_id=session_id,
                start_time=start,
            )
            return result

        context = self._collect_related_context(
            [c["chunk"].id for c in chunks if c["chunk"].id is not None],
            graph_depth=graph_depth,
        )

        result = {
            "chunks": chunks,
            "entities": context["entities"],
            "relations": context["relations"],
            "source_documents": self._collect_source_documents(chunks),
        }
        self._usage_logger.log_query(
            tool_name="search",
            mode=mode,
            params={
                "text": text,
                "top_k": top_k,
                "include_graph": True,
                "graph_depth": graph_depth,
            },
            result=result,
            session_id=session_id,
            start_time=start,
        )
        return result

    def explore(self, mode: str, session_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """统一图谱探索入口.

        Args:
            mode: entity | neighbors | subgraph | path | provenance | conflicts
            session_id: 可选会话跟踪 ID
            **kwargs: 各模式所需参数

        Returns:
            统一包装的结果字典，只有对应 mode 的字段有值.
        """
        start = time.time()
        result: dict[str, Any] = {"success": True, "mode": mode}

        if mode == "entity":
            result.update(self._explore_entity(**kwargs))
        elif mode == "neighbors":
            result.update(self._explore_neighbors(**kwargs))
        elif mode == "subgraph":
            result.update(self._explore_subgraph(**kwargs))
        elif mode == "path":
            result.update(self._explore_path(**kwargs))
        elif mode == "provenance":
            result.update(self._explore_provenance(**kwargs))
        elif mode == "conflicts":
            result.update(self._explore_conflicts(**kwargs))
        else:
            raise ValueError(f"Unsupported explore mode: {mode}")

        self._usage_logger.log_query(
            tool_name="explore",
            mode=mode,
            params={"mode": mode, **kwargs},
            result=result,
            session_id=session_id,
            start_time=start,
        )
        return result

    def read(
        self,
        doc_id: str | None = None,
        chapter: str | None = None,
        chapter_regex: str | None = None,
        concept: str | None = None,
        chunk_db_id: int | None = None,
        with_entities: bool = True,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """统一内容读取入口.

        Args:
            doc_id: 文档 ID
            chapter: 章节标题子串匹配
            chapter_regex: 章节标题正则匹配
            concept: 概念名匹配
            chunk_db_id: 直接按 block DB ID 查询
            with_entities: 是否返回关联实体/关系
            session_id: 可选会话跟踪 ID

        Returns:
            {"query_type": ..., "content": ..., "chunks": ..., "entities": ..., "relations": ...}
        """
        start = time.time()
        if chunk_db_id is not None:
            result = self._read_by_chunk(chunk_db_id, with_entities=with_entities)
        elif chapter is not None or chapter_regex is not None or concept is not None:
            chunks = self._svc.query_chunks(
                doc_id=doc_id,
                chapter=chapter,
                chapter_regex=chapter_regex,
                concept=concept,
            )
            result = self._build_read_result(chunks, with_entities=with_entities)
        else:
            raise ValueError("Provide chunk_db_id, or doc_id with chapter/chapter_regex/concept")

        self._usage_logger.log_query(
            tool_name="read",
            mode=None,
            params={
                "doc_id": doc_id,
                "chapter": chapter,
                "chapter_regex": chapter_regex,
                "concept": concept,
                "chunk_db_id": chunk_db_id,
                "with_entities": with_entities,
            },
            result=result,
            session_id=session_id,
            start_time=start,
        )
        return result

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _explore_entity(
        self,
        entity_type: str | None = None,
        name: str | None = None,
        name_pattern: str | None = None,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        """查询实体本身."""
        if name and entity_type:
            entity = self._svc.get_entity(entity_type, name)
            if entity:
                entity = copy.deepcopy(entity)
                self._svc._apply_doc_properties(entity, doc_id)
                return {"count": 1, "entities": [entity.to_dict()]}
            return {"count": 0, "entities": []}

        entities = self._svc.find_entities(
            entity_type=entity_type, name_pattern=name_pattern, doc_id=doc_id
        )
        return {"count": len(entities), "entities": [e.to_dict() for e in entities]}

    def _explore_neighbors(
        self,
        entity_type: str,
        name: str,
        rel_type: str | None = None,
        direction: str = "out",
        doc_id: str | None = None,
        depth: int = 1,
    ) -> dict[str, Any]:
        """查询实体邻居."""
        if depth <= 0:
            raise ValueError("depth must be > 0 for neighbors mode")

        # 当前只支持 depth=1 的邻居查询；depth>1 时退化为 subgraph
        if depth > 1:
            return self._explore_subgraph(
                entity_type=entity_type, name=name, depth=depth, rel_types=None, doc_id=doc_id
            )

        neighbors = self._svc.get_neighbors(entity_type, name, rel_type, direction, doc_id)
        return {
            "center": {"entity_type": entity_type, "name": name},
            "neighbor_count": len(neighbors),
            "neighbors": [
                {
                    "entity": copy.deepcopy(nentity).to_dict(),
                    "relation": rel,
                    "relation_properties": rel_props,
                }
                for nentity, rel, rel_props in neighbors
            ],
        }

    def _explore_subgraph(
        self,
        entity_type: str,
        name: str,
        depth: int = Config.PLUGIN_SUBGRAPH_DEPTH,
        rel_types: set[str] | None = None,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        """查询子图."""
        subgraph = self._svc.get_subgraph(entity_type, name, depth, rel_types)
        return {
            "center": {"entity_type": entity_type, "name": name},
            "depth": depth,
            "subgraph": {
                "node_count": subgraph.node_count,
                "edge_count": subgraph.edge_count,
                "entities": [copy.deepcopy(e).to_dict() for e in subgraph.entities()],
                "relations": [copy.deepcopy(r).to_dict() for r in subgraph.relations()],
            },
        }

    def _explore_path(
        self,
        from_type: str,
        from_name: str,
        to_type: str,
        to_name: str,
        max_depth: int = Config.PLUGIN_GRAPH_MAX_DEPTH,
    ) -> dict[str, Any]:
        """查询两实体间路径."""
        paths = self._svc.find_path(from_type, from_name, to_type, to_name, max_depth)

        def _parse_nid(nid: str) -> dict[str, str]:
            parts = nid.split("::", 1)
            return {"entity_type": parts[0], "name": parts[1] if len(parts) > 1 else nid}

        return {
            "from_entity": {"entity_type": from_type, "name": from_name},
            "to_entity": {"entity_type": to_type, "name": to_name},
            "max_depth": max_depth,
            "path_count": len(paths),
            "paths": [[_parse_nid(nid) for nid in p] for p in paths],
        }

    def _explore_provenance(
        self,
        entity_type: str,
        name: str,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        """查询实体来源."""
        entity = self._svc.get_entity(entity_type, name)
        if not entity:
            return {"provenance": []}

        provenance: list[dict[str, Any]] = []
        seen: set[tuple[str, str, int]] = set()

        chunks = self._svc.find_chunks_by_entity(entity_type, name, doc_id)
        for chunk in chunks:
            key = (chunk.doc_id, chunk.chapter_title, chunk.id)
            if key in seen:
                continue
            seen.add(key)
            provenance.append(
                {
                    "doc_id": chunk.doc_id,
                    "source_chapter": chunk.chapter_title,
                    "chunk_db_id": chunk.id,
                    "content_preview": chunk.content[:500],
                }
            )

        return {
            "entity": {"entity_type": entity_type, "name": name},
            "provenance": provenance,
        }

    def _explore_conflicts(
        self,
        entity_type: str | None = None,
        name: str | None = None,
        limit: int = Config.PLUGIN_DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        """查询冲突日志."""
        conflicts = self._svc.get_conflict_logs(entity_type, name, limit)
        return {"conflicts": conflicts}

    def _read_by_chunk(self, chunk_db_id: int, with_entities: bool = True) -> dict[str, Any]:
        """按 chunk_db_id 读取内容."""
        block = self._svc.db.get_block_by_db_id(chunk_db_id)
        if not block:
            raise ValueError(f"Block {chunk_db_id} not found")

        chunk = Chunk(
            id=block["id"],
            doc_id=block["doc_id"],
            chunk_id=block["block_id"],
            content=block["content"],
            chunk_type="text",
            chapter_title=block["metadata"].get("section_title", ""),
            metadata=block["metadata"],
            status=block["status"],
        )

        if not with_entities:
            return {
                "query_type": "chunk",
                "doc_id": chunk.doc_id,
                "chapter_title": chunk.chapter_title,
                "content": chunk.content,
                "total_chars": len(chunk.content),
                "chunks": [{"chunk_db_id": chunk.id, "chunk_id": chunk.chunk_id}],
                "entities": [],
                "relations": [],
            }

        context = self._collect_entities_for_chunk(chunk_db_id)
        return {
            "query_type": "chunk",
            "doc_id": chunk.doc_id,
            "chapter_title": chunk.chapter_title,
            "content": chunk.content,
            "total_chars": len(chunk.content),
            "chunks": [{"chunk_db_id": chunk.id, "chunk_id": chunk.chunk_id}],
            "entities": context["entities"],
            "relations": context["relations"],
        }

    def _build_read_result(self, chunks: list[Chunk], with_entities: bool = True) -> dict[str, Any]:
        """构建章节/概念查询结果."""
        if not chunks:
            raise ValueError("No content found for the given query")

        chunks.sort(key=lambda ck: (ck.id, ck.chunk_id))
        full_content = "\n\n---\n\n".join(ck.content for ck in chunks)

        result: dict[str, Any] = {
            "query_type": "chapter",
            "doc_id": chunks[0].doc_id,
            "chapter_title": chunks[0].chapter_title,
            "content": full_content,
            "total_chars": len(full_content),
            "chunks": [
                {"chunk_db_id": ck.id, "chunk_id": ck.chunk_id, "chapter_title": ck.chapter_title}
                for ck in chunks
            ],
        }

        if with_entities:
            chunk_ids = [ck.id for ck in chunks if ck.id is not None]
            context = self._collect_related_context(chunk_ids, graph_depth=0)
            result["entities"] = context["entities"]
            result["relations"] = context["relations"]
        else:
            result["entities"] = []
            result["relations"] = []

        return result

    def _collect_related_context(
        self,
        chunk_db_ids: list[int],
        graph_depth: int = Config.PLUGIN_SUBGRAPH_DEPTH,
    ) -> dict[str, list[Any]]:
        """基于 chunk 集合聚合相关实体/关系/子图（返回对象，由调用方序列化）."""
        entities: list[Any] = []
        relations: list[Any] = []
        seen_entities: set[str] = set()
        seen_relations: set[str] = set()

        for db_id in chunk_db_ids:
            chunk_context = self._collect_entities_for_chunk(db_id, graph_depth=graph_depth)
            for entity in chunk_context["entities"]:
                eid = f"{entity.entity_type}::{entity.name}"
                if eid not in seen_entities:
                    seen_entities.add(eid)
                    entities.append(entity)

            for relation in chunk_context["relations"]:
                rel_key = (
                    f"{relation.from_type}::{relation.from_name}--"
                    f"{relation.rel_type}--{relation.to_type}::{relation.to_name}"
                )
                if rel_key not in seen_relations:
                    seen_relations.add(rel_key)
                    relations.append(relation)

        return {"entities": entities, "relations": relations}

    def _collect_entities_for_chunk(
        self,
        chunk_db_id: int,
        graph_depth: int = 0,
    ) -> dict[str, list[Any]]:
        """收集单个 chunk 关联的实体/关系（含可选子图扩展）."""
        entities: list[Any] = []
        relations: list[Any] = []
        seen_entities: set[str] = set()
        seen_relations: set[str] = set()

        for ref in self._svc.bridge.get_entities(chunk_db_id):
            eid = f"{ref.entity_type}::{ref.entity_name}"
            if eid in seen_entities:
                continue
            seen_entities.add(eid)

            entity = self._svc.graph.get_entity(ref.entity_type, ref.entity_name)
            if entity:
                entities.append(copy.deepcopy(entity))

                if graph_depth > 0:
                    subgraph = self._svc.graph.get_subgraph(
                        ref.entity_type, ref.entity_name, depth=graph_depth
                    )
                    for sg_entity in subgraph.entities():
                        sg_eid = f"{sg_entity.entity_type}::{sg_entity.name}"
                        if sg_eid not in seen_entities:
                            seen_entities.add(sg_eid)
                            entities.append(copy.deepcopy(sg_entity))

                    for sg_relation in subgraph.relations():
                        rel_key = (
                            f"{sg_relation.from_type}::{sg_relation.from_name}--"
                            f"{sg_relation.rel_type}--"
                            f"{sg_relation.to_type}::{sg_relation.to_name}"
                        )
                        if rel_key not in seen_relations:
                            seen_relations.add(rel_key)
                            relations.append(copy.deepcopy(sg_relation))

                for rel in self._svc.graph.get_entity_relations(
                    entity.entity_type, entity.name, direction="both"
                ):
                    rel_key = (
                        f"{rel.from_type}::{rel.from_name}--"
                        f"{rel.rel_type}--{rel.to_type}::{rel.to_name}"
                    )
                    if rel_key not in seen_relations:
                        seen_relations.add(rel_key)
                        relations.append(copy.deepcopy(rel))

        return {"entities": entities, "relations": relations}

    def _collect_source_documents(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从 chunk 结果聚合来源文档信息."""
        seen: set[str] = set()
        docs: list[dict[str, Any]] = []
        for item in chunks:
            chunk = item["chunk"]
            if chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            doc = self._svc.get_document(chunk.doc_id)
            if doc:
                docs.append(
                    {
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "total_pages": doc.total_pages,
                    }
                )
        return docs
