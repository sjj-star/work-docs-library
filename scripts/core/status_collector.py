"""status_collector 模块.

为 status 工具收集数据库、向量索引、知识图谱、配置、数据质量等
结构化统计信息。
"""

import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Config

if TYPE_CHECKING:
    from .knowledge_base_service import KnowledgeBaseService

logger = logging.getLogger(__name__)


class StatusCollector:
    """结构化状态收集器."""

    def __init__(self, svc: "KnowledgeBaseService") -> None:
        """初始化.

        Args:
            svc: KnowledgeBaseService 实例，提供 db / vec / graph 访问。
        """
        self.svc = svc

    # ------------------------------------------------------------------
    # documents
    # ------------------------------------------------------------------

    def collect_documents_status(self, top_n: int = 20) -> dict[str, Any]:
        """文档级状态."""
        docs = self.svc.list_documents()
        status_dist = self.svc.db.count_documents_by_status()

        # 每个文档的 block/向量/图谱统计
        block_dist = {item["doc_id"]: item for item in self.svc.db.count_blocks_by_doc()}
        vector_by_doc = self._vectors_by_doc()
        graph_by_doc = self.svc.graph.stats_by_doc()

        doc_list = []
        for doc in docs:
            doc_id = doc.doc_id
            blocks = block_dist.get(
                doc_id,
                {"total": 0, "pending": 0, "embedded": 0, "done": 0, "skipped": 0, "failed": 0},
            )
            doc_list.append(
                {
                    "doc_id": doc_id,
                    "title": doc.title,
                    "status": doc.status,
                    "total_pages": doc.total_pages,
                    "chapters": len(doc.chapters) if doc.chapters else 0,
                    "blocks": blocks,
                    "vectors": vector_by_doc.get(doc_id, 0),
                    "graph_entities": graph_by_doc.get(doc_id, {}).get("entities", 0),
                    "graph_relations": graph_by_doc.get(doc_id, {}).get("relations", 0),
                    "extracted_at": doc.extracted_at,
                }
            )

        failed = [d for d in doc_list if d["status"] == "failed"]
        return {
            "success": True,
            "scope": "documents",
            "summary": {
                "total_documents": len(docs),
                "status_distribution": status_dist,
            },
            "documents": doc_list[:top_n] if top_n > 0 else doc_list,
            "failed_documents": failed[:top_n] if top_n > 0 else failed,
        }

    # ------------------------------------------------------------------
    # vectors
    # ------------------------------------------------------------------

    def collect_vectors_status(self) -> dict[str, Any]:
        """向量索引状态."""
        index_info = self.svc.vec.index_info()
        block_status = self.svc.db.count_blocks_by_status()
        db_embedded = int(block_status.get("embedded", 0))
        db_pending = int(block_status.get("pending", 0))
        db_failed = int(block_status.get("failed", 0))
        faiss_count = int(index_info["total_vectors"])
        return {
            "success": True,
            "scope": "vectors",
            "index": index_info,
            "database": {
                "embedded_blocks": db_embedded,
                "pending_blocks": db_pending,
                "failed_blocks": db_failed,
            },
            "consistency": {
                "ok": db_embedded == faiss_count,
                "delta": db_embedded - faiss_count,
            },
            "by_document": sorted(
                [{"doc_id": k, "vectors": v} for k, v in self._vectors_by_doc().items()],
                key=lambda x: x["doc_id"],
            ),
        }

    def _vectors_by_doc(self) -> dict[str, int]:
        """计算每个文档在 FAISS 中的向量数量."""
        ids = self.svc.vec.list_ids()
        if not ids:
            return {}
        result: dict[str, int] = defaultdict(int)
        # 批量查询 block 所属 doc_id
        for db_id in ids:
            block = self.svc.db.get_block_by_db_id(db_id)
            if block:
                result[block["doc_id"]] += 1
        return dict(result)

    # ------------------------------------------------------------------
    # graph
    # ------------------------------------------------------------------

    def collect_graph_status(self) -> dict[str, Any]:
        """知识图谱状态."""
        stats = self.svc.graph.stats()
        global_path = Config.DB_PATH.parent / Config.GRAPH_OUTPUT_DIR / "global.json"
        subgraph_dir = Config.DB_PATH.parent / Config.GRAPH_OUTPUT_DIR
        subgraph_files = []
        if subgraph_dir.exists():
            subgraph_files = sorted(
                [p.name for p in subgraph_dir.glob("*.json") if p.name != "global.json"]
            )

        return {
            "success": True,
            "scope": "graph",
            "summary": {
                "node_count": stats["nodes"],
                "edge_count": stats["edges"],
                **self.svc.graph.verified_stats(),
                **self.svc.graph.low_confidence_stats(),
                **self.svc.graph.source_gaps(),
                "global_json_exists": global_path.exists(),
                "global_json_size_bytes": global_path.stat().st_size if global_path.exists() else 0,
            },
            "entity_type_distribution": dict(self.svc.graph.entity_type_distribution()),
            "relation_type_distribution": dict(self.svc.graph.relation_type_distribution()),
            "by_document": sorted(
                [
                    {"doc_id": k, "entities": v["entities"], "relations": v["relations"]}
                    for k, v in self.svc.graph.stats_by_doc().items()
                ],
                key=lambda x: x["doc_id"],
            ),
            "document_subgraph_files": subgraph_files,
        }

    # ------------------------------------------------------------------
    # blocks
    # ------------------------------------------------------------------

    def collect_blocks_status(self, top_n: int = 20) -> dict[str, Any]:
        """content_blocks 状态."""
        global_status = self.svc.db.count_blocks_by_status()
        global_status["total"] = sum(global_status.values())
        by_doc = self.svc.db.count_blocks_by_doc()
        pending = self.svc.db.get_pending_blocks_summary(top_n)
        return {
            "success": True,
            "scope": "blocks",
            "summary": global_status,
            "by_document": by_doc,
            "pending_blocks": pending,
        }

    # ------------------------------------------------------------------
    # headings
    # ------------------------------------------------------------------

    def collect_headings_status(self) -> dict[str, Any]:
        """heading_maps 状态."""
        return {
            "success": True,
            "scope": "headings",
            "summary": self.svc.db.count_headings(),
            "by_document": self.svc.db.count_headings_by_doc(),
        }

    # ------------------------------------------------------------------
    # conflicts
    # ------------------------------------------------------------------

    def collect_conflicts_status(self, top_n: int = 20) -> dict[str, Any]:
        """冲突日志状态."""
        total = self.svc.db.count_conflict_logs()
        recent = self.svc.db.get_recent_conflicts(top_n)
        # 按 entity 聚合
        by_entity: dict[str, int] = defaultdict(int)
        for log in recent:
            key = f"{log.get('entity_type', '')}::{log.get('name', '')}"
            by_entity[key] += 1
        return {
            "success": True,
            "scope": "conflicts",
            "summary": {"total": total},
            "recent_conflicts": recent,
            "by_entity": dict(sorted(by_entity.items(), key=lambda x: -x[1])[:top_n]),
        }

    # ------------------------------------------------------------------
    # feedback
    # ------------------------------------------------------------------

    def collect_feedback_status(self, top_n: int = 20) -> dict[str, Any]:
        """反馈状态."""
        fb_stats = self.svc.db.count_feedback()
        low = self.svc.db.get_low_rating_feedback(top_n)
        return {
            "success": True,
            "scope": "feedback",
            "summary": fb_stats,
            "low_rating_records": low,
        }

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------

    def collect_config_status(self) -> dict[str, Any]:
        """运行配置状态（脱敏）."""
        cfg = Config.to_dict(mask_sensitive=True)
        allowed_dirs = [
            str(Path.cwd()),
            str(Config.DB_PATH.parent),
            str(Path(tempfile.gettempdir())),
        ]
        return {
            "success": True,
            "scope": "config",
            "config": cfg,
            "paths": {
                "db_path": str(Config.DB_PATH),
                "faiss_index_path": str(Config.FAISS_INDEX_PATH),
                "graph_output_dir": str(Config.DB_PATH.parent / Config.GRAPH_OUTPUT_DIR),
                "allowed_dirs": allowed_dirs,
            },
        }

    # ------------------------------------------------------------------
    # quality
    # ------------------------------------------------------------------

    def collect_quality_status(self) -> dict[str, Any]:
        """数据质量检查."""
        issues: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}

        # 文档失败
        failed_docs = self.svc.db.get_failed_documents(limit=100)
        if failed_docs:
            issues.append(
                {
                    "severity": "error",
                    "category": "document",
                    "message": f"存在 {len(failed_docs)} 个处理失败的文档",
                    "count": len(failed_docs),
                    "suggested_action": "使用 reprocess 重试失败文档",
                }
            )

        # block 状态
        block_status = self.svc.db.count_blocks_by_status()
        metrics["pending_blocks"] = block_status.get("pending", 0)
        metrics["failed_blocks"] = block_status.get("failed", 0)
        if metrics["pending_blocks"] > 0:
            issues.append(
                {
                    "severity": "warning",
                    "category": "block",
                    "message": f"有 {metrics['pending_blocks']} 个 content_blocks 尚未 embedding",
                    "count": metrics["pending_blocks"],
                    "suggested_action": "检查 ingest pipeline 是否完成，必要时执行 reprocess",
                }
            )

        # FAISS 一致性
        index_info = self.svc.vec.index_info()
        db_embedded = block_status.get("embedded", 0)
        faiss_count = index_info["total_vectors"]
        metrics["embedded_blocks"] = db_embedded
        metrics["faiss_vectors"] = faiss_count
        if db_embedded != faiss_count:
            msg = (
                f"DB embedded blocks ({db_embedded}) 与 "
                f"FAISS vectors ({faiss_count}) 不一致"
            )
            issues.append(
                {
                    "severity": "error",
                    "category": "vector",
                    "message": msg,
                    "count": abs(db_embedded - faiss_count),
                    "suggested_action": "删除 faiss.index 并重新处理所有文档",
                }
            )

        # 图谱 global.json
        global_path = Config.DB_PATH.parent / Config.GRAPH_OUTPUT_DIR / "global.json"
        subgraph_dir = Config.DB_PATH.parent / Config.GRAPH_OUTPUT_DIR
        has_subgraphs = bool(subgraph_dir.exists() and list(subgraph_dir.glob("*.json")))
        metrics["global_json_exists"] = global_path.exists()
        if not global_path.exists() and has_subgraphs:
            issues.append(
                {
                    "severity": "error",
                    "category": "graph",
                    "message": "存在文档子图但缺少 global.json",
                    "suggested_action": "执行 rebuild_global_graph 重建全局图",
                }
            )

        # 图谱质量
        metrics.update(self.svc.graph.low_confidence_stats())
        metrics.update(self.svc.graph.source_gaps())
        metrics.update(self.svc.graph.orphan_node_stats())
        if metrics["low_confidence_entities"] > 0:
            issues.append(
                {
                    "severity": "warning",
                    "category": "graph",
                    "message": f"存在 {metrics['low_confidence_entities']} 个低置信度实体",
                    "count": metrics["low_confidence_entities"],
                    "suggested_action": "使用 graph_query 查看并人工验证",
                }
            )
        if metrics["entities_without_source"] > 0 or metrics["relations_without_source"] > 0:
            src_msg = (
                f"存在 {metrics['entities_without_source']} 个无来源实体和 "
                f"{metrics['relations_without_source']} 个无来源关系"
            )
            issues.append(
                {
                    "severity": "warning",
                    "category": "graph",
                    "message": src_msg,
                    "suggested_action": "检查数据完整性",
                }
            )

        # 冲突
        conflict_total = self.svc.db.count_conflict_logs()
        metrics["conflict_logs"] = conflict_total
        if conflict_total > 0:
            issues.append(
                {
                    "severity": "warning",
                    "category": "conflict",
                    "message": f"存在 {conflict_total} 条冲突日志",
                    "count": conflict_total,
                    "suggested_action": "使用 graph_conflicts 查看详情",
                }
            )

        # 反馈
        fb_stats = self.svc.db.count_feedback()
        metrics["feedback_total"] = fb_stats["total"]
        metrics["feedback_low_rating"] = fb_stats["low_rating_count"]
        if fb_stats["low_rating_count"] > 0:
            issues.append(
                {
                    "severity": "warning",
                    "category": "feedback",
                    "message": f"存在 {fb_stats['low_rating_count']} 条低分反馈",
                    "count": fb_stats["low_rating_count"],
                    "suggested_action": "使用 graph_query/feedback 查看并修正",
                }
            )

        score = max(
            0, 100 - len(issues) * 5 - sum(i.get("count", 0) for i in issues if i.get("count"))
        )
        return {
            "success": True,
            "scope": "quality",
            "score": score,
            "metrics": metrics,
            "issues": issues,
        }

    # ------------------------------------------------------------------
    # ingest pipeline
    # ------------------------------------------------------------------

    def collect_ingest_pipeline_status(self) -> dict[str, Any]:
        """每个文档的 ingest pipeline 阶段推断."""
        docs = self.svc.list_documents()
        block_dist = {item["doc_id"]: item for item in self.svc.db.count_blocks_by_doc()}
        graph_by_doc = self.svc.graph.stats_by_doc()
        vector_by_doc = self._vectors_by_doc()

        pipelines = []
        for doc in docs:
            doc_id = doc.doc_id
            blocks = block_dist.get(
                doc_id, {"total": 0, "pending": 0, "embedded": 0, "done": 0, "failed": 0}
            )
            total = blocks["total"]
            embedded = blocks["embedded"]
            entities = graph_by_doc.get(doc_id, {}).get("entities", 0)
            relations = graph_by_doc.get(doc_id, {}).get("relations", 0)
            vectors = vector_by_doc.get(doc_id, 0)
            pipelines.append(
                {
                    "doc_id": doc_id,
                    "title": doc.title,
                    "status": doc.status,
                    "stages": {
                        "parsed": total > 0,
                        "batches_built": total > 0,
                        "extracted": entities > 0 or relations > 0,
                        "embedded": total > 0 and total == embedded and vectors == total,
                        "done": doc.status == "done",
                    },
                    "blocks": blocks,
                    "vectors": vectors,
                    "entities": entities,
                    "relations": relations,
                }
            )
        return {
            "success": True,
            "scope": "ingest_pipeline",
            "pipelines": pipelines,
        }

    # ------------------------------------------------------------------
    # all
    # ------------------------------------------------------------------

    def collect_all(self, top_n: int = 20) -> dict[str, Any]:
        """返回所有 scope 的数据（不含 overview）."""
        return {
            "success": True,
            "scope": "all",
            "documents": self.collect_documents_status(top_n),
            "vectors": self.collect_vectors_status(),
            "graph": self.collect_graph_status(),
            "blocks": self.collect_blocks_status(top_n),
            "headings": self.collect_headings_status(),
            "conflicts": self.collect_conflicts_status(top_n),
            "feedback": self.collect_feedback_status(top_n),
            "config": self.collect_config_status(),
            "quality": self.collect_quality_status(),
            "ingest_pipeline": self.collect_ingest_pipeline_status(),
        }
