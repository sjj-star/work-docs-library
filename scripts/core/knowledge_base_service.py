"""KnowledgeBaseService — 统一服务层.

封装 KnowledgeDB + VectorIndex + GraphStore + DocGraphPipeline，
为 Plugin 工具和上层应用提供一致的 API 入口。
"""

import logging

from .config import Config
from .db import KnowledgeDB
from .doc_graph_pipeline import DocGraphPipeline
from .embedding_client import EmbeddingClient
from .graph_store import GraphEntity, NetworkXGraphStore, SubGraphView
from .models import Chunk, Document
from .vector_index import VectorIndex

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """统一知识库服务层.

    职责：
    - 文档导入（ingest / reprocess）
    - Chunk 查询（结构化查询 + 语义搜索）
    - 图谱查询（实体、邻居、路径、子图）
    - 文档状态管理
    """

    def __init__(
        self,
        db: KnowledgeDB | None = None,
        vec: VectorIndex | None = None,
        graph_store: NetworkXGraphStore | None = None,
    ) -> None:
        """初始化服务层，自动加载持久化的索引和图谱."""
        self.db = db or KnowledgeDB()
        self.vec = vec or VectorIndex(dim=Config.EMBEDDING_DIMENSION)
        self.graph = graph_store or NetworkXGraphStore()
        if graph_store is None:
            self._load_all_graphs()

    # ------------------------------------------------------------------
    # 文档管理
    # ------------------------------------------------------------------

    def _load_all_graphs(self) -> None:
        """加载所有已保存的文档图谱到全局内存图，实现跨文档实体互通."""
        global_path = Config.DB_PATH.parent / "graphs" / "global.json"
        if global_path.exists():
            try:
                self.graph.load(global_path)
                logger.info(f"全局图谱已加载 | {self.graph.stats()}")
                return
            except Exception as e:
                logger.warning(f"加载全局图谱失败 | error={e}")

        # 兼容：如果没有 global.json，逐个加载文档子图
        graphs_dir = Config.DB_PATH.parent / "graphs"
        if not graphs_dir.exists():
            return
        for path in sorted(graphs_dir.glob("*.json")):
            if path.name == "global.json":
                continue
            try:
                temp = NetworkXGraphStore()
                temp.load(path)
                self._merge_graph(temp)
            except Exception as e:
                logger.warning(f"加载图谱失败 | path={path} | error={e}")
        if self.graph.stats()["nodes"] > 0:
            logger.info(f"已从文档子图构建全局图谱 | {self.graph.stats()}")

    def _merge_graph(self, other: NetworkXGraphStore) -> None:
        """将另一个图谱合并到全局图（同名同类型实体自动去重，属性合并）."""
        for entity in other.all_entities():
            self.graph.add_entity(entity)
        for relation in other.all_relations():
            self.graph.add_relation(relation)

    def _save_global_graph(self) -> None:
        """保存全局合并图谱."""
        global_path = Config.DB_PATH.parent / "graphs" / "global.json"
        global_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph.save(global_path)

    def ingest_document(self, path: str, dry_run: bool = False, force: bool = False) -> list[str]:
        """导入文档（支持单文件或目录）.

        Args:
            path: 文件或目录路径
            dry_run: 仅扫描，不实际处理
            force: 强制重新处理，忽略缓存

        Returns:
            成功处理的 doc_id 列表
        """
        # 使用独立的 graph_store，避免污染全局图
        pipe = DocGraphPipeline(db=self.db, vec=self.vec, graph_store=NetworkXGraphStore())
        try:
            doc_ids = pipe.ingest(str(path), dry_run=dry_run, force=force)
            # 重建全局图：清空后从所有文档子图重新加载，确保无幽灵数据
            self.graph.clear()
            graphs_dir = Config.DB_PATH.parent / "graphs"
            if graphs_dir.exists():
                for gp in sorted(graphs_dir.glob("*.json")):
                    if gp.name == "global.json":
                        continue
                    try:
                        temp = NetworkXGraphStore()
                        temp.load(gp)
                        self._merge_graph(temp)
                    except Exception as e:
                        logger.warning(f"加载子图失败 | path={gp} | error={e}")
            self._save_global_graph()
            return doc_ids
        finally:
            pipe.close()

    def list_documents(self) -> list[Document]:
        """列出所有已导入的文档."""
        return self.db.list_documents()

    def get_document(self, doc_id: str) -> Document | None:
        """按 doc_id 获取文档信息."""
        return self.db.get_document(doc_id)

    def get_document_progress(self, doc_id: str) -> dict[str, int]:
        """获取文档处理进度统计.

        Returns:
            {"total": int, "done": int, "embedded": int,
             "skipped": int, "pending": int, "failed": int}
        """
        with self.db._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as c FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchone()["c"]

            def _count(status: str) -> int:
                return conn.execute(
                    "SELECT COUNT(*) as c FROM chunks WHERE doc_id = ? AND status = ?",
                    (doc_id, status),
                ).fetchone()["c"]

            return {
                "total": total,
                "done": _count("done"),
                "embedded": _count("embedded"),
                "skipped": _count("skipped"),
                "pending": _count("pending"),
                "failed": _count("failed"),
            }

    def reprocess_document(self, doc_id: str) -> str:
        """强制重新处理指定文档.

        Returns:
            处理后的 doc_id

        Raises:
            ValueError: 文档不存在
        """
        doc = self.db.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document {doc_id} not found")
        # 从全局图中精确移除旧文档贡献
        self.graph.remove_document_contributions(doc_id)
        # 使用独立的 graph_store 处理
        pipe = DocGraphPipeline(db=self.db, vec=self.vec, graph_store=NetworkXGraphStore())
        try:
            result = pipe._process_one(doc.source_path, dry_run=False, force=True)
            if not result:
                raise RuntimeError(f"Reprocess failed for {doc_id}")
            # 将重新处理后的图谱合并到全局图
            graph_path = Config.DB_PATH.parent / "graphs" / f"{doc_id}.json"
            if graph_path.exists():
                temp = NetworkXGraphStore()
                temp.load(graph_path)
                self._merge_graph(temp)
            self._save_global_graph()
            return result
        finally:
            pipe.close()

    # ------------------------------------------------------------------
    # Chunk 查询
    # ------------------------------------------------------------------

    def search_semantic(self, text: str, top_k: int = 5) -> list[dict]:
        """语义向量搜索.

        Returns:
            每个结果包含 score 和 chunk 基本信息
            [{"score": float, "chunk": Chunk}, ...]
        """
        embedder = EmbeddingClient()
        try:
            emb = embedder.embed([str(text)])[0]
            hits = self.vec.search(emb, top_k=top_k)
            results = []
            for db_id, score in hits:
                chunk = self.db.get_chunk_by_db_id(db_id)
                if chunk:
                    results.append({"score": round(float(score), 4), "chunk": chunk})
            return results
        finally:
            embedder.close()

    def query_chunks(
        self,
        doc_id: str | None = None,
        chapter: str | None = None,
        chapter_regex: str | None = None,
        keyword: str | None = None,
        concept: str | None = None,
        top_k: int = 10,
    ) -> list[Chunk]:
        """统一 chunk 结构化查询.

        Args:
            doc_id: 文档 ID（chapter/concept 查询必需）
            chapter: 章节标题子串匹配
            chapter_regex: 章节标题正则匹配
            keyword: 关键词匹配（跨文档）
            concept: 概念名匹配（需 doc_id）
            top_k: 最大返回数量

        Returns:
            Chunk 列表

        Raises:
            ValueError: 缺少必需参数
        """
        results: list[Chunk] = []
        if chapter is not None:
            if not doc_id:
                raise ValueError("chapter query requires doc_id")
            results = self.db.query_by_chapter(doc_id, chapter)
        elif chapter_regex:
            if not doc_id:
                raise ValueError("chapter_regex query requires doc_id")
            results = self.db.query_by_chapter_regex(doc_id, chapter_regex)
        elif keyword:
            results = self.db.query_by_keyword(keyword)
        elif concept:
            if not doc_id:
                raise ValueError("concept query requires doc_id")
            results = self.db.query_by_concept(doc_id, concept)
        else:
            raise ValueError("Provide chapter, chapter_regex, keyword, or concept")

        return results[:top_k]

    def get_chunk_content(
        self,
        chunk_db_id: int | None = None,
        doc_id: str | None = None,
        chapter: str | None = None,
    ) -> dict:
        """获取 chunk 完整内容.

        Args:
            chunk_db_id: 直接按 chunk DB ID 查询
            doc_id: 文档 ID（配合 chapter）
            chapter: 章节标题

        Returns:
            {"query_type": str, "chunks": list[Chunk], "content": str}

        Raises:
            ValueError: 参数不合法或找不到内容
        """
        if chunk_db_id is not None:
            chunk = self.db.get_chunk_by_db_id(int(chunk_db_id))
            if not chunk:
                raise ValueError(f"Chunk {chunk_db_id} not found")
            return {
                "query_type": "chunk",
                "chunks": [chunk],
                "content": chunk.content,
            }

        if not doc_id:
            raise ValueError("Provide chunk_db_id, or doc_id with chapter")

        if chapter is not None:
            chunks = self.db.query_by_chapter(doc_id, chapter)
            query_type = "chapter"
        else:
            raise ValueError("Provide chapter with doc_id")

        if not chunks:
            raise ValueError("No content found for the given query")

        chunks.sort(key=lambda ck: (ck.id, ck.chunk_id))
        full_content = "\n\n---\n\n".join(ck.content for ck in chunks)
        return {
            "query_type": query_type,
            "chunks": chunks,
            "content": full_content,
        }

    # ------------------------------------------------------------------
    # 图谱查询
    # ------------------------------------------------------------------

    def get_entity(self, entity_type: str, name: str) -> GraphEntity | None:
        """按类型和名称获取实体."""
        return self.graph.get_entity(entity_type, name)

    def find_entities(
        self,
        entity_type: str | None = None,
        name_pattern: str | None = None,
    ) -> list[GraphEntity]:
        """搜索实体.

        Args:
            entity_type: 精确匹配实体类型
            name_pattern: 名称子串匹配（大小写不敏感）

        Returns:
            匹配的实体列表
        """
        if entity_type and not name_pattern:
            return self.graph.find_by_type(entity_type)
        if name_pattern:
            types = {entity_type} if entity_type else None
            return self.graph.search_entities(name_pattern, entity_types=types)
        return self.graph.all_entities()

    def get_neighbors(
        self,
        entity_type: str,
        name: str,
        rel_type: str | None = None,
        direction: str = "out",
    ) -> list[tuple[GraphEntity, str]]:
        """获取实体的邻居节点."""
        return self.graph.get_neighbors(entity_type, name, rel_type, direction)

    def find_path(
        self,
        from_type: str,
        from_name: str,
        to_type: str,
        to_name: str,
        max_depth: int = 3,
    ) -> list[list[str]]:
        """查找两实体间的路径.

        Returns:
            路径列表，每条路径是节点 ID 列表（如 ["Module::A", "Signal::B"]）
        """
        return self.graph.find_path(from_type, from_name, to_type, to_name, max_depth)

    def get_subgraph(
        self,
        center_type: str,
        center_name: str,
        depth: int = 1,
        rel_types: set[str] | None = None,
    ) -> SubGraphView:
        """获取以某实体为中心的子图."""
        return self.graph.get_subgraph(center_type, center_name, depth, rel_types)

    def graph_stats(self) -> dict[str, int]:
        """返回当前内存中图谱的统计信息."""
        return self.graph.stats()

    def load_document_graph(self, doc_id: str) -> None:
        """增量加载指定文档的图谱到全局图（兼容旧接口）."""
        graph_path = Config.DB_PATH.parent / "graphs" / f"{doc_id}.json"
        if graph_path.exists():
            temp = NetworkXGraphStore()
            temp.load(graph_path)
            self._merge_graph(temp)
            logger.info(f"已加载文档图谱 | doc_id={doc_id} | {self.graph.stats()}")
        else:
            logger.warning(f"文档图谱不存在 | doc_id={doc_id} | path={graph_path}")

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------


