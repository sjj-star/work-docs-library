"""KnowledgeBaseService — 统一服务层.

封装 KnowledgeDB + VectorIndex + GraphStore + DocGraphPipeline，
为 Plugin 工具和上层应用提供一致的 API 入口。
"""

import copy
import logging
import shutil
from collections.abc import Callable
from typing import Any

from .bridge import EntityChunkBridge as _EntityChunkBridge
from .bridge import _EntityRef
from .config import Config
from .db import KnowledgeDB
from .doc_graph_pipeline import DocGraphPipeline
from .embedding_client import EmbeddingClient
from .graph_store import GraphEntity, NetworkXGraphStore, SubGraphView
from .models import Chunk, Document
from .reranker import LLMReranker
from .sparse_index import BM25SparseIndex
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
        self._embedder: EmbeddingClient | None = None
        self._sparse_index: BM25SparseIndex | None = None
        self._reranker: LLMReranker | None = None
        self.bridge = _EntityChunkBridge()
        self.bridge.rebuild(self.db)
        if graph_store is None:
            self._load_all_graphs()

    # ------------------------------------------------------------------
    # 文档管理
    # ------------------------------------------------------------------

    def _load_all_graphs(self) -> None:
        """加载所有已保存的文档图谱到全局内存图，实现跨文档实体互通."""
        global_path = Config.DB_PATH.parent / Config.GRAPH_OUTPUT_DIR / "global.json"
        doc_count = len(self.db.list_documents())
        if global_path.exists():
            try:
                self.graph.load(global_path)
                stats = self.graph.stats()
                logger.info(f"全局图谱已加载 | {stats}")
                # 启动校验：节点数异常少且 documents 表有数据时自动重建
                if stats["nodes"] < 10 and doc_count > 0:
                    logger.warning(
                        f"全局图谱节点数异常 | nodes={stats['nodes']} | "
                        f"docs={doc_count} | 触发自动重建"
                    )
                    self.rebuild_global_graph()
                return
            except Exception as e:
                logger.warning(f"加载全局图谱失败 | error={e}")

    def _merge_graph(self, other: NetworkXGraphStore) -> None:
        """将另一个图谱合并到全局图（同名同类型实体自动去重，属性合并）."""
        for entity in other.all_entities():
            self.graph.add_entity(entity)
        for relation in other.all_relations():
            self.graph.add_relation(relation)

    def _save_global_graph(self) -> None:
        """保存全局合并图谱.

        保存前自动备份旧文件，失败时回滚到备份。
        """
        global_path = Config.DB_PATH.parent / Config.GRAPH_OUTPUT_DIR / "global.json"
        global_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path = global_path.with_suffix(".json.bak")
        if global_path.exists():
            shutil.copy2(global_path, backup_path)
        try:
            self.graph.save(global_path)
        except Exception:
            if backup_path.exists():
                backup_path.replace(global_path)
            raise

    def rebuild_global_graph(self) -> dict[str, int]:
        """全量重建全局图：清空后从所有子图重新加载.

        用于修复全局图与磁盘子图之间的不一致。

        Returns:
            重建后的图谱统计信息
        """
        logger.info("开始全量重建全局图谱...")
        self.graph.clear()
        graphs_dir = Config.DB_PATH.parent / Config.GRAPH_OUTPUT_DIR
        loaded = 0
        if graphs_dir.exists():
            for gp in sorted(graphs_dir.glob("*.json")):
                if gp.name == "global.json":
                    continue
                try:
                    temp = NetworkXGraphStore()
                    temp.load(gp)
                    self._merge_graph(temp)
                    loaded += 1
                except Exception as e:
                    logger.warning(f"加载子图失败 | path={gp} | error={e}")
        self._save_global_graph()
        stats = self.graph.stats()
        logger.info(f"全局图谱重建完成 | loaded={loaded} | {stats}")
        return stats

    def _get_embedder(self) -> EmbeddingClient:
        """获取复用的 EmbeddingClient 实例（懒加载）."""
        if self._embedder is None:
            self._embedder = EmbeddingClient()
        return self._embedder

    def _get_sparse_index(self) -> BM25SparseIndex:
        """获取复用的 BM25SparseIndex 实例（懒加载 + 自动失效）."""
        current_total = sum(self.db.count_blocks_by_status().values())
        cached_total = (
            self._sparse_index.index_info().get("num_blocks", 0)
            if self._sparse_index is not None
            else -1
        )
        if cached_total != current_total:
            self._sparse_index = BM25SparseIndex(self.db)
        assert self._sparse_index is not None
        return self._sparse_index

    def _get_reranker(self) -> LLMReranker:
        """获取复用的 LLMReranker 实例（懒加载）."""
        if self._reranker is None:
            self._reranker = LLMReranker()
        return self._reranker

    _ALLOWED_RETRIEVERS: dict[str, str] = {
        "semantic": "search_semantic",
        "hybrid": "search_hybrid",
        "reranked": "search_reranked",
    }

    def _get_retriever(self, retriever: str) -> Callable[..., list[dict]]:
        """按名称返回检索方法 callable，并在不支持时尽早抛出清晰错误."""
        method_name = self._ALLOWED_RETRIEVERS.get(retriever)
        if method_name is None:
            allowed = ", ".join(sorted(self._ALLOWED_RETRIEVERS))
            raise ValueError(f"Unsupported retriever: {retriever}. Allowed: {allowed}")
        return getattr(self, method_name)

    def close(self) -> None:
        """关闭资源."""
        if self._embedder is not None:
            self._embedder.close()
            self._embedder = None

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
            # 增量合并：只加载新导入的文档子图到全局图
            failed_doc_ids: list[str] = []
            for doc_id in doc_ids:
                graph_path = Config.DB_PATH.parent / Config.GRAPH_OUTPUT_DIR / f"{doc_id}.json"
                if graph_path.exists():
                    try:
                        temp = NetworkXGraphStore()
                        temp.load(graph_path)
                        self._merge_graph(temp)
                    except Exception as e:
                        failed_doc_ids.append(doc_id)
                        logger.warning(f"加载子图失败 | path={graph_path} | error={e}")
            if failed_doc_ids:
                # 回滚本次 ingest 中已成功合并的子图，避免全局图部分合并
                for doc_id in doc_ids:
                    if doc_id not in failed_doc_ids:
                        self.graph.remove_document_contributions(doc_id)
                raise RuntimeError(f"部分子图加载失败，已回滚全局图变更: {failed_doc_ids}")
            self._save_global_graph()
            # 内容变更后使稀疏索引缓存失效
            self._sparse_index = None
            # 同步 bridge 索引（失败时记录警告，重启后自动恢复）
            for doc_id in doc_ids:
                try:
                    self._sync_bridge_for_doc(doc_id)
                except Exception as e:
                    logger.warning(f"Bridge 同步失败，重启后自动恢复 | doc_id={doc_id} | error={e}")
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
        """获取文档处理进度统计（基于 content_blocks）.

        Returns:
            {"total": int, "done": int, "embedded": int,
             "skipped": int, "pending": int, "failed": int}
        """
        with self.db._connect() as conn:
            block_count = conn.execute(
                "SELECT COUNT(*) as c FROM content_blocks WHERE doc_id = ?", (doc_id,)
            ).fetchone()["c"]

            def _block_count(status: str) -> int:
                return conn.execute(
                    "SELECT COUNT(*) as c FROM content_blocks WHERE doc_id = ? AND status = ?",
                    (doc_id, status),
                ).fetchone()["c"]

            return {
                "total": block_count,
                "done": _block_count("done"),
                "embedded": _block_count("embedded"),
                "skipped": _block_count("skipped"),
                "pending": _block_count("pending"),
                "failed": _block_count("failed"),
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

        # 保存全局图快照，用于失败回滚
        backup_g = copy.deepcopy(self.graph._g)
        backup_index = {k: set(v) for k, v in self.graph._property_index.items()}

        try:
            # 从全局图中精确移除旧文档贡献
            self.graph.remove_document_contributions(doc_id)
            # 使用独立的 graph_store 处理
            pipe = DocGraphPipeline(db=self.db, vec=self.vec, graph_store=NetworkXGraphStore())
            try:
                result = pipe._process_one(doc.source_path, dry_run=False, force=True)
                if not result:
                    raise RuntimeError(f"Reprocess failed for {doc_id}")
                # 将重新处理后的图谱合并到全局图
                graph_path = Config.DB_PATH.parent / Config.GRAPH_OUTPUT_DIR / f"{doc_id}.json"
                if graph_path.exists():
                    temp = NetworkXGraphStore()
                    temp.load(graph_path)
                    self._merge_graph(temp)
                self._save_global_graph()
                # 内容变更后使稀疏索引缓存失效
                self._sparse_index = None
                # 完整性校验：保存后节点数不应低于 reprocess 前（允许小幅波动）
                after_nodes = self.graph.stats()["nodes"]
                before_nodes = backup_g.number_of_nodes()
                if after_nodes < before_nodes * 0.5:
                    logger.warning(
                        f"全局图节点数异常下降 | before={before_nodes} after={after_nodes} | "
                        f"触发重建 | doc_id={doc_id}"
                    )
                    self.rebuild_global_graph()
                # 同步 bridge 索引（失败时记录警告，重启后自动恢复）
                try:
                    self._sync_bridge_for_doc(doc_id)
                except Exception as e:
                    logger.warning(f"Bridge 同步失败，重启后自动恢复 | doc_id={doc_id} | error={e}")
                return result
            finally:
                pipe.close()
        except Exception:
            # 回滚全局图到处理前状态
            self.graph._g = backup_g
            self.graph._property_index = backup_index
            self._save_global_graph()
            raise

    def _sync_bridge_for_doc(self, doc_id: str) -> None:
        """同步指定文档的 bridge 索引（detach 旧 + attach 新）.

        在 ingest / reprocess 完成后调用，确保 bridge 与 SQLite 一致.
        """
        # 先 detach 该文档所有已索引的 blocks
        for db_id in list(self.bridge.get_chunk_ids()):
            block = self.db.get_block_by_db_id(db_id)
            if block and block.get("doc_id") == doc_id:
                self.bridge.detach(db_id)
        # 再 attach 该文档当前的所有 content_blocks
        for block in self.db.query_blocks_by_doc(doc_id):
            refs = self.bridge.extract_refs(block)
            self.bridge.attach(block["id"], refs)
        logger.debug(f"Bridge 已同步 | doc_id={doc_id}")

    # ------------------------------------------------------------------
    # Chunk 查询
    # ------------------------------------------------------------------

    def search_semantic(self, text: str, top_k: int = Config.PLUGIN_SEARCH_TOP_K) -> list[dict]:
        """语义向量搜索（基于 content_blocks 偏移量）.

        Returns:
            每个结果包含 score 和 chunk 基本信息
            [{"score": float, "chunk": Chunk}, ...]
        """
        embedder = self._get_embedder()
        emb = embedder.embed([str(text)])[0]
        hits = self.vec.search(emb, top_k=top_k)
        results = []
        for block_db_id, score in hits:
            block = self.db.get_block_by_db_id(block_db_id)
            if block:
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
                results.append({"score": round(float(score), 4), "chunk": chunk})
        return results

    def search_hybrid(self, text: str, top_k: int = Config.PLUGIN_SEARCH_TOP_K) -> list[dict]:
        """混合检索：BM25 + FAISS 向量，RRF 融合.

        Returns:
            每个结果包含 score 和 chunk 基本信息
            [{"score": float, "chunk": Chunk}, ...]
        """
        from .hybrid_retriever import RRFFusionRetriever

        sparse = self._get_sparse_index()
        retriever = RRFFusionRetriever(self.vec, sparse)
        hits = retriever.search(text, self._get_embedder(), top_k=top_k)
        results = []
        for block_db_id, score in hits:
            chunk = self._get_chunk(block_db_id)
            if chunk:
                results.append({"score": round(float(score), 4), "chunk": chunk})
        return results

    def search_reranked(
        self,
        text: str,
        top_k: int = Config.PLUGIN_SEARCH_TOP_K,
        candidate_k: int | None = None,
    ) -> list[dict]:
        """混合检索 + LLM 重排序.

        Args:
            text: 查询文本
            top_k: 最终返回结果数量
            candidate_k: 混合检索召回数量，默认 PLUGIN_SEARCH_TOP_K * 4

        Returns:
            [{"score": float, "chunk": Chunk}, ...]
        """
        if candidate_k is None:
            candidate_k = top_k * 4

        candidates = self.search_hybrid(text, top_k=candidate_k)
        passages = [(c["chunk"].id, c["chunk"].content) for c in candidates]
        try:
            reranked = self._get_reranker().rank(text, passages)
        except Exception:
            logger.exception("Reranking failed, returning hybrid results")
            reranked = [(c["chunk"].id, c["score"]) for c in candidates]

        results = []
        for block_db_id, score in reranked[:top_k]:
            chunk = self._get_chunk(block_db_id)
            if chunk:
                results.append({"score": round(float(score), 4), "chunk": chunk})
        return results

    def evaluate_dataset(
        self,
        dataset_name: str,
        retriever: str = "semantic",
        top_k: int = Config.PLUGIN_SEARCH_TOP_K,
    ) -> dict[str, Any]:
        """Evaluate a dataset using the specified retriever."""
        from .evaluation import EvalHarness

        # 尽早校验 retriever，避免在 harness 内部才暴露不清晰的错误
        self._get_retriever(retriever)

        dataset = self.db.load_eval_dataset(dataset_name)
        harness = EvalHarness(self)
        return harness.run_retrieval_eval(dataset, retriever=retriever, top_k=top_k)

    def query_chunks(
        self,
        doc_id: str | None = None,
        chapter: str | None = None,
        chapter_regex: str | None = None,
        concept: str | None = None,
        include_children: bool = False,
        top_k: int = Config.PLUGIN_QUERY_TOP_K,
    ) -> list[Chunk]:
        """统一 chunk 结构化查询（基于 content_blocks + heading_maps）.

        Args:
            doc_id: 文档 ID（chapter/concept 查询必需）
            chapter: 章节标题子串匹配
            chapter_regex: 章节标题正则匹配
            concept: 概念名匹配（需 doc_id）
            include_children: 是否包含子章节内容（仅 chapter 查询有效）
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
            if include_children:
                blocks = self.db.query_by_heading_recursive(doc_id, chapter)
            else:
                blocks = self.db.query_by_heading(doc_id, chapter)
            for block in blocks:
                results.append(
                    Chunk(
                        id=block["id"],
                        doc_id=block["doc_id"],
                        chunk_id=block["block_id"],
                        content=block["content"],
                        chunk_type="text",
                        chapter_title=block["metadata"].get("section_title", ""),
                        metadata=block["metadata"],
                        status=block["status"],
                    )
                )
        elif chapter_regex:
            if not doc_id:
                raise ValueError("chapter_regex query requires doc_id")
            # 利用 heading_maps 索引缩小范围，避免全表扫描
            import re

            compiled = re.compile(chapter_regex)
            # 先查 heading_maps 获取候选标题
            candidate_titles: set[str] = set()
            for hm in self.db.query_heading_maps_by_doc(doc_id):
                if compiled.search(hm["heading_title"]):
                    candidate_titles.add(hm["heading_title"])
            # 聚合匹配的 block_db_ids
            seen_ids: set[int] = set()
            all_block_ids: list[int] = []
            for title in candidate_titles:
                for block in self.db.query_by_heading(doc_id, title):
                    bid = block["id"]
                    if bid not in seen_ids:
                        seen_ids.add(bid)
                        all_block_ids.append(bid)
            # 按 seq_index 查询 content_blocks
            for bid in sorted(all_block_ids):
                block = self.db.get_block_by_db_id(bid)
                if block:
                    results.append(
                        Chunk(
                            id=block["id"],
                            doc_id=block["doc_id"],
                            chunk_id=block["block_id"],
                            content=block["content"],
                            chunk_type="text",
                            chapter_title=block["metadata"].get("section_title", ""),
                            metadata=block["metadata"],
                            status=block["status"],
                        )
                    )
        elif concept:
            if not doc_id:
                raise ValueError("concept query requires doc_id")
            # 基于 content_blocks 的概念查询
            for block in self.db.query_blocks_by_doc(doc_id):
                entities = block["metadata"].get("extracted_entities", [])
                for ent in entities:
                    if ent.get("name") == concept:
                        results.append(
                            Chunk(
                                id=block["id"],
                                doc_id=block["doc_id"],
                                chunk_id=block["block_id"],
                                content=block["content"],
                                chunk_type="text",
                                chapter_title=block["metadata"].get("section_title", ""),
                                metadata=block["metadata"],
                                status=block["status"],
                            )
                        )
                        break
        else:
            raise ValueError("Provide chapter, chapter_regex, or concept")

        return results[:top_k]

    def get_chunk_content(
        self,
        chunk_db_id: int | None = None,
        doc_id: str | None = None,
        chapter: str | None = None,
    ) -> dict:
        """获取 chunk 完整内容（基于 content_blocks + heading_maps）.

        Args:
            chunk_db_id: 直接按 block DB ID 查询
            doc_id: 文档 ID（配合 chapter）
            chapter: 章节标题

        Returns:
            {"query_type": str, "chunks": list[Chunk], "content": str}

        Raises:
            ValueError: 参数不合法或找不到内容
        """
        if chunk_db_id is not None:
            block = self.db.get_block_by_db_id(int(chunk_db_id))
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
            return {
                "query_type": "chunk",
                "chunks": [chunk],
                "content": chunk.content,
            }

        if not doc_id:
            raise ValueError("Provide chunk_db_id, or doc_id with chapter")

        if chapter is not None:
            blocks = self.db.query_by_heading(doc_id, chapter)
            chunks = []
            for block in blocks:
                chunks.append(
                    Chunk(
                        id=block["id"],
                        doc_id=block["doc_id"],
                        chunk_id=block["block_id"],
                        content=block["content"],
                        chunk_type="text",
                        chapter_title=block["metadata"].get("section_title", ""),
                        metadata=block["metadata"],
                        status=block["status"],
                    )
                )
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

    @staticmethod
    def _apply_doc_properties(entity: GraphEntity, doc_id: str | None) -> GraphEntity:
        """如果指定了 doc_id，用 doc_properties 中的快照替换 properties.

        返回深拷贝，避免修改内存中的全局图节点。
        """
        if doc_id and entity.doc_properties and doc_id in entity.doc_properties:
            entity = copy.deepcopy(entity)
            entity.properties = dict(entity.doc_properties[doc_id])
        return entity

    def find_entities(
        self,
        entity_type: str | None = None,
        name_pattern: str | None = None,
        doc_id: str | None = None,
    ) -> list[GraphEntity]:
        """搜索实体.

        Args:
            entity_type: 精确匹配实体类型
            name_pattern: 名称子串匹配（大小写不敏感）
            doc_id: 可选，指定文档 ID 以获取该文档中的原始属性快照

        Returns:
            匹配的实体列表（深拷贝，禁止外部修改全局图）
        """
        results: list[GraphEntity]
        if entity_type and not name_pattern:
            results = self.graph.find_by_type(entity_type)
        elif name_pattern:
            types = {entity_type} if entity_type else None
            results = self.graph.search_entities(name_pattern, entity_types=types)
        else:
            results = self.graph.all_entities()
        if doc_id:
            for e in results:
                self._apply_doc_properties(e, doc_id)
        return [copy.deepcopy(e) for e in results]

    def get_neighbors(
        self,
        entity_type: str,
        name: str,
        rel_type: str | None = None,
        direction: str = "out",
        doc_id: str | None = None,
    ) -> list[tuple[GraphEntity, str, dict]]:
        """获取实体的邻居节点."""
        results = self.graph.get_neighbors(entity_type, name, rel_type, direction)
        if doc_id:
            for i, (entity, rt, props) in enumerate(results):
                self._apply_doc_properties(entity, doc_id)
        return [(copy.deepcopy(entity), rt, copy.deepcopy(props)) for entity, rt, props in results]

    def find_path(
        self,
        from_type: str,
        from_name: str,
        to_type: str,
        to_name: str,
        max_depth: int = Config.PLUGIN_GRAPH_MAX_DEPTH,
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
        depth: int = Config.PLUGIN_SUBGRAPH_DEPTH,
        rel_types: set[str] | None = None,
    ) -> SubGraphView:
        """获取以某实体为中心的子图."""
        return self.graph.get_subgraph(center_type, center_name, depth, rel_types)

    def graph_stats(self) -> dict[str, int]:
        """返回当前内存中图谱的统计信息."""
        return self.graph.stats()

    # -- 跨粒度桥接原子操作（机制层，无策略参数）--

    def _entity_to_chunks(self, entity_type: str, entity_name: str) -> set[int]:
        """反向查询：提及该实体的所有 chunk_db_id（O(1)）."""
        return self.bridge.get_chunks(_EntityRef(entity_type, entity_name))

    def _get_chunk(self, chunk_db_id: int) -> Chunk | None:
        """按 ID 获取 content_block（深拷贝隔离）."""
        block = self.db.get_block_by_db_id(chunk_db_id)
        if block:
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
            return copy.deepcopy(chunk)
        return None

    def find_chunks_by_entity(
        self, entity_type: str, name: str, doc_id: str | None = None
    ) -> list[Chunk]:
        """查找包含指定实体的所有 blocks（通过桥接索引，O(1) 反向查询）.

        Args:
            entity_type: 实体类型
            name: 实体名称
            doc_id: 可选，限定只查某文档的 blocks

        Returns:
            Chunk 列表（深拷贝，按 doc_id, chunk_id 排序）
        """
        chunk_ids = self._entity_to_chunks(entity_type, name)
        chunks: list[Chunk] = []
        for cid in chunk_ids:
            block = self.db.get_block_by_db_id(cid)
            if block:
                if doc_id is None or block.get("doc_id") == doc_id:
                    chunks.append(
                        copy.deepcopy(
                            Chunk(
                                id=block["id"],
                                doc_id=block["doc_id"],
                                chunk_id=block["block_id"],
                                content=block["content"],
                                chunk_type="text",
                                chapter_title=block["metadata"].get("section_title", ""),
                                metadata=block["metadata"],
                                status=block["status"],
                            )
                        )
                    )
        chunks.sort(key=lambda c: (c.doc_id, c.chunk_id))
        return chunks

    # -- 图谱动态更新（CRUD）--

    def add_entity(self, entity: GraphEntity) -> list[dict]:
        """添加或更新实体. 返回冲突日志列表."""
        # 保存操作前快照，用于失败时恢复而非删除
        old_entity = self.graph.get_entity(entity.entity_type, entity.name)
        conflicts = self.graph.add_entity(entity)
        save_ok = False
        try:
            self._save_global_graph()
            save_ok = True
            if conflicts:
                self.db.insert_conflict_logs(conflicts)
        except Exception:
            # 持久化失败时恢复到操作前状态
            if old_entity is not None:
                self.graph.update_entity(
                    entity.entity_type,
                    entity.name,
                    properties=old_entity.properties,
                    confidence=old_entity.confidence,
                    verified=old_entity.verified,
                    feedback_score=old_entity.feedback_score,
                )
            else:
                self.graph.delete_entity(entity.entity_type, entity.name)
            if save_ok:
                try:
                    self._save_global_graph()
                except Exception:
                    logger.exception("回滚全局图后再次保存失败")
            raise
        return conflicts

    def update_entity(
        self,
        entity_type: str,
        name: str,
        properties: dict | None = None,
        confidence: float | None = None,
        verified: bool | None = None,
        feedback_score: int | None = None,
    ) -> bool:
        """更新实体属性."""
        ok = self.graph.update_entity(
            entity_type, name, properties, confidence, verified, feedback_score
        )
        if ok:
            self._save_global_graph()
        return ok

    def delete_entity(self, entity_type: str, name: str) -> bool:
        """删除实体."""
        ok = self.graph.delete_entity(entity_type, name)
        if ok:
            self._save_global_graph()
        return ok

    def add_relation(self, relation) -> list[dict]:
        """添加或更新关系. 返回冲突日志列表."""
        # 保存操作前快照，用于失败时恢复而非删除
        old_relation = self.graph.get_relation(
            relation.from_type,
            relation.from_name,
            relation.to_type,
            relation.to_name,
            relation.rel_type,
        )
        conflicts = self.graph.add_relation(relation)
        save_ok = False
        try:
            self._save_global_graph()
            save_ok = True
            if conflicts:
                self.db.insert_conflict_logs(conflicts)
        except Exception:
            # 持久化失败时恢复到操作前状态
            if old_relation is not None:
                self.graph.update_relation(
                    relation.from_type,
                    relation.from_name,
                    relation.to_type,
                    relation.to_name,
                    relation.rel_type,
                    properties=old_relation.properties,
                    confidence=old_relation.confidence,
                    verified=old_relation.verified,
                    feedback_score=old_relation.feedback_score,
                )
            else:
                self.graph.delete_relation(
                    relation.from_type,
                    relation.from_name,
                    relation.to_type,
                    relation.to_name,
                    relation.rel_type,
                )
            if save_ok:
                try:
                    self._save_global_graph()
                except Exception:
                    logger.exception("回滚全局图后再次保存失败")
            raise
        return conflicts

    def update_relation(
        self,
        from_type: str,
        from_name: str,
        to_type: str,
        to_name: str,
        rel_type: str,
        properties: dict | None = None,
        confidence: float | None = None,
        verified: bool | None = None,
    ) -> bool:
        """更新关系属性."""
        ok = self.graph.update_relation(
            from_type, from_name, to_type, to_name, rel_type, properties, confidence, verified
        )
        if ok:
            self._save_global_graph()
        return ok

    def delete_relation(
        self,
        from_type: str,
        from_name: str,
        to_type: str,
        to_name: str,
        rel_type: str,
    ) -> bool:
        """删除关系."""
        ok = self.graph.delete_relation(from_type, from_name, to_type, to_name, rel_type)
        if ok:
            self._save_global_graph()
        return ok

    def verify_entity(self, entity_type: str, name: str, verified: bool = True) -> bool:
        """标记实体验证状态."""
        ok = self.graph.verify_entity(entity_type, name, verified)
        if ok:
            self._save_global_graph()
        return ok

    # -- 冲突日志查询 --

    def get_conflict_logs(
        self,
        entity_type: str | None = None,
        name: str | None = None,
        limit: int = Config.PLUGIN_DEFAULT_LIMIT,
    ) -> list[dict]:
        """查询冲突日志."""
        return self.db.query_conflict_logs(entity_type, name, limit)

    # -- 反馈 --

    def submit_feedback(
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
        """提交反馈. 返回反馈记录 ID."""
        feedback_id = self.db.insert_feedback(
            rating=rating,
            entity_type=entity_type,
            entity_name=entity_name,
            relation_type=relation_type,
            relation_from_type=relation_from_type,
            relation_from_name=relation_from_name,
            relation_to_type=relation_to_type,
            relation_to_name=relation_to_name,
            comment=comment,
        )
        # 同步更新内存中的 feedback_score
        if entity_type and entity_name:
            score = self.db.get_entity_feedback_score(entity_type, entity_name)
            self.graph.update_entity(entity_type, entity_name, feedback_score=score)
            self._save_global_graph()
        elif (
            relation_type
            and relation_from_type
            and relation_from_name
            and relation_to_type
            and relation_to_name
        ):
            score = self.db.get_relation_feedback_score(
                relation_type,
                relation_from_type,
                relation_from_name,
                relation_to_type,
                relation_to_name,
            )
            self.graph.update_relation(
                relation_from_type,
                relation_from_name,
                relation_to_type,
                relation_to_name,
                relation_type,
                feedback_score=score,
            )
            self._save_global_graph()
        return feedback_id

    def get_feedback(
        self,
        entity_type: str | None = None,
        entity_name: str | None = None,
        limit: int = Config.PLUGIN_DEFAULT_LIMIT,
    ) -> list[dict]:
        """查询反馈."""
        return self.db.query_feedback(entity_type, entity_name, limit)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
