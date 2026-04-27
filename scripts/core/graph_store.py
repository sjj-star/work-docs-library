"""文档图谱存储抽象层 + NetworkX 实现.

支持实体（节点）和关系（边）的 CRUD、子图查询、JSON 序列化
预留 Neo4j 迁移接口.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


def _normalize_sids(value):
    """将各种格式的 source_doc_ids 统一转为 set[str]."""
    if isinstance(value, set):
        return value
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, list):
        return set(value)
    return set()


# ---------------------------------------------------------------------------
# 图谱 Schema 常量
# ---------------------------------------------------------------------------

# 节点类型
NODE_FEATURE = "Feature"
NODE_MODULE = "Module"
NODE_INTERFACE = "Interface"
NODE_SIGNAL = "Signal"
NODE_REGISTER = "Register"
NODE_REGISTER_FIELD = "RegisterField"
NODE_SCENARIO = "Scenario"
NODE_CLOCK_DOMAIN = "ClockDomain"
NODE_RESET_DOMAIN = "ResetDomain"
NODE_MEMORY_MAP = "MemoryMap"
NODE_TEST_CASE = "TestCase"

ALL_NODE_TYPES = {
    NODE_FEATURE,
    NODE_MODULE,
    NODE_INTERFACE,
    NODE_SIGNAL,
    NODE_REGISTER,
    NODE_REGISTER_FIELD,
    NODE_SCENARIO,
    NODE_CLOCK_DOMAIN,
    NODE_RESET_DOMAIN,
    NODE_MEMORY_MAP,
    NODE_TEST_CASE,
}

# 关系类型
REL_IMPLEMENTS = "IMPLEMENTS"
REL_CONTAINS = "CONTAINS"
REL_HAS_INTERFACE = "HAS_INTERFACE"
REL_HAS_SIGNAL = "HAS_SIGNAL"
REL_HAS_REGISTER = "HAS_REGISTER"
REL_HAS_FIELD = "HAS_FIELD"
REL_REQUIRES = "REQUIRES"
REL_VALIDATES = "VALIDATES"
REL_BELONGS_TO = "BELONGS_TO"
REL_AT_ADDRESS = "AT_ADDRESS"
REL_CONNECTS_TO = "CONNECTS_TO"
REL_DOCUMENTED_BY = "DOCUMENTED_BY"

ALL_REL_TYPES = {
    REL_IMPLEMENTS,
    REL_CONTAINS,
    REL_HAS_INTERFACE,
    REL_HAS_SIGNAL,
    REL_HAS_REGISTER,
    REL_HAS_FIELD,
    REL_REQUIRES,
    REL_VALIDATES,
    REL_BELONGS_TO,
    REL_AT_ADDRESS,
    REL_CONNECTS_TO,
    REL_DOCUMENTED_BY,
}


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class GraphEntity:
    """图谱实体（节点）."""

    entity_type: str
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    source_doc_ids: set[str] = field(default_factory=set)
    source_chapter: str = ""

    def to_dict(self) -> dict[str, Any]:
        """to_dict 函数."""
        return {
            "type": self.entity_type,
            "name": self.name,
            "properties": self.properties,
            "source_doc_ids": sorted(list(self.source_doc_ids)),
            "source_chapter": self.source_chapter,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GraphEntity":
        """from_dict 函数."""
        return cls(
            entity_type=d.get("type", ""),
            name=d.get("name", ""),
            properties=d.get("properties", {}),
            source_doc_ids=_normalize_sids(d.get("source_doc_ids", [])),
            source_chapter=d.get("source_chapter", ""),
        )


@dataclass
class GraphRelation:
    """图谱关系（边）."""

    rel_type: str
    from_name: str
    to_name: str
    from_type: str = ""
    to_type: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    source_doc_ids: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        """to_dict 函数."""
        return {
            "type": self.rel_type,
            "from": self.from_name,
            "to": self.to_name,
            "from_type": self.from_type,
            "to_type": self.to_type,
            "properties": self.properties,
            "source_doc_ids": sorted(list(self.source_doc_ids)),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GraphRelation":
        """from_dict 函数."""
        return cls(
            rel_type=d.get("type", ""),
            from_name=d.get("from", ""),
            to_name=d.get("to", ""),
            from_type=d.get("from_type", ""),
            to_type=d.get("to_type", ""),
            properties=d.get("properties", {}),
            source_doc_ids=_normalize_sids(d.get("source_doc_ids", [])),
        )


# ---------------------------------------------------------------------------
# 抽象接口
# ---------------------------------------------------------------------------


class GraphStore(ABC):
    """图谱存储抽象接口，预留 NetworkX / Neo4j 双实现."""

    @abstractmethod
    def add_entity(self, entity: GraphEntity) -> None:
        """添加或更新实体节点."""
        ...

    @abstractmethod
    def add_relation(self, relation: GraphRelation) -> None:
        """添加关系边."""
        ...

    @abstractmethod
    def get_entity(self, entity_type: str, name: str) -> GraphEntity | None:
        """按类型和名称获取实体."""
        ...

    @abstractmethod
    def get_neighbors(
        self,
        entity_type: str,
        name: str,
        rel_type: str | None = None,
        direction: str = "out",  # "out", "in", "both"
    ) -> list[tuple[GraphEntity, str]]:
        """获取邻居节点.

        返回: [(neighbor_entity, rel_type), ...].
        """
        ...

    @abstractmethod
    def get_subgraph(
        self,
        center_type: str,
        center_name: str,
        depth: int = 1,
        rel_types: set[str] | None = None,
    ) -> "SubGraphView":
        """获取以某节点为中心的子图视图."""
        ...

    @abstractmethod
    def find_by_type(self, entity_type: str) -> list[GraphEntity]:
        """按类型查找所有实体."""
        ...

    @abstractmethod
    def find_by_property(self, entity_type: str, key: str, value: Any) -> list[GraphEntity]:
        """按属性查找实体."""
        ...

    @abstractmethod
    def all_entities(self) -> list[GraphEntity]:
        """获取所有实体."""
        ...

    @abstractmethod
    def all_relations(self) -> list[GraphRelation]:
        """获取所有关系."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """持久化到文件."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """从文件加载."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空图谱."""
        ...

    @abstractmethod
    def remove_document_contributions(self, doc_id: str) -> None:
        """从图谱中移除指定文档贡献的所有节点和边."""
        ...

    @abstractmethod
    def remove_chapter_contributions(self, doc_id: str, chapter_title: str) -> None:
        """从图谱中移除指定文档指定章节贡献的所有节点."""
        ...

    @abstractmethod
    def stats(self) -> dict[str, int]:
        """返回图谱统计信息."""
        ...

    @abstractmethod
    def find_path(
        self,
        from_type: str,
        from_name: str,
        to_type: str,
        to_name: str,
        max_depth: int = 3,
    ) -> list[list[str]]:
        """查找从起点到终点的所有路径（BFS，限制深度）.

        Returns:
            每条路径是节点 ID 列表，如 [["Module::A", "Signal::B", "Register::C"], ...]
        """
        ...

    @abstractmethod
    def search_entities(
        self, query: str, entity_types: set[str] | None = None
    ) -> list[GraphEntity]:
        """按名称模糊搜索实体（大小写不敏感子串匹配）.

        Args:
            query: 搜索关键字
            entity_types: 限制搜索的实体类型集合，None 表示不限

        Returns:
            匹配的实体列表
        """
        ...


# ---------------------------------------------------------------------------
# NetworkX 实现
# ---------------------------------------------------------------------------


class NetworkXGraphStore(GraphStore):
    """基于 NetworkX 的内存图谱存储，JSON 序列化持久化."""

    def __init__(self) -> None:
        """初始化 NetworkXGraphStore."""
        self._g = nx.DiGraph()
        logger.info("NetworkX 图谱存储已初始化")

    # -- 内部辅助 --

    def _node_id(self, entity_type: str, name: str) -> str:
        return f"{entity_type}::{name}"

    def _parse_node_id(self, node_id: str) -> tuple[str, str]:
        parts = node_id.split("::", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else ("", node_id)

    # -- 实体操作 --

    def add_entity(self, entity: GraphEntity) -> None:
        """add_entity 函数."""
        nid = self._node_id(entity.entity_type, entity.name)
        if nid in self._g:
            # 合并 properties，source 字段为空时保留旧值
            existing = self._g.nodes[nid]
            merged_props = dict(existing.get("properties", {}))
            merged_props.update(entity.properties)
            existing["properties"] = merged_props
            # 合并来源文档集合
            existing_sids = _normalize_sids(existing.get("source_doc_ids", set()))
            existing_sids.update(entity.source_doc_ids)
            existing["source_doc_ids"] = existing_sids
            existing["source_chapter"] = entity.source_chapter or existing.get("source_chapter", "")
        else:
            self._g.add_node(
                nid,
                entity_type=entity.entity_type,
                name=entity.name,
                properties=entity.properties,
                source_doc_ids=set(entity.source_doc_ids),
                source_chapter=entity.source_chapter,
            )
        logger.debug(f"添加实体 | {nid}")

    def get_entity(self, entity_type: str, name: str) -> GraphEntity | None:
        """get_entity 函数."""
        nid = self._node_id(entity_type, name)
        if nid not in self._g:
            return None
        data = self._g.nodes[nid]
        return GraphEntity(
            entity_type=data.get("entity_type", entity_type),
            name=data.get("name", name),
            properties=data.get("properties", {}),
            source_doc_ids=_normalize_sids(data.get("source_doc_ids", set())),
            source_chapter=data.get("source_chapter", ""),
        )

    def find_by_type(self, entity_type: str) -> list[GraphEntity]:
        """find_by_type 函数."""
        results = []
        for nid, data in self._g.nodes(data=True):
            if data.get("entity_type") == entity_type:
                results.append(
                    GraphEntity(
                        entity_type=data.get("entity_type", ""),
                        name=data.get("name", ""),
                        properties=data.get("properties", {}),
                        source_doc_ids=_normalize_sids(data.get("source_doc_ids", set())),
                        source_chapter=data.get("source_chapter", ""),
                    )
                )
        return results

    def find_by_property(self, entity_type: str, key: str, value: Any) -> list[GraphEntity]:
        """find_by_property 函数."""
        results = []
        for nid, data in self._g.nodes(data=True):
            if data.get("entity_type") == entity_type:
                props = data.get("properties", {})
                if props.get(key) == value:
                    results.append(
                        GraphEntity(
                            entity_type=data.get("entity_type", ""),
                            name=data.get("name", ""),
                            properties=props,
                            source_doc_ids=_normalize_sids(data.get("source_doc_ids", set())),
                            source_chapter=data.get("source_chapter", ""),
                        )
                    )
        return results

    def all_entities(self) -> list[GraphEntity]:
        """all_entities 函数."""
        results = []
        for nid, data in self._g.nodes(data=True):
            results.append(
                GraphEntity(
                    entity_type=data.get("entity_type", ""),
                    name=data.get("name", ""),
                    properties=data.get("properties", {}),
                    source_doc_ids=_normalize_sids(data.get("source_doc_ids", set())),
                    source_chapter=data.get("source_chapter", ""),
                )
            )
        return results

    # -- 关系操作 --

    def add_relation(self, relation: GraphRelation) -> None:
        """add_relation 函数."""
        from_nid = self._node_id(relation.from_type or "", relation.from_name)
        to_nid = self._node_id(relation.to_type or "", relation.to_name)

        # 确保节点存在（自动创建空节点）
        if from_nid not in self._g:
            self._g.add_node(
                from_nid,
                entity_type=relation.from_type or "",
                name=relation.from_name,
                source_doc_ids=set(),
            )
        if to_nid not in self._g:
            self._g.add_node(
                to_nid,
                entity_type=relation.to_type or "",
                name=relation.to_name,
                source_doc_ids=set(),
            )

        # 检查是否已存在相同边，合并 source_doc_ids
        existing = self._g.get_edge_data(from_nid, to_nid)
        if existing:
            existing_sids = _normalize_sids(existing.get("source_doc_ids", set()))
            existing_sids.update(relation.source_doc_ids)
            existing["source_doc_ids"] = existing_sids
            merged_props = dict(existing.get("properties", {}))
            merged_props.update(relation.properties)
            existing["properties"] = merged_props
        else:
            self._g.add_edge(
                from_nid,
                to_nid,
                rel_type=relation.rel_type,
                properties=relation.properties,
                source_doc_ids=set(relation.source_doc_ids),
            )
        logger.debug(f"添加关系 | {from_nid} -[{relation.rel_type}]-> {to_nid}")

    def get_neighbors(
        self,
        entity_type: str,
        name: str,
        rel_type: str | None = None,
        direction: str = "out",
    ) -> list[tuple[GraphEntity, str]]:
        """get_neighbors 函数."""
        nid = self._node_id(entity_type, name)
        if nid not in self._g:
            return []

        results = []

        if direction in ("out", "both"):
            for _, target, edge_data in self._g.out_edges(nid, data=True):
                if rel_type is None or edge_data.get("rel_type") == rel_type:
                    t_data = self._g.nodes[target]
                    results.append(
                        (
                            GraphEntity(
                                entity_type=t_data.get("entity_type", ""),
                                name=t_data.get("name", ""),
                                properties=t_data.get("properties", {}),
                                source_doc_ids=_normalize_sids(t_data.get("source_doc_ids", set())),
                                source_chapter=t_data.get("source_chapter", ""),
                            ),
                            edge_data.get("rel_type", ""),
                        )
                    )

        if direction in ("in", "both"):
            for source, _, edge_data in self._g.in_edges(nid, data=True):
                if rel_type is None or edge_data.get("rel_type") == rel_type:
                    s_data = self._g.nodes[source]
                    results.append(
                        (
                            GraphEntity(
                                entity_type=s_data.get("entity_type", ""),
                                name=s_data.get("name", ""),
                                properties=s_data.get("properties", {}),
                                source_doc_ids=_normalize_sids(s_data.get("source_doc_ids", set())),
                                source_chapter=s_data.get("source_chapter", ""),
                            ),
                            edge_data.get("rel_type", ""),
                        )
                    )

        return results

    def all_relations(self) -> list[GraphRelation]:
        """all_relations 函数."""
        results = []
        for u, v, data in self._g.edges(data=True):
            u_type, u_name = self._parse_node_id(u)
            v_type, v_name = self._parse_node_id(v)
            results.append(
                GraphRelation(
                    rel_type=data.get("rel_type", ""),
                    from_name=u_name,
                    to_name=v_name,
                    from_type=u_type,
                    to_type=v_type,
                    properties=data.get("properties", {}),
                )
            )
        return results

    # -- 子图查询 --

    def get_subgraph(
        self,
        center_type: str,
        center_name: str,
        depth: int = 1,
        rel_types: set[str] | None = None,
    ) -> "SubGraphView":
        """get_subgraph 函数."""
        nid = self._node_id(center_type, center_name)
        if nid not in self._g:
            return SubGraphView(nx.DiGraph())

        # BFS 收集节点
        nodes_to_include = {nid}
        current_layer = {nid}

        for _ in range(depth):
            next_layer = set()
            for node in current_layer:
                # 出边
                for _, target, data in self._g.out_edges(node, data=True):
                    if rel_types is None or data.get("rel_type") in rel_types:
                        next_layer.add(target)
                # 入边
                for source, _, data in self._g.in_edges(node, data=True):
                    if rel_types is None or data.get("rel_type") in rel_types:
                        next_layer.add(source)
            nodes_to_include.update(next_layer)
            current_layer = next_layer

        sub = self._g.subgraph(nodes_to_include).copy()
        return SubGraphView(sub)

    # -- 持久化 --

    def save(self, path: Path) -> None:
        """使用 node-link 格式导出为 JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._g, edges="edges")
        # 序列化前将 set 转为 list
        for node in data.get("nodes", []):
            sids = node.get("source_doc_ids")
            if isinstance(sids, set):
                node["source_doc_ids"] = sorted(list(sids))
        for edge in data.get("edges", []):
            sids = edge.get("source_doc_ids")
            if isinstance(sids, set):
                edge["source_doc_ids"] = sorted(list(sids))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(
            f"图谱已保存 | nodes={self._g.number_of_nodes()} | "
            f"edges={self._g.number_of_edges()} | path={path}"
        )

    def load(self, path: Path) -> None:
        """从 node-link JSON 加载."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._g = nx.node_link_graph(data, edges="edges")
        # 反序列化后将 list 转回 set
        for nid in self._g.nodes():
            sids = self._g.nodes[nid].get("source_doc_ids", [])
            if isinstance(sids, list):
                self._g.nodes[nid]["source_doc_ids"] = set(sids)
        for u, v in self._g.edges():
            sids = self._g.edges[u, v].get("source_doc_ids", [])
            if isinstance(sids, list):
                self._g.edges[u, v]["source_doc_ids"] = set(sids)
        logger.info(
            f"图谱已加载 | nodes={self._g.number_of_nodes()} | "
            f"edges={self._g.number_of_edges()} | path={path}"
        )

    def clear(self) -> None:
        """Clear 函数."""
        self._g.clear()
        logger.info("图谱已清空")

    def stats(self) -> dict[str, int]:
        """Stats 函数."""
        return {
            "nodes": self._g.number_of_nodes(),
            "edges": self._g.number_of_edges(),
        }

    def remove_document_contributions(self, doc_id: str) -> None:
        """从全局图中移除指定文档贡献的节点和边."""
        if not doc_id:
            return

        # 移除该文档贡献的节点（如无其他文档引用）
        nodes_to_remove = []
        for nid, data in list(self._g.nodes(data=True)):
            sids = _normalize_sids(data.get("source_doc_ids", set()))
            if doc_id in sids:
                sids.discard(doc_id)
                if sids:
                    self._g.nodes[nid]["source_doc_ids"] = sids
                else:
                    nodes_to_remove.append(nid)

        # 移除孤儿节点（一并移除其关联边）
        self._g.remove_nodes_from(nodes_to_remove)

        # 移除该文档贡献的边（如无其他文档引用）
        edges_to_remove = []
        for u, v, data in list(self._g.edges(data=True)):
            sids = _normalize_sids(data.get("source_doc_ids", set()))
            if doc_id in sids:
                sids.discard(doc_id)
                if sids:
                    self._g.edges[u, v]["source_doc_ids"] = sids
                else:
                    edges_to_remove.append((u, v))

        self._g.remove_edges_from(edges_to_remove)
        logger.info(
            f"已移除文档贡献 | doc_id={doc_id} | "
            f"剩余 nodes={self._g.number_of_nodes()} | edges={self._g.number_of_edges()}"
        )

    def remove_chapter_contributions(self, doc_id: str, chapter_title: str) -> None:
        """从全局图中移除指定文档指定章节的实体贡献（关联边一并移除）."""
        if not doc_id or not chapter_title:
            return

        nodes_to_remove = []
        for nid, data in list(self._g.nodes(data=True)):
            sids = _normalize_sids(data.get("source_doc_ids", set()))
            if doc_id in sids and data.get("source_chapter") == chapter_title:
                sids.discard(doc_id)
                if sids:
                    self._g.nodes[nid]["source_doc_ids"] = sids
                else:
                    nodes_to_remove.append(nid)

        # 移除节点会自动移除关联边
        self._g.remove_nodes_from(nodes_to_remove)
        logger.info(
            f"已移除章节贡献 | doc_id={doc_id} | chapter={chapter_title} | "
            f"剩余 nodes={self._g.number_of_nodes()} | edges={self._g.number_of_edges()}"
        )

    def find_path(
        self,
        from_type: str,
        from_name: str,
        to_type: str,
        to_name: str,
        max_depth: int = 3,
    ) -> list[list[str]]:
        """find_path 函数."""
        from_nid = self._node_id(from_type, from_name)
        to_nid = self._node_id(to_type, to_name)

        if from_nid not in self._g or to_nid not in self._g:
            return []

        # 限制 max_depth 防止爆炸（默认 3，最大 6）
        max_depth = min(max(max_depth, 1), 6)
        all_paths: list[list[str]] = []

        # BFS 收集所有简单路径（不重复节点），限制深度
        def _bfs_paths(start: str, target: str, max_d: int) -> list[list[str]]:
            queue: list[tuple[str, list[str]]] = [(start, [start])]
            paths: list[list[str]] = []
            while queue:
                current, path = queue.pop(0)
                if current == target and len(path) > 1:
                    paths.append(path.copy())
                    continue
                if len(path) >= max_d + 1:
                    continue
                for _, nxt, _ in self._g.out_edges(current, data=True):
                    if nxt not in path:
                        queue.append((nxt, path + [nxt]))
            return paths

        all_paths = _bfs_paths(from_nid, to_nid, max_depth)
        return all_paths

    def search_entities(
        self, query: str, entity_types: set[str] | None = None
    ) -> list[GraphEntity]:
        """search_entities 函数."""
        q = query.lower()
        results = []
        for nid, data in self._g.nodes(data=True):
            et = data.get("entity_type", "")
            name = data.get("name", "")
            if entity_types and et not in entity_types:
                continue
            if q in name.lower():
                results.append(
                    GraphEntity(
                        entity_type=et,
                        name=name,
                        properties=data.get("properties", {}),
                        source_doc_ids=_normalize_sids(data.get("source_doc_ids", set())),
                        source_chapter=data.get("source_chapter", ""),
                    )
                )
        return results


# ---------------------------------------------------------------------------
# 子图视图
# ---------------------------------------------------------------------------


class SubGraphView:
    """子图视图，提供只读查询接口."""

    def __init__(self, g: nx.DiGraph) -> None:
        """初始化 SubGraphView."""
        self._g = g

    def to_text_context(self) -> str:
        """将子图转换为文本上下文，供 LLM 使用."""
        lines = []
        for nid, data in self._g.nodes(data=True):
            entity_type = data.get("entity_type", "")
            name = data.get("name", "")
            props = data.get("properties", {})
            prop_str = ", ".join(f"{k}={v}" for k, v in props.items() if v)
            lines.append(f"- [{entity_type}] {name}" + (f" ({prop_str})" if prop_str else ""))

        if self._g.number_of_edges() > 0:
            lines.append("\n关系:")
            for u, v, data in self._g.edges(data=True):
                u_name = self._g.nodes[u].get("name", u)
                v_name = self._g.nodes[v].get("name", v)
                rel = data.get("rel_type", "")
                lines.append(f"  {u_name} --[{rel}]--> {v_name}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """to_dict 函数."""
        return nx.node_link_data(self._g, edges="edges")

    def entities(self) -> list[GraphEntity]:
        """Entities 函数."""
        results = []
        for nid, data in self._g.nodes(data=True):
            results.append(
                GraphEntity(
                    entity_type=data.get("entity_type", ""),
                    name=data.get("name", ""),
                    properties=data.get("properties", {}),
                    source_doc_ids=_normalize_sids(data.get("source_doc_ids", set())),
                    source_chapter=data.get("source_chapter", ""),
                )
            )
        return results

    def relations(self) -> list[GraphRelation]:
        """Relations 函数."""
        results = []
        for u, v, data in self._g.edges(data=True):
            u_type = self._g.nodes[u].get("entity_type", "")
            u_name = self._g.nodes[u].get("name", "")
            v_type = self._g.nodes[v].get("entity_type", "")
            v_name = self._g.nodes[v].get("name", "")
            results.append(
                GraphRelation(
                    rel_type=data.get("rel_type", ""),
                    from_name=u_name,
                    to_name=v_name,
                    from_type=u_type,
                    to_type=v_type,
                    properties=data.get("properties", {}),
                )
            )
        return results

    @property
    def node_count(self) -> int:
        """node_count 函数."""
        return self._g.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """edge_count 函数."""
        return self._g.number_of_edges()
