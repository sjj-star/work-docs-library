"""文档图谱存储抽象层 + NetworkX 实现.

支持实体（节点）和关系（边）的 CRUD、子图查询、JSON 序列化
预留 Neo4j 迁移接口.
"""

import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

from .config import Config

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
NODE_PARAMETER = "Parameter"
NODE_PRODUCT = "Product"
# CPU/DSP 处理器架构实体（C28x+CLA）
NODE_INSTRUCTION = "Instruction"
NODE_INSTRUCTION_GROUP = "InstructionGroup"
NODE_ADDRESSING_MODE = "AddressingMode"
NODE_OPERAND = "Operand"
NODE_ARCHITECTURE_STATE = "ArchitectureState"
NODE_PIPELINE_STAGE = "PipelineStage"
NODE_FUNCTIONAL_UNIT = "FunctionalUnit"
NODE_INTERRUPT = "Interrupt"
NODE_EXCEPTION = "Exception"
NODE_MEMORY_REGION = "MemoryRegion"
NODE_SHADOW_REGISTER = "ShadowRegister"
NODE_CPU_MODE = "CPU_Mode"
NODE_CLA_TASK = "CLA_Task"
NODE_PERIPHERAL = "Peripheral"

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
    NODE_PARAMETER,
    NODE_PRODUCT,
    NODE_INSTRUCTION,
    NODE_INSTRUCTION_GROUP,
    NODE_ADDRESSING_MODE,
    NODE_OPERAND,
    NODE_ARCHITECTURE_STATE,
    NODE_PIPELINE_STAGE,
    NODE_FUNCTIONAL_UNIT,
    NODE_INTERRUPT,
    NODE_EXCEPTION,
    NODE_MEMORY_REGION,
    NODE_SHADOW_REGISTER,
    NODE_CPU_MODE,
    NODE_CLA_TASK,
    NODE_PERIPHERAL,
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
# IC 设计深层关系
REL_DRIVES = "DRIVES"
REL_DRIVEN_BY = "DRIVEN_BY"
REL_TIMING_PATH = "TIMING_PATH"
REL_CLOCK_GATED_BY = "CLOCK_GATED_BY"
REL_RESET_BY = "RESET_BY"
REL_PARAMETERIZED_BY = "PARAMETERIZED_BY"
REL_INSTANCE_OF = "INSTANCE_OF"
# CPU/DSP 处理器架构关系（C28x+CLA）
REL_ISA_HAS_INSTRUCTION = "ISA_HAS_INSTRUCTION"
REL_INSTRUCTION_BELONGS_TO_GROUP = "INSTRUCTION_BELONGS_TO_GROUP"
REL_INSTRUCTION_USES_MODE = "INSTRUCTION_USES_MODE"
REL_OPERAND_HAS_MODE = "OPERAND_HAS_MODE"
REL_INSTRUCTION_READS_REGISTER = "INSTRUCTION_READS_REGISTER"
REL_INSTRUCTION_WRITES_REGISTER = "INSTRUCTION_WRITES_REGISTER"
REL_INSTRUCTION_MODIFIES_STATE = "INSTRUCTION_MODIFIES_STATE"
REL_INSTRUCTION_EXECUTED_IN = "INSTRUCTION_EXECUTED_IN"
REL_INTERRUPT_TRIGGERS = "INTERRUPT_TRIGGERS"
REL_EXCEPTION_RAISES = "EXCEPTION_RAISES"
REL_STATE_HAS_REGISTER = "STATE_HAS_REGISTER"
REL_MODULE_IMPLEMENTS_INSTRUCTION = "MODULE_IMPLEMENTS_INSTRUCTION"
REL_HAS_PERIPHERAL = "HAS_PERIPHERAL"
REL_PERIPHERAL_HAS_REGISTER = "PERIPHERAL_HAS_REGISTER"
REL_CLA_HAS_TASK = "CLA_HAS_TASK"
REL_TASK_USES_INSTRUCTION = "TASK_USES_INSTRUCTION"
REL_MEMORY_MAPS_TO = "MEMORY_MAPS_TO"
REL_UNIT_EXECUTES = "UNIT_EXECUTES"
REL_STAGE_PRODUCES = "STAGE_PRODUCES"

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
    REL_DRIVES,
    REL_DRIVEN_BY,
    REL_TIMING_PATH,
    REL_CLOCK_GATED_BY,
    REL_RESET_BY,
    REL_PARAMETERIZED_BY,
    REL_INSTANCE_OF,
    REL_ISA_HAS_INSTRUCTION,
    REL_INSTRUCTION_BELONGS_TO_GROUP,
    REL_INSTRUCTION_USES_MODE,
    REL_OPERAND_HAS_MODE,
    REL_INSTRUCTION_READS_REGISTER,
    REL_INSTRUCTION_WRITES_REGISTER,
    REL_INSTRUCTION_MODIFIES_STATE,
    REL_INSTRUCTION_EXECUTED_IN,
    REL_INTERRUPT_TRIGGERS,
    REL_EXCEPTION_RAISES,
    REL_STATE_HAS_REGISTER,
    REL_MODULE_IMPLEMENTS_INSTRUCTION,
    REL_HAS_PERIPHERAL,
    REL_PERIPHERAL_HAS_REGISTER,
    REL_CLA_HAS_TASK,
    REL_TASK_USES_INSTRUCTION,
    REL_MEMORY_MAPS_TO,
    REL_UNIT_EXECUTES,
    REL_STAGE_PRODUCES,
}


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """返回当前 ISO 格式时间戳."""
    return datetime.now().isoformat()


@dataclass
class GraphEntity:
    """图谱实体（节点）."""

    entity_type: str
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    doc_properties: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_doc_ids: set[str] = field(default_factory=set)
    source_chapter: str = ""
    confidence: float = 1.0
    verified: bool = False
    created_at: str = ""
    updated_at: str = ""
    feedback_score: int = 0

    def to_dict(self) -> dict[str, Any]:
        """to_dict 函数."""
        return {
            "type": self.entity_type,
            "name": self.name,
            "properties": self.properties,
            "doc_properties": self.doc_properties,
            "source_doc_ids": sorted(list(self.source_doc_ids)),
            "source_chapter": self.source_chapter,
            "confidence": self.confidence,
            "verified": self.verified,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "feedback_score": self.feedback_score,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GraphEntity":
        """from_dict 函数."""
        return cls(
            entity_type=d.get("type", ""),
            name=d.get("name", ""),
            properties=d.get("properties", {}),
            doc_properties=d.get("doc_properties", {}),
            source_doc_ids=_normalize_sids(d.get("source_doc_ids", [])),
            source_chapter=d.get("source_chapter", ""),
            confidence=d.get("confidence", 1.0),
            verified=d.get("verified", False),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            feedback_score=d.get("feedback_score", 0),
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
    doc_properties: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_doc_ids: set[str] = field(default_factory=set)
    source_chapter: str = ""
    confidence: float = 1.0
    verified: bool = False
    created_at: str = ""
    updated_at: str = ""
    feedback_score: int = 0

    def to_dict(self) -> dict[str, Any]:
        """to_dict 函数."""
        return {
            "type": self.rel_type,
            "from": self.from_name,
            "to": self.to_name,
            "from_type": self.from_type,
            "to_type": self.to_type,
            "properties": self.properties,
            "doc_properties": self.doc_properties,
            "source_doc_ids": sorted(list(self.source_doc_ids)),
            "source_chapter": self.source_chapter,
            "confidence": self.confidence,
            "verified": self.verified,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "feedback_score": self.feedback_score,
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
            doc_properties=d.get("doc_properties", {}),
            source_doc_ids=_normalize_sids(d.get("source_doc_ids", [])),
            source_chapter=d.get("source_chapter", ""),
            confidence=d.get("confidence", 1.0),
            verified=d.get("verified", False),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            feedback_score=d.get("feedback_score", 0),
        )


# ---------------------------------------------------------------------------
# 抽象接口
# ---------------------------------------------------------------------------


class GraphStore(ABC):
    """图谱存储抽象接口，预留 NetworkX / Neo4j 双实现."""

    @abstractmethod
    def add_entity(self, entity: GraphEntity) -> list[dict]:
        """添加或更新实体节点. 返回冲突日志列表（属性覆盖事件）."""
        ...

    @abstractmethod
    def add_relation(self, relation: GraphRelation) -> list[dict]:
        """添加关系边. 返回冲突日志列表（属性覆盖事件）."""
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
    ) -> list[tuple[GraphEntity, str, dict]]:
        """获取邻居节点.

        返回: [(neighbor_entity, rel_type, rel_properties), ...].
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
        rel_types: set[str] | None = None,
    ) -> list[list[str]]:
        """查找从起点到终点的所有路径（BFS，限制深度）.

        Args:
            from_type: 起点实体类型
            from_name: 起点实体名称
            to_type: 终点实体类型
            to_name: 终点实体名称
            max_depth: 最大搜索深度
            rel_types: 仅遍历指定关系类型的边，None 表示不限

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

    @abstractmethod
    def update_entity(
        self,
        entity_type: str,
        name: str,
        properties: dict[str, Any] | None = None,
        confidence: float | None = None,
        verified: bool | None = None,
        feedback_score: int | None = None,
    ) -> bool:
        """更新实体属性.

        Returns:
            是否成功找到并更新实体.
        """
        ...

    @abstractmethod
    def delete_entity(self, entity_type: str, name: str) -> bool:
        """删除实体（级联删除关联边）.

        Returns:
            是否成功删除.
        """
        ...

    @abstractmethod
    def update_relation(
        self,
        from_type: str,
        from_name: str,
        to_type: str,
        to_name: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
        confidence: float | None = None,
        verified: bool | None = None,
        feedback_score: int | None = None,
    ) -> bool:
        """更新关系属性.

        Returns:
            是否成功找到并更新关系.
        """
        ...

    @abstractmethod
    def delete_relation(
        self,
        from_type: str,
        from_name: str,
        to_type: str,
        to_name: str,
        rel_type: str,
    ) -> bool:
        """删除关系.

        Returns:
            是否成功删除.
        """
        ...

    @abstractmethod
    def verify_entity(self, entity_type: str, name: str, verified: bool = True) -> bool:
        """标记实体验证状态.

        Returns:
            是否成功找到并更新.
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
        # 属性索引: {(entity_type, key, value): {nid, ...}}
        self._property_index: dict[tuple[str, str, Any], set[str]] = {}
        self._lock = threading.Lock()
        logger.info("NetworkX 图谱存储已初始化")

    # -- 内部辅助 --

    def _node_id(self, entity_type: str, name: str) -> str:
        return f"{entity_type}::{name}"

    def _parse_node_id(self, node_id: str) -> tuple[str, str]:
        parts = node_id.split("::", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else ("", node_id)

    def _node_to_entity(self, nid: str, data: dict) -> GraphEntity:
        """将 NetworkX 节点数据转换为 GraphEntity."""
        return GraphEntity(
            entity_type=data.get("entity_type", ""),
            name=data.get("name", nid),
            properties=data.get("properties", {}),
            doc_properties=data.get("doc_properties", {}),
            source_doc_ids=_normalize_sids(data.get("source_doc_ids", set())),
            source_chapter=data.get("source_chapter", ""),
            confidence=data.get("confidence", 1.0),
            verified=data.get("verified", False),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            feedback_score=data.get("feedback_score", 0),
        )

    def _edge_to_relation(self, u: str, v: str, data: dict) -> GraphRelation:
        """将 NetworkX 边数据转换为 GraphRelation."""
        u_type, u_name = self._parse_node_id(u)
        v_type, v_name = self._parse_node_id(v)
        return GraphRelation(
            rel_type=data.get("rel_type", ""),
            from_name=u_name,
            to_name=v_name,
            from_type=u_type,
            to_type=v_type,
            properties=data.get("properties", {}),
            doc_properties=data.get("doc_properties", {}),
            source_doc_ids=_normalize_sids(data.get("source_doc_ids", set())),
            confidence=data.get("confidence", 1.0),
            verified=data.get("verified", False),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            feedback_score=data.get("feedback_score", 0),
        )

    # -- 实体操作 --

    def _add_to_property_index(self, nid: str, entity_type: str, properties: dict) -> None:
        """将实体属性加入索引."""
        for k, v in properties.items():
            key = (entity_type, k, v)
            if key not in self._property_index:
                self._property_index[key] = set()
            self._property_index[key].add(nid)

    def _remove_from_property_index(self, nid: str, entity_type: str, properties: dict) -> None:
        """从索引中移除实体属性."""
        for k, v in properties.items():
            key = (entity_type, k, v)
            if key in self._property_index:
                self._property_index[key].discard(nid)
                if not self._property_index[key]:
                    del self._property_index[key]

    def add_entity(self, entity: GraphEntity) -> list[dict]:
        """add_entity 函数. 返回冲突日志列表."""
        with self._lock:
            return self._add_entity_unsafe(entity)

    def _add_entity_unsafe(self, entity: GraphEntity) -> list[dict]:
        """无锁的 add_entity 实现（调用方必须持有 _lock）."""
        nid = self._node_id(entity.entity_type, entity.name)
        conflicts: list[dict] = []
        now = _now_iso()
        doc_id = next(iter(entity.source_doc_ids), "") if entity.source_doc_ids else ""
        if nid in self._g:
            existing = self._g.nodes[nid]
            # 先移除旧属性索引
            old_props = dict(existing.get("properties", {}))
            self._remove_from_property_index(nid, entity.entity_type, old_props)
            # 合并 properties，检测冲突
            merged_props = old_props
            for k, v in entity.properties.items():
                if k in merged_props and merged_props[k] != v:
                    conflicts.append(
                        {
                            "entity_type": entity.entity_type,
                            "name": entity.name,
                            "property_key": k,
                            "old_value": merged_props[k],
                            "new_value": v,
                            "timestamp": now,
                            "doc_id": doc_id,
                        }
                    )
                merged_props[k] = v
            existing["properties"] = merged_props
            # 加入新属性索引
            self._add_to_property_index(nid, entity.entity_type, merged_props)
            # 合并来源文档集合
            existing_sids = _normalize_sids(existing.get("source_doc_ids", set()))
            existing_sids.update(entity.source_doc_ids)
            existing["source_doc_ids"] = existing_sids
            existing["source_chapter"] = entity.source_chapter or existing.get("source_chapter", "")
            # 保存文档级属性快照
            existing_doc_props = existing.get("doc_properties", {})
            if doc_id:
                existing_doc_props[doc_id] = dict(entity.properties)
            existing["doc_properties"] = existing_doc_props
            # 更新元数据（取最小 confidence，保持 verified 为 True）
            existing["confidence"] = min(existing.get("confidence", 1.0), entity.confidence)
            existing["verified"] = existing.get("verified", False) or entity.verified
            existing["updated_at"] = now
        else:
            doc_props = {doc_id: dict(entity.properties)} if doc_id else {}
            self._g.add_node(
                nid,
                entity_type=entity.entity_type,
                name=entity.name,
                properties=entity.properties,
                doc_properties=doc_props,
                source_doc_ids=set(entity.source_doc_ids),
                source_chapter=entity.source_chapter,
                confidence=entity.confidence,
                verified=entity.verified,
                created_at=entity.created_at or now,
                updated_at=now,
                feedback_score=entity.feedback_score,
            )
            self._add_to_property_index(nid, entity.entity_type, entity.properties)
        logger.debug(f"添加实体 | {nid} | conflicts={len(conflicts)}")
        return conflicts

    def get_entity(self, entity_type: str, name: str) -> GraphEntity | None:
        """get_entity 函数."""
        nid = self._node_id(entity_type, name)
        if nid not in self._g:
            return None
        return self._node_to_entity(nid, self._g.nodes[nid])

    def find_by_type(self, entity_type: str) -> list[GraphEntity]:
        """find_by_type 函数."""
        results = []
        for nid, data in self._g.nodes(data=True):
            if data.get("entity_type") == entity_type:
                results.append(self._node_to_entity(nid, data))
        return results

    def find_by_property(self, entity_type: str, key: str, value: Any) -> list[GraphEntity]:
        """find_by_property 函数. 使用属性索引加速（O(1) 查找）."""
        index_key = (entity_type, key, value)
        nids = self._property_index.get(index_key, set())
        results = []
        for nid in nids:
            if nid in self._g:
                results.append(self._node_to_entity(nid, self._g.nodes[nid]))
        return results

    def all_entities(self) -> list[GraphEntity]:
        """all_entities 函数."""
        return [self._node_to_entity(nid, data) for nid, data in self._g.nodes(data=True)]

    # -- 关系操作 --

    def add_relation(self, relation: GraphRelation) -> list[dict]:
        """add_relation 函数. 返回冲突日志列表."""
        with self._lock:
            return self._add_relation_unsafe(relation)

    def _add_relation_unsafe(self, relation: GraphRelation) -> list[dict]:
        """无锁的 add_relation 实现（调用方必须持有 _lock）."""
        from_nid = self._node_id(relation.from_type or "", relation.from_name)
        to_nid = self._node_id(relation.to_type or "", relation.to_name)
        conflicts: list[dict] = []
        now = _now_iso()
        doc_id = next(iter(relation.source_doc_ids), "") if relation.source_doc_ids else ""

        # 确保节点存在（自动创建空节点）
        if from_nid not in self._g:
            self._g.add_node(
                from_nid,
                entity_type=relation.from_type or "",
                name=relation.from_name,
                source_doc_ids=set(),
                confidence=1.0,
                verified=False,
                created_at=now,
                updated_at=now,
                feedback_score=0,
            )
        if to_nid not in self._g:
            self._g.add_node(
                to_nid,
                entity_type=relation.to_type or "",
                name=relation.to_name,
                source_doc_ids=set(),
                confidence=1.0,
                verified=False,
                created_at=now,
                updated_at=now,
                feedback_score=0,
            )

        # 检查是否已存在相同边，合并 source_doc_ids
        existing = self._g.get_edge_data(from_nid, to_nid)
        if existing:
            # 检测属性冲突
            merged_props = dict(existing.get("properties", {}))
            for k, v in relation.properties.items():
                if k in merged_props and merged_props[k] != v:
                    conflicts.append(
                        {
                            "from_type": relation.from_type,
                            "from_name": relation.from_name,
                            "to_type": relation.to_type,
                            "to_name": relation.to_name,
                            "rel_type": relation.rel_type,
                            "property_key": k,
                            "old_value": merged_props[k],
                            "new_value": v,
                            "timestamp": now,
                            "doc_id": doc_id,
                        }
                    )
                merged_props[k] = v
            existing["properties"] = merged_props
            existing_sids = _normalize_sids(existing.get("source_doc_ids", set()))
            existing_sids.update(relation.source_doc_ids)
            existing["source_doc_ids"] = existing_sids
            # 保存文档级属性快照
            existing_doc_props = existing.get("doc_properties", {})
            if doc_id:
                existing_doc_props[doc_id] = dict(relation.properties)
            existing["doc_properties"] = existing_doc_props
            existing["confidence"] = min(existing.get("confidence", 1.0), relation.confidence)
            existing["verified"] = existing.get("verified", False) or relation.verified
            existing["updated_at"] = now
        else:
            doc_props = {doc_id: dict(relation.properties)} if doc_id else {}
            self._g.add_edge(
                from_nid,
                to_nid,
                rel_type=relation.rel_type,
                properties=relation.properties,
                doc_properties=doc_props,
                source_doc_ids=set(relation.source_doc_ids),
                confidence=relation.confidence,
                verified=relation.verified,
                created_at=relation.created_at or now,
                updated_at=now,
                feedback_score=relation.feedback_score,
            )
        logger.debug(
            f"添加关系 | {from_nid} -[{relation.rel_type}]-> {to_nid} | conflicts={len(conflicts)}"
        )
        return conflicts

    def get_neighbors(
        self,
        entity_type: str,
        name: str,
        rel_type: str | None = None,
        direction: str = "out",
    ) -> list[tuple[GraphEntity, str, dict]]:
        """get_neighbors 函数.

        返回: [(neighbor_entity, rel_type, rel_properties), ...].
        """
        nid = self._node_id(entity_type, name)
        if nid not in self._g:
            return []

        results = []

        if direction in ("out", "both"):
            for _, target, edge_data in self._g.out_edges(nid, data=True):
                if rel_type is None or edge_data.get("rel_type") == rel_type:
                    results.append(
                        (
                            self._node_to_entity(target, self._g.nodes[target]),
                            edge_data.get("rel_type", ""),
                            dict(edge_data.get("properties", {})),
                        )
                    )

        if direction in ("in", "both"):
            for source, _, edge_data in self._g.in_edges(nid, data=True):
                if rel_type is None or edge_data.get("rel_type") == rel_type:
                    results.append(
                        (
                            self._node_to_entity(source, self._g.nodes[source]),
                            edge_data.get("rel_type", ""),
                            dict(edge_data.get("properties", {})),
                        )
                    )

        return results

    def get_entity_relations(
        self,
        entity_type: str,
        name: str,
        direction: str = "both",
    ) -> list[GraphRelation]:
        """获取与指定实体相关的所有完整关系.

        Args:
            entity_type: 实体类型
            name: 实体名称
            direction: 方向 out/in/both

        Returns:
            GraphRelation 列表
        """
        nid = self._node_id(entity_type, name)
        if nid not in self._g:
            return []
        results: list[GraphRelation] = []
        if direction in ("out", "both"):
            for _, target, data in self._g.out_edges(nid, data=True):
                results.append(self._edge_to_relation(nid, target, data))
        if direction in ("in", "both"):
            for source, _, data in self._g.in_edges(nid, data=True):
                results.append(self._edge_to_relation(source, nid, data))
        return results

    def all_relations(self) -> list[GraphRelation]:
        """all_relations 函数."""
        return [self._edge_to_relation(u, v, data) for u, v, data in self._g.edges(data=True)]

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
        with self._lock:
            self._save_unsafe(path)

    def _save_unsafe(self, path: Path) -> None:
        """无锁的 save 实现（调用方必须持有 _lock）."""
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
        with self._lock:
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
            # 重建属性索引
            self._property_index.clear()
            for nid, data in self._g.nodes(data=True):
                et = data.get("entity_type", "")
                props = data.get("properties", {})
                self._add_to_property_index(nid, et, props)
            logger.info(
                f"图谱已加载 | nodes={self._g.number_of_nodes()} | "
                f"edges={self._g.number_of_edges()} | path={path}"
            )

    def clear(self) -> None:
        """Clear 函数."""
        with self._lock:
            self._g.clear()
            self._property_index.clear()
            logger.info("图谱已清空")

    def stats(self) -> dict[str, int]:
        """Stats 函数."""
        return {
            "nodes": self._g.number_of_nodes(),
            "edges": self._g.number_of_edges(),
        }

    def remove_document_contributions(self, doc_id: str) -> None:
        """从全局图中移除指定文档贡献的节点和边."""
        with self._lock:
            self._remove_document_contributions_unsafe(doc_id)

    def _remove_document_contributions_unsafe(self, doc_id: str) -> None:
        """无锁的 remove_document_contributions 实现（调用方必须持有 _lock）."""
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

    def find_path(
        self,
        from_type: str,
        from_name: str,
        to_type: str,
        to_name: str,
        max_depth: int = 3,
        rel_types: set[str] | None = None,
    ) -> list[list[str]]:
        """find_path 函数."""
        from_nid = self._node_id(from_type, from_name)
        to_nid = self._node_id(to_type, to_name)

        if from_nid not in self._g or to_nid not in self._g:
            return []

        # 限制 max_depth 防止爆炸
        max_depth = min(max(max_depth, 1), Config.GRAPH_MAX_PATH_DEPTH)

        # BFS 收集所有简单路径（不重复节点），限制深度，支持关系过滤
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
                for _, nxt, edge_data in self._g.out_edges(current, data=True):
                    if nxt not in path:
                        if rel_types is None or edge_data.get("rel_type") in rel_types:
                            queue.append((nxt, path + [nxt]))
            return paths

        return _bfs_paths(from_nid, to_nid, max_depth)

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
                results.append(self._node_to_entity(nid, data))
        return results

    # -- 动态更新（CRUD）--

    def update_entity(
        self,
        entity_type: str,
        name: str,
        properties: dict[str, Any] | None = None,
        confidence: float | None = None,
        verified: bool | None = None,
        feedback_score: int | None = None,
    ) -> bool:
        """更新实体属性."""
        with self._lock:
            return self._update_entity_unsafe(
                entity_type, name, properties, confidence, verified, feedback_score
            )

    def _update_entity_unsafe(
        self,
        entity_type: str,
        name: str,
        properties: dict[str, Any] | None = None,
        confidence: float | None = None,
        verified: bool | None = None,
        feedback_score: int | None = None,
    ) -> bool:
        """无锁的 update_entity 实现（调用方必须持有 _lock）."""
        nid = self._node_id(entity_type, name)
        if nid not in self._g:
            return False
        now = _now_iso()
        if properties is not None:
            old_props = dict(self._g.nodes[nid].get("properties", {}))
            self._remove_from_property_index(nid, entity_type, old_props)
            self._g.nodes[nid]["properties"] = properties
            self._add_to_property_index(nid, entity_type, properties)
        if confidence is not None:
            self._g.nodes[nid]["confidence"] = confidence
        if verified is not None:
            self._g.nodes[nid]["verified"] = verified
        if feedback_score is not None:
            self._g.nodes[nid]["feedback_score"] = feedback_score
        self._g.nodes[nid]["updated_at"] = now
        return True

    def delete_entity(self, entity_type: str, name: str) -> bool:
        """删除实体（级联删除关联边）."""
        with self._lock:
            return self._delete_entity_unsafe(entity_type, name)

    def _delete_entity_unsafe(self, entity_type: str, name: str) -> bool:
        """无锁的 delete_entity 实现（调用方必须持有 _lock）."""
        nid = self._node_id(entity_type, name)
        if nid not in self._g:
            return False
        # 移除属性索引
        old_props = dict(self._g.nodes[nid].get("properties", {}))
        self._remove_from_property_index(nid, entity_type, old_props)
        self._g.remove_node(nid)
        return True

    def update_relation(
        self,
        from_type: str,
        from_name: str,
        to_type: str,
        to_name: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
        confidence: float | None = None,
        verified: bool | None = None,
        feedback_score: int | None = None,
    ) -> bool:
        """更新关系属性."""
        with self._lock:
            return self._update_relation_unsafe(
                from_type,
                from_name,
                to_type,
                to_name,
                rel_type,
                properties,
                confidence,
                verified,
                feedback_score,
            )

    def _update_relation_unsafe(
        self,
        from_type: str,
        from_name: str,
        to_type: str,
        to_name: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
        confidence: float | None = None,
        verified: bool | None = None,
        feedback_score: int | None = None,
    ) -> bool:
        """无锁的 update_relation 实现（调用方必须持有 _lock）."""
        from_nid = self._node_id(from_type, from_name)
        to_nid = self._node_id(to_type, to_name)
        if not self._g.has_edge(from_nid, to_nid):
            return False
        data = self._g.edges[from_nid, to_nid]
        if data.get("rel_type") != rel_type:
            return False
        now = _now_iso()
        if properties is not None:
            data["properties"] = properties
        if confidence is not None:
            data["confidence"] = confidence
        if verified is not None:
            data["verified"] = verified
        if feedback_score is not None:
            data["feedback_score"] = feedback_score
        data["updated_at"] = now
        return True

    def delete_relation(
        self,
        from_type: str,
        from_name: str,
        to_type: str,
        to_name: str,
        rel_type: str,
    ) -> bool:
        """删除关系."""
        with self._lock:
            return self._delete_relation_unsafe(from_type, from_name, to_type, to_name, rel_type)

    def _delete_relation_unsafe(
        self,
        from_type: str,
        from_name: str,
        to_type: str,
        to_name: str,
        rel_type: str,
    ) -> bool:
        """无锁的 delete_relation 实现（调用方必须持有 _lock）."""
        from_nid = self._node_id(from_type, from_name)
        to_nid = self._node_id(to_type, to_name)
        if not self._g.has_edge(from_nid, to_nid):
            return False
        data = self._g.edges[from_nid, to_nid]
        if data.get("rel_type") != rel_type:
            return False
        self._g.remove_edge(from_nid, to_nid)
        return True

    def verify_entity(self, entity_type: str, name: str, verified: bool = True) -> bool:
        """标记实体验证状态."""
        with self._lock:
            nid = self._node_id(entity_type, name)
            if nid not in self._g:
                return False
            self._g.nodes[nid]["verified"] = verified
            self._g.nodes[nid]["updated_at"] = _now_iso()
            return True


# ---------------------------------------------------------------------------
# 子图视图
# ---------------------------------------------------------------------------


class SubGraphView:
    """子图视图，提供只读查询接口."""

    def __init__(self, g: nx.DiGraph) -> None:
        """初始化 SubGraphView."""
        self._g = g

    def _node_to_entity(self, nid: str, data: dict) -> GraphEntity:
        """将 NetworkX 节点数据转换为 GraphEntity."""
        return GraphEntity(
            entity_type=data.get("entity_type", ""),
            name=data.get("name", nid),
            properties=data.get("properties", {}),
            doc_properties=data.get("doc_properties", {}),
            source_doc_ids=_normalize_sids(data.get("source_doc_ids", set())),
            source_chapter=data.get("source_chapter", ""),
            confidence=data.get("confidence", 1.0),
            verified=data.get("verified", False),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            feedback_score=data.get("feedback_score", 0),
        )

    def _edge_to_relation(self, u: str, v: str, data: dict) -> GraphRelation:
        """将 NetworkX 边数据转换为 GraphRelation."""
        u_type = self._g.nodes[u].get("entity_type", "")
        u_name = self._g.nodes[u].get("name", u)
        v_type = self._g.nodes[v].get("entity_type", "")
        v_name = self._g.nodes[v].get("name", v)
        return GraphRelation(
            rel_type=data.get("rel_type", ""),
            from_name=u_name,
            to_name=v_name,
            from_type=u_type,
            to_type=v_type,
            properties=data.get("properties", {}),
            doc_properties=data.get("doc_properties", {}),
            source_doc_ids=_normalize_sids(data.get("source_doc_ids", set())),
            confidence=data.get("confidence", 1.0),
            verified=data.get("verified", False),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            feedback_score=data.get("feedback_score", 0),
        )

    def to_text_context(self) -> str:
        """将子图转换为文本上下文，供 LLM 使用."""
        lines = []
        for nid, data in self._g.nodes(data=True):
            entity_type = data.get("entity_type", "")
            name = data.get("name", "")
            props = data.get("properties", {})
            prop_str = ", ".join(f"{k}={v}" for k, v in props.items() if v)
            sids = sorted(_normalize_sids(data.get("source_doc_ids", set())))
            sources = f" [来源: {', '.join(sids)}]" if sids else ""
            verified_mark = " ✓" if data.get("verified", False) else ""
            lines.append(
                f"- [{entity_type}] {name}{verified_mark}{sources}"
                + (f" ({prop_str})" if prop_str else "")
            )

        if self._g.number_of_edges() > 0:
            lines.append("\n关系:")
            for u, v, data in self._g.edges(data=True):
                u_name = self._g.nodes[u].get("name", u)
                v_name = self._g.nodes[v].get("name", v)
                rel = data.get("rel_type", "")
                rel_props = data.get("properties", {})
                rel_prop_str = ", ".join(f"{k}={v}" for k, v in rel_props.items() if v)
                rel_extra = f" ({rel_prop_str})" if rel_prop_str else ""
                lines.append(f"  {u_name} --[{rel}]{rel_extra}--> {v_name}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """to_dict 函数."""
        return nx.node_link_data(self._g, edges="edges")

    def entities(self) -> list[GraphEntity]:
        """Entities 函数."""
        return [self._node_to_entity(nid, data) for nid, data in self._g.nodes(data=True)]

    def relations(self) -> list[GraphRelation]:
        """Relations 函数."""
        return [self._edge_to_relation(u, v, data) for u, v, data in self._g.edges(data=True)]

    @property
    def node_count(self) -> int:
        """node_count 函数."""
        return self._g.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """edge_count 函数."""
        return self._g.number_of_edges()
