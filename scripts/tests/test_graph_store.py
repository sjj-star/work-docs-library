"""graph_store 模块单元测试."""

import pytest
from core.graph_store import (
    GraphEntity,
    GraphRelation,
    NetworkXGraphStore,
)


@pytest.fixture
def graph_store():
    """返回一个全新的 NetworkXGraphStore 实例."""
    return NetworkXGraphStore()


@pytest.fixture
def sample_entities():
    """返回一组示例实体."""
    return [
        GraphEntity(
            entity_type="Module",
            name="TOP",
            properties={"description": "top module"},
            source_doc_ids={"doc1"},
            source_chapter="ch1",
        ),
        GraphEntity(
            entity_type="Module",
            name="SUB",
            properties={"description": "sub module"},
            source_doc_ids={"doc1"},
            source_chapter="ch1",
        ),
        GraphEntity(
            entity_type="Signal",
            name="CLK",
            properties={"width": 1, "clock": True},
            source_doc_ids={"doc1"},
            source_chapter="ch2",
        ),
    ]


@pytest.fixture
def sample_relations():
    """返回一组示例关系."""
    return [
        GraphRelation(
            rel_type="CONTAINS",
            from_name="TOP",
            to_name="SUB",
            from_type="Module",
            to_type="Module",
        ),
        GraphRelation(
            rel_type="HAS_SIGNAL",
            from_name="TOP",
            to_name="CLK",
            from_type="Module",
            to_type="Signal",
        ),
    ]


# ---------------------------------------------------------------------------
# 1. GraphEntity / GraphRelation 数据模型
# ---------------------------------------------------------------------------


def test_graph_entity_to_dict_from_dict():
    """GraphEntity to_dict / from_dict 往返一致."""
    e = GraphEntity(
        entity_type="Register",
        name="CTRL",
        properties={"addr": "0x00", "width": 32},
        source_doc_ids={"doc1"},
        source_chapter="ch3",
    )
    d = e.to_dict()
    assert d == {
        "type": "Register",
        "name": "CTRL",
        "properties": {"addr": "0x00", "width": 32},
        "doc_properties": {},
        "source_doc_ids": ["doc1"],
        "source_chapter": "ch3",
        "confidence": 1.0,
        "verified": False,
        "created_at": "",
        "updated_at": "",
        "feedback_score": 0,
    }
    restored = GraphEntity.from_dict(d)
    assert restored == e


def test_graph_relation_to_dict_from_dict():
    """GraphRelation to_dict / from_dict 往返一致."""
    r = GraphRelation(
        rel_type="HAS_REGISTER",
        from_name="TOP",
        to_name="CTRL",
        from_type="Module",
        to_type="Register",
        properties={"access": "RW"},
    )
    d = r.to_dict()
    assert d == {
        "type": "HAS_REGISTER",
        "from": "TOP",
        "to": "CTRL",
        "from_type": "Module",
        "to_type": "Register",
        "properties": {"access": "RW"},
        "doc_properties": {},
        "source_doc_ids": [],
        "source_chapter": "",
        "confidence": 1.0,
        "verified": False,
        "created_at": "",
        "updated_at": "",
        "feedback_score": 0,
    }
    restored = GraphRelation.from_dict(d)
    assert restored == r


def test_graph_entity_defaults():
    """GraphEntity 默认属性为空."""
    e = GraphEntity(entity_type="Module", name="A")
    assert e.properties == {}
    assert e.source_doc_ids == set()
    assert e.source_chapter == ""


def test_graph_relation_defaults():
    """GraphRelation 默认属性为空."""
    r = GraphRelation(rel_type="CONTAINS", from_name="A", to_name="B")
    assert r.from_type == ""
    assert r.to_type == ""
    assert r.properties == {}


# ---------------------------------------------------------------------------
# 2. 实体 CRUD
# ---------------------------------------------------------------------------


def test_add_entity_and_get_entity(graph_store, sample_entities):
    """add_entity 后可通过 get_entity 正确读取."""
    for e in sample_entities:
        graph_store.add_entity(e)

    top = graph_store.get_entity("Module", "TOP")
    assert top is not None
    assert top.name == "TOP"
    assert top.properties == {"description": "top module"}

    clk = graph_store.get_entity("Signal", "CLK")
    assert clk is not None
    assert clk.properties["clock"] is True


def test_get_entity_missing(graph_store):
    """查询不存在的实体返回 None."""
    assert graph_store.get_entity("Module", "NONEXISTENT") is None


def test_find_by_type(graph_store, sample_entities):
    """find_by_type 按类型过滤."""
    for e in sample_entities:
        graph_store.add_entity(e)

    modules = graph_store.find_by_type("Module")
    assert len(modules) == 2
    names = {m.name for m in modules}
    assert names == {"TOP", "SUB"}

    signals = graph_store.find_by_type("Signal")
    assert len(signals) == 1
    assert signals[0].name == "CLK"

    empty = graph_store.find_by_type("Register")
    assert empty == []


def test_find_by_property(graph_store, sample_entities):
    """find_by_property 按属性键值过滤."""
    for e in sample_entities:
        graph_store.add_entity(e)

    results = graph_store.find_by_property("Signal", "clock", True)
    assert len(results) == 1
    assert results[0].name == "CLK"

    results = graph_store.find_by_property("Signal", "clock", False)
    assert results == []

    results = graph_store.find_by_property("Module", "description", "top module")
    assert len(results) == 1
    assert results[0].name == "TOP"


# ---------------------------------------------------------------------------
# 3. 关系与邻居查询
# ---------------------------------------------------------------------------


def test_add_relation_and_get_neighbors_out(graph_store, sample_entities, sample_relations):
    """Out 方向邻居查询."""
    for e in sample_entities:
        graph_store.add_entity(e)
    for r in sample_relations:
        graph_store.add_relation(r)

    neighbors = graph_store.get_neighbors("Module", "TOP", direction="out")
    assert len(neighbors) == 2
    names = {n.name for n, _, _ in neighbors}
    assert names == {"SUB", "CLK"}


def test_get_neighbors_in(graph_store, sample_entities, sample_relations):
    """In 方向邻居查询."""
    for e in sample_entities:
        graph_store.add_entity(e)
    for r in sample_relations:
        graph_store.add_relation(r)

    neighbors = graph_store.get_neighbors("Module", "SUB", direction="in")
    assert len(neighbors) == 1
    assert neighbors[0][0].name == "TOP"
    assert neighbors[0][1] == "CONTAINS"
    assert neighbors[0][2] == {}  # rel_properties


def test_get_neighbors_both(graph_store, sample_entities, sample_relations):
    """Both 方向邻居查询."""
    for e in sample_entities:
        graph_store.add_entity(e)
    for r in sample_relations:
        graph_store.add_relation(r)

    # 再给 SUB 加一条出边，使其既有入边又有出边
    graph_store.add_relation(
        GraphRelation(
            rel_type="HAS_SIGNAL",
            from_name="SUB",
            to_name="CLK",
            from_type="Module",
            to_type="Signal",
        )
    )

    neighbors = graph_store.get_neighbors("Module", "SUB", direction="both")
    # 入边: TOP --CONTAINS--> SUB
    # 出边: SUB --HAS_SIGNAL--> CLK
    assert len(neighbors) == 2
    names = {n.name for n, _, _ in neighbors}
    assert names == {"TOP", "CLK"}


def test_get_neighbors_filtered_by_rel_type(graph_store, sample_entities, sample_relations):
    """按关系类型过滤邻居."""
    for e in sample_entities:
        graph_store.add_entity(e)
    for r in sample_relations:
        graph_store.add_relation(r)

    neighbors = graph_store.get_neighbors("Module", "TOP", rel_type="CONTAINS", direction="out")
    assert len(neighbors) == 1
    assert neighbors[0][0].name == "SUB"


def test_get_neighbors_missing_node(graph_store):
    """对不存在的节点查询邻居返回空列表."""
    assert graph_store.get_neighbors("Module", "X", direction="out") == []
    assert graph_store.get_neighbors("Module", "X", direction="in") == []
    assert graph_store.get_neighbors("Module", "X", direction="both") == []


# ---------------------------------------------------------------------------
# 4. 全量枚举
# ---------------------------------------------------------------------------


def test_all_entities(graph_store, sample_entities):
    """all_entities 返回所有实体."""
    for e in sample_entities:
        graph_store.add_entity(e)
    all_e = graph_store.all_entities()
    assert len(all_e) == 3
    names = {e.name for e in all_e}
    assert names == {"TOP", "SUB", "CLK"}


def test_all_relations(graph_store, sample_entities, sample_relations):
    """all_relations 返回所有关系."""
    for e in sample_entities:
        graph_store.add_entity(e)
    for r in sample_relations:
        graph_store.add_relation(r)

    all_r = graph_store.all_relations()
    assert len(all_r) == 2
    types = {r.rel_type for r in all_r}
    assert types == {"CONTAINS", "HAS_SIGNAL"}


# ---------------------------------------------------------------------------
# 5. 子图查询
# ---------------------------------------------------------------------------


def _build_chain_store(graph_store):
    """构建链式拓扑: A -> B -> C -> D, 并额外给 A -> C."""
    for name in ("A", "B", "C", "D"):
        graph_store.add_entity(GraphEntity(entity_type="Module", name=name))
    graph_store.add_relation(
        GraphRelation(
            rel_type="CONTAINS",
            from_name="A",
            to_name="B",
            from_type="Module",
            to_type="Module",
        )
    )
    graph_store.add_relation(
        GraphRelation(
            rel_type="CONTAINS",
            from_name="B",
            to_name="C",
            from_type="Module",
            to_type="Module",
        )
    )
    graph_store.add_relation(
        GraphRelation(
            rel_type="CONTAINS",
            from_name="C",
            to_name="D",
            from_type="Module",
            to_type="Module",
        )
    )
    graph_store.add_relation(
        GraphRelation(
            rel_type="HAS_SIGNAL",
            from_name="A",
            to_name="C",
            from_type="Module",
            to_type="Module",
        )
    )


def test_get_subgraph_depth_1(graph_store):
    """depth=1 时只包含中心节点及直接邻居."""
    _build_chain_store(graph_store)
    sg = graph_store.get_subgraph("Module", "B", depth=1)
    names = {e.name for e in sg.entities()}
    assert names == {"A", "B", "C"}
    assert sg.node_count == 3
    # NetworkX subgraph 会保留所有选中节点之间的边，因此 A->C 也会被包含
    assert sg.edge_count == 3  # A->B, B->C, A->C


def test_get_subgraph_depth_2(graph_store):
    """depth=2 时包含两层邻居."""
    _build_chain_store(graph_store)
    sg = graph_store.get_subgraph("Module", "B", depth=2)
    names = {e.name for e in sg.entities()}
    assert names == {"A", "B", "C", "D"}
    assert sg.node_count == 4


def test_get_subgraph_rel_type_filter(graph_store):
    """rel_types 过滤只遍历指定类型的边."""
    _build_chain_store(graph_store)
    # 只沿 CONTAINS 边展开，A->C 的 HAS_SIGNAL 不应被遍历
    sg = graph_store.get_subgraph("Module", "A", depth=2, rel_types={"CONTAINS"})
    names = {e.name for e in sg.entities()}
    assert names == {"A", "B", "C"}
    # 若包含 HAS_SIGNAL，则 C 可达，depth=2 时 D 也可达
    sg2 = graph_store.get_subgraph("Module", "A", depth=2, rel_types={"CONTAINS", "HAS_SIGNAL"})
    names2 = {e.name for e in sg2.entities()}
    assert names2 == {"A", "B", "C", "D"}


def test_get_subgraph_missing_center(graph_store):
    """中心节点不存在时返回空 SubGraphView."""
    sg = graph_store.get_subgraph("Module", "MISSING", depth=1)
    assert sg.node_count == 0
    assert sg.edge_count == 0
    assert sg.entities() == []
    assert sg.relations() == []


# ---------------------------------------------------------------------------
# 6. 持久化 / 清空 / 统计
# ---------------------------------------------------------------------------


def test_save_and_load(tmp_path, graph_store, sample_entities, sample_relations):
    """Save 后 load 应恢复完全一致的数据."""
    for e in sample_entities:
        graph_store.add_entity(e)
    for r in sample_relations:
        graph_store.add_relation(r)

    path = tmp_path / "graph.json"
    graph_store.save(path)
    assert path.exists()

    new_store = NetworkXGraphStore()
    new_store.load(path)

    assert new_store.stats() == {"nodes": 3, "edges": 2}
    assert {e.name for e in new_store.all_entities()} == {"TOP", "SUB", "CLK"}
    assert len(new_store.all_relations()) == 2

    top = new_store.get_entity("Module", "TOP")
    assert top is not None
    assert top.properties == {"description": "top module"}


def test_clear(graph_store, sample_entities, sample_relations):
    """Clear 后图谱为空."""
    for e in sample_entities:
        graph_store.add_entity(e)
    for r in sample_relations:
        graph_store.add_relation(r)

    assert graph_store.stats()["nodes"] == 3
    graph_store.clear()
    assert graph_store.stats() == {"nodes": 0, "edges": 0}
    assert graph_store.all_entities() == []
    assert graph_store.all_relations() == []


def test_stats(graph_store, sample_entities, sample_relations):
    """Stats 返回正确的节点和边数."""
    assert graph_store.stats() == {"nodes": 0, "edges": 0}

    for e in sample_entities:
        graph_store.add_entity(e)
    assert graph_store.stats() == {"nodes": 3, "edges": 0}

    for r in sample_relations:
        graph_store.add_relation(r)
    assert graph_store.stats() == {"nodes": 3, "edges": 2}


# ---------------------------------------------------------------------------
# 7. SubGraphView 只读接口
# ---------------------------------------------------------------------------


def test_subgraph_view_to_text_context(graph_store):
    """to_text_context 生成预期文本."""
    graph_store.add_entity(GraphEntity(entity_type="Module", name="M1", properties={"a": 1}))
    graph_store.add_entity(GraphEntity(entity_type="Signal", name="S1", properties={}))
    graph_store.add_relation(
        GraphRelation(
            rel_type="HAS_SIGNAL",
            from_name="M1",
            to_name="S1",
            from_type="Module",
            to_type="Signal",
        )
    )
    sg = graph_store.get_subgraph("Module", "M1", depth=1)
    text = sg.to_text_context()
    assert "[Module] M1 (a=1)" in text
    assert "[Signal] S1" in text
    assert "M1 --[HAS_SIGNAL]--> S1" in text


def test_subgraph_view_to_dict(graph_store, sample_entities, sample_relations):
    """to_dict 返回 networkx node_link_data 格式."""
    for e in sample_entities:
        graph_store.add_entity(e)
    for r in sample_relations:
        graph_store.add_relation(r)
    sg = graph_store.get_subgraph("Module", "TOP", depth=1)
    d = sg.to_dict()
    assert "nodes" in d
    assert "edges" in d
    assert len(d["nodes"]) == sg.node_count
    assert len(d["edges"]) == sg.edge_count


def test_subgraph_view_entities(graph_store, sample_entities):
    """Entities 返回 GraphEntity 列表."""
    for e in sample_entities:
        graph_store.add_entity(e)
    sg = graph_store.get_subgraph("Module", "TOP", depth=0)
    ents = sg.entities()
    assert len(ents) == 1
    assert ents[0].name == "TOP"
    assert isinstance(ents[0], GraphEntity)


def test_subgraph_view_relations(graph_store, sample_entities, sample_relations):
    """Relations 返回 GraphRelation 列表."""
    for e in sample_entities:
        graph_store.add_entity(e)
    for r in sample_relations:
        graph_store.add_relation(r)
    sg = graph_store.get_subgraph("Module", "TOP", depth=1)
    rels = sg.relations()
    assert len(rels) == 2
    assert all(isinstance(r, GraphRelation) for r in rels)


def test_subgraph_view_counts(graph_store, sample_entities, sample_relations):
    """node_count / edge_count 属性正确."""
    for e in sample_entities:
        graph_store.add_entity(e)
    for r in sample_relations:
        graph_store.add_relation(r)
    sg = graph_store.get_subgraph("Module", "TOP", depth=1)
    assert sg.node_count == 3
    assert sg.edge_count == 2


# ---------------------------------------------------------------------------
# 8. 实体去重：同名同类型合并属性
# ---------------------------------------------------------------------------


def test_entity_deduplication_merges_properties(graph_store):
    """重复添加同名同类型实体应合并属性."""
    e1 = GraphEntity(entity_type="Module", name="X", properties={"a": 1, "b": 2})
    e2 = GraphEntity(entity_type="Module", name="X", properties={"b": 3, "c": 4})
    graph_store.add_entity(e1)
    graph_store.add_entity(e2)

    ent = graph_store.get_entity("Module", "X")
    assert ent is not None
    assert ent.properties == {"a": 1, "b": 3, "c": 4}
    # 后添加的实体 source_doc_id 覆盖空值
    assert ent.name == "X"


def test_entity_deduplication_preserves_source_when_empty(graph_store):
    """合并时若新实体 source 为空则保留旧值."""
    e1 = GraphEntity(
        entity_type="Module",
        name="X",
        properties={"a": 1},
        source_doc_ids={"doc1"},
        source_chapter="ch1",
    )
    e2 = GraphEntity(
        entity_type="Module",
        name="X",
        properties={"b": 2},
        source_doc_ids=set(),
        source_chapter="",
    )
    graph_store.add_entity(e1)
    graph_store.add_entity(e2)

    ent = graph_store.get_entity("Module", "X")
    assert ent.properties == {"a": 1, "b": 2}
    assert ent.source_doc_ids == {"doc1"}
    assert ent.source_chapter == "ch1"


# ---------------------------------------------------------------------------
# 9. 关系自动创建缺失节点
# ---------------------------------------------------------------------------


def test_relation_auto_creates_nodes(graph_store):
    """add_relation 在节点不存在时自动创建空节点."""
    r = GraphRelation(
        rel_type="CONTAINS",
        from_name="PARENT",
        to_name="CHILD",
        from_type="Module",
        to_type="Module",
    )
    graph_store.add_relation(r)

    assert graph_store.stats() == {"nodes": 2, "edges": 1}

    parent = graph_store.get_entity("Module", "PARENT")
    assert parent is not None
    assert parent.properties == {}

    child = graph_store.get_entity("Module", "CHILD")
    assert child is not None
    assert child.properties == {}

    relations = graph_store.all_relations()
    assert len(relations) == 1
    assert relations[0].rel_type == "CONTAINS"
    assert relations[0].from_name == "PARENT"
    assert relations[0].to_name == "CHILD"


# ---------------------------------------------------------------------------
# 10. 路径搜索
# ---------------------------------------------------------------------------


def _build_path_graph(graph_store):
    """构建路径测试图: A -> B -> C -> D, A -> C."""
    for name in ("A", "B", "C", "D"):
        graph_store.add_entity(GraphEntity(entity_type="Module", name=name))
    graph_store.add_relation(
        GraphRelation(
            rel_type="CONTAINS",
            from_name="A",
            to_name="B",
            from_type="Module",
            to_type="Module",
        )
    )
    graph_store.add_relation(
        GraphRelation(
            rel_type="CONTAINS",
            from_name="B",
            to_name="C",
            from_type="Module",
            to_type="Module",
        )
    )
    graph_store.add_relation(
        GraphRelation(
            rel_type="CONTAINS",
            from_name="C",
            to_name="D",
            from_type="Module",
            to_type="Module",
        )
    )
    graph_store.add_relation(
        GraphRelation(
            rel_type="HAS_SIGNAL",
            from_name="A",
            to_name="C",
            from_type="Module",
            to_type="Module",
        )
    )


def test_find_path_direct(graph_store):
    """find_path 找到直接路径."""
    _build_path_graph(graph_store)
    paths = graph_store.find_path("Module", "A", "Module", "B", max_depth=3)
    assert len(paths) == 1
    assert paths[0] == ["Module::A", "Module::B"]


def test_find_path_multiple_routes(graph_store):
    """find_path 找到多条路径."""
    _build_path_graph(graph_store)
    paths = graph_store.find_path("Module", "A", "Module", "C", max_depth=3)
    # A->B->C 和 A->C
    assert len(paths) == 2
    route_ids = {tuple(p) for p in paths}
    assert ("Module::A", "Module::C") in route_ids
    assert ("Module::A", "Module::B", "Module::C") in route_ids


def test_find_path_longer_depth(graph_store):
    """find_path 限制深度时过滤长路径."""
    _build_path_graph(graph_store)
    # depth=2 时 A->B->C->D (4 nodes) 和 A->C->D (3 nodes) 中，
    # 只有 A->C->D (3 nodes, depth=2 edges) 可达
    paths = graph_store.find_path("Module", "A", "Module", "D", max_depth=2)
    assert len(paths) == 1
    assert paths[0] == ["Module::A", "Module::C", "Module::D"]
    # depth=3 时两条路径都可达
    paths = graph_store.find_path("Module", "A", "Module", "D", max_depth=3)
    assert len(paths) == 2
    route_ids = {tuple(p) for p in paths}
    assert ("Module::A", "Module::B", "Module::C", "Module::D") in route_ids
    assert ("Module::A", "Module::C", "Module::D") in route_ids


def test_find_path_missing_node(graph_store):
    """find_path 起点或终点不存在时返回空."""
    graph_store.add_entity(GraphEntity(entity_type="Module", name="A"))
    assert graph_store.find_path("Module", "A", "Module", "X", max_depth=3) == []
    assert graph_store.find_path("Module", "X", "Module", "A", max_depth=3) == []


def test_find_path_max_depth_capped(graph_store):
    """find_path max_depth 超过 6 会被截断到 6."""
    graph_store.add_entity(GraphEntity(entity_type="Module", name="A"))
    graph_store.add_entity(GraphEntity(entity_type="Module", name="B"))
    graph_store.add_relation(
        GraphRelation(
            rel_type="CONTAINS",
            from_name="A",
            to_name="B",
            from_type="Module",
            to_type="Module",
        )
    )
    # max_depth=10 会被截断到 6，但仍能找到路径
    paths = graph_store.find_path("Module", "A", "Module", "B", max_depth=10)
    assert len(paths) == 1


# ---------------------------------------------------------------------------
# 11. 实体搜索
# ---------------------------------------------------------------------------


def test_search_entities_by_name_pattern(graph_store, sample_entities):
    """search_entities 按名称子串匹配."""
    for e in sample_entities:
        graph_store.add_entity(e)
    results = graph_store.search_entities("CL")
    assert len(results) == 1
    assert results[0].name == "CLK"


def test_search_entities_case_insensitive(graph_store, sample_entities):
    """search_entities 大小写不敏感."""
    for e in sample_entities:
        graph_store.add_entity(e)
    results = graph_store.search_entities("top")
    assert len(results) == 1
    assert results[0].name == "TOP"


def test_search_entities_by_type(graph_store, sample_entities):
    """search_entities 限制实体类型."""
    for e in sample_entities:
        graph_store.add_entity(e)
    # "O" matches "TOP" but not "SUB"
    results = graph_store.search_entities("O", entity_types={"Module"})
    assert len(results) == 1
    assert results[0].name == "TOP"

    results = graph_store.search_entities("O", entity_types={"Signal"})
    assert len(results) == 0


def test_search_entities_empty_query(graph_store, sample_entities):
    """search_entities 空查询匹配所有（在类型限制下）."""
    for e in sample_entities:
        graph_store.add_entity(e)
    results = graph_store.search_entities("", entity_types={"Module"})
    assert len(results) == 2


def test_search_entities_no_match(graph_store, sample_entities):
    """search_entities 无匹配时返回空列表."""
    for e in sample_entities:
        graph_store.add_entity(e)
    assert graph_store.search_entities("NONEXISTENT") == []


# ---------------------------------------------------------------------------
# 10. CRUD 与冲突检测
# ---------------------------------------------------------------------------


def test_update_entity(graph_store):
    """update_entity 更新属性."""
    graph_store.add_entity(GraphEntity(entity_type="Module", name="M1", properties={"a": 1}))
    ok = graph_store.update_entity("Module", "M1", properties={"a": 2, "b": 3}, confidence=0.8)
    assert ok is True
    e = graph_store.get_entity("Module", "M1")
    assert e.properties == {"a": 2, "b": 3}
    assert e.confidence == 0.8


def test_update_entity_missing(graph_store):
    """update_entity 对不存在的实体返回 False."""
    assert graph_store.update_entity("Module", "X", properties={"a": 1}) is False


def test_delete_entity(graph_store):
    """delete_entity 删除实体并级联删除关联边."""
    graph_store.add_entity(GraphEntity(entity_type="Module", name="A"))
    graph_store.add_entity(GraphEntity(entity_type="Module", name="B"))
    graph_store.add_relation(
        GraphRelation(
            rel_type="CONTAINS", from_name="A", to_name="B", from_type="Module", to_type="Module"
        )
    )
    assert graph_store.stats()["edges"] == 1
    ok = graph_store.delete_entity("Module", "A")
    assert ok is True
    assert graph_store.stats()["nodes"] == 1
    assert graph_store.stats()["edges"] == 0


def test_delete_entity_missing(graph_store):
    """delete_entity 对不存在的实体返回 False."""
    assert graph_store.delete_entity("Module", "X") is False


def test_verify_entity(graph_store):
    """verify_entity 标记验证状态."""
    graph_store.add_entity(GraphEntity(entity_type="Module", name="M1"))
    assert graph_store.get_entity("Module", "M1").verified is False
    assert graph_store.verify_entity("Module", "M1", True) is True
    assert graph_store.get_entity("Module", "M1").verified is True
    assert graph_store.verify_entity("Module", "M1", False) is True
    assert graph_store.get_entity("Module", "M1").verified is False


def test_add_entity_conflict_detection(graph_store):
    """add_entity 合并时检测属性冲突."""
    graph_store.add_entity(GraphEntity(entity_type="Register", name="R1", properties={"width": 32}))
    conflicts = graph_store.add_entity(
        GraphEntity(entity_type="Register", name="R1", properties={"width": 64})
    )
    assert len(conflicts) == 1
    assert conflicts[0]["property_key"] == "width"
    assert conflicts[0]["old_value"] == 32
    assert conflicts[0]["new_value"] == 64
    # 属性应被更新
    e = graph_store.get_entity("Register", "R1")
    assert e.properties["width"] == 64


def test_add_relation_conflict_detection(graph_store):
    """add_relation 合并时检测属性冲突."""
    graph_store.add_entity(GraphEntity(entity_type="Module", name="A"))
    graph_store.add_entity(GraphEntity(entity_type="Module", name="B"))
    graph_store.add_relation(
        GraphRelation(
            rel_type="CONTAINS",
            from_name="A",
            to_name="B",
            from_type="Module",
            to_type="Module",
            properties={"order": 1},
        )
    )
    conflicts = graph_store.add_relation(
        GraphRelation(
            rel_type="CONTAINS",
            from_name="A",
            to_name="B",
            from_type="Module",
            to_type="Module",
            properties={"order": 2},
        )
    )
    assert len(conflicts) == 1
    assert conflicts[0]["property_key"] == "order"


# ---------------------------------------------------------------------------
# 11. 路径搜索增强
# ---------------------------------------------------------------------------


def test_find_path_with_rel_types_filter(graph_store):
    """find_path 支持按关系类型过滤."""
    for name in ("A", "B", "C", "D"):
        graph_store.add_entity(GraphEntity(entity_type="Module", name=name))
    graph_store.add_relation(
        GraphRelation(
            rel_type="CONTAINS", from_name="A", to_name="B", from_type="Module", to_type="Module"
        )
    )
    graph_store.add_relation(
        GraphRelation(
            rel_type="HAS_SIGNAL", from_name="B", to_name="C", from_type="Module", to_type="Module"
        )
    )
    graph_store.add_relation(
        GraphRelation(
            rel_type="CONTAINS", from_name="C", to_name="D", from_type="Module", to_type="Module"
        )
    )

    # 不过滤：A->B->C->D
    paths = graph_store.find_path("Module", "A", "Module", "D", max_depth=4)
    assert len(paths) == 1

    # 只走 CONTAINS：A 无法到 D（B->C 是 HAS_SIGNAL）
    paths_filtered = graph_store.find_path(
        "Module", "A", "Module", "D", max_depth=4, rel_types={"CONTAINS"}
    )
    assert len(paths_filtered) == 0


# ---------------------------------------------------------------------------
# 12. 属性索引优化
# ---------------------------------------------------------------------------


def test_find_by_property_uses_index(graph_store):
    """find_by_property 通过索引正确返回结果."""
    graph_store.add_entity(GraphEntity(entity_type="Register", name="R1", properties={"width": 32}))
    graph_store.add_entity(GraphEntity(entity_type="Register", name="R2", properties={"width": 64}))
    graph_store.add_entity(GraphEntity(entity_type="Signal", name="S1", properties={"width": 32}))

    results = graph_store.find_by_property("Register", "width", 32)
    assert len(results) == 1
    assert results[0].name == "R1"

    results = graph_store.find_by_property("Register", "width", 64)
    assert len(results) == 1
    assert results[0].name == "R2"

    results = graph_store.find_by_property("Signal", "width", 32)
    assert len(results) == 1
    assert results[0].name == "S1"


def test_property_index_updated_on_delete(graph_store):
    """删除实体后属性索引应同步更新."""
    graph_store.add_entity(GraphEntity(entity_type="Register", name="R1", properties={"width": 32}))
    assert len(graph_store.find_by_property("Register", "width", 32)) == 1
    graph_store.delete_entity("Register", "R1")
    assert len(graph_store.find_by_property("Register", "width", 32)) == 0


def test_property_index_updated_on_update(graph_store):
    """update_entity 后属性索引应同步更新."""
    graph_store.add_entity(GraphEntity(entity_type="Register", name="R1", properties={"width": 32}))
    graph_store.update_entity("Register", "R1", properties={"width": 64})
    assert len(graph_store.find_by_property("Register", "width", 32)) == 0
    assert len(graph_store.find_by_property("Register", "width", 64)) == 1


# ---------------------------------------------------------------------------
# 13. 子图文本化增强
# ---------------------------------------------------------------------------


def test_subgraph_text_includes_sources_and_verified(graph_store):
    """to_text_context 包含来源和验证标记."""
    graph_store.add_entity(
        GraphEntity(entity_type="Module", name="M1", source_doc_ids={"doc1"}, verified=True)
    )
    graph_store.add_entity(
        GraphEntity(entity_type="Signal", name="S1", source_doc_ids={"doc1", "doc2"})
    )
    graph_store.add_relation(
        GraphRelation(
            rel_type="HAS_SIGNAL",
            from_name="M1",
            to_name="S1",
            from_type="Module",
            to_type="Signal",
            properties={"width": 32},
        )
    )

    sg = graph_store.get_subgraph("Module", "M1", depth=1)
    text = sg.to_text_context()
    assert "[来源: doc1]" in text
    assert "✓" in text  # verified mark
    assert "width=32" in text


# ---------------------------------------------------------------------------
# 14. doc_properties（文档级属性快照）
# ---------------------------------------------------------------------------


def test_entity_doc_properties_on_create(graph_store):
    """实体首次创建时 doc_properties 保存原始属性."""
    e = GraphEntity(
        entity_type="Register",
        name="CTRL",
        properties={"addr": "0x1000"},
        source_doc_ids={"doc_a"},
    )
    graph_store.add_entity(e)
    result = graph_store.get_entity("Register", "CTRL")
    assert result is not None
    assert result.doc_properties == {"doc_a": {"addr": "0x1000"}}


def test_entity_doc_properties_on_merge(graph_store):
    """实体合并时 doc_properties 追加不丢失."""
    graph_store.add_entity(
        GraphEntity(
            entity_type="Register",
            name="CTRL",
            properties={"addr": "0x1000"},
            source_doc_ids={"doc_a"},
        )
    )
    graph_store.add_entity(
        GraphEntity(
            entity_type="Register",
            name="CTRL",
            properties={"addr": "0x2000"},
            source_doc_ids={"doc_b"},
        )
    )
    result = graph_store.get_entity("Register", "CTRL")
    assert result is not None
    assert result.doc_properties == {
        "doc_a": {"addr": "0x1000"},
        "doc_b": {"addr": "0x2000"},
    }
    # properties 仍为合并后值（新值覆盖）
    assert result.properties == {"addr": "0x2000"}


def test_relation_doc_properties_on_create(graph_store):
    """关系首次创建时 doc_properties 保存原始属性."""
    r = GraphRelation(
        rel_type="HAS_REGISTER",
        from_name="TOP",
        to_name="CTRL",
        from_type="Module",
        to_type="Register",
        properties={"access": "RW"},
        source_doc_ids={"doc_a"},
    )
    graph_store.add_relation(r)
    rels = graph_store.all_relations()
    assert len(rels) == 1
    assert rels[0].doc_properties == {"doc_a": {"access": "RW"}}


def test_relation_doc_properties_on_merge(graph_store):
    """关系合并时 doc_properties 追加不丢失."""
    graph_store.add_relation(
        GraphRelation(
            rel_type="HAS_REGISTER",
            from_name="TOP",
            to_name="CTRL",
            from_type="Module",
            to_type="Register",
            properties={"access": "RW"},
            source_doc_ids={"doc_a"},
        )
    )
    graph_store.add_relation(
        GraphRelation(
            rel_type="HAS_REGISTER",
            from_name="TOP",
            to_name="CTRL",
            from_type="Module",
            to_type="Register",
            properties={"access": "RO"},
            source_doc_ids={"doc_b"},
        )
    )
    rels = graph_store.all_relations()
    assert len(rels) == 1
    assert rels[0].doc_properties == {
        "doc_a": {"access": "RW"},
        "doc_b": {"access": "RO"},
    }


def test_doc_properties_persistence(graph_store, tmp_path):
    """doc_properties 正确序列化和反序列化."""
    graph_store.add_entity(
        GraphEntity(
            entity_type="Register",
            name="CTRL",
            properties={"addr": "0x1000"},
            source_doc_ids={"doc_a"},
        )
    )
    graph_store.add_entity(
        GraphEntity(
            entity_type="Register",
            name="CTRL",
            properties={"addr": "0x2000"},
            source_doc_ids={"doc_b"},
        )
    )
    path = tmp_path / "graph.json"
    graph_store.save(path)

    g2 = NetworkXGraphStore()
    g2.load(path)
    result = g2.get_entity("Register", "CTRL")
    assert result is not None
    assert result.doc_properties == {
        "doc_a": {"addr": "0x1000"},
        "doc_b": {"addr": "0x2000"},
    }


def test_conflict_log_includes_doc_id(graph_store):
    """冲突日志包含引发冲突的 doc_id."""
    graph_store.add_entity(
        GraphEntity(
            entity_type="Register",
            name="CTRL",
            properties={"addr": "0x1000"},
            source_doc_ids={"doc_a"},
        )
    )
    conflicts = graph_store.add_entity(
        GraphEntity(
            entity_type="Register",
            name="CTRL",
            properties={"addr": "0x2000"},
            source_doc_ids={"doc_b"},
        )
    )
    assert len(conflicts) == 1
    assert conflicts[0]["doc_id"] == "doc_b"


# ---------------------------------------------------------------------------
# 15. Product 实体类型
# ---------------------------------------------------------------------------


def test_product_in_all_node_types():
    """Product 实体类型存在于 ALL_NODE_TYPES."""
    from core.graph_store import ALL_NODE_TYPES

    assert "Product" in ALL_NODE_TYPES
