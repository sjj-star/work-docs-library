"""EntityExtractor 单元测试."""

from core.doc_graph_pipeline import EntityExtractor


def test_build_batch_requests_replaces_doc_context():
    """_build_batch_requests 应正确替换 {{doc_context}} 占位符."""
    extractor = EntityExtractor(batch_client=None)
    batches = [[{"title": "GPIO Registers", "content": "GPIO content"}]]
    doc_context = "以下章节来自 文档《TMS320F28379D TRM》，产品型号 TMS320F28379D。\n\n---\n"

    requests = extractor._build_batch_requests(
        batches, image_base_dir=None, doc_context=doc_context
    )

    assert len(requests) == 1
    user_content = requests[0]["body"]["messages"][1]["content"]
    text_parts = [c["text"] for c in user_content if c["type"] == "text"]
    full_text = "".join(text_parts)

    assert "以下章节来自 文档《TMS320F28379D TRM》" in full_text
    assert "产品型号 TMS320F28379D" in full_text
    assert "GPIO Registers" in full_text
    assert "GPIO content" in full_text


def test_build_batch_requests_empty_doc_context():
    """doc_context 为空字符串时，占位符应被替换为空，不影响章节内容."""
    extractor = EntityExtractor(batch_client=None)
    batches = [[{"title": "Clock Domain", "content": "Clock content"}]]

    requests = extractor._build_batch_requests(batches, image_base_dir=None, doc_context="")

    assert len(requests) == 1
    user_content = requests[0]["body"]["messages"][1]["content"]
    text_parts = [c["text"] for c in user_content if c["type"] == "text"]
    full_text = "".join(text_parts)

    assert "{{doc_context}}" not in full_text
    assert "Clock Domain" in full_text
    assert "Clock content" in full_text


def test_build_batch_requests_multiple_batches():
    """多个 batch 应共享同一份 doc_context."""
    extractor = EntityExtractor(batch_client=None)
    batches = [
        [{"title": "Ch1", "content": "content1"}],
        [{"title": "Ch2", "content": "content2"}],
    ]
    doc_context = "以下章节来自 文档《Test》。\n\n---\n"

    requests = extractor._build_batch_requests(
        batches, image_base_dir=None, doc_context=doc_context
    )

    assert len(requests) == 2
    for req in requests:
        user_content = req["body"]["messages"][1]["content"]
        text_parts = [c["text"] for c in user_content if c["type"] == "text"]
        full_text = "".join(text_parts)
        assert "文档《Test》" in full_text


def test_prompt_contains_isa_entity_types():
    """系统提示词应包含 ISA 方向核心实体类型."""
    extractor = EntityExtractor(batch_client=None)
    batches = [[{"title": "Instruction Set", "content": "MAC instruction description."}]]

    requests = extractor._build_batch_requests(batches, image_base_dir=None, doc_context="")
    assert len(requests) == 1
    system_msg = requests[0]["body"]["messages"][0]["content"]

    # ISA 方向核心实体类型
    assert "Instruction" in system_msg
    assert "InstructionGroup" in system_msg
    assert "AddressingMode" in system_msg
    assert "Operand" in system_msg
    assert "ArchitectureState" in system_msg
    assert "PipelineStage" in system_msg
    assert "FunctionalUnit" in system_msg
    assert "Interrupt" in system_msg
    assert "Exception" in system_msg
    assert "CPU_Mode" in system_msg
    assert "CLA_Task" in system_msg


def test_prompt_contains_peripheral_entity_types():
    """系统提示词应包含外设寄存器方向核心实体类型."""
    extractor = EntityExtractor(batch_client=None)
    batches = [[{"title": "ePWM Registers", "content": "TBCTL register."}]]

    requests = extractor._build_batch_requests(batches, image_base_dir=None, doc_context="")
    assert len(requests) == 1
    system_msg = requests[0]["body"]["messages"][0]["content"]

    # 外设寄存器方向核心实体类型
    assert "Peripheral" in system_msg
    assert "Module" in system_msg
    assert "Register" in system_msg
    assert "RegisterField" in system_msg
    assert "ShadowRegister" in system_msg
    assert "MemoryRegion" in system_msg
    assert "Signal" in system_msg


def test_prompt_contains_isa_relations():
    """系统提示词应包含 ISA 方向核心关系类型."""
    extractor = EntityExtractor(batch_client=None)
    batches = [[{"title": "Test", "content": "test"}]]

    requests = extractor._build_batch_requests(batches, image_base_dir=None, doc_context="")
    system_msg = requests[0]["body"]["messages"][0]["content"]

    # ISA 方向核心关系
    assert "ISA_HAS_INSTRUCTION" in system_msg
    assert "INSTRUCTION_READS_REGISTER" in system_msg
    assert "INSTRUCTION_WRITES_REGISTER" in system_msg
    assert "INSTRUCTION_MODIFIES_STATE" in system_msg
    assert "INSTRUCTION_BELONGS_TO_GROUP" in system_msg
    assert "INSTRUCTION_USES_MODE" in system_msg
    assert "OPERAND_HAS_MODE" in system_msg
    assert "INTERRUPT_TRIGGERS" in system_msg
    assert "MODULE_IMPLEMENTS_INSTRUCTION" in system_msg
    assert "CLA_HAS_TASK" in system_msg
    assert "TASK_USES_INSTRUCTION" in system_msg
    assert "STATE_HAS_REGISTER" in system_msg
    assert "MEMORY_MAPS_TO" in system_msg
    assert "UNIT_EXECUTES" in system_msg
    assert "STAGE_PRODUCES" in system_msg


def test_prompt_contains_peripheral_relations():
    """系统提示词应包含外设寄存器方向核心关系类型."""
    extractor = EntityExtractor(batch_client=None)
    batches = [[{"title": "Test", "content": "test"}]]

    requests = extractor._build_batch_requests(batches, image_base_dir=None, doc_context="")
    system_msg = requests[0]["body"]["messages"][0]["content"]

    # 外设寄存器方向核心关系
    assert "HAS_REGISTER" in system_msg
    assert "HAS_FIELD" in system_msg
    assert "PERIPHERAL_HAS_REGISTER" in system_msg
    assert "HAS_PERIPHERAL" in system_msg
    assert "CONTAINS" in system_msg
    assert "HAS_SIGNAL" in system_msg
    assert "CLOCK_GATED_BY" in system_msg
    assert "RESET_BY" in system_msg


def test_prompt_contains_isa_properties():
    """系统提示词应包含 ISA 方向专属属性说明."""
    extractor = EntityExtractor(batch_client=None)
    batches = [[{"title": "Test", "content": "test"}]]

    requests = extractor._build_batch_requests(batches, image_base_dir=None, doc_context="")
    system_msg = requests[0]["body"]["messages"][0]["content"]

    # ISA 专属属性
    assert "opcode" in system_msg
    assert "format" in system_msg
    assert "cycle_count" in system_msg
    assert "affected_flags" in system_msg
    assert "addressing_modes" in system_msg
    assert "delay_slots" in system_msg
    assert "atomic" in system_msg
    assert "repeatable" in system_msg
    assert "vector_address" in system_msg
    assert "trigger_source" in system_msg
    assert "pipeline_stall_condition" in system_msg
    assert "只提取当前 chunk" in system_msg


def test_prompt_contains_peripheral_properties():
    """系统提示词应包含外设寄存器方向专属属性说明."""
    extractor = EntityExtractor(batch_client=None)
    batches = [[{"title": "Test", "content": "test"}]]

    requests = extractor._build_batch_requests(batches, image_base_dir=None, doc_context="")
    system_msg = requests[0]["body"]["messages"][0]["content"]

    # 外设寄存器专属属性
    assert "write_semantics" in system_msg
    assert "read_semantics" in system_msg
    assert "field_semantics" in system_msg
    assert "shadow_of" in system_msg
    assert "update_condition" in system_msg
    assert "config_sequence" in system_msg
    assert "dependency" in system_msg
    assert "trigger_condition" in system_msg


def test_prompt_contains_code_generation_section():
    """系统提示词应包含代码生成导向提取章节."""
    extractor = EntityExtractor(batch_client=None)
    batches = [[{"title": "Test", "content": "test"}]]

    requests = extractor._build_batch_requests(batches, image_base_dir=None, doc_context="")
    system_msg = requests[0]["body"]["messages"][0]["content"]

    assert "代码生成导向提取" in system_msg
    assert "汇编代码生成支撑" in system_msg
    assert "裸机C代码生成支撑" in system_msg
    assert "初始化/配置序列" in system_msg
    assert "使能依赖关系" in system_msg
    assert "流水线冲突" in system_msg


def test_prompt_step1_classification_restructured():
    """系统提示词的 Step 1 应包含重构后的 8 个分类."""
    extractor = EntityExtractor(batch_client=None)
    batches = [[{"title": "Test", "content": "test"}]]

    requests = extractor._build_batch_requests(batches, image_base_dir=None, doc_context="")
    system_msg = requests[0]["body"]["messages"][0]["content"]

    # 8 个 Step 1 分类
    assert "寄存器表" in system_msg
    assert "外设功能描述" in system_msg
    assert "ISA 架构描述" in system_msg
    assert "指令详细参考" in system_msg
    assert "指令集概览" in system_msg
    assert "CLA 描述" in system_msg
    assert "概述/介绍" in system_msg
    assert "其他" in system_msg


def test_prompt_pruned_irrelevant_types():
    """系统提示词应已精简与目标方向关联较弱的实体类型."""
    extractor = EntityExtractor(batch_client=None)
    batches = [[{"title": "Test", "content": "test"}]]

    requests = extractor._build_batch_requests(batches, image_base_dir=None, doc_context="")
    system_msg = requests[0]["body"]["messages"][0]["content"]

    # 被精简的类型不应出现在核心实体类型定义区域
    # 它们可能只在最终检查清单中作为反例出现
    # 截取"实体类型定义"到"关系类型定义"之间的区域作为核心类型区域
    core_section = system_msg.split("## 关系类型定义")[0]

    # 这些类型应在核心实体类型区域中被移除
    assert "ElectricalSpec" not in core_section
    assert "ProtocolLayer" not in core_section
    assert "Workaround" not in core_section
    assert "BuildConfig" not in core_section
    # Advisory 和 Pin 可能在其他位置出现，但不应在核心实体类型表格中
    # 检查 Advisory 不在核心类型区域的表格中（通过检查它不在核心区域的显著位置）
    assert "Advisory" not in core_section.split("## 实体类型定义")[1].split("### 扩展")[0]
