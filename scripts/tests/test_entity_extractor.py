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


def test_prompt_contains_arch_entity_types():
    """系统提示词应包含新增的处理器架构实体类型."""
    extractor = EntityExtractor(batch_client=None)
    batches = [[{"title": "Instruction Set", "content": "MAC instruction description."}]]

    requests = extractor._build_batch_requests(batches, image_base_dir=None, doc_context="")
    assert len(requests) == 1
    system_msg = requests[0]["body"]["messages"][0]["content"]

    # 验证新增实体类型名称存在于系统提示词
    assert "Instruction" in system_msg
    assert "AddressingMode" in system_msg
    assert "ArchitectureState" in system_msg
    assert "PipelineStage" in system_msg
    assert "Interrupt" in system_msg
    assert "CLA_Task" in system_msg
    assert "Peripheral" in system_msg

    # 验证新增关系类型名称存在于系统提示词
    assert "INSTRUCTION_READS_REGISTER" in system_msg
    assert "INSTRUCTION_WRITES_REGISTER" in system_msg
    assert "MODULE_IMPLEMENTS_INSTRUCTION" in system_msg
    assert "INTERRUPT_TRIGGERS" in system_msg
    assert "HAS_PERIPHERAL" in system_msg


def test_prompt_contains_arch_properties():
    """系统提示词应包含 C28x+CLA 专用属性说明."""
    extractor = EntityExtractor(batch_client=None)
    batches = [[{"title": "Test", "content": "test"}]]

    requests = extractor._build_batch_requests(batches, image_base_dir=None, doc_context="")
    system_msg = requests[0]["body"]["messages"][0]["content"]

    assert "opcode" in system_msg
    assert "cycle_count" in system_msg
    assert "affected_flags" in system_msg
    assert "vector_address" in system_msg
    assert "trigger_source" in system_msg
    assert "跨层级提取" in system_msg
