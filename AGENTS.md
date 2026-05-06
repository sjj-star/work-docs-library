# work-docs-library Agent 自主式开发策略文档

> 本文档供 AI Coding Agent 阅读，是项目权威技术参考与行动准则。项目详细介绍见 `README.md`，技术方案见 `DESIGN.md`。

---

## 项目目标（一句话定义）

从 IC 前端设计技术文档（PDF）中自动提取结构化实体与关系，构建**跨文档互通**的知识图谱，同时保留向量检索能力作为补充。

**目标规模**：数百个文档，每个文档几十到万页级。全局图谱通过 NetworkX 内存存储管理，同名同类型实体自动对齐。

---

## 核心架构决策及原因

| 决策 | 原因 |
|------|------|
| **Batch API 优先** | 成本为同步 API 的 50%；超大 JSONL 自动拆分并行提交；分钟级延迟可接受 |
| **KnowledgeBaseService 统一服务层** | 封装 DB + VectorIndex + GraphStore，为 Plugin 工具和上层应用提供一致 API；禁止工具直接访问 `db._connect()` |
| **NetworkX 而非 Neo4j** | 轻量、JSON 序列化、当前查询深度不超过 3 跳，无需引入外部服务 |
| **零数据丢失** | 技术文档信息密度高，任何截断/过滤都可能导致关键寄存器/信号丢失 |
| **树形章节解析** | 按 Markdown 标题层级拆分是唯一允许的结构性拆分；禁止基于内容相关性的过滤 |
| **Multimodal 图片流式处理** | 图片（时序图、架构框图）是文档语义不可分割的一部分，必须按原文顺序嵌入文档流 |
| **SQLite + FAISS** | 单用户、本地部署、零运维；FAISS IndexFlatIP 经 L2 归一化后等效于余弦相似度 |
| **Prompt 外部化** | 所有 LLM 提示词在 `scripts/prompts/*.txt` 中，用户可编辑，无需改代码 |
| **config.json + .env 双轨配置** | `config.json` 存非敏感参数（模型、端点），`.env` 存 API Key；三层优先级（环境变量 > config.json > .env > 默认值） |

---

## 代码规范

### 强制工具链
- **ruff**：代码格式化与 lint（`ruff check scripts/` / `ruff format scripts/`）
- **pyright**：类型检查（`pyright scripts/`）
- **pytest**：测试执行（`PYTHONPATH=scripts pytest scripts/tests/ -v`）

### 日志规范
- 统一使用 `logging`，格式：`"%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"`
- 禁止随意使用 `print()` 输出调试信息

### 类型与模型
- 领域模型使用 `dataclass`（`Chunk`、`Document`、`GraphEntity`、`GraphRelation`、`SubGraphView`）
- `Chunk.status`：`pending` → `embedded` → `done`（含 `skipped`、`failed`，`ChunkStatus` StrEnum）
- `Chunk.chunk_type`：`text` / `table` / `image_desc` / `summary`（`ChunkType` StrEnum）
- `Document.status`：`pending` → `processing` → `done` / `failed`（`DocumentStatus` StrEnum）
- `GraphEntity`/`GraphRelation` 字段：`entity_type`/`rel_type`、`name`、`properties`、`doc_properties`（按文档原始属性快照）、`source_doc_ids`、`source_chapter`、`confidence`、`verified`、`created_at`、`updated_at`、`feedback_score`

### 数据库安全
- **所有 SQL 必须使用参数化查询**（`?` 占位符）
- `KnowledgeDB._connect()` 使用上下文管理器

### Prompt 管理
- 所有 LLM 提示词必须在 `scripts/prompts/*.txt` 中
- 代码通过 `Config.PROMPT_DIR / f"{name}.txt"` 读取
- 修改提示词后无需重启，下次调用自动生效

### API 规范
- **通用 API**（LLM Batch、Embedding、同步对话）：统一使用兼容 OpenAI 协议的 HTTP API，通过配置参数（`base_url`、`batch_endpoint`、`download_url_template` 等）适配不同服务商，实现厂商无感接入
- **专用 API**（PDF 解析）：`BigModelParserClient` 使用 BigModel 专有的 Expert 文件解析接口（`/files/parser/create`、`/files/parser/result`），非 OpenAI-compatible，无法直接切换至其他厂商。失败时自动 fallback 到本地 `PDFParser`
- 构造项目内部API时要尽可能使用函数式编程

### 机制与策略分离原则

> 不要擅自设计任何硬编码的策略，应当优先设计一个框架和机制，后面根据实际需求使用机制来实现策略。

- **零数据丢失是底线**：任何原始解析的文本都不得删除或丢弃

---

## 开发计划

### 当前阶段（已完成）
- ✅ DocGraphPipeline 重构：Batch API 架构、树形章节解析、multimodal 图片处理
- ✅ 零数据丢失：移除所有源数据过滤和截断
- ✅ Prompt 外部化：所有提示词在 `scripts/prompts/*.txt`
- ✅ 代码清理：删除 11 个旧文件、4 个旧 prompt 文件、7 个旧测试文件
- ✅ **API 接口重构**：新增 `KnowledgeBaseService` 统一服务层；Plugin 暴露 22 个工具。代码中额外实现了 `get_feedback()` 方法（用于汇总实体反馈评分），但尚未在 `plugin.json` 中注册为 CLI 工具
- ✅ **图谱查询增强**：`GraphStore.find_path()` BFS 路径搜索、`search_entities()` 模糊搜索
- ✅ **跨文档知识互通**：全局统一图谱 `global.json` + 文档子图快照 `{doc_id}.json`，同名同类型实体自动去重
- ✅ **章节级增量更新**：`content_hash` 指纹比较，未变章节复用实体缓存与 embedding，仅 LLM 提取变更/新增章节
- ✅ **本地 PDF 解析器 TOC 驱动式章节识别**：基于 PDF 内置 TOC 直接匹配定位 heading，解决编号/标题分离和表格噪声问题
- ✅ **本地 PDF 解析器 Markdown 格式修复**：解决标题重复和段落内换行被误分段问题，输出格式与 BigModel Expert 保持一致
- ✅ **完整章节层级构造**：使用 PDF TOC 真实层级生成 `#`/`##`/`###`/`####`，不再扁平化为两级
- ✅ **段落内换行保留原始 `\n`**：LLM 直接理解换行语义，零额外 token，无 HTML 标签污染
- ✅ **ChapterParser 多级树重构**：栈结构构建真正的多级树；去掉 preface 传播，每个节点保留自己的原始 content
- ✅ **BatchBuilder 接收扁平化节点**：`ChapterParser.collect_all_nodes()` 收集所有有 content 的节点，为每个 chunk 附加完整标题路径前缀；段落边界切分（`\n\n+`）替代句子边界，避免编号标题被误切开
- ✅ **数据模型清理**：移除 `Document.chunks`、`Document.metadata`（早期版本）、`Chunk.page_start/page_end` 等无功能实体字段；引入 `StrEnum` 约束状态值。注意：`Chunk.metadata` 在后续版本中重新引入，现承载 `content_hash`/`extracted_entities`/`extracted_relations`/`image_descriptions`/`embedding` 等核心缓存数据
- ✅ **数据库 schema 简化**：移除 `chapters_override`、`page_start`、`page_end` 列；新增 `_schema_meta` 版本管理表；新增 `query_by_doc()` 辅助方法
- ✅ **数据质量增强**：`GraphEntity`/`GraphRelation` 新增 `confidence`/`verified`/`created_at`/`updated_at`/`feedback_score` 字段
- ✅ **图谱动态更新接口**：Plugin 暴露 `graph_add_entity`/`graph_update_entity`/`graph_delete_entity`/`graph_add_relation`/`graph_delete_relation`/`graph_verify_entity` 等 6 个写工具。注意：`graph_update_relation` 尚未实现（如需要，可通过 `graph_delete_relation` + `graph_add_relation` 组合实现）
- ✅ **冲突检测与日志**：同名实体属性差异自动记录 `conflict_logs` 表，供人工审核
- ✅ **语义-图谱联合查询**：`search_with_graph()` 先 FAISS 语义搜索再扩展关联子图
- ✅ **chunk+实体联合返回**：`get_content_with_entities()` 返回 chunk 及其关联的图谱实体/关系
- ✅ **用户反馈机制**：`graph_feedback` 工具支持对实体/关系打分（+1/-1），`feedback_score` 实时汇总
- ✅ **IC 关系扩展**：新增 `DRIVES`/`DRIVEN_BY`/`TIMING_PATH`/`CLOCK_GATED_BY`/`RESET_BY`/`PARAMETERIZED_BY`/`INSTANCE_OF`
- ✅ **处理器架构图谱扩展（C28x+CLA）**：新增 `Instruction`/`InstructionGroup`/`AddressingMode`/`Operand`/`ArchitectureState`/`PipelineStage`/`FunctionalUnit`/`Interrupt`/`Exception`/`MemoryRegion`/`ShadowRegister`/`CPU_Mode`/`CLA_Task`/`Peripheral` 等实体类型，以及 `ISA_HAS_INSTRUCTION`/`INSTRUCTION_READS_REGISTER`/`INSTRUCTION_WRITES_REGISTER`/`INSTRUCTION_MODIFIES_STATE`/`MODULE_IMPLEMENTS_INSTRUCTION`/`INTERRUPT_TRIGGERS`/`HAS_PERIPHERAL`/`CLA_HAS_TASK` 等跨层级关系类型，支持 ISA 层级与 RTL 层级的知识互通
- ✅ **跨产品外设变体建模**：`GraphEntity`/`GraphRelation` 新增 `doc_properties` 字段，保存每个文档的原始属性快照；引入 `Product` 实体类型，文档解析时自动提取产品型号并建立 `Product --[HAS_MODULE]--> Module` 关系；查询接口支持 `doc_id` 参数以获取指定文档的原始属性
- ✅ **属性索引优化**：`NetworkXGraphStore` 内部维护 `property_index`，`find_by_property()` 从 O(N) 降至 O(1)
- ✅ **Pipeline 四阶段拆分**：`_process_one` 拆分为 `stage1_parse` / `stage2_build_jsonl` / `stage3_submit_batches` / `stage4_ingest_results`，每个中间产物（result.md / requests.jsonl / batch_info.json / results.jsonl / 子图谱 JSON）均可独立执行、人为干预、重新执行
- ✅ 283 个测试全部通过

### 下一阶段（精确到下一步）
1. **可视化**：图谱可视化导出（Graphviz / D3.js）
2. **评估体系**：实体提取准确率、关系提取召回率的自动化评估
3. **DOCX/XLSX 接入 pipeline**：当前解析器代码存在但未接入 `DocGraphPipeline`

---

## Agent 自主开发原则

### 可以自主决定的事项
- 代码重构（变量重命名、函数拆分、模块移动）
- 新增单元测试或调整测试结构
- 更新日志格式、错误提示信息
- 优化算法实现（不改变输入输出语义）
- 更新本文档（AGENTS.md）和 README.md、DESIGN.md

### 必须询问用户的事项
- 修改核心数据模型（`Chunk`、`Document`、`GraphEntity`、`GraphRelation` 的字段）
- 修改数据库 Schema（新增/删除/修改表或字段）
- 修改配置系统的优先级或解析逻辑
- 新增外部依赖（`requirements.txt` 变更）
- 修改 Pipeline 核心流程（解析 → 章节树 → batch 构建 → 实体提取 → 图谱 → 向量化）
- 删除已有功能或文件
- 涉及真实 API 调用的操作（测试除外）
- 修改 `plugin.json` 插件定义（除非用户明确要求）

---

## 文件修改权限规则

| 文件/目录 | 权限 | 说明 |
|-----------|------|------|
| `scripts/core/*.py` | ⚠️ 需批准 | 核心模块，修改前必须说明影响范围 |
| `scripts/prompts/*.txt` | ✅ 可改 | 提示词文件，运行时读取 |
| `scripts/tests/*.py` | ✅ 可改 | 测试文件 |
| `scripts/parsers/*.py` | ⚠️ 需批准 | 解析器影响数据输入质量 |
| `plugin.json` | ⚠️ 需批准 | 插件定义，用户明确要求时可修改 |
| `config.json` | ✅ 可改 | 用户持久化配置模板 |
| `README.md` / `AGENTS.md` / `DESIGN.md` | ✅ 可改 | 文档必须随代码同步更新 |
| `knowledge_base/` | ❌ 禁止 | 运行时生成数据 |

---

## 测试策略

### 核心原则
- **Mock 优先**：所有涉及外部 API 的测试使用 Fake 客户端，**禁止调用真实 API**
- **回归即修复**：任何导致测试失败的变更必须当场修复
- **283 个测试用例必须全部通过**

### 测试文件清单
| 测试文件 | 说明 |
|----------|------|
| `test_plugin_router.py` | Plugin 工具路由、参数解析 |
| `test_pdf_parser.py` | PDF 解析核心测试 |
| `test_office_parser.py` | DOCX / XLSX 解析测试 |
| `test_db.py` | SQLite 操作、事务管理 |
| `test_vector_index.py` | FAISS 索引增删查、持久化 |
| `test_llm_client.py` | LLM 客户端 Mock |
| `test_config_json.py` | 配置优先级、凭证注入 |
| `test_models.py` | 数据模型测试（含 StrEnum） |
| `test_chapter_parser.py` | ChapterParser 树形章节解析测试（含 Markdown heading / 代码块保护 / TOC 行识别） |
| `test_image_utils.py` | 图片压缩工具测试 |
| `test_graph_store.py` | NetworkX 图谱存储 CRUD、冲突检测、属性索引、子图、路径搜索、持久化、`doc_properties` 测试 |
| `test_batch_clients.py` | Batch API 客户端（Kimi + BigModel）Mock 测试 |
| `test_knowledge_base_service.py` | KnowledgeBaseService 统一服务层测试 |
| `test_entity_extractor.py` | EntityExtractor multimodal batch 请求构建测试 |
| `test_batch_builder.py` | BatchBuilder 切分保护与空 content 过滤测试 |
| `test_parsed_docs_jsonl.py` | 真实文档端到端 JSONL 生成测试 |
| `test_pipeline_stages.py` | 四阶段 pipeline 拆分测试 |

### Mock 方法
使用 `monkeypatch.setattr` 替换客户端类方法：
```python
# 示例：mock BatchClient
monkeypatch.setattr(
    "core.batch_clients.BatchClient.submit_parallel_batches",
    lambda self, reqs: [{"entities": [], "relationships": [], "image_descriptions": {}}]
)
```

---

## 配置系统说明

### 三层优先级（高 → 低）
```
环境变量（Kimi CLI 运行时注入，如 llm.api_key）
  ↓
config.json（用户持久化配置，项目根目录）
  ↓
.env 文件（环境变量回退，如 WORKDOCS_LLM_API_KEY）
  ↓
代码硬编码默认值
```

### 关键配置项速查

完整配置列表见 `scripts/core/config.py`。以下为常用配置项：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `WORKDOCS_LLM_API_KEY` | 空 | LLM Batch API Key（实体提取） |
| `WORKDOCS_LLM_BASE_URL` | `https://api.moonshot.cn/v1` | LLM Base URL |
| `WORKDOCS_LLM_MODEL` | `kimi-k2.5` | 对话模型 |
| `WORKDOCS_EMBEDDING_API_KEY` | 空 | Embedding API Key（向量化） |
| `WORKDOCS_EMBEDDING_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4` | Embedding Base URL |
| `WORKDOCS_EMBEDDING_MODEL` | `embedding-3` | 向量化模型 |
| `WORKDOCS_EMBEDDING_DIMENSION` | `1024` | 向量维度 |
| `WORKDOCS_PARSER_API_KEY` | 空 | PDF 解析 API Key（BigModel 专用） |
| `WORKDOCS_LLM_BATCH_MAX_CHARS` | `10000` | 每个 batch 最大文本字符数 |
| `WORKDOCS_LLM_BATCH_TIMEOUT` | `3600` | Batch API 轮询超时（秒） |
| `WORKDOCS_LLM_VISION_MAX_EDGE` | `1024` | 图片压缩最长边（px） |
| `WORKDOCS_LLM_VISION_QUALITY` | `85` | JPEG 压缩质量 1-100 |
| `WORKDOCS_BATCH_MAX_FILE_SIZE_MB` | `100` | 单个 JSONL 文件大小上限 |
| `WORKDOCS_BATCH_PARALLEL_WORKERS` | `4` | 并行 batch 提交线程数 |
| `WORKDOCS_GRAPH_MAX_PATH_DEPTH` | `6` | 图谱路径搜索最大深度 |

> 注：`config.json` 的 `config_file` 由 `plugin.json` 指定，Kimi CLI 可自动管理。`.env` 文件支持双路径加载：项目根目录和 `scripts/` 目录下的 `.env` 均会被读取（后者优先级更高）。

---

## 官方 API 开发文档

+ [Kimi Code CLI 插件](https://moonshotai.github.io/kimi-cli/zh/customization/plugins.html)
+ [Kimi API 概述](https://platform.kimi.com/docs/api/overview)
+ [Kimi 模型参数参考](https://platform.kimi.com/docs/api/models-overview)
+ [Kimi 使用 Batch API 批量处理任务](https://platform.kimi.com/docs/guide/use-batch-api)
+ [BigModel API 使用概述](https://docs.bigmodel.cn/cn/api/introduction)
+ [BigModel 结构化输出](https://docs.bigmodel.cn/cn/guide/capabilities/struct-output)
+ [BigModel Embedding-3](https://docs.bigmodel.cn/cn/guide/models/embedding/embedding-3)
+ [BigModel 批量处理](https://docs.bigmodel.cn/cn/guide/tools/batch)
+ [BigModel 新文件解析服务](https://docs.bigmodel.cn/cn/guide/tools/file-parser)
