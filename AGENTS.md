# work-docs-library Agent 自主式开发策略文档

> 本文档供 AI Coding Agent 阅读，是项目权威技术参考与行动准则。
> - 项目详细介绍、安装与使用指南 → `README.md`
> - 技术方案决策、设计意图与权衡 → `DESIGN.md`
> - 本规范与上述两份文档**必须同步更新**，具体规则见下方「文档同步规范」。

---

## 项目目标（一句话定义）

从 IC 前端设计技术文档（PDF）中自动提取结构化实体与关系，构建**跨文档互通**的知识图谱，同时具有文档内容块的向量检索能力增强知识库。

**目标规模**：数百个文档，每个文档几十到万页级。全局图谱通过 NetworkX 内存存储管理，知识图谱检索与文档内容向量检索互通关联。

---

## 技术参考索引

以下主题在 `DESIGN.md` 和 `README.md` 中有详细论述，本处仅提供索引，**禁止在此重复展开**：

| 主题 | 权威文档 | 说明 |
|------|---------|------|
| 核心架构决策（Batch API、NetworkX、零数据丢失等 11 项） | `DESIGN.md` 第 1–10 章 | 每项决策的选择原因、实现细节与权衡 |
| Pipeline 六阶段拆分与中间产物持久化 | `DESIGN.md` 第 19 章 | 产物清单、状态管理、开发约束 |
| 防御性编程与状态安全原则（7 条） | `DESIGN.md` 第 20 章 | 2026-04 审计 21 项缺陷的教训总结 |
| Chunk/Sub-chunk 概念与生成链路 | `DESIGN.md` 第 16 章 | 概念定义、继承表、切分策略、设计意图 |
| Prompt 设计演进与策略约束 | `DESIGN.md` 第 22 章 | 三步提取流程、代码排除规则、属性格式规范、变更历史 |
| 配置系统与完整配置项速查 | `README.md`「配置说明」 | 四层优先级、全部活跃配置项（~45 项） |
| 环境隔离与测试基础设施 | `README.md`「开发与测试」 | conftest.py 三重隔离机制 |
| 官方 API 开发文档链接 | `README.md`「参考资源」 | Kimi / BigModel API 文档 |

---

## 代码规范

### 强制工具链
- **ruff**：代码格式化与 lint（`venv/bin/ruff check scripts/` / `venv/bin/ruff format scripts/`）
- **pyright**：类型检查（`venv/bin/pyright scripts/`）
- **pytest**：测试执行（`PYTHONPATH=scripts venv/bin/python -m pytest scripts/tests/ -v`）

### 日志规范
- 统一使用 `logging`，格式：`"%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"`
- 禁止随意使用 `print()` 输出调试信息

### 类型与模型
- 领域模型使用 `dataclass`（`Chunk`、`Document`、`GraphEntity`、`GraphRelation`、`SubGraphView`）
- `Chunk.status`：`pending` → `embedded` → `done`（含 `skipped`、`failed`，`ChunkStatus` StrEnum）
- `Chunk.chunk_type`：`text` / `table` / `image_desc`（`ChunkType` StrEnum）
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
- 具体 Prompt 策略约束（实体类型、关系类型、属性格式、排除规则等）→ 见 `DESIGN.md` 第 22 章

---

## 文档同步规范

> **核心原则**：三份文档各有明确边界。代码变更后，Agent 必须按触发矩阵同步更新对应文档，禁止「只改代码不改文档」或「三份文档各自维护同一张表格」。

### 三份文档的职责边界

| 文档 | 面向读者 | 应包含的内容 | 禁止包含的内容 |
|------|---------|-------------|---------------|
| `AGENTS.md` | AI Coding Agent | 编码规范、测试策略、文件权限、自主/询问边界、文档同步规范本身 | 详细设计论证、用户使用说明、完整配置表格 |
| `DESIGN.md` | 技术开发者/审计者 | 架构决策（选择原因+权衡）、Pipeline 设计、数据模型、Prompt 设计意图与变更历史、审计教训 | 安装步骤、工具命令参考、配置速查表 |
| `README.md` | 终端用户/使用者 | 安装指南、快速开始、工具命令、配置说明、已知限制 | 内部代码规范、详细设计论证 |

### 代码变更 → 文档更新触发矩阵

| 代码变更类型 | 必须更新的文档 | 禁止出现的动作 |
|-------------|---------------|---------------|
| 修改配置项（增/删/改默认值/改说明） | `README.md`「配置说明」为唯一源头 → 同步到 `config.json` 模板 | 在 `AGENTS.md` 中独立维护配置表格；只更新一份导致另一份过期 |
| 修改 Pipeline 阶段或产物路径 | `DESIGN.md` 第 19 章 | 产物路径在三份文档中不一致 |
| 修改数据模型字段 | `DESIGN.md` 对应数据模型章节 | `AGENTS.md` 的「类型与模型」字段列表与代码 dataclass 不一致 |
| 修改 Prompt 策略 | `DESIGN.md` 第 22 章（设计意图+变更历史） | 只改 prompt 文件不改 `DESIGN.md` |
| 新增/删除实体或关系类型 | `DESIGN.md` Schema 扩展策略 + 第 22 章 | `AGENTS.md`「开发计划」只记完成标记，不记设计决策 |
| 修复数据损坏/状态安全类 bug | `DESIGN.md` 第 20 章（审计教训） | 修复后不在文档中记录教训 |
| 修改测试基础设施 | `README.md`「开发与测试」 | 在 `AGENTS.md`「测试策略」中重复粘贴实现代码 |
| 修改 `.venv` / `plugin.json` / 安装方式 | `README.md`「安装」+ `plugin.json` | 安装路径在三份文档中不一致 |

### 关键约束

1. **配置表唯一源头**：`README.md`「配置说明」是全部活跃配置项的唯一权威表格。`AGENTS.md` 不再维护配置表，改为引用 `README.md`。`config.json` 模板必须与表格同步。
2. **测试数字统一引用**：「XX 个测试全部通过」等易变数字，以最近一次 pytest 输出为准。三份文档中出现时应保持一致。
3. **禁止三份文档独立维护同一张表格**：如 Pipeline 产物映射表、已知限制列表、实体类型清单等，必须只在一份文档中维护，其他文档用索引引用。
4. **修改后自检**：完成代码修改和文档更新后，运行 `grep -n` 检查三份文档中是否出现同一配置项/路径的不同默认值或不同说明。

---

## Agent 自主开发原则

### 可以自主决定的事项
- 代码重构（变量重命名、函数拆分、模块移动）
- 新增单元测试或调整测试结构
- 更新日志格式、错误提示信息
- 优化算法实现（不改变输入输出语义）
- 更新本文档（AGENTS.md）和 README.md、DESIGN.md（按「文档同步规范」执行）
- 不过度考虑项目开发的版本兼容性

### 必须询问用户的事项
- 修改核心数据模型（`Chunk`、`Document`、`GraphEntity`、`GraphRelation` 的字段）
- 修改数据库 Schema（新增/删除/修改表或字段）
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
| `plugin.json` | ⚠️ 需批准 | 插件定义，用户明确要求时可修改（已完成 v1.1.0 重构：30→22 工具） |
| `config.json` | ✅ 可改 | 用户持久化配置模板 |
| `README.md` / `AGENTS.md` / `DESIGN.md` | ✅ 可改 | 文档必须随代码同步更新（按「文档同步规范」执行） |
| `knowledge_base/` | ❌ 禁止 | 运行时生成数据 |

---

## 测试策略

### 核心原则
- **Mock 优先**：所有涉及外部 API 的测试使用 Fake 客户端，**禁止调用真实 API**
- **环境隔离**：`scripts/tests/conftest.py` 在模块级别完成三重隔离（清除环境变量、阻止 `load_dotenv` 重新加载 `.env`、重定向 Config 默认路径到临时目录），确保测试不依赖外部 `.env` 或生产数据。详见 `README.md`「开发与测试」
- **回归即修复**：任何导致测试失败的变更必须当场修复
- **352 个测试用例必须全部通过**（2 个 skipped 为正常：真实文档参数集为空）

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
| `test_knowledge_base_service_queries.py` | 语义-图谱联合查询、chunk+实体联合返回测试 |
| `test_bigmodel_parser_client.py` | BigModel 解析客户端全路径覆盖测试 |
| `test_entity_extractor.py` | EntityExtractor multimodal batch 请求构建测试 |
| `test_batch_builder.py` | BatchBuilder 切分保护与空 content 过滤测试 |
| `test_parsed_docs_jsonl.py` | 真实文档端到端 JSONL 生成测试 |
| `test_pipeline_stages.py` | 六阶段 pipeline 拆分测试 |

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

## 开发计划

### 当前阶段（已完成）
- ✅ DocGraphPipeline 重构：Batch API 架构、树形章节解析、multimodal 图片处理
- ✅ 零数据丢失：移除所有源数据过滤和截断
- ✅ Prompt 外部化：所有提示词在 `scripts/prompts/*.txt`
- ✅ 代码清理：删除 11 个旧文件、4 个旧 prompt 文件、7 个旧测试文件
- ✅ **API 接口重构**：新增 `KnowledgeBaseService` 统一服务层；Plugin 暴露 22 个优化工具
- ✅ **图谱查询增强**：`GraphStore.find_path()` BFS 路径搜索、`search_entities()` 模糊搜索
- ✅ **跨文档知识互通**：全局统一图谱 `global.json` + 文档子图快照 `{doc_id}.json`，同名同类型实体自动去重
- ✅ **章节级增量更新**：`content_hash` 指纹比较，未变章节复用实体缓存与 embedding，仅 LLM 提取变更/新增章节
- ✅ **本地 PDF 解析器 TOC 驱动式章节识别**
- ✅ **完整章节层级构造**
- ✅ **段落内换行保留原始 `\n`**
- ✅ **ChapterParser 多级树重构**
- ✅ **BatchBuilder 接收扁平化节点**
- ✅ **数据模型清理**
- ✅ **数据库 schema 简化**
- ✅ **数据质量增强**：`GraphEntity`/`GraphRelation` 新增 `confidence`/`verified`/`created_at`/`updated_at`/`feedback_score` 字段
- ✅ **图谱动态更新接口**：Plugin 暴露 4 个写工具
- ✅ **冲突检测与日志**
- ✅ **语义-图谱联合查询**
- ✅ **chunk+实体联合返回**
- ✅ **用户反馈机制**
- ✅ **IC 关系扩展**
- ✅ **处理器架构图谱扩展（C28x+CLA）**
- ✅ **多文档类型图谱扩展**
- ✅ **跨产品外设变体建模**
- ✅ **属性索引优化**
- ✅ **跨粒度桥接索引**
- ✅ **Pipeline 六阶段拆分**
- ✅ **Embedding 同步单文本 API**
- ✅ **环境隔离三重机制**：彻底根治 `.env` 污染测试环境的问题（`fc7fb38` 未根治，通过 conftest.py 清除+阻止 load_dotenv+临时目录重定向彻底解决）
- ✅ **存储粒度与查询粒度解耦（方案C）**：引入 `content_blocks` 表作为存储粒度（按 `##` section 聚合后切分），`heading_maps` 表作为查询粒度（`##`/`###` 共享同一 block 集合），batch 数量减少 40-50%，API 成本降低
- ✅ **FAISS ID 偏移**：`_BLOCK_FAISS_OFFSET = 10_000_000` 避免 content_blocks 与兼容层 chunks 在 FAISS 中冲突
- ✅ **352 个测试全部通过**（2 个 skipped 为正常）

### 下一阶段（精确到下一步）
1. **可视化**：图谱可视化导出（Graphviz / D3.js）
2. **评估体系**：实体提取准确率、关系提取召回率的自动化评估
3. **DOCX/XLSX 接入 pipeline**：当前解析器代码存在但未接入 `DocGraphPipeline`
