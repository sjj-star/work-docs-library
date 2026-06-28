# work-docs-library Agent 自主式开发策略文档

> 本文档供 AI Coding Agent 阅读，是项目权威技术参考与行动准则。
> - 项目详细介绍、安装与使用指南 → `README.md`
> - 技术方案决策、设计意图与权衡 → `DESIGN.md`
> - 本规范与上述两份文档**必须同步更新**，具体规则见下方「文档同步规范」。

---

## 项目概述

本项目是一个面向 **IC 前端设计技术文档（PDF）** 的自动化知识提取 Pipeline，以 **Kimi Code CLI Plugin** 形式运行。核心能力包括：

- **智能文档解析**：PDF 通过 BigModel Expert API 解析为 Markdown + 图片；失败时自动 fallback 到本地 PDFParser（PyMuPDF + TOC 驱动）
- **知识图谱构建**：自动提取结构化实体（Feature、Module、Register、Signal、Instruction、Interrupt、PipelineStage、Peripheral 等）和关系（IMPLEMENTS、CONTAINS、HAS_REGISTER、INSTRUCTION_READS_REGISTER、MODULE_IMPLEMENTS_INSTRUCTION、INTERRUPT_TRIGGERS 等），构建跨文档互通的知识图谱
- **混合语义检索**：基于 FAISS 的稠密向量检索 + BM25 稀疏索引 RRF 融合，支持 LLM 交叉编码器重排序
- **Batch API 架构**：LLM 实体提取通过 Batch API 提交（成本为同步 API 的 50%），Embedding 通过同步单文本 API 完成
- **章节级增量更新**：按章节 `content_hash` 指纹比较，未变章节复用缓存，仅对变更/新增章节进行 LLM 提取
- **Multimodal 图片理解**：LLM 直接分析文档中的图片，生成文字描述用于向量化
- **评估框架**：EvalDataset/EvalQuestion + LLM-as-judge + retrieval metrics + EvalHarness
- **Agentic 查询规划**：通过外部 Skill 编排 `AgenticSearchPlanner` 查询分解与多步检索

**目标规模**：数百个文档，每个文档几十到万页级。全局图谱通过 NetworkX 内存存储管理，知识图谱检索与文档内容向量检索互通关联。

---

## AI Agent 原生插件原则

> 本插件是 Kimi Code 等 AI Agent 的 **MCP 扩展**，不是独立 SaaS。设计和新增功能时必须内化这一事实。

### 核心含义
1. **LLM 是外部 Agent 的资源，不是插件的固定成本**
   - 评估框架、智能体检索、查询改写、`AgenticSearchPlanner` 查询分解等需要 LLM 评判或推理的能力，优先以 **Skill 编排 + 原子 MCP 工具** 实现，由外部 Agent 承担 LLM 调用成本，而不是在插件内部启动完整 Agent 运行时。
   - 新增的 `skills/agentic-search/SKILL.md` 即通过 Skill 编排 `AgenticSearchPlanner` 查询分解策略，由外部 Agent 承担 LLM 推理成本。
2. **复杂策略进 Skill，通用机制进代码**
   - Python 层只提供机制：评估指标计算、检索器接口、单步搜索、chunk 读取、图谱查询。
   - Skill 层编排策略：评估流程、ReAct/Self-Ask 步骤、查询分解、结果批判。
3. **MCP 工具保持原子性与可组合性**
   - 每个 MCP 工具应像 Unix 工具一样完成一件事：搜索、读取、评分、记录。
   - 避免在单个 MCP 工具内部做多轮 LLM 推理或隐藏状态机。
4. **状态安全优先于智能**
   - 数据变更类操作保留在 `admin_tools.py`，不暴露为 MCP 写工具，保留人工/审计边界。
5. **可观测性应面向 Agent 理解**
   - 评估结果、检索中间步骤、冲突日志、来源信息以结构化 JSON 返回，方便外部 Agent 做下一步决策，而不是仅面向人类阅读。

### 设计决策边界
- 需要新增 LLM 调用流程时，先问：**这个流程应该由外部 Agent 通过 Skill 编排，还是必须由插件内部无干预完成？**
- 只有当流程需要批处理、离线运行、或无 Agent 在场时，才在插件内部实现 LLM 调用。
- **当前无例外**：MCP 工具面已精简为 5 个原子工具（`search` / `explore` / `read` / `ingest` / `status`），插件内部不做 LLM 合成或智能路由。`AgenticSearchPlanner` 的查询分解与 `LLMReranker` 的 passage scoring 等 LLM 推理均移至外部 Skill 编排，由外部 Agent 承担 LLM 调用成本。

---

## 技术栈与运行时架构

### 核心依赖
| 类别 | 技术/库 | 版本要求 |
|------|--------|---------|
| 语言 | Python | >= 3.11 |
| 虚拟环境 | uv / venv | — |
| PDF 解析 | PyMuPDF, pdfplumber, PyPDF2 | >=1.23, >=0.10, >=3.0 |
| 向量检索 | FAISS (CPU), NumPy | >=1.7.4, >=1.24 |
| 图谱存储 | NetworkX | >=3.0 |
| 元数据存储 | SQLite (标准库 sqlite3) | — |
| 混合检索 | rank-bm25 | >=0.2.2 |
| 本地重排序 | sentence-transformers | >=3.0.0 |
| 图片处理 | Pillow | >=10.0 |
| HTTP 客户端 | requests | >=2.31 |
| 环境变量 | python-dotenv | >=1.0 |
| 测试 | pytest | >=9.0 |

### 可选依赖
| 类别 | 技术/库 | 版本要求 | 说明 |
|------|--------|---------|------|
| Office 文档解析 | python-docx, openpyxl | >=1.1, >=3.1 | 尚未接入主 pipeline，单独使用需 `pip install -e ".[office]"` |

### 四存储系统架构
| 存储 | 职责 | 持久化文件 | 原子性保证 |
|------|------|-----------|-----------|
| **SQLite** | 文档元数据、content_blocks、heading_maps | `knowledge_base/workdocs.db` | 单连接事务 |
| **FAISS** | 向量索引（语义搜索） | `knowledge_base/faiss.index`（IndexIDMap2） | `fcntl` 进程锁 + 事务 + 临时文件 rename |
| **NetworkX** | 全局知识图谱（实体+关系） | `knowledge_base/graphs/{doc_id}.json` + `global.json` | 内存操作 + 文件原子写入 |
| **Bridge** | block ↔ 实体 双向索引 | 纯内存（重启从 SQLite 重建） | 内存级 |

### Pipeline 六阶段
1. **解析**（PDF → Markdown + images）
2. **构建 LLM Batch JSONL**（Markdown → 树形章节 → content_blocks + heading_maps → batch requests）
3. **提交 LLM Batch API**（增量过滤 → 并行提交 → 保存原始结果）
4. **解析入库**（结果文件 → entities/relations → GraphStore + SQLite，不含向量化）
5. **构建 Embedding JSONL**（从 SQLite 查询待向量化的 blocks）
6. **同步 Embedding 向量化**（逐条调用 Embedding API → SQLite + FAISS）

详见 `README.md`「架构概览」和 `DESIGN.md` 第 19 章。

---

## 目录结构与代码组织

```
work-docs-library/
├── kimi.plugin.json              # Kimi Code 新规范插件 Manifest（MCP server + Skill）
├── skills/
│   ├── using-workdocs/
│   │   └── SKILL.md              # 入口：总览、规则、何时调用子 skill
│   ├── ingesting-workdocs/
│   │   └── SKILL.md              # 文档入库/更新/重试工作流
│   └── exploring-workdocs/
│       └── SKILL.md              # 语义搜索 + 图谱联合查询工作流
├── pyproject.toml                # Python 项目配置、依赖、ruff/pyright/pytest 设置
├── scripts/
│   ├── mcp_server.py             # MCP stdio server（JSON-RPC，stdout 隔离）
│   ├── plugin_router.py          # Plugin 工具函数库（被 mcp_server / admin_tools 复用）
│   ├── admin_tools.py            # 不暴露为 MCP 的内部管理命令入口
│   ├── .env / .env.example       # 环境变量（凭证等，gitignored）
│   ├── prompts/                  # LLM 提示词文件（运行时读取，无需重启）
│   │   ├── entity_extraction_system.txt
│   │   └── entity_extraction_user.txt
│   ├── core/                     # 业务逻辑层
│   │   ├── config.py             # 统一配置中心（.env / 环境变量 → 默认值）
│   │   ├── doc_graph_pipeline.py # ⭐ DocGraphPipeline 主管道（六阶段）
│   │   ├── knowledge_base_service.py  # 统一服务层封装（DB + VectorIndex + GraphStore + Bridge）
│   │   ├── batch_clients.py      # BaseBatchClient + BatchClient（通用 OpenAI-compatible Batch API）
│   │   ├── llm_chat_client.py    # LLM 同步对话客户端（Chat 模式回退）
│   │   ├── embedding_client.py   # Embedding 同步单文本客户端
│   │   ├── bigmodel_parser_client.py  # BigModel Expert 文件解析（专用非兼容 API）
│   │   ├── graph_store.py        # NetworkX 图谱存储（CRUD、冲突检测、属性索引、路径搜索）
│   │   ├── db.py                 # SQLite 数据库操作
│   │   ├── vector_index.py       # FAISS 向量索引管理
│   │   ├── models.py             # 数据模型（Chunk、Document、EvalDataset、EvalQuestion）
│   │   ├── enums.py              # StrEnum 定义（ChunkStatus、DocumentStatus、ChunkType）
│   │   ├── evaluation.py         # 评估框架（EvalDataset/EvalQuestion + LLM-as-judge + EvalHarness）
│   │   ├── sparse_index.py       # BM25 稀疏索引
│   │   ├── hybrid_retriever.py   # RRF 稠密+稀疏混合检索
│   │   ├── reranker.py           # LLM 交叉编码器重排序
│   │   └── agentic_search.py     # AgenticSearchPlanner 查询分解机制
│   ├── parsers/                  # IO / 解析层
│   │   ├── pdf_parser.py         # PDF 本地解析器（PyMuPDF + TOC 驱动章节识别 + 表格/图片检测）
│   │   ├── office_parser.py      # DOCX / XLSX 解析器（代码存在，尚未接入 pipeline）
│   │   └── image_utils.py        # 图片压缩与三分类（彩色/灰度/黑白）
│   ├── tests/                    # pytest 测试集（506 passed, 0 skipped）
│   │   ├── conftest.py           # 三重环境隔离（清除 WORKDOCS_ 环境变量、阻止 load_dotenv、临时目录重定向）
│   │   ├── fixtures/             # 测试 fixture（PDF 页样本、解析输出样本）
│   │   └── test_*.py             # 各模块测试文件
├── knowledge_base/               # 运行时自动生成数据（❌ 禁止手动修改）
│   ├── workdocs.db               # SQLite
│   ├── faiss.index               # FAISS 向量索引（IndexIDMap2，直接存储 block_db_id）
│   ├── parsed/<doc_id>/          # Stage1 解析输出
│   ├── batch/                    # Stage2/3/5/6 中间产物
│   └── graphs/                   # Stage4 子图快照 + global.json
└── .venv/                        # Python 虚拟环境
```

> 新增项目级 Skill：`skills/agentic-search/SKILL.md`（Agentic Search 查询分解工作流，体现「复杂策略进 Skill，通用机制进代码」原则）。

---

## 构建与测试命令

### 环境安装
```bash
# 方式一：uv（推荐）
cd /path/to/work-docs-library
uv sync

# 方式二：pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 代码质量检查（提交前必须执行）
```bash
# 代码格式化与 lint
cd /path/to/work-docs-library
./.venv/bin/ruff check scripts/
./.venv/bin/ruff format scripts/

# 类型检查
./.venv/bin/pyright scripts/
```

### 测试执行
```bash
# 完整测试集（当前状态：506 passed, 0 skipped, 0 failed）
cd /path/to/work-docs-library
PYTHONPATH=scripts ./.venv/bin/python -m pytest scripts/tests/ -v

# 仅运行核心基础设施测试（<5s）
PYTHONPATH=scripts ./.venv/bin/python -m pytest \
  scripts/tests/test_graph_store.py \
  scripts/tests/test_vector_index.py \
  scripts/tests/test_db.py \
  scripts/tests/test_knowledge_base_service.py \
  scripts/tests/test_knowledge_base_service_queries.py -v

# 仅运行 Pipeline 集成测试（<5s）
PYTHONPATH=scripts ./.venv/bin/python -m pytest \
  scripts/tests/test_pipeline_stages.py \
  scripts/tests/test_plugin_router.py \
  scripts/tests/test_pdf_parser.py -v
```

---

## 代码规范

### 强制工具链
- **ruff**：代码格式化与 lint（配置见 `pyproject.toml [tool.ruff]`）
  - target-version: py311, line-length: 100
  - select: E, W, F, I, N, D, UP
  - per-file-ignores: `scripts/tests/*` 忽略模块级 docstring 要求
- **pyright**：类型检查（`scripts/` 为 include，`scripts/tests/fixtures` 和 `__pycache__` 为 exclude）
- **pytest**：测试执行（`pythonpath = ["scripts"]`）

### 日志规范
- 统一使用 `logging`，格式：`"%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"`
- 禁止随意使用 `print()` 输出调试信息

### 类型与模型
- 领域模型使用 `dataclass`（`Chunk`、`Document`、`GraphEntity`、`GraphRelation`、`SubGraphView`、`EvalQuestion`、`EvalDataset`）
- `Chunk.status`：`pending` → `embedded` → `done`（含 `skipped`、`failed`，`ChunkStatus` StrEnum）
- `Chunk.chunk_type`：`text` / `table` / `image_desc` / `summary`（`ChunkType` StrEnum）
- `Document.status`：`pending` → `processing` → `batch_submitted` → `done` / `failed`（`DocumentStatus` StrEnum）
- `GraphEntity`/`GraphRelation` 字段：`entity_type`/`rel_type`、`name`、`properties`、`doc_properties`（按文档原始属性快照）、`source_doc_ids`、`source_chapter`、`confidence`、`verified`、`created_at`、`updated_at`、`feedback_score`

### 数据库安全
- **所有 SQL 必须使用参数化查询**（`?` 占位符）
- `KnowledgeDB._connect()` 使用上下文管理器

### Prompt 管理
- 所有 LLM 提示词必须在 `scripts/prompts/*.txt` 中
- 代码通过 `Config.PROMPT_DIR / f"{name}.txt"` 读取
- 修改提示词后无需重启，下次调用自动生效

### API 规范
- **通用 API**（LLM Batch、Embedding、同步对话）：统一使用兼容 OpenAI 协议的 HTTP API，通过配置参数适配不同服务商
- **专用 API**（PDF 解析）：`BigModelParserClient` 使用 BigModel 专有的 Expert 文件解析接口，非 OpenAI-compatible，失败时自动 fallback 到本地 `PDFParser`
- 构造项目内部 API 时要尽可能使用函数式编程

### 机制与策略分离原则
> 不要擅自设计任何硬编码的策略，应当优先设计一个框架和机制，后面根据实际需求使用机制来实现策略。

- **零数据丢失是底线**：任何原始解析的文本都不得删除或丢弃
- 具体 Prompt 策略约束 → 见 `DESIGN.md` 第 22 章

---

## 测试策略

### 核心原则
- **Mock 优先**：所有涉及外部 API 的测试使用 Fake 客户端，**禁止调用真实 API**
- **环境隔离**：`scripts/tests/conftest.py` 在模块级别完成三重隔离：
  1. 清除所有 `WORKDOCS_` 前缀和含 `.` 的环境变量
  2. 阻止 `load_dotenv` 重新加载 `.env` 文件
  3. 重定向 Config 默认路径到临时目录（DB、FAISS、Graph 均隔离）
- **回归即修复**：任何导致测试失败的变更必须当场修复
- **506 个测试用例必须全部通过**（0 skipped）

### 测试文件清单
| 测试文件 | 用例数 | 说明 |
|----------|--------|------|
| `test_plugin_router.py` | 55 | Plugin 工具路由、参数解析、路径沙箱 |
| `test_pdf_parser.py` | 71 | PDF 解析核心测试（含 14 个真实页面 fixture） |
| `test_borderless_table_extractor.py` | 3 | AMBA 风格无边框表格提取单元测试 |
| `test_table_utils.py` | 4 | Markdown 表格规范化单元测试 |
| `test_office_parser.py` | 3 | DOCX / XLSX 解析测试 |
| `test_db.py` | 15 | SQLite 操作、事务管理 |
| `test_vector_index.py` | 14 | FAISS IndexIDMap2 增删查、事务 |
| `test_llm_client.py` | 9 | LLM 客户端 Mock |
| `test_config_env.py` | 13 | 环境变量配置优先级、默认值、敏感 key 脱敏 |
| `test_chapter_parser.py` | 20 | ChapterParser 树形章节解析测试 |
| `test_image_utils.py` | 13 | 图片压缩工具测试 |
| `test_graph_store.py` | 77 | NetworkX 图谱存储 CRUD、冲突检测、属性索引、子图、路径搜索、持久化 |
| `test_batch_clients.py` | 19 | Batch API 客户端 Mock 测试 |
| `test_knowledge_base_service.py` | 21 | KnowledgeBaseService 统一服务层测试 |
| `test_knowledge_base_service_queries.py` | 3 | 语义-图谱联合查询、block+实体联合返回测试 |
| `test_bigmodel_parser_client.py` | 8 | BigModel 解析客户端全路径覆盖测试 |
| `test_content_blocks.py` | 10 | 内容块切分、heading_maps 构建 |
| `test_batch_builder.py` | 14 | BatchBuilder 切分保护与空 content 过滤测试 |
| `test_parsed_docs_jsonl.py` | 2 | 真实文档端到端 JSONL 生成测试 |
| `test_pipeline_stages.py` | 31 | 六阶段 pipeline 拆分测试 |
| `test_audit_issues.py` | 10 | 生产 bug/审计问题定向回归测试（含 FAISS/SQLite 原子性） |
| `test_mcp_server.py` | 10 | MCP Server 工具注册与 JSON-RPC 调用测试（含 5 个 MCP 工具白名单校验） |
| `test_status_tool.py` | 13 | 结构化状态仪表盘各 scope 测试 |
| `test_evaluation.py` | 21 | 评估框架（EvalDataset/EvalQuestion + LLM-as-judge + EvalHarness） |
| `test_sparse_index.py` | 9 | BM25 稀疏索引构建与搜索 |
| `test_hybrid_retriever.py` | 3 | RRF 稠密+稀疏混合检索 |
| `test_reranker.py` | 9 | LLM 交叉编码器重排序 |
| `test_agentic_search.py` | 11 | AgenticSearchPlanner 查询分解机制 |

### Mock 方法
使用 `monkeypatch.setattr` 替换客户端类方法：
```python
monkeypatch.setattr(
    "core.batch_clients.BatchClient.submit_parallel_batches",
    lambda self, reqs: [{"entities": [], "relationships": [], "image_descriptions": {}}]
)
```

---

## 安全注意事项

### 1. API 密钥管理
- API Key 存储于 `scripts/.env`（gitignored），**禁止**硬编码到源码或提交到版本控制
- 新版 `kimi.plugin.json` 不再使用旧 `plugin.json` 的 `inject` 字段；API Key 通过 `.env` 或环境变量配置，由 `Config` 自行读取
- `Config.to_dict()` 对 `LLM_API_KEY`/`EMBEDDING_API_KEY`/`PARSER_API_KEY` 始终脱敏为 `***`；`tool_config` 忽略用户传入的 `mask_sensitive` 参数，强制返回脱敏配置

### 2. 数据库安全
- 所有 SQL 必须使用参数化查询（`?` 占位符），禁止字符串拼接 SQL
- SQLite 数据库文件 `knowledge_base/workdocs.db` 为单用户本地场景设计，不建议多进程并发写入

### 3. 存储系统一致性
- FAISS 索引操作使用 `fcntl.flock` 进程级排他锁，修改前调用 `_reload()` 加载磁盘最新状态
- 多存储系统（SQLite + FAISS + NetworkX + Bridge）之间**无分布式事务**，极端崩溃场景可通过 `reprocess` 或 `rebuild_global_graph` 重建
- `knowledge_base/` 目录为运行时生成数据，**禁止**手动修改其中的 `.db`、`.index`、`.json` 文件

### 4. 外部 API 调用安全
- 测试环境**绝对禁止**调用真实 API（BigModel Expert、Kimi Batch、Embedding 等）
- `conftest.py` 已清除所有 `WORKDOCS_` 环境变量并阻止 `.env` 加载，防止测试意外使用生产凭证
- Batch API 超大 JSONL 自动按 100MB 拆分，防止单文件过大导致网络/内存问题

### 5. 文件系统安全
- 所有图谱/索引持久化使用「临时文件 + `os.replace`」原子写入，避免崩溃导致文件损坏
- `Config.setup_logging()` 将日志输出到 stderr；**MCP server 的 stdout 必须仅用于 JSON-RPC 消息**，严禁 print 或日志混入 stdout
- Plugin 所有用户传入的文件路径必须通过 `_resolve_allowed_path()` 沙箱校验，禁止跳出允许目录或包含 `..`（已新增回归测试）
- `BigModelParserClient` 解压 ZIP 时校验条目名，禁止 `..`、绝对路径、盘符，防止路径遍历
- MCP 工具面仅暴露适合 Agent 自主调用的读取/导入类工具；数据改写/管理类功能放入 `scripts/admin_tools.py`，不进入 MCP，保留人工/审计边界

---

## 文件修改权限规则

| 文件/目录 | 权限 | 说明 |
|-----------|------|------|
| `scripts/core/*.py` | ⚠️ 需批准 | 核心模块，修改前必须说明影响范围 |
| `scripts/prompts/*.txt` | ✅ 可改 | 提示词文件，运行时读取 |
| `scripts/tests/*.py` | ✅ 可改 | 测试文件 |
| `scripts/parsers/*.py` | ⚠️ 需批准 | 解析器影响数据输入质量 |
| `kimi.plugin.json` | ⚠️ 需批准 | Kimi Code 新规范插件 Manifest |
| `scripts/mcp_server.py` | ⚠️ 需批准 | MCP stdio server，协议层变更影响插件可用性 |
| `skills/using-workdocs/SKILL.md` | ⚠️ 需批准 | 入口 Skill，影响 Agent 使用行为 |
| `skills/ingesting-workdocs/SKILL.md` | ⚠️ 需批准 | 入库工作流 Skill |
| `skills/exploring-workdocs/SKILL.md` | ⚠️ 需批准 | 查询+图谱联合工作流 Skill |

| `README.md` / `AGENTS.md` / `DESIGN.md` | ✅ 可改 | 文档必须随代码同步更新 |
| `knowledge_base/` | ❌ 禁止 | 运行时生成数据 |

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
- 新增外部依赖（`pyproject.toml` 变更）
- 修改 Pipeline 核心流程（解析 → 章节树 → batch 构建 → 实体提取 → 图谱 → 向量化）
- 删除已有功能或文件
- 涉及真实 API 调用的操作（测试除外）
- 修改 `kimi.plugin.json` 插件 Manifest（除非用户明确要求）

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
| 修改配置项（增/删/改默认值/改说明） | `README.md`「配置说明」为唯一源头 → 同步到 `scripts/.env.example` 模板 | 在 `AGENTS.md` 中独立维护配置表格；只更新一份导致另一份过期 |
| 修改 Pipeline 阶段或产物路径 | `DESIGN.md` 第 19 章 | 产物路径在三份文档中不一致 |
| 修改数据模型字段 | `DESIGN.md` 对应数据模型章节 | `AGENTS.md` 的「类型与模型」字段列表与代码 dataclass 不一致 |
| 修改 Prompt 策略 | `DESIGN.md` 第 22 章（设计意图+变更历史） | 只改 prompt 文件不改 `DESIGN.md` |
| 新增/删除实体或关系类型 | `DESIGN.md` Schema 扩展策略 + 第 22 章 | `AGENTS.md`「开发计划」只记完成标记，不记设计决策 |
| 修复数据损坏/状态安全类 bug | `DESIGN.md` 第 20 章（审计教训） | 修复后不在文档中记录教训 |
| 修改测试基础设施 | `README.md`「开发与测试」 | 在 `AGENTS.md`「测试策略」中重复粘贴实现代码 |
| 修改 `.venv` / `kimi.plugin.json` / 安装方式 | `README.md`「安装」+ `kimi.plugin.json` | 安装路径在三份文档中不一致 |

### 关键约束

1. **配置表唯一源头**：`README.md`「配置说明」是全部活跃配置项的唯一权威表格。`AGENTS.md` 不再维护配置表，改为引用 `README.md`。`scripts/.env.example` 模板必须与表格同步。
2. **测试数字统一引用**：「XX 个测试全部通过」等易变数字，以最近一次 pytest 输出为准。三份文档中出现时应保持一致。
3. **禁止三份文档独立维护同一张表格**：如 Pipeline 产物映射表、已知限制列表、实体类型清单等，必须只在一份文档中维护，其他文档用索引引用。
4. **修改后自检**：完成代码修改和文档更新后，运行 `grep -n` 检查三份文档中是否出现同一配置项/路径的不同默认值或不同说明。

---

## 技术参考索引

以下主题在 `DESIGN.md` 和 `README.md` 中有详细论述，本处仅提供索引，**禁止在此重复展开**：

| 主题 | 权威文档 | 说明 |
|------|---------|------|
| 核心架构决策（Batch API、NetworkX、零数据丢失等 12 项） | `DESIGN.md` 第 1–10 章及第 25 章 | 每项决策的选择原因、实现细节与权衡 |
| Pipeline 六阶段拆分与中间产物持久化 | `DESIGN.md` 第 19 章 | 产物清单、状态管理、开发约束 |
| 防御性编程与状态安全原则（7 条） | `DESIGN.md` 第 20 章（审计教训与新增开发原则） | 2026-04 审计 21 项缺陷的教训总结 |
| Block/Sub-block 概念与生成链路 | `DESIGN.md` 第 16 章 | 概念定义、继承表、切分策略、设计意图 |
| Prompt 设计演进与策略约束 | `DESIGN.md` 第 22 章 | 三步提取流程、代码排除规则、属性格式规范、变更历史 |
| 配置系统与完整配置项速查 | `README.md`「配置说明」 | 三层优先级、全部活跃配置项（约 62 项） |
| 环境隔离与测试基础设施 | `README.md`「开发与测试」 | conftest.py 三重隔离机制 |
| 官方 API 开发文档链接 | `README.md`「参考资源」 | Kimi / BigModel API 文档 |

---

## 开发计划

### 当前阶段（已完成）
- ✅ DocGraphPipeline 重构：Batch API 架构、树形章节解析、multimodal 图片处理
- ✅ 零数据丢失：移除所有源数据过滤和截断
- ✅ Prompt 外部化：所有提示词在 `scripts/prompts/*.txt`
- ✅ 代码清理：删除 11 个旧文件、4 个旧 prompt 文件、7 个旧测试文件
- ✅ **API 接口重构**：新增 `KnowledgeBaseService` 统一服务层；Plugin 暴露 5 个原子 MCP 工具（`search` / `explore` / `read` / `ingest` / `status`）
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
- ✅ **图谱动态更新接口**：Plugin 内部提供 4 个图谱写工具（admin_tools / plugin_router），按最小权限原则未暴露为 MCP
- ✅ **冲突检测与日志**
- ✅ **语义-图谱联合查询**
- ✅ **block+实体联合返回**
- ✅ **用户反馈机制**
- ✅ **IC 关系扩展**
- ✅ **处理器架构图谱扩展（C28x+CLA）**
- ✅ **多文档类型图谱扩展**
- ✅ **跨产品外设变体建模**
- ✅ **属性索引优化**
- ✅ **跨粒度桥接索引**
- ✅ **Pipeline 六阶段拆分**
- ✅ **Embedding 同步单文本 API**
- ✅ **环境隔离三重机制**：彻底根治 `.env` 污染测试环境的问题
- ✅ **存储粒度与查询粒度解耦（方案C）**：引入 `content_blocks` 表作为存储粒度，`heading_maps` 表作为查询粒度，batch 数量减少 40-50%
- ✅ **FAISS 索引重构为 IndexIDMap2**：直接使用 block_db_id 作为存储 ID，移除 `_BLOCK_FAISS_OFFSET` 与手动 `_id_map`
- ✅ **506 个测试全部通过**
- ✅ **PDF Parser 表格检测增强（Milestone 1-4）**：`find_tables(strategy="lines_strict")`、caption-gated 预筛选、位域图重叠保护、全部 14 个 Magic Number 配置化（已移除 PyMuPDF4LLM fallback）
- ✅ **PDF Parser 图片检测增强（Milestone 2）**：`page.get_image_info()` 过滤链、双路径提取
- ✅ **性能基准测试**：TI (219页) 10.3s/0表格 → 46.2s/68表格；AMBA (585页) 8.7s/0表格 → 93.3s/22表格
- ✅ **多进程并行化可行性分析**：4-worker 加速比 TI 2.07x / AMBA 1.61x，但 Amdahl 定律限制整体收益仅 ~1.3x，**否决实施**
- ✅ **AMBA 空跑根因分析**：251 页触发 find_tables → 234 页 Type A 空跑，根因为 lines_strict 对无竖线表格检测率仅 9%
- ✅ **GapsFirstScanner 重构**：Caption-driven linear extractor 替代 UZN 方案，~600 行替代 2037 行，性能提升 5-10x
- ✅ **Hard Separator + Zone 模型**：header/footer/heading/caption/body_text 构成 y 轴硬分割，提取只在 zone 内部搜索
- ✅ **Cluster-based 图片渲染**：每个 drawing cluster 独立渲染为图片，修复过度合并问题
- ✅ **Orphan Zone 检测**：无 Caption 表格通过 `_classify_table_style` 启发式检测，命中率 90.5%
- ✅ **自适应表格策略**：`_classify_table_style` 识别 grid/horizontal 风格；grid 走 `find_tables(strategy="lines_strict")`，horizontal（含零高度线）走 `BorderlessTableExtractor`
- ✅ **预计算优化**：grid 页面每页一次 `find_tables` 全页扫描，结果复用于所有 caption/orphan zone，避免 142 次重复调用
- ✅ **AMBA 零高度线调研**：AMBA 表格水平线为 height=0.0 的零高度矢量线，`find_tables` 任何策略均无法识别，确认为 PyMuPDF C 库层面限制
- ✅ **`tab.cells` 空伪表格防御**：`find_tables()` 返回 cells=[] 的伪表格导致 `tab.bbox` 触发 `ValueError`，添加防御性跳过
- ✅ **`_fix_drawing_rect` 前零高度线统计**：在 rect 扩展前统计原始 drawing 的 `height==0.0`，避免检测永远失效
- ✅ **`_classify_table_style` v2/v3**：去掉 `len<10` 门槛，增加 drawing 风格分类（h/v/other）、文本密度过滤（<0.02 text/pt）、零高度线过滤；`horizontal` 风格直接路由到 `BorderlessTableExtractor`，不做 `find_tables` fallback
- ✅ **空跑率优化**：TI 15.9%→1.4%，AMBA 92.9%→0.0%，SPRUI07 10.4%→8.8%，DC_UG 31.0%→28.6%
- ✅ **AMBA 零高度线表格提取**：新增 `scripts/parsers/borderless_table_extractor.py`，基于横线 + 文本 x 对齐重建 Markdown 表格；`GapsFirstScanner` 识别 `horizontal` 后直接进入该提取器，不再尝试 `find_tables`
- ✅ **sprui07 page 692 纵向堆叠时序图修复**：cluster 合并改为连通分量 + 最近 caption 分量，防止只渲染底部子图
- ✅ **审计选项 A 最小紧急修复**：路径沙箱、强制脱敏、ZIP 路径遍历防护、FAISS/SQLite 写入顺序、KBService 原子失败处理、GraphStore 深拷贝、`fitz.open` 上下文管理、Batch 超时与失败抛错
- ✅ **解析器输出格式改为 PNG**：矢量图/光栅图/表格区域统一输出 PNG（无损高保真），移除 `PARSER_IMAGE_JPEG_QUALITY` 配置；LLM API 发送时的压缩由 `_compress_image_to_base64` 独立处理（三层分类策略不变）
- ✅ **评估框架**：`EvalDataset`/`EvalQuestion` + LLM-as-judge（Faithfulness / ContextPrecision / ContextRecall）+ retrieval metrics（HitRate / MRR / NDCG）+ `EvalHarness`
- ✅ **BM25 稀疏索引**：新增 `scripts/core/sparse_index.py`，支持从 SQLite blocks 或内存 block 列表构建
- ✅ **RRF 混合检索**：新增 `scripts/core/hybrid_retriever.py`，FAISS 稠密检索与 BM25 稀疏检索 RRF 融合
- ✅ **LLM 交叉编码器重排序**：新增 `scripts/core/reranker.py`，`LLMReranker` 通过同步 LLM 调用对候选 passage 打分
- ✅ **AgenticSearchPlanner 查询分解机制**：新增 `scripts/core/agentic_search.py`，LLM 将自然语言问题分解为 semantic / hybrid / reranked / graph / chapter / metadata / synthesize 多步搜索计划
- ✅ **Agentic Search Skill**：新增项目级 Skill `skills/agentic-search/SKILL.md`，由外部 Agent 编排查询分解与多步检索
- ✅ **MCP 工具面精简为 5 个原子工具**：`search`（聚合 semantic/hybrid/reranked）、`explore`（聚合 entity/neighbors/subgraph/path/provenance/conflicts）、`read`、`ingest`、`status`

### 下一阶段（精确到下一步）
1. **可视化**：图谱可视化导出（Graphviz / D3.js）
2. **评估体系细化**：评测数据集沉淀、端到端 RAG 评估流程产品化、指标看板
3. **DOCX/XLSX 接入 pipeline**：当前解析器代码存在但未接入 `DocGraphPipeline`
4. **系统化加固（审计选项 B）**：跨存储事务回滚、VectorIndex 原子重建、参数限流、解析阈值配置化、fuzz/property 测试

### 已完成的非表格项

- ✅ **Kimi Code 新插件规范迁移**：旧 `plugin.json` 替换为 `kimi.plugin.json`；新增 `scripts/mcp_server.py` 暴露 5 个 MCP 工具；`scripts/admin_tools.py` 承载数据改写/管理命令；新增 `skills/using-workdocs/SKILL.md`
- ✅ **移除 config.json 配置机制**：配置来源统一为 `.env`/环境变量 → 默认值

### PDF Parser 表格检测已知问题与下一步方向

**已完成的分析**（详见 `DESIGN.md` 第 26 章）：
- `_classify_table_style` 对 AMBA 表格 100% 正确分类为 `horizontal`，但 `find_tables` 任何策略均返回 0 表格
- 根因：AMBA 表格水平线为 height=0.0 的零高度矢量线，PyMuPDF 内部算法无法识别
- `text` 策略产生大量整页级误报（64×12 伪表格），不可用

**当前状态**（2026-06-10）：
- Grid 表格（TI/SPRUI07）：`lines_strict` 有效，自适应策略已优化
- Horizontal 表格（AMBA/DC_UG）：统一由 `BorderlessTableExtractor` 处理，不再使用 `find_tables(strategy="lines")`（实测 AMBA 该策略零产出）
- 路由原则：先由 `_classify_table_style` 判断风格，再直接调用最优方法，不做 fallback
- 误触发成本：预计算机制使 TABLE_CAPTION_RE 误匹配的影响从 "142 次 clip 调用" 降至 "一次全页扫描"
- 空跑率：TI 1.4% / AMBA 0.0% / SPRUI07 8.8% / DC_UG 28.6%（v2 算法，详见 DESIGN.md 26.8.4）

**已修复问题**：
- `tab.cells` 空伪表格：防御性跳过，避免 `ValueError: min() iterable argument is empty`
- `_fix_drawing_rect` 副作用：在 rect 扩展前统计原始零高度线，避免检测永远失效

**下一步方向**（按优先级）：
1. **收紧 TABLE_CAPTION_RE**：排除包含 describes/shows/lists/gives/provides 等动词的引用句（进一步减少不必要的全页扫描）
2. **跳过跨页续表**：检测 "Continued from previous page" 直接跳过 find_tables（预估可跳过 ~33 页）
3. **自研 horizontal 表格提取**：不依赖 `find_tables`，基于文本块对齐分析实现 AMBA 风格表格识别（投入大，优先级低）
4. **to_markdown 性能**：TI 中 to_markdown 耗时比 find_tables 还高 31%，但为 PyMuPDF C 实现，外部优化空间极小
