# work-docs-library Agent 自主式开发策略文档

> 本文档供 AI Coding Agent 阅读，是项目权威技术参考与行动准则。
> - 项目详细介绍、安装与使用指南 → `README.md`
> - 技术方案决策、设计意图与权衡 → `DESIGN.md`
> - 本规范与上述两份文档**必须同步更新**，具体规则见下方「文档同步规范」。

---

## 项目概述

本项目是面向 **IC 前端设计技术文档（PDF）** 的自动化知识提取 Pipeline，以 **Kimi Code CLI Plugin** 形式运行。

核心能力：
- **文档解析**：BigModel Expert API + 本地 PyMuPDF fallback，TOC 驱动章节识别
- **知识图谱**：自动提取实体（Module、Register、Signal、Instruction 等）与关系，构建跨文档互通的 NetworkX 全局图谱
- **混合检索**：FAISS 稠密向量 + BM25 稀疏索引 RRF 融合，可选重排序
- **Batch API 架构**：LLM 提取走 Batch API 降低成本，Embedding 走同步单文本 API
- **增量更新**：按章节 `content_hash` 指纹比较，未变章节复用缓存
- **Agentic 工作流**：外部 Skill 编排 `search`/`explore`/`read`/`ingest`/`status` 五个原子 MCP 工具完成复杂任务

目标规模：数百个文档，每个几十到万页级。详细架构与安装使用见 `README.md` 和 `DESIGN.md`。

---

## AI Agent 原生插件原则

> 本插件是 Kimi Code 等 AI Agent 的 **MCP 扩展**，不是独立 SaaS。设计和新增功能时必须内化这一事实。

### 核心含义
1. **LLM 是外部 Agent 的资源，不是插件的固定成本**
   - 需要 LLM 评判或推理的能力（查询分解、结果综合、评估 judge、ReAct/Self-Ask 步骤）优先以 **Skill 编排 + 原子 MCP 工具** 实现，由外部 Agent 承担 LLM 调用成本，而不是在插件内部启动完整 Agent 运行时。
   - 示例：`AgenticSearchPlanner` 的查询分解、`LLMReranker` 的 passage scoring 均由外部 Skill 编排，插件只提供可组合的检索/读取/评分机制。
2. **复杂策略进 Skill，通用机制进代码**
   - Python 层只提供机制：检索器接口、chunk 读取、图谱 CRUD、评估指标计算、日志记录。
   - Skill 层编排策略：何时调用 `search` vs `explore`、如何组合多步结果、是否使用 `reranked`、如何综合答案。
3. **MCP 工具保持原子性与可组合性**
   - 每个 MCP 工具应像 Unix 工具一样完成一件事。
   - 当前仅暴露 5 个原子工具：`search` / `explore` / `read` / `ingest` / `status`。不在单个工具内部隐藏多轮 LLM 推理或状态机。
4. **状态安全优先于智能**
   - 数据变更类操作（实体/关系增删改、反馈、重建全局图、reprocess）保留在 `admin_tools.py`，不暴露为 MCP 写工具，保留人工/审计边界。
5. **可观测性面向 Agent**
   - 状态、日志、冲突、来源、评估结果均以结构化 JSON 返回，方便外部 Agent 做下一步决策，而不是仅面向人类阅读。

### 设计决策边界
- 需要新增 LLM 调用流程时，先问：**这个流程应该由外部 Agent 通过 Skill 编排，还是必须由插件内部无干预完成？**
- 只有当流程需要批处理、离线运行、或无 Agent 在场时，才在插件内部实现 LLM 调用。
- **当前无例外**：MCP 工具面保持 5 个原子工具；插件内部不做 LLM 合成或智能路由。

---

## 技术栈与运行时架构

### 四存储系统
| 存储 | 职责 | 持久化文件 |
|------|------|-----------|
| **SQLite** | 文档元数据、content_blocks、heading_maps、usage_logs | `knowledge_base/workdocs.db` |
| **FAISS** | 向量索引 | `knowledge_base/faiss.index` |
| **NetworkX** | 全局知识图谱 | `knowledge_base/graphs/{doc_id}.json` + `global.json` |
| **Bridge** | block ↔ 实体 双向索引 | 纯内存（重启从 SQLite 重建） |

### Pipeline 六阶段
解析 → 构建 Batch JSONL → 提交 Batch API → 解析入库 → 构建 Embedding JSONL → 同步 Embedding。详见 `DESIGN.md` 第 19 章。

依赖与版本见 `pyproject.toml`。

---

## 目录结构与代码组织

```
work-docs-library/
├── kimi.plugin.json              # 插件 Manifest
├── skills/                       # Agent Skill 文档
│   ├── using-workdocs/           # 入口 Skill
│   ├── ingesting-workdocs/       # 入库工作流
│   ├── exploring-workdocs/       # 查询工作流
│   ├── agentic-search/           # 多跳检索工作流
│   ├── synthesizing-workdocs/    # 综合报告工作流
│   └── fixing-workdocs/          # 错误修正工作流
├── scripts/
│   ├── mcp_server.py             # MCP stdio server
│   ├── plugin_router.py          # 工具函数库
│   ├── admin_tools.py            # 内部管理命令
│   ├── prompts/                  # LLM 提示词
│   ├── core/                     # 业务逻辑层
│   ├── parsers/                  # 解析层
│   └── tests/                    # pytest 测试集
└── knowledge_base/               # 运行时生成数据（❌ 禁止手动修改）
```

模块职责见源码 docstring，详细架构见 `DESIGN.md`。

---

## 构建与测试命令

```bash
# 提交前必须执行
cd /path/to/work-docs-library
./.venv/bin/ruff check scripts/
./.venv/bin/ruff format scripts/
./.venv/bin/pyright scripts/
PYTHONPATH=scripts ./.venv/bin/python -m pytest scripts/tests/ -v
```

环境安装见 `README.md`「安装」。

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
- `Chunk.chunk_type`：`text` / `table` / `image_desc`（`ChunkType` StrEnum）
- `Document.status`：`pending` → `processing` → `batch_submitted` → `done` / `failed`（`DocumentStatus` StrEnum）
- `GraphEntity`/`GraphRelation` 字段：`entity_type`/`rel_type`、`name`、`properties`、`doc_properties`（按文档原始属性快照）、`source_doc_ids`、`source_chapter`、`confidence`、`verified`、`created_at`、`updated_at`、`feedback_score`；其中 `confidence`/`feedback_score` 仅作参考，不用于默认排序或过滤
- 跟踪与审计表：`usage_logs`（统一记录工具调用、向量/实体/关系命中、问题标记）、`block_activation`（向量 block 命中计数）

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
- **536 个测试用例必须全部通过**（0 skipped）

### 测试文件清单

> 完整的测试文件清单与分类见 `README.md`「开发与测试」（唯一权威详表），此处不再重复维护，避免双份漂移。

**当前状态**：536 passed, 0 skipped, 0 failed。

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
| 配置系统与完整配置项速查 | `README.md`「配置说明」 | 三层优先级、全部活跃配置项（以 README 配置表为准） |
| 环境隔离与测试基础设施 | `README.md`「开发与测试」 | conftest.py 三重隔离机制 |
| 官方 API 开发文档链接 | `README.md`「参考资源」 | Kimi / BigModel API 文档 |

---

## 开发计划

### 当前阶段（已完成）
- ✅ Pipeline：Batch API、章节树、content_blocks/heading_maps 双粒度、六阶段拆分
- ✅ 解析：BigModel Expert + 本地 PyMuPDF fallback、TOC 驱动章节、表格/图片检测、GapsFirstScanner + BorderlessTableExtractor
- ✅ 存储：SQLite + FAISS IndexIDMap2 + NetworkX 全局图 + Bridge 双向索引
- ✅ 检索：语义搜索、BM25 稀疏索引、RRF 混合检索、可选重排序
- ✅ 接口：`KnowledgeBaseService` 统一服务层；MCP 仅暴露 `search`/`explore`/`read`/`ingest`/`status` 5 个原子工具
- ✅ 质量：`confidence`/`verified`/`feedback_score`、冲突日志、`usage_logs`/`block_activation` 使用跟踪、`fixing-workdocs` 修正闭环
- ✅ API 错误处理：统一 `APIClient` + Provider + `RetryPolicy`，429/5xx 重试，401/402/403/配额类 429 快速失败；Embedding 超长拆分、单条失败隔离
- ✅ Skill：`using-workdocs`、`ingesting-workdocs`、`exploring-workdocs`、`agentic-search`、`synthesizing-workdocs`、`fixing-workdocs`
- ✅ 测试：529 passed, 0 skipped, 0 failed

详细历史见 git log 与 `DESIGN.md` 各章节。

### 下一阶段（精确到下一步）
1. **可视化**：图谱可视化导出（Graphviz / D3.js）
2. **评估体系细化**：评测数据集沉淀、端到端 RAG 评估流程产品化
3. **DOCX/XLSX 接入 pipeline**
4. **系统化加固**：跨存储事务回滚、VectorIndex 原子重建、参数限流、fuzz/property 测试

### PDF Parser 表格检测
- Grid 表格用 `find_tables(strategy="lines_strict")`；Horizontal（AMBA 等零高度线）用 `BorderlessTableExtractor`
- 各文档空跑率与下一步优化（收紧 TABLE_CAPTION_RE、跳过跨页续表等）详见 `DESIGN.md` 第 26 章（唯一源头，避免双份维护）
- `to_markdown` 性能：TI 中 to_markdown 耗时比 find_tables 还高 31%，但为 PyMuPDF C 实现，外部优化空间极小
