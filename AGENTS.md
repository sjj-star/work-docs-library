# work-docs-library Agent 指南

> 本文档供 AI Coding Agent 阅读。详细说明见 `README.md`。

## 项目概述

`work-docs-library` 是面向技术文档（PDF、Word、Excel）的自动化知识提取与检索 pipeline，以 **Kimi Code CLI Plugin** 形式运行。

核心能力：多格式文档解析、SQLite 结构化存储、FAISS 向量语义索引、LLM 自动摘要、Agent 批量协作工作流（checkpoint/resume）。

项目主要使用 **中文** 编写注释与文档。

---

## 技术栈与运行时架构

| 层级 | 依赖 |
|------|------|
| 解析 | `pymupdf`、`python-docx`、`openpyxl` |
| 向量检索 | `faiss-cpu`、`numpy` |
| 数据库 | SQLite（标准库 `sqlite3`） |
| LLM API | `requests`（OpenAI-compatible HTTP API） |
| 环境配置 | `python-dotenv` + `config.json`（三层优先级） |
| 测试 | `pytest` |

- **Python 版本要求**：>= 3.11
- **平台**：主要测试于 Linux，兼容 macOS/Windows

### 双模式运行时架构

系统根据配置自动选择两种互斥的操作模式：

1. **LLM API Flow**（高质量处理模式）
   - 前提：同时配置了 LLM 对话模型和 Embedding 模型
   - 行为：调用 `LLMAPIIngestionPipeline`，执行**两阶段处理**：
     - **Phase A**（Parse & Embed）：解析 → 存储 chunks → 嵌入 → `status="embedded"`
     - **Phase B**（LLM Enhance）：层次化文本总结、图像分析、章节摘要 → 持久化到 DB → `status="done"`
   - **断点续传**：若 Phase B 中断，再次 `ingest` 会检测到已有 `embedded` chunks 并跳过 Phase A 直接续传
   - 异常处理：任何阶段失败时文档状态变为 `"failed"`

2. **Agent Skill Flow**（高效批处理模式）
   - 前提：仅配置了 Embedding 模型
   - 行为：调用 `CompatibilityIngestionPipeline`，仅执行向量化，不自动总结
   - 后续需通过 `agent_batch_helper.py` 将 pending chunk 分批导出，由外部 Agent 阅读并回写摘要

```
文档 → Parser → Document/Chunk → SQLite
                 ↓
          FlowSelector 选择模式
                 ↓
    LLM API Flow  ←→  Agent Skill Flow
                 ↓
         Embedding → FAISS 向量索引
                 ↓
       Query / Search CLI / Plugin Tool
```

### 存储布局

- `knowledge_base/workdocs.db` — SQLite 数据库（4 张表）
- `knowledge_base/faiss.index` — FAISS 向量索引
- `knowledge_base/id_map.json` — FAISS 内部 ID 到 chunk 的映射
- `knowledge_base/images/<file_hash>/` — 从 PDF 提取的图片
- `auto_batches/<doc_id>/` — `agent_batch_helper.py` 生成的批次文件（checkpoint/resume）

> 完整数据库 Schema、数据流向、查询接口、FAISS 细节见 `README.md` → **数据库与存储架构**。

---

## 配置系统

三层优先级（高 → 低）：**环境变量(Kimi CLI注入) > `config.json` > `.env`**。

- `config.json`：主配置入口，Kimi CLI 通过 `plugin.json` 的 `inject` 自动注入 OAuth token。被 `.gitignore` 忽略。
- `config.example.json`：配置模板（提交到仓库）。
- `.env`：开发/独立运行回退配置。

`_resolve_config(env_name, json_path, default)` 实现解析逻辑，完整说明见 `README.md` → **配置说明**。

---

## 数据库 Schema（开发者速查）

### 核心表

| 表 | 关键字段 | 开发注意 |
|---|----------|----------|
| `documents` | `doc_id` (MD5), `source_path`, `status` | `status`: `pending` → `processing` → `done`/`failed` |
| `chunks` | `id` (PK=chunk_db_id), `doc_id`, `content`, `metadata`, `status` | `metadata` 存 embedding、images、vision_desc；`status`: `pending` → `embedded` → `done` |
| `chapter_summaries` | `doc_id` + `chapter_title`, `summary`, `concepts`, `relationships` | `upsert_chapter_summary()` 使用 `ON CONFLICT` 更新 |
| `concept_index` | `doc_id` + `concept_name`, `definition`, `first_mentioned_page` | `upsert_concept()` 使用 `ON CONFLICT IGNORE` |

> **⚠️ 状态语义差异**：`documents.status="done"` 在 LLM API Flow 中表示全阶段完成；在 Agent Skill Flow 中表示嵌入完成（后续摘要由外部 Agent 补充）。

### 查询接口速查

| 方法 | 用途 |
|------|------|
| `query_by_page(doc_id, ps, pe)` | 页码重叠查询 |
| `query_by_chapter(doc_id, title)` | 章节标题 LIKE 匹配 |
| `query_by_keyword(keyword)` | 关键词 LIKE 搜索 |
| `query_by_concept(doc_id, concept)` | 概念名联合搜索 |
| `get_chunk_by_db_id(db_id)` | 精确单条 |
| `get_embedded_but_unsummarized_chunks(doc_id?)` | **Agent Batch 核心**：获取待总结的 chunks |
| `get_pending_chunks(doc_id?)` | 获取尚未嵌入的 chunks |
| `update_chunk_metadata(db_id, metadata)` | 更新 chunk metadata JSON |

> 完整 Schema 说明、表关系图、状态生命周期见 `README.md` → **数据库与存储架构**。

---

## 目录结构与代码组织

```
work-docs-library/
├── plugin.json                   # Kimi CLI Plugin 定义（10 个 tools，含 inject 配置）
├── config.example.json           # 配置模板
├── scripts/
│   ├── plugin_router.py          # Plugin 统一入口：stdin JSON → stdout JSON
│   ├── main.py                   # 独立 CLI 主入口
│   ├── doc_extractor.py          # 传统 CLI（ingest/status/query/search/toc…）
│   ├── agent_batch_helper.py     # Agent 批量协作 CLI（list/dump/apply/filter/progress/auto）
│   ├── requirements.txt          # Python 依赖
│   ├── .env.example / .env       # 环境变量模板 / 实际配置（gitignored）
│   ├── prompts/
│   │   ├── summarize.txt         # LLM chunk 摘要提示词
│   │   ├── structural_summarize.txt  # 章节级结构化摘要提示词
│   │   └── filter_config.json    # 低价值内容过滤规则
│   ├── core/                     # 业务逻辑层
│   │   ├── config.py             # 配置中心（三层优先级解析）
│   │   ├── flow_selector.py      # 双模式流程选择器
│   │   ├── models.py             # 数据模型：Chapter、Chunk、Document
│   │   ├── db.py                 # KnowledgeDB：SQLite 增删改查
│   │   ├── vector_index.py       # VectorIndex：FAISS 管理
│   │   ├── pipeline.py           # IngestionPipeline：通用摄入管道
│   │   ├── llm_api_pipeline.py   # LLMAPIIngestionPipeline（LLM API Flow）
│   │   ├── compatibility_pipeline.py  # CompatibilityIngestionPipeline（Agent Skill Flow）
│   │   ├── llm_chat_client.py    # LLMChatClient：对话/总结/图像分析
│   │   ├── embedding_client.py   # EmbeddingClient：向量化
│   │   ├── llm_client.py         # 旧版通用客户端（向后兼容）
│   │   ├── context_manager.py    # 上下文窗口管理
│   │   └── chapter_editor.py     # 交互式章节编辑器
│   ├── parsers/                  # IO / 解析层
│   │   ├── pdf_parser.py         # PDF 解析器（pymupdf）
│   │   ├── office_parser.py      # DOCX / XLSX 解析
│   │   └── image_utils.py        # 图片压缩
│   └── tests/                    # pytest 测试集（179 个用例）
└── venv/                         # Python 虚拟环境
```

### 核心模块职责速查

| 模块 | 关键类/函数 | 职责 |
|------|-------------|------|
| `core/config.py` | `Config` | 统一配置中心，三层优先级：环境变量 > `config.json` > `.env` |
| `core/flow_selector.py` | `FlowSelector` | 根据配置自动选择 `LLM_API_FLOW` 或 `AGENT_SKILL_FLOW` |
| `core/models.py` | `Chapter`, `Chunk`, `Document` | 领域模型，使用 `dataclass` 定义 |
| `core/db.py` | `KnowledgeDB` | SQLite 操作，参数化查询防注入 |
| `core/vector_index.py` | `VectorIndex` | FAISS 索引封装，维度不匹配时自动重建 |
| `core/pipeline.py` | `IngestionPipeline` | 通用摄入管道：扫描 → 解析 → chunk → 嵌入 → 入库 |
| `core/llm_api_pipeline.py` | `LLMAPIIngestionPipeline` | LLM API Flow 专用，两阶段处理，支持断点续传 |
| `core/compatibility_pipeline.py` | `CompatibilityIngestionPipeline` | Agent Skill Flow 兼容管道，仅向量化 |
| `core/llm_chat_client.py` | `LLMChatClient` | 对话客户端，支持聊天、总结、vision 描述、思考模式 |
| `core/embedding_client.py` | `EmbeddingClient` | 向量化客户端，支持批处理、首次维度验证 |
| `parsers/pdf_parser.py` | `PDFParser` | PDF 文本/图片/矢量图提取，含图表区域启发式识别 |
| `parsers/office_parser.py` | `OfficeParser` | DOCX / XLSX 解析（DOCX 目前解析为单 chunk） |

---

## 构建与测试

```bash
# 环境准备
cd /path/to/work-docs-library
python3 -m venv venv
source venv/bin/activate
pip install -r scripts/requirements.txt

# 配置
cp scripts/.env.example scripts/.env
# 或复制 config.example.json → config.json（Plugin 模式）

# 验证配置
PYTHONPATH=scripts ./venv/bin/python scripts/main.py --validate-config dummy_path

# 运行全部测试（179 个用例）
PYTHONPATH=scripts ./venv/bin/python -m pytest scripts/tests/ -v

# 独立 CLI
PYTHONPATH=scripts ./venv/bin/python scripts/main.py /path/to/document.pdf --verbose
PYTHONPATH=scripts ./venv/bin/python scripts/doc_extractor.py ingest --path ./docs
PYTHONPATH=scripts ./venv/bin/python scripts/agent_batch_helper.py auto --doc-id <HASH> --filter
```

> 完整 CLI 参考、程序化使用示例见 `README.md` → **CLI 参考 / 程序化使用**。

---

## 代码风格指南

### 日志规范
- **诊断/进度/错误信息统一使用 `logging`**，格式：`"%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"`
- CLI 格式化结果展示保留 `print()`，禁止随意使用 `print()` 输出调试信息

### 类型与模型
- 领域模型使用 `dataclass`（`Chapter`、`Chunk`、`Document`）
- `Chunk.status`：`pending`、`embedded`、`done`、`skipped`、`failed`
- `Document.status`：`pending`、`processing`、`done`、`failed`

### 数据库安全
- **所有 SQL 必须使用参数化查询**（`?` 占位符），禁止字符串拼接
- `KnowledgeDB._connect()` 使用上下文管理器管理连接生命周期

### 路径安全
- 文件读取操作需校验路径在 skill 根目录内：`Path(path).resolve().relative_to(skill_root)`

### 配置访问
- 统一通过 `core.config.Config` 读取配置
- 数值型配置通过 `_initialize_numeric_configs()` 初始化，无效值自动回退到默认值

### 向后兼容
- `core/llm_client.py` 中的旧版客户端保留，供存量代码导入
- 新代码优先使用 `core/llm_chat_client.py` 和 `core/embedding_client.py`

---

## 测试策略

| 测试文件 | 说明 |
|----------|------|
| `test_pdf_parser.py` | 图表提取核心测试（44+ 用例） |
| `test_plugin_router.py` | FlowSelector、get_content、失败恢复、断点续传 |
| `test_agent_batch_helper.py` | 批处理、断点续传、过滤规则、race condition |
| `test_pipeline.py` | 文档处理流程测试 |
| `test_db.py` | SQLite 操作、事务管理 |
| `test_vector_index.py` | FAISS 索引增删查、持久化、维度迁移 |
| `test_llm_client.py` | LLM 对话、总结、图像分析 Mock |
| `test_dual_client.py` | 双客户端架构、配置验证、Kimi 适配 |
| `test_integration.py` | 端到端集成测试 |
| `test_config_json.py` | 配置优先级、凭证注入、文件加载 |

### Mock 规范
- 测试 LLM/Embedding 客户端时，**禁止直接调用真实 API**
- 使用 `monkeypatch.setattr(Config, ...)` 而非 `monkeypatch.setenv` 来 mock 配置，避免 `.env` 加载顺序导致 mock 失效

> 完整测试分类、性能指标、稳定性分析见 `README.md` → **开发与测试**。

---

## Agent Batch 机制（开发者必读）

`batch_*.txt` 是 Agent Skill Flow 的核心协作媒介——由脚本生成、由 Agent 阅读、由 Agent 产出 `batch_*.json` 回写数据库。

### 核心流程

1. `run_auto_summarize()` 过滤低价值 chunk，按同章节优先分组，写入 `batch_001.txt`
2. Agent 阅读 txt，产出 `batch_001.json`：`[{chunk_db_id, summary, keywords, entities, relationships}]`
3. 再次调用 `run_auto_summarize()`：发现 json 存在 → `_apply_json_file()` → 回写 DB → 删除 json → 继续下一 batch

### 关键设计

- **分组逻辑**：`_smart_batch()` 同章节优先，目标 25000 字符/12 个 chunk，尾部小 batch 合并
- **图片注入**：`_enrich_batch_with_images()` 检查 `metadata["images"]`，若无 `vision_desc` 则提示 `AGENT VISION REQUIRED`
- **断点续传**：`checkpoint.json` 记录 batch_map 和已应用批次，resume 时直接续传
- **文件生命周期**：全部完成后 txt/json/checkpoint 全部清理

> 完整流程图、文件格式详细说明、闭环机制见 `README.md` → **Agent Batch 机制详解**。

---

## 章节综合机制（`synthesize_chapters`）

`synthesize_chapters` 与 `auto_summarize` 是**两个独立的 Agent 协作流水线**，执行顺序上 `auto_summarize` 先、`synthesize_chapters` 后。

| 对比项 | `auto_summarize` | `synthesize_chapters` |
|--------|-----------------|----------------------|
| **处理对象** | 单个 chunk（`status='embedded'`） | 整个章节（多个 `status='done'` 的 chunk 聚合） |
| **输入** | chunk 原始 `content` | chunk 的 `summary`、`entities`、`relationships`、`vision_insights` |
| **输出** | chunk 级 `summary` + `keywords` | 章节级结构化摘要：`summary`、`concepts`、`relationships`、`key_figures`、`key_tables` |
| **写入表** | `chunks` | `chapter_summaries` + `concept_index` |
| **文件前缀** | `batch_*.txt` / `batch_*.json` | `chapter_synthesis_*.txt` / `chapter_synthesis_*.json` |
| **Checkpoint** | `checkpoint.json` | `chapter_checkpoint.json` |

> 完整对比说明、txt/json 格式详细说明见 `README.md` → **章节综合机制**。

---

## 已知限制与常见陷阱

1. **DOCX 单 chunk 限制**：`.docx` 文件目前被解析为单个 `Chunk`，超大文档可能导致嵌入/token 超限。
2. **矢量图提取**：PDF 中的矢量图不会通过 `page.get_images()` 直接提取。项目通过识别 `Figure X-X.` 标题来渲染周围区域作为补偿。
3. **FAISS 与 SQLite 非原子**：极端情况下（进程崩溃、磁盘满），可能出现 FAISS 索引与 SQLite 元数据不一致。可通过 `reprocess` 命令重建文档解决。
4. **Vision API 开销**：开启 `auto_vision=1` 后，每个包含图片的页面都会调用一次 Vision API，费用可能较高。
5. **LLM 客户端 timeout**：默认 HTTP timeout 为 120 秒，已增加 3 次指数退避重试（1s / 2s / 4s）。
6. **Embedding 维度不可变**：FAISS 索引创建后维度固定。更换模型导致维度变化时，必须删除旧索引并重新处理文档。
7. **过滤规则**：`agent_batch_helper.py filter/auto` 依赖 `scripts/prompts/filter_config.json`，修改规则后无需重启，下次运行自动生效。
8. **后台运行限制**：Kimi CLI 插件架构为单次 subprocess 调用，无法真正后台运行。超长文档依赖断点续传降低重试成本。

> 安全性分析、代码风格详细分析见 `README.md` → **功能稳定性 / 安全性 / 代码风格分析**。

---

## Plugin 工具速查

| Tool | 说明 | 关键参数 |
|------|------|----------|
| `ingest` | 提取并存储文档 | `path`, `dry_run`, `auto_chapter`, `agent_mode` |
| `search` | 语义向量搜索 | `text`, `top_k` |
| `query` | 结构化查询 | `doc_id`, `page`, `chapter`, `keyword`, `concept` |
| `get_content` | 获取完整未截断内容 | `doc_id` + `page`/`chapter`/`chunk_db_id` |
| `status` | 列出所有文档 | 无 |
| `toc` | 目录/标题搜索 | `doc_id` 或 `match` |
| `auto_summarize` | Agent 批量总结流水线 | `doc_id`, `batch_size`, `filter` |
| `synthesize_chapters` | 章节综合 | `doc_id` |
| `progress` | 处理进度 | `doc_id` |
| `reprocess` | 强制重新处理 | `doc_id` |

**注意**：`plugin.json` 中所有 tool 的 `command` 固定为 `venv/bin/python3`，不依赖系统 PATH。

---

## 相关文件索引

| 文件 | 作用 |
|------|------|
| `plugin.json` | Kimi CLI Plugin 工具定义（10 个 tools，含 inject 配置） |
| `config.example.json` | 配置模板 |
| `scripts/plugin_router.py` | Plugin 运行时路由（stdin JSON → stdout JSON） |
| `scripts/main.py` | 独立 CLI 主入口 |
| `scripts/doc_extractor.py` | 传统 CLI（子命令：ingest/status/query/search/toc…） |
| `scripts/agent_batch_helper.py` | Agent 批量协作 CLI（子命令：list/dump/apply/filter/progress/auto） |
| `scripts/core/config.py` | 配置中心（三层优先级解析） |
| `scripts/core/flow_selector.py` | 双模式流程选择器 |
| `scripts/core/models.py` | 数据模型 |
| `scripts/core/db.py` | SQLite 数据库操作 |
| `scripts/core/vector_index.py` | FAISS 向量索引 |
| `scripts/core/pipeline.py` | 通用摄入管道 |
| `scripts/core/llm_api_pipeline.py` | LLM API Flow 专用管道 |
| `scripts/core/compatibility_pipeline.py` | Agent Skill Flow 兼容管道 |
| `scripts/core/llm_chat_client.py` | LLM 对话客户端 |
| `scripts/core/embedding_client.py` | Embedding 向量化客户端 |
| `scripts/parsers/pdf_parser.py` | PDF 解析器 |
| `scripts/parsers/office_parser.py` | Office 解析器 |
| `scripts/prompts/filter_config.json` | 低价值 chunk 过滤规则 |
| `scripts/requirements.txt` | Python 依赖清单 |
| `scripts/.env.example` | 环境变量模板 |
