# work-docs-library Agent 指南

> 本文档供 AI Coding Agent 阅读。假设读者对项目一无所知。详细说明见 `README.md`。

## 项目概述

`work-docs-library` 是一个面向技术文档（PDF、Word、Excel）的自动化知识提取与检索 pipeline，以 **Kimi Code CLI Plugin（Skill）** 形式运行。

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
| 环境配置 | `python-dotenv` |
| 测试 | `pytest` |

- **Python 版本要求**：>= 3.11
- **平台**：主要测试于 Linux，兼容 macOS/Windows

### 双模式运行时架构

系统根据 `.env` 配置自动选择两种互斥的操作模式：

1. **LLM API Flow**（高质量处理模式）
   - 前提：同时配置了 LLM 对话模型和 Embedding 模型
   - 行为：调用 `LLMAPIIngestionPipeline`，执行**两阶段处理**：
     - **Phase A**（Parse & Embed）：解析 → 存储 chunks → 嵌入 → `status="embedded"`
     - **Phase B**（LLM Enhance）：层次化文本总结、图像分析、章节摘要 → 持久化到 DB → `status="done"`
   - **断点续传**：若 Phase B 中断，再次 `ingest` 会检测到已有 `embedded` chunks 并跳过 Phase A 直接续传
   - 异常处理：任何阶段失败时文档状态变为 `"failed"`，不会卡住 `"processing"`

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

- `knowledge_base/workdocs.db` — SQLite 数据库（4 张表：documents、chunks、chapter_summaries、concept_index）
- `knowledge_base/faiss.index` — FAISS 向量索引
- `knowledge_base/id_map.json` — FAISS 内部 ID 到 chunk 的映射
- `knowledge_base/images/<file_hash>/` — 从 PDF 提取的图片
- `auto_batches/<doc_id>/` — `agent_batch_helper.py` 生成的批次文件（checkpoint/resume）

> 完整数据库 Schema、数据流向、查询接口、FAISS 细节见 `README.md` → **数据库与存储架构**。

---

## 数据库 Schema（开发者速查）

### `documents` — 文档元数据
| 字段 | 类型 | 开发注意 |
|------|------|----------|
| `doc_id` (PK) | `TEXT` | MD5 内容哈希，也是 `reprocess` 定位键 |
| `source_path` (UNIQUE) | `TEXT` | `get_document_by_path()` 的查询键 |
| `chapters` | `TEXT` (JSON) | 无目录时默认为 `[{"title":"全文","start_page":1,"end_page":N}]` |
| `chapters_override` | `TEXT` (JSON) | 人工覆盖，优先级高于 `chapters` |
| `status` | `TEXT` | `pending` → `processing` → `done` / `failed` / `embedded` |

### `chunks` — 内容块（核心表）
| 字段 | 类型 | 开发注意 |
|------|------|----------|
| `id` (PK) | `INTEGER AUTOINCREMENT` | **这就是 `chunk_db_id`**，FAISS `id_map` 映射目标，也是 `get_content`/`apply` 的键 |
| `doc_id` (FK) | `TEXT` | → `documents(doc_id)` |
| `chunk_id` | `TEXT` | 逻辑 ID，如 `page_3_text` |
| `content` | `TEXT` | **原始提取内容，永不覆盖** |
| `chunk_type` | `TEXT` | `text` / `table` / `image_desc` / `summary` |
| `chapter_title` | `TEXT` | 无目录时默认为 `"全文"` |
| `keywords` | `TEXT` | JSON list 或逗号分隔 |
| `summary` | `TEXT` | Phase B 或 Agent 回写后才填充 |
| `metadata` | `TEXT` (JSON) | `{"embedding": [...], "images": [{"path":"...","vision_desc":"..."}]}` |
| `status` | `TEXT` | `pending` → `embedded` → `done` / `skipped` / `failed` |

**索引**：`idx_chunks_doc(doc_id)`、`idx_chunks_type(chunk_type)`、`idx_chunks_chapter(doc_id, chapter_title)`

### `chapter_summaries` — 章节级 LLM 摘要
| 字段 | 开发注意 |
|------|----------|
| `doc_id` + `chapter_title` | 复合定位；`upsert_chapter_summary()` 使用 `ON CONFLICT` 更新 |
| `summary` | `_persist_llm_outputs()` 写入 |
| `status` | `pending` / `done` |

### `concept_index` — 概念/关键词索引
| 字段 | 开发注意 |
|------|----------|
| `doc_id` + `concept_name` (UNIQUE) | `upsert_concept()` 使用 `ON CONFLICT IGNORE` |

### 数据流向（开发视角）

```
Parser → Document/Chunk → insert_chunk() → status="pending"
                          ↓
                    EmbeddingClient.embed()
                          ↓
                    update_chunk_embedding() + vec.add()
                    update_chunk_status("embedded")
                          ↓
              ┌───────────┴───────────┐
              ▼                       ▼
    LLM API Flow (Phase B)     Agent Skill Flow
    _llm_enhance_document()    agent_batch_helper
              ↓                       ↓
    _persist_llm_outputs()    _apply_json_file()
    update_chunk_summary()    update_chunk_summary()
    update_chunk_keywords()   update_chunk_keywords()
    update_chunk_status()     set_chunk_done()
    ("done")                  ("done")
```

### 查询接口速查（KnowledgeDB 方法）

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

> 完整 Schema 说明、表关系图、原始数据 vs LLM 增强数据存储位置、状态生命周期、FAISS 细节见 `README.md` → **数据库与存储架构**。

---

## 目录结构与代码组织

```
work-docs-library/
├── plugin.json                   # Kimi CLI Plugin 定义（10 个 tools）
├── scripts/
│   ├── plugin_router.py          # Plugin 统一入口：stdin JSON → stdout JSON
│   ├── main.py                   # 独立 CLI 主入口
│   ├── doc_extractor.py          # 传统 CLI（ingest/status/query/search/toc/chapter-edit…）
│   ├── agent_batch_helper.py     # Agent 批量协作 CLI（list/dump/apply/filter/progress/auto）
│   ├── requirements.txt          # Python 依赖
│   ├── .env.example / .env       # 环境变量模板 / 实际配置（gitignored）
│   ├── prompts/
│   │   ├── summarize.txt         # LLM chunk 摘要提示词
│   │   ├── structural_summarize.txt  # 章节级结构化摘要提示词
│   │   └── filter_config.json    # 低价值内容过滤规则
│   ├── core/                     # 业务逻辑层
│   │   ├── config.py             # 配置中心
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
│   └── tests/                    # pytest 测试集（166 个用例）
└── venv/                         # Python 虚拟环境
```

### 核心模块职责速查

| 模块 | 关键类/函数 | 职责 |
|------|-------------|------|
| `core/config.py` | `Config` | 统一读取 `.env`，所有配置项以 `WORKDOCS_` 为前缀 |
| `core/flow_selector.py` | `FlowSelector` | 根据配置自动选择 `LLM_API_FLOW` 或 `AGENT_SKILL_FLOW` |
| `core/models.py` | `Chapter`, `Chunk`, `Document` | 领域模型，使用 `dataclass` 定义 |
| `core/db.py` | `KnowledgeDB` | SQLite 操作，参数化查询防注入，`_connect()` 为上下文管理器 |
| `core/vector_index.py` | `VectorIndex` | FAISS 索引封装，维度不匹配时自动重建，兼容旧版 dict 格式 `id_map` |
| `core/pipeline.py` | `IngestionPipeline` | 通用摄入管道：扫描 → 解析 → chunk → 嵌入 → 入库 |
| `core/llm_api_pipeline.py` | `LLMAPIIngestionPipeline` | LLM API Flow 专用，两阶段处理，支持断点续传 |
| `core/compatibility_pipeline.py` | `CompatibilityIngestionPipeline` | Agent Skill Flow 兼容管道，仅向量化 |
| `core/llm_chat_client.py` | `LLMChatClient` | 对话客户端，支持聊天、总结、vision 描述、思考模式 |
| `core/embedding_client.py` | `EmbeddingClient` | 向量化客户端，支持批处理、首次维度验证 |
| `parsers/pdf_parser.py` | `PDFParser` | PDF 文本/图片/矢量图提取，含图表区域启发式识别 |
| `parsers/office_parser.py` | `OfficeParser` | DOCX / XLSX 解析（DOCX 目前解析为单 chunk） |

---

## 构建与测试命令

### 环境准备

```bash
cd /path/to/work-docs-library
python3 -m venv venv
source venv/bin/activate
pip install -r scripts/requirements.txt
```

### 配置环境变量

```bash
cp scripts/.env.example scripts/.env
# 编辑 scripts/.env，填入 API Key
```

配置加载顺序（后加载的优先级更高）：
1. `~/.kimi/plugins/work-docs-library/.env`
2. `~/.kimi/plugins/work-docs-library/scripts/.env`

### 验证配置

```bash
PYTHONPATH=scripts ./venv/bin/python scripts/main.py --validate-config dummy_path
```

### 运行测试

```bash
# 运行全部测试（166 个用例）
PYTHONPATH=scripts ./venv/bin/python -m pytest scripts/tests/ -v

# 运行单个测试文件
PYTHONPATH=scripts ./venv/bin/python -m pytest scripts/tests/test_pdf_parser.py -v
```

### 独立 CLI 使用

```bash
# 处理文档（自动选择模式）
PYTHONPATH=scripts ./venv/bin/python scripts/main.py /path/to/document.pdf --verbose

# 传统 CLI
PYTHONPATH=scripts ./venv/bin/python scripts/doc_extractor.py ingest --path ./docs
PYTHONPATH=scripts ./venv/bin/python scripts/doc_extractor.py search --text "AHB bus arbitration" --top-k 5

# Agent 批量摘要
PYTHONPATH=scripts ./venv/bin/python scripts/agent_batch_helper.py auto --doc-id <DOC_HASH> --output-dir ./auto_batches --filter
```

### Plugin 模式

项目通过 `plugin.json` 注册为 Kimi Code CLI Plugin，`plugin_router.py` 作为统一入口：
- 从 `stdin` 读取 JSON 参数
- `sys.argv[1]` 为 tool 名称（如 `ingest`、`search`、`auto_summarize`）
- 结果以 JSON 写入 `stdout`
- 日志写入 `stderr`

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
- 统一通过 `core.config.Config` 读取环境变量
- 数值型配置通过 `_initialize_numeric_configs()` 初始化，无效值自动回退到默认值

### 向后兼容
- `core/llm_client.py` 中的旧版客户端保留，供存量代码导入
- 新代码优先使用 `core/llm_chat_client.py` 和 `core/embedding_client.py`

---

## 测试策略

| 测试文件 | 说明 |
|----------|------|
| `test_pdf_parser.py` | 图表提取核心测试（44+ 用例） |
| `test_plugin_router.py` | FlowSelector、get_content、失败恢复、断点续传（12 用例） |
| `test_agent_batch_helper.py` | 批处理、断点续传、过滤规则、race condition |
| `test_pipeline.py` | 文档处理流程测试 |
| `test_db.py` | SQLite 操作、事务管理 |
| `test_vector_index.py` | FAISS 索引增删查、持久化、维度迁移 |
| `test_llm_client.py` | LLM 对话、总结、图像分析 Mock |
| `test_dual_client.py` | 双客户端架构、配置验证、Kimi 适配 |
| `test_integration.py` | 端到端集成测试 |

### Mock 规范
- 测试 LLM/Embedding 客户端时，**禁止直接调用真实 API**
- 使用 `monkeypatch.setattr(Config, ...)` 而非 `monkeypatch.setenv` 来 mock 配置，避免 `.env` 加载顺序导致 mock 失效

> 完整测试分类、真实 fixtures 列表、性能指标见 `README.md` → **开发与测试**。

---

## Agent Batch 机制详解（开发者必读）

`batch_*.txt` 是 **Agent Skill Flow** 的核心协作媒介——由脚本生成、由 Agent 阅读、由 Agent 产出 `batch_*.json` 回写数据库。

### 触发入口

`agent_batch_helper.py` 的 `run_auto_summarize()` 函数（被 `tool_auto_summarize` 调用），面向 `status='embedded' AND summary IS NULL` 的 chunks。

### 分组逻辑：`_smart_batch()`

```python
def _smart_batch(rows, target_chars=25000, max_chunks=12, min_chunks=3):
```

1. **同章节优先**：同一章节的 chunk 尽量放在同一 batch，只要总字符 ≤ `target_chars` 且 chunk 数 ≤ `max_chunks`
2. **大章节拆分**：超出限制时按顺序拆分
3. **尾部合并**：最后一个 batch 若 chunk 数 < `min_chunks`，合并到前一个 batch

输出：`batch_map = [[db_id1, db_id2, ...], ...]`，持久化到 `checkpoint.json`。

### txt 文件格式

```text
--- BATCH CHAPTER CONTEXT ---
Chapter: System Architecture
Previous chunk summary: The DMA controller uses...
--------------------------------------------------------------------------------

--- CHUNK_DB_ID=47 | page_3_text | System Architecture P3 ---
The AHB bus matrix supports up to 16 masters...

================================================================================
```

**关键元素**：
- `CHUNK_DB_ID=47`：来自 `chunks.id`，Agent 回传 JSON 时必须原样带回
- `BATCH CHAPTER CONTEXT`：同章节最近一个 `done` chunk 的 summary，让 Agent 保持理解连贯性
- 图片路径：`_enrich_batch_with_images()` 检查 `metadata["images"]`，若 `vision_desc` 不存在则提示 `AGENT VISION REQUIRED`

### checkpoint.json 结构

```json
{
  "doc_id": "...",
  "batch_map": [[47, 48, 49], [50, 51, 52]],
  "done_chunk_ids": [47, 48, 49],
  "total_batches": 5,
  "applied_batches": 1
}
```

**Resume 逻辑**：
- 若当前 `embedded` chunk ID 是 checkpoint `batch_map` 的子集 → 直接续传
- 若不是（filter 规则变更 / 重新嵌入）→ 废弃旧 checkpoint，重新分组

### 闭环流程

1. `run_auto_summarize()` 写入 `batch_001.txt`
2. Agent 阅读 txt，产出 `batch_001.json`：`[{"chunk_db_id": 47, "summary": "...", "keywords": "..."}]`
3. 再次调用 `run_auto_summarize()`：
   - 发现 `batch_001.json` 存在
   - `_apply_json_file()` → `update_chunk_summary()` + `update_chunk_keywords()` + `set_chunk_done()`
   - 删除 `batch_001.json`，继续 `batch_002.txt`

### 文件生命周期

| 阶段 | batch_001.txt | batch_001.json | checkpoint.json |
|------|---------------|----------------|-----------------|
| 初始生成 | ✅ | ❌ | ✅ |
| Agent 处理中 | ✅ | ✅ | ✅ |
| 自动应用后 | ✅ | ❌（已删除） | ✅ |
| 全部完成后 | ❌（全部删除） | ❌ | ❌ |

> 完整的 batch 机制流程图、文件格式详细说明、图片注入逻辑见 `README.md` → **Agent Batch 机制详解**。

---

## 已知限制与常见陷阱

1. **DOCX 单 chunk 限制**：`.docx` 文件目前被解析为单个 `Chunk`，超大文档可能导致嵌入/token 超限。
2. **矢量图提取**：PDF 中的矢量图不会通过 `page.get_images()` 直接提取。项目通过识别 `Figure X-X.` 标题来渲染周围区域作为补偿，若文档无 Figure Caption 可能遗漏。
3. **FAISS 与 SQLite 非原子**：极端情况下（进程崩溃、磁盘满），可能出现 FAISS 索引与 SQLite 元数据不一致。可通过 `reprocess` 命令重建文档解决。
4. **Vision API 开销**：开启 `WORKDOCS_AUTO_VISION=1` 后，每个包含图片的页面都会调用一次 Vision API，费用可能较高。
5. **LLM 客户端 timeout**：默认 HTTP timeout 为 120 秒，已增加 3 次指数退避重试（1s / 2s / 4s），但极大文档或极慢网络仍可能超时。
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

## 快速参考：环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `WORKDOCS_LLM_PROVIDER` | `openai` | LLM 提供商（`openai`/`kimi`/自定义） |
| `WORKDOCS_LLM_API_KEY` | 空 | LLM API 密钥 |
| `WORKDOCS_LLM_BASE_URL` | `https://api.openai.com/v1` | LLM API Base URL |
| `WORKDOCS_LLM_MODEL` | `gpt-4o-mini` | 对话模型名称 |
| `WORKDOCS_LLM_THINKING_ENABLED` | `0` | Kimi 思考模式开关 |
| `WORKDOCS_EMBEDDING_PROVIDER` | `openai` | Embedding 提供商 |
| `WORKDOCS_EMBEDDING_API_KEY` | 空 | Embedding API 密钥（可与 LLM 不同） |
| `WORKDOCS_EMBEDDING_BASE_URL` | 空 | Embedding API Base URL |
| `WORKDOCS_EMBEDDING_MODEL` | `text-embedding-3-small` | 嵌入模型名称 |
| `WORKDOCS_EMBEDDING_DIMENSION` | `1536` | 嵌入向量维度（作为 `dimensions` 参数传递） |
| `WORKDOCS_CONTEXT_STRATEGY` | `smart` | 上下文策略：`recent`/`keyword`/`smart`/`truncate` |
| `WORKDOCS_COST_OPTIMIZATION` | `balanced` | 成本优化：`aggressive`/`balanced`/`quality` |
| `WORKDOCS_IMAGE_MAX_EDGE` | `1024` | 图片压缩后最大边长（px） |
| `WORKDOCS_IMAGE_QUALITY` | `85` | JPEG 压缩质量 |
| `WORKDOCS_BATCH_SIZE` | `4` | 嵌入 API 批处理大小 |
| `WORKDOCS_AUTO_VISION` | `0` | 是否自动调用 Vision API 描述图片 |

---

## 相关文件索引

| 文件 | 作用 |
|------|------|
| `plugin.json` | Kimi CLI Plugin 工具定义（10 个 tools） |
| `scripts/plugin_router.py` | Plugin 运行时路由（stdin JSON → stdout JSON） |
| `scripts/main.py` | 独立 CLI 主入口 |
| `scripts/doc_extractor.py` | 传统 CLI（子命令：ingest/status/query/search/toc/list-pending/write-summary/reprocess…） |
| `scripts/agent_batch_helper.py` | Agent 批量协作 CLI（子命令：list/dump/apply/filter/progress/auto） |
| `scripts/core/config.py` | 配置中心 |
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
