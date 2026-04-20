# work-docs-library Agent 指南

> 本文档供 AI Coding Agent 阅读。假设读者对项目一无所知。

## 项目概述

`work-docs-library` 是一个面向技术文档（PDF、Word、Excel）的自动化知识提取与检索 pipeline，以 **Kimi Code CLI Plugin（Skill）** 形式运行。

核心能力：
- 多格式文档解析（PDF 含图片/矢量图区域提取、DOCX、XLSX）
- 结构化存储（SQLite 存储文档元数据、章节、文本块 chunk）
- 向量语义索引（FAISS + Embedding API）
- LLM 自动摘要与关键词生成
- Agent 批量协作工作流（checkpoint/resume 的长文档摘要流水线）

项目主要使用 **中文** 编写注释与文档。

---

## 技术栈与运行时架构

### 技术栈

| 层级 | 依赖 |
|------|------|
| 解析 | `pymupdf`（PDF 主解析）、`pdfplumber`、`PyPDF2`、`python-docx`、`openpyxl` |
| 图像 | `Pillow`（压缩、格式转换） |
| 向量检索 | `faiss-cpu`、`numpy` |
| 数据库 | SQLite（标准库 `sqlite3`） |
| LLM API | `requests`（直接调用 OpenAI-compatible HTTP API） |
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
     - **Phase B**（LLM Enhance）：层次化文本总结、图像详细分析、章节摘要生成 → 持久化到 DB → `status="done"`
   - **断点续传**：若 Phase B 中断，再次 `ingest` 会检测到已有 `embedded` chunks 并跳过 Phase A 直接续传
   - 异常处理：任何阶段失败时文档状态变为 `"failed"`，不会卡住 `"processing"`
   - 典型耗时：单章节总结 5–10 秒，层次化总结 15–30 秒，图像分析 3–5 秒/图

2. **Agent Skill Flow**（高效批处理模式）
   - 前提：仅配置了 Embedding 模型（LLM 配置缺失或被注释）
   - 行为：调用 `CompatibilityIngestionPipeline`，仅执行向量化，不自动总结
   - 后续需通过 `agent_batch_helper.py` 将 pending chunk 分批导出，由外部 Agent 阅读并回写摘要

**数据流**：
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

- `knowledge_base/workdocs.db` — SQLite 数据库（文档、chunk、章节摘要、概念索引）
- `knowledge_base/faiss.index` — FAISS 向量索引
- `knowledge_base/id_map.json` — FAISS 内部 ID 到 chunk 的映射
- `knowledge_base/images/<file_hash>/` — 从 PDF 提取的图片
- `auto_batches/<doc_id>/` — `agent_batch_helper.py` 生成的批次文件（checkpoint/resume）

---

## 目录结构与代码组织

```
work-docs-library/
├── plugin.json                   # Kimi CLI Plugin 定义（tools 接口）
├── scripts/
│   ├── plugin_router.py          # Plugin 统一入口：读取 stdin JSON → 调用对应 tool → stdout JSON
│   ├── main.py                   # 独立 CLI 主入口（支持 --validate-config、--dry-run）
│   ├── doc_extractor.py          # 传统 CLI（ingest/status/query/search/toc/chapter-edit 等子命令）
│   ├── agent_batch_helper.py     # Agent 批量协作 CLI（list/dump/apply/filter/progress/auto）
│   ├── auto_batch_summarizer.py  # 自动化批次总结辅助逻辑
│   ├── requirements.txt          # Python 依赖
│   ├── .env.example              # 环境变量模板
│   ├── .env                      # 实际配置（gitignored，优先加载）
│   ├── prompts/
│   │   ├── summarize.txt         # LLM chunk 摘要提示词
│   │   ├── structural_summarize.txt  # 章节级结构化摘要提示词
│   │   └── filter_config.json    # 低价值内容过滤规则（always_skip / heuristic_skip）
│   ├── core/                     # 业务逻辑层
│   │   ├── config.py             # 配置中心：读取 .env，定义 Config 类
│   │   ├── flow_selector.py      # 智能流程选择器（LLM_API_FLOW / AGENT_SKILL_FLOW）
│   │   ├── models.py             # 数据模型：Chapter、Chunk、Document（dataclass）
│   │   ├── db.py                 # KnowledgeDB：SQLite 增删改查、事务管理
│   │   ├── vector_index.py       # VectorIndex：FAISS 加载、添加、删除、搜索、持久化
│   │   ├── pipeline.py           # IngestionPipeline：通用扫描→解析→chunk→嵌入→入库
│   │   ├── llm_api_pipeline.py   # LLMAPIIngestionPipeline（LLM API Flow 专用）
│   │   ├── compatibility_pipeline.py  # CompatibilityIngestionPipeline（Agent Skill Flow）
│   │   ├── llm_chat_client.py    # LLMChatClient：对话/总结/图像分析专用客户端
│   │   ├── embedding_client.py   # EmbeddingClient：向量化专用客户端（批处理、动态维度）
│   │   ├── llm_client.py         # 旧版通用客户端（_BaseClient / ChatClient / EmbeddingClient），保持向后兼容
│   │   ├── context_manager.py    # 上下文窗口管理（recent/keyword/smart/truncate 策略）
│   │   └── chapter_editor.py     # ChapterEditor：基于 input() 的交互式章节增删改
│   ├── parsers/                  # IO / 解析层
│   │   ├── pdf_parser.py         # PDFParser：基于 pymupdf，含大量启发式图表区域识别
│   │   ├── office_parser.py      # OfficeParser：DOCX / XLSX 解析
│   │   └── image_utils.py        # 图片压缩和格式转换
│   └── tests/                    # pytest 测试集
│       ├── conftest.py           # 全局 fixture（插入 sys.path、设置 caplog 级别）
│       ├── fixtures/pdf_pages/   # 真实技术文档单页 PDF 测试样本（TI、AMBA、VCS 等）
│       └── test_*.py             # 各模块测试（见下表）
└── venv/                         # Python 虚拟环境
```

### 核心模块职责速查

| 模块 | 关键类/函数 | 职责 |
|------|-------------|------|
| `core/config.py` | `Config` | 统一读取 `.env`，所有配置项以 `WORKDOCS_` 为前缀的环境变量 |
| `core/flow_selector.py` | `FlowSelector` | 根据配置自动选择 `LLM_API_FLOW` 或 `AGENT_SKILL_FLOW` |
| `core/models.py` | `Chapter`, `Chunk`, `Document` | 领域模型，使用 `dataclass` 定义 |
| `core/db.py` | `KnowledgeDB` | SQLite 操作，使用参数化查询防注入，`_connect()` 为上下文管理器 |
| `core/vector_index.py` | `VectorIndex` | FAISS 索引封装，支持维度不匹配时自动重建，兼容旧版 dict 格式 `id_map` |
| `core/pipeline.py` | `IngestionPipeline` | 通用摄入管道：扫描 → 解析 → chunk → 嵌入 → 入库 |
| `core/llm_api_pipeline.py` | `LLMAPIIngestionPipeline` | LLM API Flow 专用，支持层次化总结和图像分析 |
| `core/compatibility_pipeline.py` | `CompatibilityIngestionPipeline` | Agent Skill Flow 兼容管道，仅向量化 |
| `core/llm_chat_client.py` | `LLMChatClient` | 对话客户端，支持聊天、总结、vision 描述、思考模式 |
| `core/embedding_client.py` | `EmbeddingClient` | 向量化客户端，支持批处理、首次维度验证 |
| `parsers/pdf_parser.py` | `PDFParser` | PDF 文本/图片/矢量图提取，含图表区域启发式识别 |
| `parsers/office_parser.py` | `OfficeParser` | DOCX / XLSX 解析（DOCX 目前解析为单 chunk） |

---

## 构建与测试命令

> ⚠️ **前置要求**：首次安装后必须创建 Python 虚拟环境并安装依赖，否则 Kimi CLI 调用插件工具时会因缺少依赖而失败。

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

- **诊断/进度/错误信息统一使用 `logging`**，格式：
  ```python
  "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
  ```
- CLI 格式化结果展示与交互式 TUI 保留 `print()`
- 禁止随意使用 `print()` 输出调试信息

### 类型与模型

- 领域模型使用 `dataclass`（`Chapter`、`Chunk`、`Document`）
- 类型注解覆盖较全，核心函数参数/返回值尽量标注
- `Chunk.status` 枚举值：`pending`、`embedded`、`done`、`skipped`、`failed`
- `Document.status` 枚举值：`pending`、`processing`、`done`、`failed`

### 数据库安全

- **所有 SQL 必须使用参数化查询**（`?` 占位符），禁止字符串拼接
- `KnowledgeDB._connect()` 使用上下文管理器管理连接生命周期（`yield conn` + `conn.commit()` + `conn.close()`）

### 路径安全

- 文件读取操作需校验路径在 skill 根目录内：
  ```python
  resolved = Path(path).resolve()
  relative_to(skill_root)
  ```
- 已应用于 `write-embedding`、`agent_batch_helper apply --input` 等接口

### 配置访问

- 统一通过 `core.config.Config` 读取环境变量
- 数值型配置在类定义后通过 `_initialize_numeric_configs()` 初始化，支持安全解析（无效值自动回退到默认值并记录 warning）

### 向后兼容

- `core/llm_client.py` 中的旧版客户端保留，供存量代码导入
- 新代码优先使用 `core/llm_chat_client.py` 和 `core/embedding_client.py`

---

## 测试策略

### 测试结构

| 测试文件 | 说明 |
|----------|------|
| `test_pdf_parser.py` | 图表提取核心测试（1117 行，44+ 用例），含真实 PDF 单页 fixtures |
| `test_context_manager.py` | 上下文窗口管理测试（461 行） |
| `test_doc_extractor.py` | CLI 功能、查询、搜索测试 |
| `test_agent_batch_helper.py` | 批处理、断点续传、过滤规则测试 |
| `test_pipeline.py` | 文档处理流程测试 |
| `test_db.py` | SQLite 操作、事务管理测试 |
| `test_vector_index.py` | FAISS 索引增删查、持久化、维度迁移测试 |
| `test_llm_client.py` | LLM 对话、总结、图像分析 Mock 测试 |
| `test_dual_client.py` | 双客户端架构、配置验证、Kimi 适配测试 |
| `test_integration.py` | 端到端集成测试 |
| `test_chapter_editor.py` | 交互式章节编辑器测试 |
| `test_office_parser.py` | DOCX/XLSX 解析测试 |
| `test_image_utils.py` | 图片压缩工具测试 |
| `test_models.py` | 数据模型序列化测试 |

### 真实 Fixtures

测试使用真实技术文档的单页 PDF：
- TI DMA 控制器：`sprui07_page_*.pdf`
- TI 处理器：`tms320f28035_page_*.pdf`
- AMBA 总线规范：`amba_ahb_page_*.pdf`、`amba_axi_page_*.pdf`
- VCS 用户指南：`vcs_ug_page_*.pdf`
- Concept 用户指南：`spru430f_page_*.pdf`

### Mock 规范

- 测试 LLM/Embedding 客户端时，**禁止直接调用真实 API**
- 使用 `monkeypatch.setattr(Config, ...)` 而非 `monkeypatch.setenv` 来 mock 配置，避免 `.env` 加载顺序导致 mock 失效

---

## 安全注意事项

| 风险 | 等级 | 说明与缓解措施 |
|------|------|----------------|
| SQL 注入 | **低** | 全部使用参数化查询，无字符串拼接 |
| 路径遍历 | **中→低** | `ingest --path` 功能上允许任意路径；`write-embedding` / `agent_batch_helper apply` 已增加 `resolve()` + `relative_to(skill_root)` 校验 |
| API Key 泄漏 | **低** | Key 存储于 `.env`，未硬编码到代码中；`.env` 在 `.gitignore` 中 |
| JSON 反序列化 | **低** | 使用标准库 `json.load()`，无自定义反序列化 |
| 资源耗尽 | **中** | 大 PDF 的 diagram 渲染、大 DOCX 的单 chunk 合并可能消耗较多内存/磁盘；生产环境建议限制单文件大小 |

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
