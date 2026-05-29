# work-docs-library

通用化技术文档知识库管理工具。

本项目是一个面向技术文档（**当前仅支持 PDF**）的自动化知识提取 pipeline，以 **Kimi Code CLI Plugin** 形式运行。它支持：

- **智能文档解析**：PDF 通过 BigModel Expert API 解析为 Markdown 文本 + 图片，保留完整格式；失败时自动 fallback 到本地 PDFParser
- **知识图谱构建**：自动提取实体（Feature、Module、Register、Signal、Instruction、Interrupt、PipelineStage、Peripheral 等）和关系（IMPLEMENTS、CONTAINS、HAS_REGISTER、INSTRUCTION_READS_REGISTER、MODULE_IMPLEMENTS_INSTRUCTION、INTERRUPT_TRIGGERS 等），构建可查询的跨层级知识图谱（RTL ↔ ISA）
- **向量语义检索**：基于 FAISS 的语义向量索引，支持相似度搜索
- **Batch API 架构**：所有 LLM 调用通过 Batch API 提交，成本为同步 API 的 50%，支持超大 JSONL 自动拆分并行处理
- **章节级增量更新**：文档修订后，按章节 `content_hash` 指纹比较，未变章节复用实体缓存与 embedding，仅对变更/新增章节进行 LLM 提取，万页级文档变更一页时成本降低 99%+
- **Multimodal 图片理解**：LLM 直接分析文档中的图片（时序图、架构框图、寄存器表等），生成文字描述用于向量化

---

> ⚠️ **前置要求**：本项目依赖 Python 虚拟环境。首次安装后，请务必执行 [安装步骤](#安装) 创建 `.venv` 并安装依赖，否则 Kimi CLI 调用插件工具时会因缺少依赖而失败。

---

## 目录

1. [架构概览](#架构概览)
2. [目录结构](#目录结构)
3. [安装](#安装)
4. [快速开始](#快速开始)
5. [Plugin 工具说明](#plugin-工具说明)
6. [配置说明](#配置说明)
7. [核心模块说明](#核心模块说明)
8. [开发与测试](#开发与测试)
9. [已知限制与注意事项](#已知限制与注意事项)

---

## 架构概览

### DocGraphPipeline 六阶段架构

```mermaid
flowchart TB
    subgraph Stage1["阶段1: 文档解析"]
        S1A[PDF] --> S1B["BigModel Expert API\n(非 OpenAI-compatible)"]
        S1B -->|失败 fallback| S1C["本地 PDFParser\n(PyMuPDF + TOC 驱动)"]
        S1B --> S1D["result.md\n+ images/"]
        S1C --> S1D
    end

    subgraph Stage2["阶段2: 构建 LLM Batch JSONL"]
        S2A[result.md] --> S2B["ChapterParser\nparse_tree() → 多级树"]
        S2B --> S2C["_collect_section_content()\n按 ## 聚合 + 子孙 content"]
        S2C --> S2D["_build_content_blocks_and_maps()\n切分 → content_blocks + heading_maps"]
        S2D --> S2E["BatchBuilder\nbuild_batches()"]
        S2E -->|段落边界切分\n保护代码块/表格| S2F["sub-batch 列表"]
        S2F --> S2G["EntityExtractor\n_build_multimodal_content()\n流式嵌入图片 base64"]
        S2G --> S2H["{doc_id}.jsonl"]
        S2E --> S2I["{doc_id}_batch_info.json\nrequest → section 映射"]
    end

    subgraph Stage3["阶段3: 提交 LLM Batch API"]
        S3A["{doc_id}.jsonl\n(支持用户编辑后读取)"] --> S3B["读取 JSONL + _incremental_analysis()\ncontent_hash 指纹比对 + 增量过滤"]
        S3B -->|仅变更/新增章节| S3C["submit_parallel_batches()\n自动拆分 >100MB"]
        S3C --> S3D["LLM Batch API\n(Kimi / BigModel)"]
        S3D --> S3E["{doc_id}_results.jsonl\n原始 JSON 返回"]
        S3B --> S3F["{doc_id}_incremental.json\n增量摘要 + hash 校验"]
    end

    subgraph Stage4["阶段4: 解析入库（不含向量化）"]
        S4A["{doc_id}_results.jsonl"] --> S4B["extract_from_results_file()\n解析 entities/relations/image_descriptions"]
        S4B -->|复用未变 section 缓存| S4C["GraphStore.add_entity()\n.add_relation()"]
        S4C --> S4D["同名同类型去重合并\ndoc_properties[doc_id] 快照"]
        S4D --> S4E["文档子图\n{doc_id}.json"]
        S4C --> S4F["Product 实体提取\nHAS_MODULE 关系建立"]
        S4B --> S4G["_save_blocks_to_db()\n插入 content_blocks + heading_maps"]
        S4G --> S4H["content_block status: embedded"]
        S4G --> S4I["metadata.extracted_entities
缓存实体引用"]
    end

    subgraph Stage5["阶段5: 构建 Embedding JSONL"]
        S5A["SQLite content_blocks\n(status=embedded)"] --> S5B["查询无 embedding 的 blocks"]
        S5B --> S5D["每个 block 生成单文本 request\ncustom_id 编码 db_id + offset"]
        S5D --> S5E["{doc_id}_embed.jsonl"]
    end

    subgraph Stage6["阶段6: 同步 Embedding 向量化"]
        S6A["{doc_id}_embed.jsonl"] --> S6B["同步 Embedding API\n(BigModel embedding-3)"]
        S6B --> S6E["解析 embeddings"]
        S6E --> S6F["update_blocks_embedded_batch()\nSQLite metadata.embedding"]
        S6F --> S6G["FAISS IndexFlatIP\nadd_batch() (db_id + offset)"]
        S6G --> S6H["content_block status: done"]
    end

    subgraph Global["全局图谱重建"]
        G1["所有 {doc_id}.json"] --> G2["clear() + 遍历加载\nNetworkX DiGraph"]
        G2 --> G3["global.json\n跨文档实体对齐"]
    end

    S1D --> S2A
    S2G --> S3A
    S2H -.-> S4A
    S3E --> S4A
    S3F -.->|hash 校验| S4A
    S4E --> G1
    S4H --> S5A
    S5E --> S6A
    S5F -.-> S6E
    S6H --> G1
```

**数据流说明：**

1. **阶段1（解析）**：`BigModelParserClient` 调用 **BigModel 专用** Expert API 解析 PDF，输出 Markdown 文本（含 `![alt](images/xxx.jpg)` 图片引用）+ `images/` 目录（⚠️ 该 API 非 OpenAI-compatible，仅支持 BigModel 厂商；失败时自动 fallback 到本地 `PDFParser`，输出格式完全一致）
2. **阶段2（构建 LLM Batch）**：`ChapterParser.parse_tree()` 将 Markdown 解析为树形章节结构（`#` 文档标题，`##` 章节，`###+` 子章节）。`_collect_section_content()` 按 `##` section 递归聚合自身 content + 所有子孙 content（保留 Markdown 层级）。`_build_content_blocks_and_maps()` 将聚合后的 content 按 `max_chars` 切分为 content_blocks，同时构建 `heading_maps`（`##` 和 `###` 都映射到同一 section 的 block 集合）。`BatchBuilder.build_batches()` 接收 content_blocks 列表，超长内容按段落边界（`\n\n+`）切分为 sub-batch，同时保护代码块/表格不被截断。`EntityExtractor` 流式解析图片引用，按原文顺序构建 multimodal content（文本 → `[image_id: alt]` → base64 图片），生成 `{doc_id}.jsonl` + `{doc_id}_batch_info.json`
3. **阶段3（提交 LLM Batch）**：**优先读取** `batch/{doc_id}.jsonl`（支持用户编辑后重新提交），结合 `batch/{doc_id}_batch_info.json` 做增量过滤，仅对变更/新增章节的 requests 提交 Batch API。超大 JSONL 自动按 100MB 拆分并行提交。结果保存为 `{doc_id}_results.jsonl`，增量摘要保存为 `{doc_id}_incremental.json` 供阶段4校验一致性
4. **阶段4（解析入库，不含向量化）**：从 `results.jsonl` 解析 `entities`/`relationships`/`image_descriptions`，复用未变 section 的缓存实体/关系。`GraphStore`（NetworkX）构建图谱，**同名同类型实体自动去重合并**，每个文档保存独立子图 `graphs/{doc_id}.json`。同时保存每个文档的原始属性快照到 `doc_properties[doc_id]`，支持按文档精确查询。提取产品型号建立 `Product --[HAS_MODULE]--> Module` 关系。content_blocks 和 heading_maps 写入 SQLite，content_block 状态设为 `embedded`；每个 block 的 `metadata.extracted_entities` 缓存该 block 中提及的实体引用，作为后续跨粒度桥接索引的唯一数据源
5. **阶段5（构建 Embedding JSONL）**：从 SQLite 查询状态为 `embedded` 且暂无 `metadata.embedding` 的 content_blocks，每个 block 生成一个单文本 request（`custom_id` 编码 `db_id + _BLOCK_FAISS_OFFSET`），生成 `{doc_id}_embed.jsonl`
6. **阶段6（同步 Embedding 向量化）**：读取 `{doc_id}_embed.jsonl`，逐条调用同步 Embedding API（默认 BigModel `embedding-3`），解析 `custom_id` 还原 block `db_id`（减去 `_BLOCK_FAISS_OFFSET`），结果写入 SQLite `metadata.embedding` + FAISS `IndexFlatIP`（block db_id 加偏移存入），content_block 状态更新为 `done`
7. **跨粒度桥接索引**：`KnowledgeBaseService` 内部维护 `_EntityChunkBridge`，在 `__init__` 时从所有 chunks 的 `metadata.extracted_entities` 全量构建 `chunk_db_id ↔ (entity_type, entity_name)` 双向映射。`ingest_document` / `reprocess_document` 完成后自动同步。提供 O(1) 的正向查询（chunk→entities）和反向查询（entity→chunks），打通向量空间与图谱空间
8. **全局图谱重建**：`KnowledgeBaseService.ingest_document()` 完成后**全量重建**全局图 `graphs/global.json`（`clear()` + 遍历所有子图重新加载），确保无幽灵残留，实现**跨文档知识互通**

### 输入文档约束

本工具对被处理的 Markdown 文档（由 BigModel Expert 解析生成）有以下约束：

1. **图片引用格式**：必须使用标准 Markdown 格式 `![image_name](images/path.jpg)`，其中 `image_name` 将作为全局唯一的 `image_id` 使用
2. **image_name 要求**：`[]` 中的名称应有意义且唯一（如 `"Figure 1: Timing Diagram"`），不建议留空。若留空，程序将退化为内部编号
3. **图片路径**：`()` 中的路径应为相对于解析输出目录的相对路径，且该路径下必须存在对应的实际图片文件

---

## 目录结构

```
work-docs-library/
├── plugin.json                   # Kimi Code CLI Plugin 配置
├── AGENTS.md                     # Agent 开发指南（架构、策略、代码规范）
├── README.md                     # 本文件
├── config.json                   # 用户持久化配置（API 参数、模型选择等）
├── scripts/
│   ├── plugin_router.py          # Plugin 统一路由（stdin/stdout JSON）
│   ├── requirements.txt
│   ├── .env.example              # 环境变量模板
│   ├── .env                      # 实际环境变量（gitignored）
│   ├── prompts/                  # LLM 提示词文件（运行时读取，无需重启）
│   │   ├── entity_extraction_system.txt   # 实体提取 system 提示词
│   │   └── entity_extraction_user.txt     # 实体提取 user 模板
│   ├── core/                     # 业务逻辑层
│   │   ├── config.py             # 配置中心
│   │   ├── doc_graph_pipeline.py # ⭐ DocGraphPipeline 主管道
│   │   ├── batch_clients.py      # BaseBatchClient + BatchClient（通用，服务商无感）
│   │   ├── llm_chat_client.py    # LLM 对话客户端（辅助用途）
│   │   ├── embedding_client.py   # Embedding 客户端（辅助用途）
│   │   ├── bigmodel_parser_client.py  # BigModel Expert 文件解析
│   │   ├── graph_store.py        # 图谱存储（NetworkX）
│   │   ├── db.py                 # SQLite 数据库操作
│   │   ├── vector_index.py       # FAISS 向量索引管理
│   │   ├── models.py             # 数据模型 (Document/Chunk)
│   │   ├── enums.py              # StrEnum 定义 (ChunkStatus/DocumentStatus/ChunkType)
│   │   └── knowledge_base_service.py  # 统一服务层封装
│   ├── parsers/                  # IO / 解析层
│   │   ├── pdf_parser.py         # PDF 本地解析器（fallback，输出与 BigModel 一致）
│   │   ├── office_parser.py      # DOCX / XLSX 解析器（代码存在，尚未接入 pipeline）
│   │   └── image_utils.py        # 图片压缩工具
│   └── tests/                    # pytest 测试集（289 个用例）
├── knowledge_base/               # 运行时自动生成
│   ├── workdocs.db               # SQLite 元数据
│   ├── faiss.index               # FAISS 向量索引
│   ├── id_map.json               # FAISS ID 映射
│   ├── parsed/<doc_id>/          # Stage1 解析输出（result.md + images/）
│   ├── batch/                    # Stage2/3/5/6 中间产物（*.jsonl, *_info.json）
│   └── graphs/                   # Stage4 子图快照（{doc_id}.json, global.json）
├── .venv/                        # Python 虚拟环境
└── .gitignore
```

---

## 安装

### 环境要求

- Python >= 3.11
- 支持 Linux/macOS/Windows（主要测试于 Linux）

### 安装步骤

推荐在虚拟环境中安装，避免与系统包管理冲突：

```bash
cd ~/.kimi/plugins/work-docs-library

# 方式一：使用 uv（推荐，自动创建虚环境并安装依赖）
uv sync

# 方式二：使用 pip（需手动创建虚环境）
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 方式三：手动安装依赖（向后兼容）
pip install -r scripts/requirements.txt
```

### 配置

复制环境变量模板并编辑：

```bash
cp scripts/.env.example scripts/.env
# 编辑 scripts/.env，填入你的 API Key
```

用户持久化配置也可写入 `config.json`（项目根目录），详见 [配置说明](#配置说明)。

---

## 快速开始

本项目以 **Kimi Code CLI Plugin** 形式运行，通过 Kimi CLI 的命令行界面调用工具。

### 1. 导入文档（完整流程）

```bash
# 在 Kimi CLI 中执行
/ingest path/to/document.pdf
```

处理流程：
1. BigModel Expert 解析 PDF → Markdown + 图片
2. 构建树形章节结构
3. 按 batch 提交到 Kimi Batch API 进行实体提取
4. 构建知识图谱并持久化
5. 向量化后写入 SQLite + FAISS

### 2. 分阶段导入（支持人工干预）

当需要审查或修正中间产物时，可使用六阶段流程。每个阶段的产物均持久化到磁盘，支持人工编辑后重新触发下游阶段。

#### 阶段1: 解析（PDF → Markdown）

```bash
/doc_parse path/to/document.pdf
```

- **输入**: PDF 文件
- **输出**: `knowledge_base/parsed/{doc_id}/result.md` + `images/`
- **干预**: 直接编辑 `result.md`（修正文本、调整标题层级、补充内容）
- **触发下一阶段**: `/doc_build_batches {doc_id}`
- **注意**: 编辑后 content_hash 会变化，阶段3 的增量分析将识别为全部变更

#### 阶段2: 构建 Batch JSONL

```bash
/doc_build_batches {doc_id}
# 可选参数: --max-chars 10000（每个 batch 最大字符数）
```

- **输入**: `parsed/{doc_id}/result.md`
- **输出**: `batch/{doc_id}.jsonl` + `batch/{doc_id}_batch_info.json`
- **产物格式**: `jsonl` 每行是一个 JSON request，body 包含 `model`/`messages`/`response_format`/`extra_body`（含 thinking 参数）
- **干预**: 编辑 `jsonl`（修改 prompt、删除不想提交的 request、调整 messages）
- **⚠️ 关键限制**:
  - 删除 requests：无需同步修改 `batch_info.json`（代码会安全忽略多余的映射条目）
  - 修改 `custom_id`：无需同步修改 `batch_info.json`（不会报错，但该 request 在增量过滤时可能不会被选中）
  - **新增 requests：必须在 `batch_info.json` 中同步添加对应的 `custom_id` → `chapter_titles` 映射**，否则 stage4 的 `chapter_map` 无法回填，导致新增实体的 `source_chapter` 为空
  - `extra_body.thinking` 会被 stage3 自动补充（无需手动添加）
- **触发下一阶段**: `/doc_submit_batches {doc_id}`

#### 阶段3: 提交 LLM Batch API（支持 Chat 回退）

```bash
/doc_submit_batches {doc_id}
# 可选参数: --file-path PATH（原始 PDF 路径，数据库无记录时必填）
#            --jsonl-path PATH（自定义 JSONL 路径）
#            --force（强制重新处理，忽略缓存）
```

- **输入**: 优先读取 `batch/{doc_id}.jsonl`（支持用户编辑后重新提交），结合 `batch_info.json` 做增量过滤
- **输出**: `batch/{doc_id}_results.jsonl` + `batch/{doc_id}_incremental.json`
- **产物格式**: `results.jsonl` 每行是一个 JSON response，`response.body.choices[0].message.content` 是 LLM 提取的 entities/relations/image_descriptions
- **干预**: 编辑 `results.jsonl`（修正 LLM 提取错误：修改 entity 名称、添加遗漏的关系、修正图片描述）
- **注意**: `incremental.json` 是机器生成的 hash 校验文件，**不要手动编辑**
- **模式切换**: 设置 `WORKDOCS_LLM_MODE=chat`（`.env` 中）或 `config.json` 中 `"llm.mode": "chat"` 可切换到同步 Chat API 模式。`.env` 优先级高于 `config.json`。Chat 模式逐条调用同步 API，结果以与 Batch API 完全一致的格式写入 `results.jsonl`，Stage 4 无需任何修改即可复用。单条失败不中断流程，适合调试或 Batch API 不可用时作为回退
- **触发下一阶段**: `/doc_ingest_results {doc_id}`

#### 阶段4: 解析入库（不含向量化）

```bash
/doc_ingest_results {doc_id}
# 可选参数: --file-path PATH（原始 PDF 路径）
#            --results-path PATH（自定义 results.jsonl 路径）
#            --force（强制重新处理）
```

- **输入**: `batch/{doc_id}_results.jsonl` + `batch/{doc_id}_batch_info.json` + `batch/{doc_id}_incremental.json`
- **输出**: SQLite content_blocks + heading_maps（content_block 状态 `embedded`）+ `graphs/{doc_id}.json`
- **干预**: 直接编辑 `graphs/{doc_id}.json`（但推荐通过 `graph_upsert_entity`/`graph_upsert_relation` 等 Plugin 工具修改，自动维护索引一致性）
- **注意**: 直接编辑子图后，必须调用 `/rebuild_global_graph` 才能同步全局图 `global.json`
- **触发下一阶段**: `/doc_build_embed_jsonl {doc_id}`

#### 阶段5: 构建 Embedding Batch JSONL

```bash
/doc_build_embed_jsonl {doc_id}
```

- **输入**: SQLite content_blocks（状态 `embedded` 且暂无 `metadata.embedding`）
- **输出**: `batch/{doc_id}_embed.jsonl`
- **分组逻辑**: 每个需要向量化的 block 生成一个独立 request，`custom_id` 编码 `db_id + _BLOCK_FAISS_OFFSET`（格式 `embed_block_{db_id + offset}`），`body.input` 为单字符串
- **产物格式**: `embed.jsonl` 每行 body.input 是单个字符串（block content）
- **干预**: 编辑 `embed.jsonl`（删除不想向量化的 chunks）
- **⚠️ 关键限制**:
  - 删除行：可直接删除，不影响其他行（每个 request 独立）
  - 新增行：不建议新增（新 chunk 需先有 db_id）
- **触发下一阶段**: `/doc_submit_embed_batches {doc_id}`

#### 阶段6: 同步 Embedding 向量化

```bash
/doc_submit_embed_batches {doc_id}
# 可选参数: --embed-jsonl-path PATH（自定义 Embedding JSONL 路径）
```

- **输入**: `batch/{doc_id}_embed.jsonl`
- **输出**: SQLite `metadata.embedding` + FAISS 向量索引
- **处理逻辑**: 读取 JSONL 逐条调用同步 Embedding API（`EmbeddingClient.embed_single()`），从 `custom_id` 解析 `db_id`（减去 `_BLOCK_FAISS_OFFSET`），结果直接入库
- **干预**: 无（此阶段纯 API 调用与结果入库）
- **content_block 状态**: `embedded` → `done`

### 3. 语义搜索

```bash
/search AH bus arbitration
```

### 4. 按章节查询

```bash
/query --doc-id <DOC_HASH> --chapter "System Architecture"
```

### 5. 查看已导入文档

```bash
/status
```

### 6. 图谱查询

图谱数据以 JSON 格式持久化，可直接读取：

```bash
# 查看生成的图谱文件
ls knowledge_base/graphs/

# 查看图谱统计
python -c "
import json, sys
with open('knowledge_base/graphs/<doc_id>.json') as f:
    g = json.load(f)
print(f'entities={len(g.get(\"nodes\", []))}, relations={len(g.get(\"edges\", []))}')
"
```

---

## Plugin 工具说明

Kimi CLI 通过 `plugin.json` 注册以下工具：

| 工具名 | 作用 |
|--------|------|
| `ingest` | 提取并存储文档（PDF），完整流程一次性执行 |
| `doc_parse` | 阶段1：PDF → Markdown + 图片（可手动调整） |
| `doc_build_batches` | 阶段2：Markdown → Batch JSONL（本地生成，不调用 API） |
| `doc_submit_batches` | 阶段3：读取 `batch/{doc_id}.jsonl`（支持用户编辑后重新提交），提交 Batch API 并保存原始结果文件 |
| `doc_build_embed_jsonl` | 阶段5：从已入库 chunks 构建 Embedding Batch JSONL（本地，可审查） |
| `doc_submit_embed_batches` | 阶段6：提交 Embedding Batch API 并解析结果入库（完成向量化） |
| `doc_ingest_results` | 阶段4：从结果文件解析实体、构建图谱、保存 chunks（不含向量化） |
| `semantic_search` | 语义向量搜索（`graph_depth=0`）+ 可选关联图谱扩展（`graph_depth>0`） |
| `query` | 按章节、关键词、概念查询 chunk |
| `status` | 列出所有已导入文档，或查看指定文档的详细状态与进度 |
| `toc` | 查看文档目录 |
| `reprocess` | 强制重新处理文档 |
| `get_content` | 获取完整未截断内容，可选同时返回关联图谱实体/关系 |
| `graph_query` | 查询知识图谱实体（`depth=0`），支持扩展邻居（`depth=1`）和子图（`depth>1`） |
| `graph_path` | 查找两实体间的路径（支持关系过滤） |
| `graph_upsert_entity` | 添加/更新图谱实体（已存在则更新，不存在则创建） |
| `graph_delete_entity` | 删除实体（级联删边） |
| `graph_upsert_relation` | 添加/更新图谱关系 |
| `graph_delete_relation` | 删除关系 |
| `graph_feedback` | 提交（`action=submit`）或查询（`action=query`）对实体/关系的反馈 |
| `graph_conflicts` | 查询冲突日志 |
| `graph_provenance` | 实体来源溯源：从图谱实体通过桥接索引 O(1) 反向查找原始文档 chunk（调试与验证） |
| `rebuild_global_graph` | 全量重建全局图谱（修复不一致） |
| `config` | 打印当前生效配置（支持脱敏） |

---

## 配置说明

### 配置优先级架构

```
1. 环境变量（.env 文件，如 WORKDOCS_LLM_API_KEY）— 用户手动配置
   ↓
2. 环境变量（Kimi CLI 运行时注入，如 llm.api_key）— 工具动态注入
   ↓
3. config.json（项目根目录）— 工具自动持久化
   ↓
4. 代码硬编码默认值
```

`config.json` 与 `.env` 为双轨配置系统：
- **`.env`**：用户手动配置，优先级最高，适合存放 API Key 等凭证，gitignored，不进入版本控制。`.env` 中的值会覆盖 Kimi CLI 注入和 `config.json`
- **环境变量**：Kimi CLI 运行时动态注入，优先级第二
- **`config.json`**：工具自动持久化，优先级第三，适合存放模型选择、端点地址等不敏感参数。由 `plugin.json` 的 `config_file` 指定路径，Kimi CLI 安装时自动注入凭证

### 完整配置参考

| 环境变量 | config.json 路径 | 默认值 | 说明 |
|---------|-----------------|--------|------|
| **LLM 配置** | | | |
| `WORKDOCS_LLM_API_KEY` | `llm.api_key` | 空 | Kimi API Key（Batch API 实体提取用） |
| `WORKDOCS_LLM_BASE_URL` | `llm.endpoint` | `https://api.moonshot.cn/v1` | Kimi Base URL |
| `WORKDOCS_LLM_MODEL` | `llm.model` | `kimi-k2.5` | 对话模型 |
| `WORKDOCS_LLM_THINKING_ENABLED` | `llm.thinking_enabled` | `0` | 是否启用 thinking 模式（`1`=`enabled`，`0`=`disabled`）。Kimi K2.6 等模型 thinking 默认开启，**必须显式传递**才能可靠关闭 |
| `WORKDOCS_LLM_MODE` | `llm.mode` | `batch` | LLM 实体提取模式：`batch`（Batch API，成本为同步的 50%，默认）或 `chat`（同步 Chat API，逐条调用，适合调试或 Batch API 不可用时回退） |
| `WORKDOCS_LLM_BATCH_ENDPOINT` | `llm.batch_endpoint` | `/v1/chat/completions` | LLM Batch API endpoint |
| `WORKDOCS_LLM_BATCH_COMPLETION_WINDOW` | `llm.completion_window` | `24h` | Batch 完成窗口（如 `24h`） |
| `WORKDOCS_LLM_BATCH_MAX_CHARS` | `llm.batch_max_chars` | `10000` | 每个 LLM batch 最大文本字符数 |
| `WORKDOCS_LLM_BATCH_TIMEOUT` | `llm.batch_timeout` | `3600` | LLM Batch API 轮询超时（秒） |
| `WORKDOCS_LLM_MAX_RETRIES` | `llm.max_retries` | `3` | LLM 同步请求最大重试次数 |
| `WORKDOCS_LLM_RETRY_BACKOFF` | `llm.retry_backoff` | `2` | LLM 重试退避系数（秒） |
| `WORKDOCS_LLM_TIMEOUT` | `llm.timeout` | `120` | LLM 同步请求超时（秒） |
| `WORKDOCS_LLM_VISION_MAX_EDGE` | `llm.vision_max_edge` | `1024` | 图片压缩最长边（px） |
| `WORKDOCS_LLM_VISION_QUALITY` | `llm.vision_quality` | `85` | JPEG 压缩质量 1-100 |
| **Embedding 配置** | | | |
| `WORKDOCS_EMBEDDING_API_KEY` | `embedding.api_key` | 空 | BigModel API Key（向量化用） |
| `WORKDOCS_EMBEDDING_BASE_URL` | `embedding.endpoint` | `https://open.bigmodel.cn/api/paas/v4` | BigModel Base URL |
| `WORKDOCS_EMBEDDING_MODEL` | `embedding.model` | `embedding-3` | 向量化模型 |
| `WORKDOCS_EMBEDDING_DIMENSION` | `embedding.dimension` | `1024` | 向量维度 |
| `WORKDOCS_EMBEDDING_BATCH_ENDPOINT` | `embedding.batch_endpoint` | `/v4/embeddings` | ~~Embedding Batch API endpoint~~（已废弃，Embedding 改为同步单文本 API） |
| `WORKDOCS_EMBED_BATCH_TIMEOUT` | `embedding.batch_timeout` | `3600` | ~~Embedding Batch API 轮询超时（秒）~~（已废弃） |
| `WORKDOCS_CHUNK_MAX_CHARS` | `chunk.max_chars` | `6000` | **兼容层 chunks 表的最大字符数上限**。仅影响 `_save_blocks_to_db()` 写入兼容层 chunks 时的切分。主存储 `content_blocks` 使用 `LLM_BATCH_MAX_CHARS`（默认 10000）控制切分 |
| `WORKDOCS_EMBED_MAX_RETRIES` | `embedding.max_retries` | `3` | Embedding 同步请求最大重试次数 |
| `WORKDOCS_EMBED_RETRY_BACKOFF` | `embedding.retry_backoff` | `2` | Embedding 重试退避系数（秒） |
| `WORKDOCS_EMBED_TIMEOUT` | `embedding.timeout` | `120` | Embedding 同步请求超时（秒） |
| **Parser 配置** | | | |
| `WORKDOCS_PARSER_API_KEY` | `parser.api_key` | 空 | BigModel Expert 解析 API Key（⚠️ 仅用于 PDF 解析，为 BigModel 专有接口） |
| `WORKDOCS_PARSER_TIMEOUT` | `parser.timeout` | `60` | 解析请求超时（秒） |
| `WORKDOCS_PARSER_MAX_RETRIES` | `parser.max_retries` | `60` | 解析轮询最大重试次数 |
| `WORKDOCS_PARSER_POLL_INTERVAL` | `parser.poll_interval` | `3` | 解析轮询间隔（秒） |
| **Batch 通用配置** | | | |
| `WORKDOCS_BATCH_POLL_INTERVAL` | `batch.poll_interval` | `10` | Batch 状态轮询间隔（秒） |
| `WORKDOCS_BATCH_MAX_POLL_RETRIES` | `batch.max_poll_retries` | `360` | Batch 状态轮询最大次数 |
| `WORKDOCS_BATCH_MAX_FILE_SIZE_MB` | `batch.max_file_size_mb` | `100` | 单个 JSONL 文件大小上限（MB） |
| `WORKDOCS_BATCH_PARALLEL_WORKERS` | `batch.parallel_workers` | `4` | 并行 batch 提交线程数 |
| `WORKDOCS_BATCH_TEMP_DIR` | `batch.temp_dir` | `batch_temp` | Batch 临时文件目录 |
| `WORKDOCS_BATCH_FILE_DOWNLOAD_TEMPLATE` | `batch.download_template` | `{base_url}/files/{file_id}/content` | Batch 结果下载 URL 模板 |
| `WORKDOCS_PARSE_OUTPUT_DIR` | `pipeline.parse_output_dir` | `parsed` | PDF 解析输出目录 |
| `WORKDOCS_BATCH_OUTPUT_DIR` | `pipeline.batch_output_dir` | `batch` | Batch 产物输出目录 |
| **Plugin 默认值** | | | |
| `WORKDOCS_PLUGIN_SEARCH_TOP_K` | `plugin.search_top_k` | `5` | 语义搜索默认返回条数 |
| `WORKDOCS_PLUGIN_QUERY_TOP_K` | `plugin.query_top_k` | `10` | 查询默认返回条数 |
| `WORKDOCS_PLUGIN_GRAPH_MAX_DEPTH` | `plugin.graph_max_depth` | `3` | 图谱查询默认最大深度 |
| `WORKDOCS_PLUGIN_SUBGRAPH_DEPTH` | `plugin.subgraph_depth` | `1` | 子图扩展默认深度 |
| `WORKDOCS_PLUGIN_DEFAULT_LIMIT` | `plugin.default_limit` | `100` | 默认分页限制 |
| **Pipeline / Graph** | | | |
| `WORKDOCS_GRAPH_MAX_PATH_DEPTH` | `graph.max_path_depth` | `6` | 图谱路径搜索最大深度 |
| `WORKDOCS_GRAPH_OUTPUT_DIR` | `graph.output_dir` | `graphs` | 图谱 JSON 输出目录 |

### 超时调节指南

处理长文档或多图文档时，若遇到 `Read timed out. (read timeout=120)` 错误，需要调大同步请求超时参数。

**什么情况下需要调节？**
- 使用 Chat 模式（`LLM_MODE=chat`）处理长文档
- 请求包含 multimodal 图片（图片越多、越大，模型处理时间越长）
- 单个 chapter 的文本超过 500KB
- 当前默认 120 秒不足以完成模型推理

**超时后会发生什么？**
- Chat 模式下，`_post()` 在超时后**不再进行无意义重试**（重试使用相同 timeout 无法解决请求过大的问题）
- 失败请求会以 500 状态码记录到 `results.jsonl`，Stage 4 会跳过该请求（对应 chapter 无实体提取结果）
- 日志会输出当前超时值和建议调节值

**如何调节？**

环境变量方式（即时生效，重启后保留需写入 `.env`）：
```bash
export WORKDOCS_LLM_TIMEOUT=300
```

config.json 方式（持久化）：
```json
{
  "llm": {
    "timeout": 300
  }
}
```

**建议值参考**：
| 文档类型 | 建议 timeout |
|---------|-------------|
| 纯文本短文档（<100KB） | 120s（默认） |
| 中等文档（100KB~500KB，少量图片） | 180~300s |
| 长文档（>500KB 或大量图片） | 300~600s |

**其他相关超时参数**：
- `WORKDOCS_LLM_BATCH_TIMEOUT=3600`：Batch API 轮询超时（通常无需调节）
- `WORKDOCS_EMBED_TIMEOUT=120`：Embedding 同步请求超时（纯文本向量化，通常无需调节）
- `WORKDOCS_PARSER_TIMEOUT=60`：PDF 解析超时（BigModel Expert 解析通常很快，无需调节）

---

## 核心模块说明

### 文档处理管道
| 模块 | 职责 |
|------|------|
| `core/doc_graph_pipeline.py` | ⭐ **DocGraphPipeline**：主管道，涵盖解析 → 章节树 → batch 构建 → multimodal 实体提取 → 结果保存 → 解析入库 → 图谱 → 向量化 |
| `core/bigmodel_parser_client.py` | BigModel Expert 文件解析客户端（⚠️ BigModel 专用 API） |
| `core/batch_clients.py` | BaseBatchClient + BatchClient（通用 OpenAI-compatible Batch API，含并行批处理与结果保存） |
| `core/llm_chat_client.py` | LLM 对话客户端（辅助用途） |
| `core/knowledge_base_service.py` | 统一服务层封装（DB + VectorIndex + GraphStore + Pipeline + `_EntityChunkBridge` 跨粒度桥接索引） |

### 数据模型与存储
| 模块 | 职责 |
|------|------|
| `core/graph_store.py` | GraphEntity / GraphRelation / NetworkXGraphStore：实体关系图谱 |
| `core/db.py` | KnowledgeDB：SQLite 增删改查 |
| `core/vector_index.py` | VectorIndex：FAISS 向量索引管理 |
| `core/models.py` | Chunk、Document 数据模型 |

### 配置
| 模块 | 职责 |
|------|------|
| `core/config.py` | 统一配置中心，`.env` / 环境变量 / `config.json` / 默认值 四层优先级 |

---

## 数据库与存储架构

### Schema

数据库位于 `knowledge_base/workdocs.db`。

#### `documents` — 文档元数据
| 字段 | 说明 |
|------|------|
| `doc_id` (PK) | 文件内容 MD5 哈希 |
| `title` | 文档标题 |
| `source_path` (UNIQUE) | 原始文件路径 |
| `file_type` | `.pdf` |
| `total_pages` | PDF 物理页数 |
| `chapters` | JSON 序列化的章节列表 |
| `extracted_at` | 处理时间戳（ISO 格式） |
| `file_hash` | 内容哈希 |
| `status` | `pending` → `done` / `failed` |

#### `content_blocks` — 内容块（方案C：存储粒度）
| 字段 | 说明 |
|------|------|
| `id` (PK, AUTOINCREMENT) | SQLite 自增 ID，FAISS 索引用它（加 `_BLOCK_FAISS_OFFSET` 偏移） |
| `doc_id` (FK) | 所属文档 |
| `block_id` | 逻辑 ID（如 `b_0`） |
| `content` | 文本内容（按 `##` section 聚合后切分） |
| `seq_index` | 全局序列号，保证文档内顺序 |
| `metadata` | JSON：嵌入向量、content_hash、**`extracted_entities`**（block→实体映射，桥接索引数据源）、缓存的关系、图片描述等 |
| `status` | `pending` → `embedded` → `done` |
| `created_at` | 创建时间戳 |

#### `heading_maps` — 标题映射（方案C：查询粒度）
| 字段 | 说明 |
|------|------|
| `id` (PK, AUTOINCREMENT) | SQLite 自增 ID |
| `doc_id` (FK) | 所属文档 |
| `heading_title` | 标题文本（如 "2.1 GPIO Configuration"） |
| `heading_level` | 标题级别（2=`##`，3=`###`…） |
| `parent_heading` | 父标题 |
| `block_db_ids` | JSON 数组，指向 `content_blocks.id` 列表 |
| `content_summary` | 内容摘要（可选） |

#### `chunks` — 内容块（兼容层，不再由 pipeline 写入）
| 字段 | 说明 |
|------|------|
| `id` (PK, AUTOINCREMENT) | SQLite 自增 ID |
| `doc_id` (FK) | 所属文档 |
| `chunk_id` | 逻辑 ID |
| `content` | 原始提取内容 |
| `chunk_type` | `text` / `table` / `image_desc` |
| `chapter_title` | 所属章节 |
| `metadata` | JSON：嵌入向量、content_hash、extracted_entities 等 |
| `status` | `pending` → `embedded` → `done` |

> ⚠️ `chunks` 表保留用于兼容层：`_save_blocks_to_db()` 按 `section_title` 聚合 content_blocks 后同时写入 chunks 表，供增量分析和 `_EntityChunkBridge` 使用。未来移除兼容层后可删除此表。

### 四存储系统架构

本项目使用四个独立的存储系统，各司其职：

| 存储 | 职责 | 持久化 | 原子性 |
|------|------|--------|--------|
| **SQLite** | 文档元数据、content_blocks、heading_maps、chunks（兼容层） | `workdocs.db` | 单连接事务（自动 commit） |
| **FAISS** | 向量索引（语义搜索） | `faiss.index` + `id_map.json` | 文件级原子写入（临时文件 + rename）+ `fcntl` 进程锁 |
| **NetworkX** | 全局知识图谱（实体+关系） | `{doc_id}.json`（子图）+ `global.json`（全局图） | 内存操作 + 文件原子写入 |
| **Bridge** | chunk/block ↔ 实体 双向索引 | 纯内存（重启从 SQLite 重建） | 内存级 |

#### `content_blocks.metadata` JSON 结构

```json
{
  "content_hash": "md5前16位",
  "section_title": "2.1 GPIO Configuration",
  "extracted_entities": [{"entity_type": "Module", "name": "DMA_Controller", "properties": {}}],
  "extracted_relations": [{"rel_type": "HAS_REGISTER", "from_name": "DMA_Controller", "to_name": "DMA_CTRL"}],
  "image_descriptions": [{"image_id": "img_001", "description": "...", "chapter_title": "..."}],
  "embedding": [0.1, 0.2, ...]
}
```

### Content Block 生命周期状态图

```
                     _save_blocks_to_db
       (insert_block + update_block_status = embedded)
  ┌─────────────────────────────────────────────────────────┐
  │                                                         ↓
pending ───────────────────────────────────────────────→ embedded
                                                               │
                                stage5: 构建 embed.jsonl       │
                                stage6: 同步 Embedding API     │
                                                               │
                    update_blocks_embedded_batch()             │
                    vec.add_batch() (db_id + offset)           │
                                                               ↓
                                                            done
```

注意：`pending` 状态在实际流程中几乎不可见——`_save_blocks_to_db` 直接插入 blocks 并设为 `embedded`。`skipped` 和 `failed` 状态当前未使用。

### 跨存储关联矩阵

| 方向 | 关联机制 | 一致性保证 |
|------|---------|-----------|
| SQLite block → FAISS | `block.id`（db_id）+ `_BLOCK_FAISS_OFFSET` → `_id_map[faiss_id]` | FAISS 已加 `fcntl` 进程锁，修改前 `_reload()` 磁盘最新状态；SQLite + FAISS 仍非分布式事务，但单进程内已防并发覆盖 |
| FAISS → SQLite | `_id_map[faiss_id]` → `db.get_block_by_db_id(db_id)` 或 `db.get_chunk_by_db_id(db_id)` | 搜索时回查 SQLite 获取最新内容 |
| SQLite block → Graph | `block.metadata["extracted_entities"]` | 实体提取时写入 block metadata，作为 Bridge 索引的数据源 |
| Graph → SQLite block | `_EntityChunkBridge._reverse[EntityRef]` | ingest/reprocess 完成后 `_sync_bridge_for_doc()` 增量同步 |
| Graph 子图 → 全局图 | `ingest_document()` 增量合并；`reprocess_document()` 先移除旧贡献再合并 | `_save_global_graph()` 原子持久化到 `global.json`。崩溃后可从子图 `rebuild_global_graph()` 重建。 |

### 状态转移安全性审计

#### SQLite ↔ FAISS 一致性（stage6）

`stage6_submit_embed_batches` 中执行两步操作：

1. `self.db.update_blocks_embedded_batch(all_items)` — 将 embedding 写入 SQLite `metadata`
2. `self.vec.add_batch(all_items)` — 将向量写入 FAISS 索引

**已修复**：第二步被包裹在 `try/except` 中。若 FAISS 写入失败（如磁盘满），自动回滚第一步——清除 SQLite 中这些 blocks 的 `metadata.embedding`，恢复 `status` 为 `embedded`。下次 stage6 会重新向量化这些 blocks。

此外，FAISS `add_batch()` / `remove_doc()` 均通过 `fcntl.flock` 加进程级排他锁，修改前调用 `_reload()` 加载磁盘最新状态，防止多进程并发覆盖。

#### 极端场景处理

| 场景 | 行为 | 恢复方法 |
|------|------|----------|
| 进程崩溃在 stage4 与 stage6 之间 | content_blocks 状态为 `embedded`，FAISS 中无对应向量 | 重新调用 `/doc_submit_embed_batches {doc_id}` |
| FAISS 索引文件损坏 | 加载时抛出 `RuntimeError` | 删除 `faiss.index` + `id_map.json`，重新处理所有文档 |
| 全局图异常（节点<10 但文档>0） | 启动时检测到全局图不完整 | 自动触发 `rebuild_global_graph()` 重建；手动调用 `/rebuild_global_graph` 亦可 |
| 子图 `{doc_id}.json` 缺失但 SQLite 存在 | 全局图缺少该文档实体 | 调用 `/doc_reprocess {doc_id}` 重新提取 |

---

## 开发与测试

### 运行测试

```bash
cd /path/to/work-docs-library
PYTHONPATH=scripts ./.venv/bin/python -m pytest scripts/tests/ -v
```

**当前状态：352 passed, 2 skipped, 0 failed。**

### 常用测试文档

- `spru924f.pdf` — TI C2000 HRPWM Reference Guide（~80页，架构框图、时序图、寄存器表）
- `DVI0045.pdf` — ARM Multi-layer AHB Technical Overview（~8页，系统框图）

### BigModel Expert 文件解析配置

项目使用 **BigModel Expert** 作为文档解析主路径：

```bash
# .env 中配置
WORKDOCS_PARSER_API_KEY=your-api-key
```

**API 说明**：
- 服务类型：`expert`（PDF 专用，保留图片，0.012元/页）
- 输出格式：ZIP（`result.md` + `images/*.jpg`）
- 图片命名：`images/{uuid}_{page}_{x}_{y}_{w}_{h}_{index}.jpg`
- 若 BigModel 失败，自动 fallback 到本地 `PDFParser`，输出格式与 BigModel 完全一致（Markdown + images/）

---

## Prompts 提示词文件

`scripts/prompts/` 目录下的文本文件被代码**运行时读取**，无需重启即可生效。修改提示词后，重新执行 `/doc_build_batches` → `/doc_submit_batches` → `/doc_ingest_results` 即可看到效果，无需重启 Kimi CLI。

### `entity_extraction_system.txt` — 实体提取 system 提示词

**被谁读取**：`EntityExtractor._load_prompt("entity_extraction_system")`

**作用**：定义 LLM 的身份、实体/关系类型、输出格式、**三步提取流程**和**质量约束**。

**当前设计要点**：

#### 三步提取流程

为减少过度提取和跨类型误提取，Prompt 要求 LLM 按以下三步执行：

1. **Step 1 — 内容分类**：先将 chunk 内容归类为 9 种类型之一（寄存器字段描述表格、信号/参数表格、代码示例、架构描述、协议状态机、勘误/电气规格、概述/介绍、指令参考、其他）
2. **Step 2 — 按类型提取**：仅在对应类型范围内提取实体
   - RegisterField **只能从寄存器字段描述表格**中提取
   - 代码示例中**禁止提取任何实体**（零容忍）
   - 禁止将描述性短语（如 "Compare A"、"Phase registers"）提取为 Register
3. **Step 3 — 关系链补全 + 去重**：确保每个非 Document 实体至少一条直接关系，同文档内 Document/Product/Module 去重

#### 代码示例排除规则（零容忍）

以下代码元素**绝对禁止**提取为实体：

| 类型 | 示例 |
|------|------|
| 局部变量 | `epwm1_tz_isr`、`temp_count` |
| 代码标签 | `Epwm1_tz_isr:` |
| 汇编地址常量 | `0x007010` |
| 宏展开值 | `EPWM1_INT` |
| 寄存器字段访问 | `EPwm1Regs.TBCTL.bit.PRDLD` |
| 位域赋值 | `.bit.XXX = YYY` |

> **关键洞察**：`EPwm1Regs.TBCTL.bit.PRDLD` 在代码中是**访问语法**，不是 RegisterField 的定义。RegisterField 的定义只存在于寄存器字段描述表格中。

#### 实体类型优先级

同一概念在同一文档中只能属于一个类型：`Module > Register > Signal > Instruction > Parameter > Feature`

#### 属性格式规范

Prompt 中显式规定格式，确保 LLM 输出统一：

| 属性 | 格式 | 示例 |
|------|------|------|
| `width` | 纯数字 | `16` |
| `access` | R/RW/R-0/W1C | `R/W` |
| `reset_value` | 十六进制字符串 | `"0x0000"` |
| `address_offset` | 十六进制字符串 | `"0x0002"` |
| `bits` | 冒号分隔 | `"15:0"` |

### `entity_extraction_user.txt` — 实体提取 user 模板

**被谁读取**：`EntityExtractor._load_prompt("entity_extraction_user")`

**作用**：提供具体任务指令和占位符。

**格式规范**：
- 必须包含 `{{chapters}}` 占位符，运行时被替换为章节文本
- 必须包含 `{{images}}` 占位符（当前替换为空字符串，图片通过 multimodal content 直接传入）

### Prompt 迭代与质量评审

**Prompt 版本化管理流程**：
1. 修改 `scripts/prompts/entity_extraction_system.txt`
2. 执行 `/doc_build_batches {doc_id}` → `/doc_submit_batches {doc_id}` → `/doc_ingest_results {doc_id}`
3. 对比全局节点/边数量变化、具体实体差异
4. 记录"预期效果 vs 实际效果"，修正 Prompt 表述

**质量评审 checklist**（每次修改后执行）：
- [ ] 全局节点数变化是否符合预期（减少误提取 → 应下降）
- [ ] 代码示例中是否仍有 Register/RegisterField 误提取
- [ ] Register 属性是否完整（address_offset、width、access、reset_value）
- [ ] bits/reset_value 格式是否统一
- [ ] 是否有孤立节点（无关系的实体）
- [ ] 跨文档合并后属性冲突是否合理

---

## 已知限制与注意事项

### 功能限制

1. **仅支持 PDF**：DOCX/XLSX 解析器代码已存在，但尚未接入 `DocGraphPipeline`
2. **Batch API 延迟**：Batch API 成本为同步 API 的 50%，但存在分钟级排队延迟
3. **JSONL 大小限制**：单个 JSONL 文件不能超过 100MB，超大文档会自动拆分为多个并行 batch
4. **Embedding 维度不可变**：FAISS 索引创建后维度固定。更换模型导致维度变化时，必须删除旧索引并重新处理
5. **FAISS 与 SQLite 非原子**：已缓解——FAISS 操作加 `fcntl` 进程锁，修改前 `_reload()` 磁盘最新状态。极端情况仍可通过 `reprocess` 重建
6. **图片压缩**：`LLM_VISION_MAX_EDGE`（默认 1024）和 `LLM_VISION_QUALITY`（默认 85）控制 base64 图片大小
7. **NetworkX 内存上限**：全局图为内存存储，数百个文档 × 万页级时可能达到 GB 级。当前单机目标规模可接受，预留 Neo4j 迁移接口
8. **输入文档约束**：Markdown 图片引用 `![name](images/path.jpg)` 中的 `name` 将作为 `image_id`，建议填写有意义的名称
9. **PDF 解析依赖 BigModel 专用 API**：文档提取主路径使用 BigModel 专有 Expert API（`/files/parser/create`），非 OpenAI-compatible，无法直接切换至其他厂商。若 BigModel 不可用，可依赖本地 `PDFParser`（PyMuPDF）作为 fallback，输出格式与 BigModel 完全一致，但解析质量可能略有差异
10. **跨产品外设变体**：同一个外设/寄存器出现在多个产品手册中时，`doc_properties` 保存每个文档的原始属性快照，全局图的 `properties` 仍为合并后值。查询时通过 `doc_id` 参数获取指定产品的精确属性。产品型号通过启发式正则从文档标题/文件名自动提取

### 实体提取质量已知问题（Prompt 迭代中）

以下问题已通过 Prompt 优化大幅缓解，但尚未完全消除：

| 问题 | 现状 | 缓解措施 | 后续方向 |
|------|------|---------|---------|
| **代码示例中的 Register 误提取** | `EPwm1Regs.AQCTLA`（无 `.bit.` 前缀）仍被提取为 Register | Prompt 已禁止 `.bit.XXX = YYY` 和 `Regs.XXX.bit.YYY` 模式，但未覆盖 `Regs.XXX` 模式 | 强化 Prompt：明确禁止提取 `Regs.` 前缀的代码访问语法 |
| **Register 属性缺失** | 14 个 Register 中 11 个 access 缺失、14 个 reset_value 缺失 | 属性格式已在 Prompt 中规范 | 在 Prompt 中增加"Register 必须提取 access/reset_value"的强制要求，或从表格列映射补全 |
| **bits 格式不统一** | 同一文档中出现 `15-4`、`15:8`、`7-0`（减号）和 `15:8`、`7:0`（冒号）混用 | Prompt 已规定 `"15:0"` 冒号格式 | 在 Prompt 最终检查清单中增加 bits 格式校验 |
| **reset_value 格式不统一** | `0`（数字）与 `0x0000`（十六进制）混用 | Prompt 已规定 `"0x0000"` 字符串格式 | 同上，在最终检查清单中增加格式校验 |
| **孤立 RegisterField** | `TBPHSH`、`Reserved` 来源为 Appendix A 而非字段描述表格 | — | 在 Prompt 中增加 Appendix/附录章节排除规则 |
| **跨文档属性合并** | 已修复：全局 `properties` 合并时比较文档信息完整性（非空属性数量），完整性高的文档优先。互补属性始终取并集。平局保留旧值。无法推断来源时保留现有值 | `doc_properties` 保存原始快照，`doc_id` 查询可获取精确属性 | 完整性评分基于属性数量而非语义权重。修改后建议调用 `/rebuild_global_graph` 重建全局图

---

## 参考资源

### 官方 API 开发文档

- [Kimi Code CLI 插件](https://moonshotai.github.io/kimi-cli/zh/customization/plugins.html)
- [Kimi API 概述](https://platform.kimi.com/docs/api/overview)
- [Kimi 模型参数参考](https://platform.kimi.com/docs/api/models-overview)
- [Kimi 使用 Batch API 批量处理任务](https://platform.kimi.com/docs/guide/use-batch-api)
- [BigModel API 使用概述](https://docs.bigmodel.cn/cn/api/introduction)
- [BigModel 结构化输出](https://docs.bigmodel.cn/cn/guide/capabilities/struct-output)
- [BigModel Embedding-3](https://docs.bigmodel.cn/cn/guide/models/embedding/embedding-3)
- [BigModel 批量处理](https://docs.bigmodel.cn/cn/guide/tools/batch)
- [BigModel 新文件解析服务](https://docs.bigmodel.cn/cn/guide/tools/file-parser)

## 许可证

MIT
