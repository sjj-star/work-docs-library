# work-docs-library 技术方案文档

> 记录每个核心设计决策的选择原因、已知限制与权衡。本文档面向开发者与 Agent，用于理解架构背后的思考。

---

## 1. 为什么 Batch API 优先？

**选择**：**实体提取**（LLM 抽取结构化实体/关系）走 Batch API；**向量化**（Embedding）走同步单文本 API。使用通用 `BatchClient`（继承 `BaseBatchClient`），通过配置参数（`api_key`/`base_url`/`batch_endpoint`/`download_url_template`）适配不同服务商的 OpenAI-compatible Batch API，无需为每个厂商单独实现客户端。

**原因**：
- **成本**：Batch API 价格为同步 API 的 50%，对于文档处理这类非实时场景，成本优势显著
- **吞吐量**：`submit_parallel_batches()` 按 100MB JSONL 限制自动切分，ThreadPoolExecutor 并行提交，可处理超大文档
- **可接受性**：实体提取的 Batch API 延迟通常为分钟级，对离线文档导入场景可接受

**为什么向量化不走 Batch API？**

项目早期曾使用 BigModel Embedding Batch API，但实践中发现两个致命问题：

1. **处理时间不可接受**：BigModel Embedding Batch API 的实际处理时间高达 **9 小时**，而同步单文本 API 仅需**分钟级**完成。9 小时的等待使 pipeline 失去实用价值。

2. **Tokenizer 不透明导致切分策略复杂**：BigModel embedding-3 的 tokenizer 不透明，无法可靠估算 token 数。项目放弃 token-based 切分，改用**纯字符数限制**（`BLOCK_MAX_CHARS = 6000`），配合段落→句子三级语义保护切分（`_split_for_embedding`），确保每段不超过 embedding API 的 3072 tokens 上限。字符数限制比 token 估算更可靠，因为 GLM tokenizer 对中文编码效率较高（1-1.5 tokens/字符），6000 字符是保守经验值。

**经验**：Batch API 的延迟不是统一的——LLM Batch API（分钟级）与 Embedding Batch API（小时级）有数量级差异。当外部 tokenizer 不透明时，**字符数限制是比 token 估算更可靠的切分策略**。

**Chat 模式回退机制**：

当 Batch API 不可用（如服务商未开通、排队过长）或需要快速调试时，可通过 `WORKDOCS_LLM_MODE=chat` 切换到同步 Chat API 模式：

- **持久化格式一致**：Chat 模式逐条读取 Stage 2 生成的 `batch/{doc_id}.jsonl`，解析 `req["body"]` 后调用同步 Chat API，将结果以与 Batch API 完全一致的格式写入 `batch/{doc_id}_results.jsonl`（`{"custom_id": "...", "response": {"status_code": 200, "body": {...}}}`）
- **Stage 4 零修改复用**：由于 `results.jsonl` 格式完全一致，`stage4_ingest_results` 和 `EntityExtractor._parse_results()` 无需任何修改即可解析 Chat 模式的输出
- **单条失败不中断**：某条请求失败时记录 `status_code: 500`，继续处理后续请求，避免整批失败
- **成本提醒**：Chat 模式价格为 Batch 的 2 倍，仅建议用于调试或 Batch API 不可用的回退场景

**权衡**：
- 实体提取延迟从秒级变为分钟级（Batch）或秒级（Chat），不适合实时交互场景
- 需要实现轮询和超时逻辑（`LLM_BATCH_TIMEOUT` 默认 3600 秒）
- 单个 JSONL 不能超过 100MB，超大文档需自动拆分

---

## 2. 为什么 NetworkX 而不是 Neo4j？

**选择**：使用 `networkx.DiGraph` 作为图谱存储引擎，JSON 序列化持久化。

**原因**：
- **轻量**：无需引入外部数据库服务，零部署成本
- **JSON 序列化**：`nx.node_link_data()` / `nx.node_link_graph()` 支持完整的图结构导出/导入，便于版本控制和调试
- **当前需求足够**：支持邻居查询、子图提取、BFS 路径搜索（`find_path`，默认深度 3，硬上限 6 可通过 `Config.GRAPH_MAX_PATH_DEPTH` 调整），NetworkX 的内存遍历性能完全满足
- **预留接口**：`GraphStore` 抽象基类已预留 Neo4j 迁移接口，但项目中**尚无 Neo4j 实现类**，当前无迁移计划

**增强功能**：
- **数据质量标记**：`GraphEntity`/`GraphRelation` 新增 `confidence`/`verified`/`created_at`/`updated_at`/`feedback_score` 字段，支持置信度追踪和人工验证
- **动态 CRUD**：支持运行时 `update_entity`/`delete_entity`/`update_relation`/`delete_relation`/`verify_entity`，Agent 可直接修正图谱
- **冲突检测**：同名实体属性差异时自动记录 `conflict_logs`，保留旧值到列表形式而非静默覆盖
- **属性索引**：内部维护 `property_index: dict[(entity_type, key, value), set[nid]]`，`find_by_property()` 从 O(N) 降至 O(1)
- **关系过滤**：`find_path()` 支持按 `rel_types` 集合过滤；`get_neighbors()` 支持按 `rel_type` 单种关系类型过滤

**Schema 扩展策略**：
- 实体类型和关系类型以 Python 常量定义（`ALL_NODE_TYPES`/`ALL_REL_TYPES`），新增类型只需在 `graph_store.py` 中追加常量并加入集合，无需修改数据库 schema 或 pipeline 核心逻辑
- pipeline 通过 `e.get("type") in ALL_NODE_TYPES` 和 `r.get("type") in ALL_REL_TYPES` 过滤 LLM 输出，schema 扩展对 pipeline 完全透明
- 提示词文件 `entity_extraction_system.txt` 独立维护实体/关系定义，修改后无需重启即可生效

**跨层级查询设计（ISA ↔ RTL）**：
- 处理器架构实体（Instruction、PipelineStage、Interrupt 等）与 RTL 实体（Module、Register、Signal 等）共存于同一图谱中
- 通过跨层级关系（如 `MODULE_IMPLEMENTS_INSTRUCTION`、`INSTRUCTION_READS_REGISTER`）桥接 ISA 描述与 RTL 实现，支持从指令追踪到模块、从寄存器追踪到流水线阶段的联合查询
- NetworkX 的子图查询（`get_subgraph`）和路径搜索（`find_path`）天然支持跨层级遍历，无需额外索引

**权衡**：
- 内存存储，单图规模受限于可用内存（当前目标文档为几十到几百页，远未触及瓶颈）
- 单进程写入为主；FAISS 已加 `fcntl` 进程级文件锁防并发覆盖，但不建议多进程并发 reprocess
- 复杂图算法（PageRank、社区发现）性能不如专用图数据库

---

## 3. 为什么零数据丢失？

**选择**：禁止过滤、截断源数据。只允许结构性拆分（按 Markdown 标题层级）。

**原因**：
- **技术文档信息密度高**：一个寄存器描述、一个信号名、一个时序参数的遗漏都可能导致设计错误
- **不可预知性**：Agent/LLM 无法预先判断哪些信息是"重要"的，截断策略必然产生遗漏
- **可复现性**：结构性拆分（按标题）是确定性的，而基于语义相关性的过滤是黑盒的

**具体措施**：
- `BatchBuilder` 按 `#` 硬边界、`##` 可合并 构建 batch，确保不跨 `#` 截断
- 图片不单独处理，按原文引用顺序嵌入到文档流中，与文本统一送入 LLM
- 图片必须先由 LLM 文字化（`image_descriptions`），再合并到 block 文本中向量化

**权衡**：
- batch 数量可能增加，导致 Batch API 调用次数上升
- 某些 batch 可能包含冗余信息，但这是确保完整性的必要成本

---

## 4. 为什么 FAISS IndexFlatIP？

**选择**：使用 `faiss.IndexFlatIP`（内积索引），并在插入/查询时对向量做 L2 归一化。

**原因**：
- **数学等效**：对向量做 L2 归一化后，内积（Inner Product）等价于余弦相似度：
  ```
  cosine_similarity(a, b) = dot(a, b) / (||a|| * ||b||) = dot(a/||a||, b/||b||)
  ```
- **简单可靠**：IndexFlatIP 是精确搜索，无近似误差；对于当前数据规模（数千到数万条），查询性能完全可接受
- **无需训练**：IndexFlatIP 是无参索引，无需像 IVF 或 HNSW 那样训练聚类参数

**权衡**：
- 时间复杂度 O(n)，数据量增长到百万级时需迁移到 IVF/HNSW
- 内存占用为原始向量的精确存储，无压缩

---

## 5. 为什么 Multimodal 图片流式处理？

**选择**：图片不按独立对象处理，而是作为文档流的一部分，按 Markdown 引用位置以 base64 嵌入文本统一处理。

**原因**：
- **语义连续性**：技术文档中的图片（时序图、架构框图、寄存器表）与 surrounding text 紧密关联，独立处理会丢失上下文
- **阅读顺序**：`EntityExtractor` 按原文出现顺序构建 multimodal content：`文本片段1` → `[image_id: alt]` → `base64 图片` → `文本片段2`
- **LLM 理解**：当前多模态 LLM（kimi-k2.5）支持图文交错输入，流式结构最符合人类阅读习惯

**实现细节**：
- 正则 `r"!\[([^\]]*)\]\(([^)]+)\)"` 匹配 Markdown 图片引用
- `image_id` 直接使用 `[]` 中的 alt 文本，要求解析器生成的 alt 有意义且唯一
- 图片压缩：`_compress_image_to_base64` 将图片最长边缩放到 `IMAGE_MAX_SIZE`（默认 1024），并基于色度距离自动选择最优存储格式（PNG 1-bit / JPEG L / JPEG RGB）

**关于 Token 计费的关键发现**（经 API 实测 + 官方文档确认）：
- Kimi Vision API 对图片采用**动态 token 计算**：**分辨率越高，token 越多**；与格式（PNG/JPEG）、颜色模式（RGB/L/1-bit）、压缩质量完全无关
- 原始大图（如 9342×5442）可达 **4176 tokens/张**；经 `IMAGE_MAX_SIZE=1024` 缩放后降至约 **825 tokens/张**，节省 **80%**
- 三层分类策略（PNG 1-bit 等）**不减少 API token 消耗**，其价值在于显著减小 base64 体积，降低 JSONL 文件大小和网络传输开销

**权衡**：
- 图片必须先由 LLM 文字化，才能进入 embedding；当前 embedding 模型（embedding-3）不支持图片输入
- base64 编码增加 JSONL 体积，超大文档更容易触发 100MB 拆分阈值；PNG 1-bit 格式可将 blackwhite 图片体积压缩约 5-9 倍，有效缓解此问题

---

## 6. 为什么 SQLite 而不是 PostgreSQL？

**选择**：使用 SQLite（标准库 `sqlite3`）作为元数据存储。

**原因**：
- **单用户场景**：本项目为本地部署的 Kimi CLI Plugin，无多用户并发需求
- **零运维**：无需安装、配置、维护数据库服务，一个 `.db` 文件即完整数据库
- **事务简单**：`KnowledgeDB._connect()` 使用上下文管理器，自动 commit/close
- **足够当前需求**：当前 Schema 包含 `_schema_meta`（版本管理）、`documents`（文档元数据）、`content_blocks`（内容块）、`heading_maps`（标题映射）、`conflict_logs`（同名实体属性冲突日志）、`feedback`（实体/关系用户反馈）六张表，无复杂查询

**权衡**：
- 单进程写入为主；FAISS 已加 `fcntl` 进程级文件锁防并发覆盖，但不建议多进程并发 reprocess
- 无内置全文检索（当前使用 FAISS 向量搜索 + 关键词 LIKE 查询，暂不依赖 SQLite FTS）
- FAISS 已加进程锁和修改前重载（`_reload()`），进程崩溃后可通过 `rebuild_global_graph` 或重启自动恢复；极端情况仍可通过 `reprocess` 重建

---

## 7. 为什么树形章节解析？

**选择**：`ChapterParser.parse_tree()` 将 Markdown 解析为树形章节结构（`#` 文档标题，`##` 章节，`###` 子章节）。

**原因**：
- **唯一允许的结构性拆分**：按 Markdown 标题层级拆分是唯一被允许的拆分方式；禁止基于内容相关性的过滤或截断
- **语义完整性**：章节是技术文档的自然语义边界，跨章节合并可能割裂逻辑单元（如一个完整的寄存器描述）

**具体规则**：
- `#` → 文档标题（书名），永不跨 `#` 合并
- `##` → 章节，硬边界（不跨 `##` 合并）
- `###` → 子章节/小节，基本 block 单位，内容较短时与同层级兄弟合并
- `#` 和第一个 `##` 之间的文字 → **移动**到第一个 `##` 的 preface（父节点 content 清空，子节点 content 前置）

> 注：BatchBuilder 的硬边界与合并规则详见第15节。

**权衡**：
- 某些 `#` 章节可能内容极少，产生小 batch；但这是保证语义边界的必要代价
- 对解析器生成的 Markdown 标题质量有依赖（BigModel Expert 通常表现良好）

---

## 8. 配置系统：config.json + .env 双轨设计

**选择**：四层优先级配置系统——`.env`（用户手动配置）> 环境变量（Kimi CLI 注入）> `config.json`（工具持久化）> 代码默认值。

**原因**：
- **`.env`**：用户手动配置，优先级最高，适合存放 API Key 等凭证，gitignored，不进入版本控制。用户手动修改的配置应覆盖工具自动注入的值
- **环境变量**：Kimi CLI 运行时动态注入，优先级第二，支持在不修改文件的情况下临时覆盖 `config.json`
- **`config.json`**：工具自动持久化，优先级第三，适合存放模型选择、端点地址等不敏感参数；由 `plugin.json` 的 `config_file` 指定路径，Kimi CLI 安装时自动注入凭证

**实现**：
- `_resolve_config(env_name, json_path, default)` 统一解析，优先级逻辑集中在一处（`BigModelParserClient` 的 `_resolve_api_key` 为历史遗留独立实现，未复用 `_resolve_config`，后续建议统一）
- 数值类型配置（dimension、batch_max_chars 等）在类定义后通过 `_initialize_numeric_configs()` 初始化

---

## 9. 为什么全局统一图谱？

**选择**：维护全局合并图谱 `global.json` + 文档子图快照 `{doc_id}.json`，`KnowledgeBaseService` 启动时自动加载全局图到内存。

**原因**：
- **跨文档实体对齐**：数百个技术文档中必然存在同名同类型实体（如 `DMA_Controller`），NetworkX 节点 ID `{type}::{name}` 天然实现自动去重与属性合并
- **查询零延迟**：所有图谱查询直接在内存全局图上执行，无需预加载或合并
- **精确更新**：`remove_document_contributions(doc_id)` 可精确移除单个文档的旧贡献，避免 reprocess 后幽灵数据残留
- **预留扩展**：节点/边属性中记录 `source_doc_ids` 集合，未来可追踪实体来源、生成来源报表

**权衡**：
- 内存存储，数百个文档 × 万页级时全局图可能达到 GB 级（当前目标规模在单机上可接受）
- 边的 `source_doc_ids` 需通过覆盖/合并管理（NetworkX `DiGraph` 同一对节点仅允许一条边；如需保留多条同类型边，需迁移到 `MultiDiGraph`）
- `save`/`load` 需处理 `set` ↔ `list` 的 JSON 序列化

---

## 10. 为什么章节级增量更新？

**选择**：文档更新时，按章节计算 `content_hash`（取 MD5 前 16 位作为指纹）比较。未变章节复用 `Chunk.metadata` 中缓存的 `extracted_entities` / `extracted_relations` / `embedding`，仅对变更/新增章节进行 LLM 提取和向量化。

**原因**：
- **万页级文档的更新成本**：全量重新处理一个万页文档的成本约 600元 + 1小时。若只变更 1 页，增量更新可将成本降至 ~10元 + 5分钟
- **技术文档的结构性**：技术文档按章节组织，章节内部高度内聚，跨章节依赖相对较少。章节是天然的最小增量单元
- **Batch API 的延迟**：避免对未变章节重复提交 Batch API，显著降低排队等待时间

**实现细节**：
- `content_blocks.metadata` 存储 `content_hash`、`extracted_entities`、`extracted_relations`、`image_descriptions`、`embedding`
- `_save_blocks_to_db()` 向量化阶段分离 `reuse_pairs`（未变 section 复用 embedding）和 `reembed_pairs`（变更/新增 section 重新调用 Embedding API）
- 全局图重建策略：`DocGraphPipeline._process_one()` 保存独立子图 `{doc_id}.json`；`KnowledgeBaseService.ingest_document()` 完成后清空内存全局图，从所有现有 `{doc_id}.json` 子图重新加载合并，确保无幽灵残留

**权衡**：
- 需要存储每个章节的实体/关系缓存，增加 SQLite 存储开销（通常 < 1%）
- 如果解析器输出不稳定（同一 PDF 多次解析结果不同），content_hash 会变化，导致不必要的重新提取
- 跨章节的实体关系（如 A 章实体指向 B 章实体）在 B 章变更时可能丢失，需重新提取 B 章恢复

---

## 11. 为什么本地 PDF 解析器使用 TOC 驱动式章节识别？

**选择**：本地 PDF 解析器（PyMuPDF fallback）使用 PDF 内置 TOC（Table of Contents）直接匹配页面文本块来识别标题，而非基于字体大小/粗体的启发式规则。

**原因**：
- **准确性高**：技术文档（如 TI Reference Guide）的标题通常有编号（如 "2.2 Configuring the HRPWM"），字体大小与正文差异不大，纯启发式规则容易误判
- **避免编号/标题分离问题**：PDF 中标题编号和标题文本可能位于不同文本块，导致匹配失败。TOC 匹配可以合并这些分离的块
- **过滤非技术章节**：目录、前言、索引等章节对知识提取价值低，可通过 `_SKIP_HEADING_TITLES` 黑名单过滤

**具体实现**：
1. `_build_toc_by_page`：按页码分组 TOC 条目，清洗特殊 Unicode 空格（如 `\u2002` EN SPACE）为普通空格
2. `_identify_headings_by_toc`：页面文本块与 TOC 条目精确匹配（normalized text）或前缀匹配（处理编号与标题分离）。匹配前 `text.replace('\n', ' ')` 处理含换行的标题文本
3. `_organize_by_chapters`：在章节开头插入标题前，检测页面 Markdown 中是否已存在该标题（`expected in page_md`），避免重复插入
4. `_fallback_heading_detection`：当 PDF 无 TOC 或 TOC 质量不佳时，回退到基于字体大小/粗体的启发式规则

**权衡**：
- 依赖 PDF 的 TOC 质量。部分扫描件或无 TOC 的 PDF 会回退到启发式规则
- 目录页条目换行问题：多行目录条目在 Markdown 中连成一行（影响小，接受此行为，详见第 12 节）

---

## 12. 为什么段落内换行保留原始 `\n`？

**选择**：`_get_page_text_blocks` 使用 `page.get_text("blocks")` 获取文本块，在 `_build_page_markdown` 中**不做替换**，直接保留文本块内的原始 `\n`。

**原因**：
- **LLM 消费原始文本**：下游 LLM 读取的是原始 Markdown 文本（非渲染后的 HTML），能直接理解 `\n` 的换行语义
- **零额外 token**：不引入 `<br>` 等 HTML 标签，不增加 token 开销
- **保留结构信息**：寄存器位域描述（`Bit 15-8: Reserved\nBit 7-0: DATA`）、代码片段、地址列表等有意为之的多行结构得以保留
- `get_text("blocks")` 已经正确分组逻辑单元（段落、表格单元格），block 之间是段落分隔，block 内部 `\n` 是段落内换行

**历史演变**：
- v1：使用 `get_text("dict")` 按 y 坐标逐行合并，破坏段落边界
- v2：使用 `get_text("blocks")`，输出时将 `\n` 替换为空格（丢失换行信息）
- v3（当前）：保留原始 `\n`，让 LLM 自行理解段落内换行

**权衡**：
- 目录页中多行条目（如 `'Preface...\n1\nIntroduction...'`）也被保留为多行；目录页对知识提取影响小，接受此行为
- 大量自动排版换行（soft line break）也被保留；LLM 能区分有意义的换行和无意义的换行

---

## 13. 为什么使用完整章节层级构造 Markdown？

**选择**：PDFParser 输出的 Markdown heading 使用 PDF TOC 中的真实层级（`#` / `##` / `###` / `####`），不再扁平化为两级。

**原因**：
- **信息保真**：技术文档通常有 3-4 级层级（如 `2.3.1 Edge Positioning`），扁平化会丢失层级语义
- **下游可利用**：多级树让 `ChapterParser` 和 `BatchBuilder` 能更精细地控制内容粒度
- **与 BigModel Expert 对齐**：BigModel 解析器输出也保留原始层级

**实现**：
- `_build_page_markdown`：`prefix = "#" * heading_level`，`heading_level` 来自 TOC 匹配结果
- `_organize_by_chapters`：移除 `min(level, 2)` 限制，使用真实 `level`
- `_fallback_heading_detection`：无 TOC 时根据编号深度推断层级（`count(".") + 1`）

**权衡**：
- 下游 `ChapterParser` 和 `BatchBuilder` 必须适配多级树（已实现）
- 某些文档的 `#` 是书名而非章节标题，需通过硬边界设计处理（详见第 15 节）

---

## 14. 为什么存储粒度与查询粒度解耦？

**选择**：引入 `content_blocks` 表作为**存储粒度**（向量化粒度，~6000 字符），引入 `heading_maps` 表作为**查询粒度**（按章节检索的入口），两者通过 `block_db_ids` 字段关联。LLM 实体提取时，`BatchBuilder` 将同一 section 的多个 content_blocks 聚合为大粒度 batch request。

**原因**：
- **向量化粒度与 LLM 粒度天然不同**：向量化 API（BigModel embedding-3）有严格的 3072 tokens 上限，而 LLM（kimi-k2.5）支持 256K 上下文。存储粒度应服从更严格的向量化上限，LLM 提取时聚合多个小 block 即可保持大粒度
- **减少向量化超限错误**：按 6000 字符切分存储，向量化时直接提交无需二次切分，避免 API 硬性报错
- **保留子章节查询能力**：`heading_maps` 中 `###` 标题映射到其父 `##` section 的所有 blocks，用户查询 "2.1.1" 仍能获取整个 2.1 section 的内容。支持 `LIKE` 子串匹配和递归子标题查询
- **独立演进**：存储粒度由 `BLOCK_MAX_CHARS` 控制，LLM 聚合粒度由 `LLM_BATCH_MAX_CHARS` 控制，查询粒度由 `heading_maps` 策略控制，三者互不阻塞

**实现**：

```
ChapterParser.parse_tree() → _collect_section_content() → _build_content_blocks_and_maps()
    → content_blocks[]（向量化粒度） + heading_maps[]（查询粒度）
    → BatchBuilder.build_batches()（按 section 聚合 blocks → LLM 粒度）
```

**content_blocks 的生成**：
1. 按 `##` section 聚合自身 content + 所有子孙 content（保留 Markdown 层级）
2. 使用 `_split_for_embedding()` 按 `BLOCK_MAX_CHARS`（默认 6000）切分为向量化粒度 blocks
3. 切分策略：段落边界 → 句子边界 → 字符硬切分（兜底），保护代码块/表格不被截断
4. 每个 block 有独立 `block_id`（`b_0`, `b_1`…）和全局 `seq_index`

**heading_maps 的映射**：
- `##`/`###`/`####` 标题 → 该 section 的所有 block_db_ids
- 支持 `LIKE '%title%'` 子串匹配查询
- 支持递归查询：输入 "2.1" 可自动包含 "2.1.1"、"2.1.2" 等子章节内容

**权衡**：
- content_blocks 数量增加 3-5 倍（粒度变细），但单条更小，FAISS 搜索精度提升
- 增量更新仍以 `###` 计算 content_hash，但 `##` 下任何 `###` 变更会导致整个 section 的 blocks 重新生成
- ~~兼容层 `chunks` 表已废弃并删除~~。`_EntityChunkBridge` 从 `content_blocks` 表全量重建

---

## 15. Content Block 的聚合、切分与入库链路

### 概念定义

| 术语 | 定义 | 示例 |
|------|------|------|
| **Section** | `ChapterParser` 按 Markdown 标题层级解析出的 `##` 级别节点，包含自身 content 和所有子孙 content | `{"title": "2.1 GPIO Configuration", "level": 2}` |
| **Content Block** | 数据库 `content_blocks` 表中的一行，是**LLM batch 和向量化的最小单位** | `{"block_id": "b_0", "content": "...", "seq_index": 0}` |
| **Sub-block** | 超长 section 被 `split_text_by_paragraphs` 拆分后的产物，属于同一 section 的不同片段 | `{"block_id": "b_1", "content": "...后半", "seq_index": 1}` |
| **Heading Map** | 数据库 `heading_maps` 表中的一行，记录标题到 block 集合的映射 | `{"heading_title": "2.1", "block_db_ids": [1, 2]}` |

### 生成链路

```
PDF → Markdown → ChapterParser.parse_tree()
    → _collect_section_content()  [递归聚合 ## + 子孙 content]
    → split_text_by_paragraphs()  [按段落边界切分]
    → content_blocks[]
    → BatchBuilder.build_batches() [构建 LLM batch]
    → _save_blocks_to_db()        [写入 SQLite]
    → content_blocks + heading_maps
```

### 聚合（`_collect_section_content`）

递归收集 `##` section 自身 content + 所有子孙 content，保留 Markdown heading 层级：

```python
def _collect_section_content(node):
    lines = []
    if node.title:
        lines.append(f"{'#' * node.level} {node.title}")
    if node.content:
        lines.append(node.content)
    for child in node.children:
        child_content = _collect_section_content(child)
        if child_content:
            lines.append(child_content)
    return "\n\n".join(lines).strip()
```

第一个 section 若 root（`#`）有 content，会将 root content 作为 preface 前置到该 section。

### 切分（`split_text_by_paragraphs`）

按优先级逐级 fallback，保护结构化块不被截断：

1. **结构化块保护**：代码块（```` ``` ````）、HTML table、Markdown table 先提取为占位符（`\x00BLOCK{N}\x00`），避免切分破坏结构
2. **段落边界**（`\n\n+`）：按空行切分，恢复占位符
3. **超长段落二次切分**：单个段落仍超限时，按句子边界（`(?<=[.!?。！？])\s+`）切分
4. **极限 fallback**：单句仍超长则按字符边界硬切分并记录 warning

### 入库（`_save_blocks_to_db`）

1. **清理旧数据**：删除旧 blocks、heading_maps，同时清理 FAISS 中旧向量（使用偏移 ID）
2. **插入 content_blocks**：每个 block 的 metadata 包含 `content_hash`、按 `section_title` 过滤的 `extracted_entities` / `extracted_relations` / `image_descriptions`
3. **插入 heading_maps**：`block_ids` 替换为 SQLite `db_id`，`##` 和 `###` 都指向同一 block 集合
4. ~~兼容层已删除~~：不再写入 `chunks` 表

### Content Block 与 Sub-block 的独立性

Sub-block 继承父 section 的属性，但有自己的 `db_id`：

| 属性 | 继承/独立 | 说明 |
|------|----------|------|
| `content` | **独立** | 拆分后的片段内容 |
| `block_id` | **独立** | `b_N` 格式 |
| `db_id` | **独立** | SQLite 自增主键，FAISS 索引用它（加 `_BLOCK_FAISS_OFFSET` 偏移） |
| `section_title` | 继承 | 与父 section 相同 |
| `content_hash` | 继承 | 父 section 的 hash（用于增量更新比对） |
| `extracted_entities` | 继承 | 父 section 缓存的实体引用 |
| `extracted_relations` | 继承 | 父 section 缓存的关系 |
| `image_descriptions` | 继承 | 父 section 合并后的图片描述 |

### 为什么 Block 粒度 = 向量化粒度？

数据库 `content_blocks` 表中的一行 = 一个向量化单位。这意味着：
- 每个 block（包括 sub-block）独立向量化
- block 在 SQLite、FAISS、`_EntityChunkBridge` 中都是**一等公民**
- 不存在"一个 block 对应多个向量"或"多个 block 合并为一个向量"的情况
- `_BLOCK_FAISS_OFFSET` 确保 block db_id 在 FAISS 中唯一（偏移后不与旧 chunks ID 冲突）

### `LLM_BATCH_MAX_CHARS` 与 `BLOCK_MAX_CHARS` 的分工

| 配置 | 默认值 | 作用阶段 | 说明 |
|------|--------|---------|------|
| `LLM_BATCH_MAX_CHARS` | 10000 | Stage 2 | 控制 LLM batch 请求的最大字符数。BatchBuilder 将同一 section 的多个 content_blocks 聚合后，若超限再切分 |
| `BLOCK_MAX_CHARS` | 6000 | Stage 2 | 控制 content_blocks 的存储切分粒度（向量化粒度）。基于 BigModel embedding-3 经验值，6000 字符通常不超过 3072 tokens 上限 |

**三层粒度解耦**：
- **存储粒度** = `BLOCK_MAX_CHARS`（~6000 字符）：content_blocks 表中的每条记录
- **LLM 提取粒度** = `LLM_BATCH_MAX_CHARS`（~10000 字符）：BatchBuilder 聚合同一 section 的多个 blocks
- **查询粒度** = `heading_maps`：任意层级标题映射到 block 集合

---

## 16. Heading Map 查询粒度与 FAISS ID 偏移

### heading_maps 的查询语义

| 查询方式 | 实现 | 返回内容 |
|---------|------|---------|
| `query_by_heading("CPU")` | `heading_maps` 表 `heading_title LIKE '%CPU%'` 子串匹配 | 匹配的 section 的所有 content_blocks（按 heading_level 排序） |
| `query_by_heading_recursive("2.1")` | 先匹配标题，再递归收集 `parent_heading = 目标标题` 的所有子标题 | 该 section 及其所有子章节的 content_blocks |
| `chapter_regex("^2\\.")` | 先查 heading_maps 缩小范围，内存中正则过滤 | 匹配标题对应的 content_blocks |
| `semantic_search("GPIO")` | FAISS 搜索 | 匹配的 content_blocks |

**关键设计**：
- `query_by_heading` 支持 `LIKE` 子串匹配，用户输入 "CPU" 可匹配 "2.1 CPU Architecture"
- `chapter_regex` 利用 heading_maps 索引（`idx_headings_title`）避免全表扫描 content_blocks
- `##`/`###`/`####` 共享同一 block 集合，保证查询时不遗漏子章节内容
- `heading_level` 和 `parent_heading` 字段保留层级关系，支持递归子标题查询

### FAISS ID 偏移（`_BLOCK_FAISS_OFFSET = 10_000_000`）

content_blocks 表的 `db_id` 从 1 开始自增。旧架构中兼容层 chunks 表也使用自增 ID，两者共享 FAISS 索引时可能冲突。

```python
# stage6 入库：block db_id 存入 FAISS 时加偏移
faiss_id = block_db_id + _BLOCK_FAISS_OFFSET  # db_id=1 → faiss_id=10000001

# semantic_search 查询：减去偏移还原 block db_id
if faiss_id >= _BLOCK_FAISS_OFFSET:
    block_db_id = faiss_id - _BLOCK_FAISS_OFFSET
```

**原因**：
- 若不加偏移，block db_id=1 和旧 chunk db_id=1 在 FAISS 中可能指向同一向量，造成数据污染
- 偏移量 10_000_000 足够大，可覆盖任何合理的 db_id 范围

**权衡**：
- 偏移增加了 `semantic_search` 和 `stage6` 的复杂度，需解析 `custom_id` 还原 block_db_id
- 未来清理旧 FAISS 索引后，可考虑是否保留偏移

---

## 为什么需要 EntityChunkBridge 双向桥接索引？

**选择**：在 `KnowledgeBaseService` 内部维护一个纯内存的 `_EntityChunkBridge`，建立 `block_db_id ↔ (entity_type, entity_name)` 的双向多对多映射。零 schema 变更、零数据模型变更，从 SQLite `content_blocks.metadata["extracted_entities"]` 构建。

**原因**：
- **补齐单向关联的缺失**：原有架构中 block → entity 的关联已存在（通过 `metadata.extracted_entities`），但 entity → block 的反向查询缺失。`graph_provenance` 此前通过逐文档遍历所有 blocks 做暴力扫描（O(N)），不可扩展
- **打通不同粒度空间**：FAISS 向量索引操作的是 block 粒度（文本片段 + embedding），NetworkX 图谱操作的是 entity 粒度（结构化实体 + 关系）。桥接索引使两者可以双向导航
- **支持策略层灵活组合**：机制层只提供原子操作（`_chunk_to_entities` / `_entity_to_chunks` / `_get_chunk` / `_semantic_hits` / `_get_subgraph`），策略层可自由组合出任意跨粒度查询（语义→图谱→语义闭环、路径文本证据链等）

**实现**：
```python
_forward:  dict[int, set[_EntityRef]]     # block_id → {EntityRef}
_reverse:  dict[_EntityRef, set[int]]     # EntityRef → {block_id}
```

**生命周期**：
- `KnowledgeBaseService.__init__` → `bridge.rebuild()` 全量构建
- `ingest_document` / `reprocess_document` 完成后 → `_sync_bridge_for_doc()` 增量同步
- `attach()` 幂等设计：同一 block 重复 attach 会先 detach 旧绑定，避免索引累积
- `detach()` 双向清理：同时清除 `_forward` 和 `_reverse`，空集合自动删除

**不同粒度关联矩阵**：

| 方向 | 粒度 | 已有/新增 | 机制 |
|------|------|----------|------|
| block → entity | 向量→图谱 | 已有 | `metadata.extracted_entities` |
| entity → block | 图谱→向量 | **新增** | `_entity_to_chunks()` O(1) |
| document → entity | 文档→图谱 | 已有 | `GraphEntity.source_doc_ids` |
| entity → document | 图谱→文档 | 已有 | `GraphEntity.source_doc_ids` |
| chapter → entity | 章节→图谱 | 已有 | `GraphEntity.source_chapter` |
| block → subgraph | 向量→子图 | 已有 | `search_with_graph()` |
| subgraph → block | 子图→向量 | **可组合** | 子图实体 → `_entity_to_chunks()` |

**权衡**：
- 内存索引，重启需重建（但 `KnowledgeBaseService` 为长生命周期单例，初始化成本可忽略：几百文档 × 几十实体）
- 不解决"实体级语义搜索"（如搜索 "GPIO" 也匹配 "General Purpose Input Output"），那是实体 embedding 的范畴，不在本次机制设计范围内

---

## 联合查询与 Agent 自主推理

**设计**：`search_with_graph()` 使用原子操作组合实现语义搜索与图谱查询的联合。

**原子操作层（机制，无策略参数）**：
- `_semantic_hits(query, top_k)` → FAISS 搜索，返回 `[(block_db_id, score), ...]`
- `_get_chunk(block_db_id)` → 深拷贝获取 Chunk 对象（内存表示）
- `_chunk_to_entities(block_db_id)` → 桥接索引正向查询，返回 `set[_EntityRef]`
- `_entity_to_chunks(entity_type, name)` → 桥接索引反向查询，返回 `set[block_db_id]`
- `get_entity(type, name)` / `get_subgraph(type, name, depth)` → 图谱空间查询

**策略组合示例**：

```
search_with_graph（语义→图谱）:
  _semantic_hits → _get_chunk → _chunk_to_entities → get_subgraph

graph_provenance 优化（图谱→向量）:
  _entity_to_chunks → _get_chunk（替换原有 O(N) 扫描）

语义→图谱→语义闭环（可扩展策略）:
  _semantic_hits → _chunk_to_entities → get_subgraph → 子图实体 _entity_to_chunks → _get_chunk
```

**block+实体联合返回**：`get_content_with_entities(block_db_id)` 使用 `_get_chunk` + `_chunk_to_entities` + `get_entity` + `get_entity_relations`，返回 block 内容 + 全局图中最新状态的关联实体/关系（深拷贝隔离）。

**Agent 自主推理能力**：
- Agent 可先用 `search_with_graph` 找到相关文本和图谱实体
- 再用 `graph_neighbors`/`graph_path`/`graph_subgraph` 做多跳推理
- 通过 `find_chunks_by_entity` 从任意实体反向查找支撑它的原始文本 blocks
- 发现错误时用 `graph_feedback` 标记，`feedback_score` 实时汇总到实体属性
- 发现缺失/错误关联时用 `graph_add_entity`/`graph_add_relation`/`graph_update_entity` 动态修正

---

## 反馈与数据质量闭环

**设计**：`feedback` 表 + `graph_feedback` 工具建立数据质量闭环。

**机制**：
- 用户对实体/关系提交 `rating`（+1 正确 / -1 错误）+ `comment`
- `get_entity_feedback_score()` 汇总评分，同步更新到 `GraphEntity.feedback_score`
- `confidence` 字段标记 LLM 提取置信度（默认 1.0）
- `verified` 字段标记人工验证状态
- 低 confidence 或负 feedback_score 的实体可被 Agent 优先复核

---

## IC 设计关系类型扩展

**新增关系类型**（覆盖 STA、CDC、电源域、参数化等场景）：

| 关系类型 | 语义 |
|---------|------|
| `DRIVES` | Signal/Module → Signal（驱动关系） |
| `DRIVEN_BY` | Signal → Signal/Module（被驱动关系） |
| `TIMING_PATH` | Entity → Entity（时序路径，STA 分析） |
| `CLOCK_GATED_BY` | Module/Signal → Signal（时钟门控控制） |
| `RESET_BY` | Module/Signal → Signal（复位来源） |
| `PARAMETERIZED_BY` | Module → Parameter（参数化配置） |
| `INSTANCE_OF` | Module → Module（实例化：instance → definition） |

---

## 17. 为什么 PDF 解析使用 BigModel 专用 API？

**选择**：PDF 解析主路径使用 `BigModelParserClient`，调用 BigModel (智谱) 专有的 Expert 文件解析 API（`/files/parser/create` + `/files/parser/result`）。

**原因**：
- **解析质量**：BigModel Expert API 在中文技术文档（如 TI Reference Guide、ARM Technical Overview）的解析上表现优异，能准确识别标题层级、保留图片、输出结构化的 Markdown
- **图片提取**：自动提取 PDF 中的矢量图/位图区域，输出 `images/` 目录 + Markdown 引用，与 pipeline 下游的 multimodal 处理无缝衔接
- **成本可控**：0.012 元/页，对于几百页的技术文档成本可接受

**限制**：
- **厂商锁定**：该 API 非 OpenAI-compatible，端点为 BigModel 专有，无法通过修改 `base_url` 切换至其他厂商（如 Kimi、OpenAI、Azure 等）
- **依赖外部服务**：需要有效的 BigModel API Key，且受限于 BigModel 服务的可用性

**缓解措施**：
- **本地 fallback**：`DocGraphPipeline` 在 `BigModelParserClient` 初始化失败或解析异常时，自动降级到本地 `PDFParser`（PyMuPDF + TOC 驱动章节识别）
- **输出格式对齐**：本地解析器的输出格式（Markdown + `images/` 目录）与 BigModel 完全一致，确保 downstream pipeline（`ChapterParser` → `BatchBuilder` → `EntityExtractor`）无需感知解析来源差异
- **零数据丢失**：无论使用哪种解析器，均遵循相同的零截断原则

**未来扩展**：
- 如需支持其他厂商的 PDF 解析服务（如 Kimi 文件解析、Azure Document Intelligence 等），需新增对应的 `ParserClient` 实现，并在 `DocGraphPipeline` 中注册为新的解析路径
- 长期可考虑抽象 `BaseParserClient` 接口，将不同厂商的解析服务统一接入，由配置决定使用哪个解析后端

---

## 18. 跨文档属性差异处理机制

### 18.1 为什么引入 `doc_properties`？

**核心问题**：不同技术文档对同一对象的描述完整性不同。文档 A 可能详细描述寄存器的所有属性（address_offset、width、access、reset_value），文档 B 可能只是引用该寄存器，只提到名称。当两者都提取为 Register 实体并入全局图时，如何保留各自原始描述，同时提供统一查询视图？

**三层数据模型**：

| 层级 | 字段 | 作用 | 示例 |
|------|------|------|------|
| 全局合并属性 | `properties` | 跨文档合并后的统一属性（冲突时后来者覆盖） | `{"width": 16, "access": "R/W"}` |
| 文档原始快照 | `doc_properties[doc_id]` | 每个文档提取时的原始属性，永不丢失 | `{"doc_A": {"width": 16}, "doc_B": {"width": 32}}` |
| 来源追踪 | `source_doc_ids` | 该实体被哪些文档提及 | `["doc_A", "doc_B"]` |

**合并策略**（`GraphStore._add_entity_unsafe`）：
1. **属性值冲突**（doc_A: width=16, doc_B: width=32）：比较两个来源文档的**信息完整性评分**（非空属性数量），评分高的一方写入 `properties`。`doc_properties` 各自保留，同时生成 `conflict_logs` 记录。完整性相同时保留旧值（保守策略）
2. **属性互补**（doc_A 有 addr，doc_B 有 access）：`properties` 取并集（新键直接追加，不受完整性影响），`doc_properties` 各自保留
3. **详细 vs 简略**（doc_A 有全部属性，doc_B 只有名称）：`properties` 保留更完整文档的属性值，`doc_properties` 保留 B 的简略快照
4. **无法推断来源**（如手动修改全局图、旧数据无 `doc_properties`）：保留现有值，保守策略

**完整性评分**：
```python
def _completeness_score(props: dict) -> int:
    return sum(1 for v in props.values() if v not in (None, "", []))
```

简单、通用、无需维护类型权重表。None、空字符串、空列表不计分。

**查询时的属性替换**（`_apply_doc_properties`）：
```python
def _apply_doc_properties(entity, doc_id):
    if doc_id and doc_id in entity.doc_properties:
        entity = copy.deepcopy(entity)  # 深拷贝，不修改全局图
        entity.properties = entity.doc_properties[doc_id]
    return entity
```

调用 `graph_query(entity_type="Register", name="TBCTL", doc_id="doc_A")` 时，返回的 `properties` 是 doc_A 的原始快照，而非全局合并值。

**冲突日志表**（`conflict_logs`）：
```sql
CREATE TABLE conflict_logs (
    id INTEGER PRIMARY KEY,
    entity_type TEXT, name TEXT,
    property_key TEXT, old_value TEXT, new_value TEXT,
    timestamp TEXT, doc_id TEXT
);
```

### 18.2 为什么引入 `Product` 实体？

文档解析时自动从产品型号格式（如 `TMS320F28379D`）提取，建立 `Product --[HAS_MODULE]--> Module` 关系，作为产品级查询入口。

**Agent 推理流程**：
```
graph_query(entity_type="Product", name="TMS320F28379D")
→ graph_neighbors(Product, rel_type="HAS_MODULE", doc_id="doc_hash")
→ graph_query(Module("DMA_Controller"), doc_id="doc_hash")
→ 获取 doc_properties["doc_hash"]["address_base"] = "0x4000"
```

**权衡**：
- 存储量增加：每个文档的每个实体多保存一份属性（通常 < 10K 实体，可接受）
- 完整性评分基于属性数量而非语义重要性。例如 doc_A 有 `description`（长文本）和 `width`，doc_B 有 `addr`、`width`、`access`，doc_B 的评分更高（3 > 2），`width` 冲突时选择 doc_B。这在绝大多数场景下合理，但极端情况下可能不反映"哪个属性更重要"
- 当旧数据无 `doc_properties` 时（如手动编辑全局图、旧版本导入的数据），无法推断属性来源，默认保留现有值。建议修改后调用 `/rebuild_global_graph` 重建
- 产品型号提取依赖启发式正则，可能误识别或漏识别

---

## 19. 为什么 Pipeline 六阶段拆分？

**选择**：将 `DocGraphPipeline._process_one` 从三阶段（`stage1_parse` / `stage2_build_jsonl` / `stage3_ingest`）拆分为六阶段（`stage1_parse` / `stage2_build_jsonl` / `stage3_submit_batches` / `stage4_ingest_results` / `stage5_build_embed_jsonl` / `stage6_submit_embed_batches`），其中 `stage3` 提交 LLM Batch API 后将**原始结果文件保存到磁盘**，`stage4` 从磁盘读取结果并解析入库（**不含向量化**），`stage5` 本地构建 Embedding 同步 API 输入 JSONL，`stage6` 调用同步 Embedding API 逐条完成向量化。

**原因**：
- **中间产物可审计**：`knowledge_base/batch/{doc_id}_results.jsonl` 包含 LLM 的原始 JSON 返回，可用于调试提取质量、审计 LLM 行为
- **阶段可独立执行**：stage3（API 调用，产生费用和延迟）与 stage4/5/6（本地处理，零成本）解耦。用户可以在 stage3 完成后审查结果，再决定是否执行 stage4
- **失败可重试**：stage4 入库失败（如数据库锁定）时，无需重新调用 Batch API（避免重复付费和排队），直接重试 stage4 即可；同理 stage6 向量化失败仅需重试 stage6
- **结果可编辑**：用户可手动修改 `batch/{doc_id}.jsonl` 后重新执行 stage3（增量过滤仍会生效），或修改 `results.jsonl` 后重新执行 stage4，修正 LLM 提取错误而无需重新调 API
- **向量化解耦**：stage4 完成后 content_blocks 状态为 `embedded`，知识图谱已可查询；stage5/stage6 独立执行向量化，不阻塞图谱使用
- **向后兼容**：`stage3_ingest()` 保留原签名，内部委托给 `stage3_submit_batches` + `stage4_ingest_results` + `stage5_build_embed_jsonl` + `stage6_submit_embed_batches`，现有调用方无需修改

**中间产物清单**：

| 阶段 | 产物 | 路径 | 说明 |
|------|------|------|------|
| Stage 1 | result.md | `parsed/{doc_id}/result.md` | 解析后的 Markdown（可人工编辑） |
| Stage 2 | requests.jsonl | `batch/{doc_id}.jsonl` | LLM Batch API 输入请求 |
| Stage 2 | batch_info.json | `batch/{doc_id}_batch_info.json` | request → chapter 映射 |
| Stage 3 | results.jsonl | `batch/{doc_id}_results.jsonl` | LLM Batch API 原始返回结果（Chat 模式下格式完全一致，Stage 4 零修改复用） |
| Stage 3 | incremental.json | `batch/{doc_id}_incremental.json` | 增量分析摘要 + result.md hash |
| Stage 4 | 子图谱 | `graphs/{doc_id}.json` | 文档级图谱快照 |
| Stage 5 | embed.jsonl | `batch/{doc_id}_embed.jsonl` | Embedding 同步 API 输入请求（单文本，custom_id 编码 db_id） |
| Stage 6 | — | — | 同步调用 Embedding API 逐条处理，结果直接入库（无中间产物文件） |

**状态管理**：
- `PROCESSING` → stage3 提交中
- `BATCH_SUBMITTED` → stage3 完成，等待 stage4
- `embedded` → stage4 完成（content_blocks 已入库，图谱已构建，但未向量化）
- `DONE` → stage6 完成（向量化已入库）

**权衡**：
- 磁盘占用增加：每个文档额外保存 results.jsonl、incremental.json、embed.jsonl 等（通常总计几十 KB 到几十 MB）
- stage3/stage4 之间由 `_incremental.json` 自动校验一致性：若 `result.md` 在 stage3 之后被修改，stage4 会检测到 hash 或 title 集合不匹配，自动删除旧的 `_incremental.json` 并重新处理，无需用户手动干预

---

## 20. 审计修复经验总结

**背景**：
- 2026-04 代码审计发现 21 项缺陷（9 个 P0 数据损坏风险、7 个 P1 可靠性缺陷、5 个 P2 代码质量问题），已全部修复并通过 283 个测试验证。
- 2026-06 再次审计发现 Critical/High 级安全与数据完整性问题，按「最小紧急修复」方案修复后测试数达到 400 个。

**关键教训**：

| 教训 | 来源缺陷 | 修复措施 |
|------|---------|---------|
| **JSON 修复正则必须验证** | `_safe_parse_json` 用 `\x01` 替代 `\1` | 单字符修复，新增边界测试 |
| **用户路径必须沙箱校验** | Plugin 工具直接接受 `path`/`output_dir` | `_resolve_allowed_path` 校验，禁止跳出允许目录 |
| **敏感配置输出必须强制脱敏** | `tool_config` 允许 `mask_sensitive=false` | 忽略请求参数，敏感 key 始终返回 `***` |
| **外部 ZIP 必须校验条目路径** | BigModel 解析结果 ZIP 未过滤 `..` | 拒绝含 `..`、绝对路径、盘符的条目，校验 resolve 后仍在输出目录 |
| **多存储写入顺序必须避免幽灵记录** | SQLite 先写、FAISS 后写导致 DB 引用缺失向量 | 改为先写 FAISS 再写 SQLite；失败时回滚 SQLite |
| **保存 + 辅助日志必须原子失败处理** | `_save_global_graph()` 成功但冲突日志写入失败 | `save_ok` 跟踪，失败时把回滚状态重新落盘 |
| **返回全局对象前必须深拷贝（含关系）** | `_apply_doc_properties_to_relation` 使用 `copy.copy` | 统一使用 `copy.deepcopy` |
| **属性合并必须深拷贝可变值** | `merged_props[k] = v` 直接引用可变对象 | 合并时 `copy.deepcopy(v)`，快照也深拷贝 |
| **删除文档贡献必须清理 per-doc 快照** | `remove_document_contributions` 保留 `doc_properties[doc_id]` | 保留节点/边时同步 `pop(doc_id, None)` |
| **资源句柄必须上下文管理** | `PDFParser.parse()` 中 `fitz.open` 异常时不关闭 | `with fitz.open(...) as doc:` |
| **并发 Future 必须超时** | `submit_parallel_batches` 中 `future.result()` 无 timeout | 传递 `timeout=timeout` |
| **Batch 失败不可静默丢数据** | `extract_from_requests` 捕获异常后返回空结果 | 记录错误后重新抛出 |
| **日志默认必须输出到 stderr** | `setup_logging` 默认 `sys.stdout` | 改为 `sys.stderr`，避免污染 plugin JSON 协议 |
| **返回全局对象前必须深拷贝** | `_apply_doc_properties` 修改原始节点 | `copy.copy` + `dict()` 深拷贝属性 |
| **删除操作必须同步清理索引** | `remove_document_contributions` 遗漏索引 | 节点移除循环中调用 `_remove_from_property_index` |
| **多存储系统操作必须原子化** | SQLite 已插入但 FAISS 向量化失败 | 统一批量写入，失败时 `remove_doc` 回滚 |
| **全局状态修改前必须快照** | `reprocess_document` 失败后半空状态 | `copy.deepcopy` 备份 `_g` + `_property_index` |
| **函数参数声明后必须使用** | `_merge_image_descriptions` 忽略 `chapter_title` | 按 `chapter_title` 过滤图片 |
| **过滤逻辑必须用完整复合键** | `_build_metadata` 按名称过滤导致跨类型污染 | `(type, name)` 元组匹配 |
| **文件写入必须原子化** | `_save_global_graph` 直接覆写 | 临时文件 + `os.replace`；失败时从 `.bak` 恢复 |
| **跨阶段产物必须校验一致性** | Stage3/Stage4 增量分析结果不一致 | 持久化 `_incremental.json` + hash 校验 |
| **配置路径加载/保存必须一致** | `_save_global_graph` 硬编码 `"graphs"` | 统一使用 `Config.GRAPH_OUTPUT_DIR` |
| **警惕外部库的导入级副作用** | `pymupdf4llm` 导入时激活 ONNX layout，静默改变 `find_tables()` 结果 | 移除 `pymupdf4llm` 依赖及未触发的 fallback 代码 |

**新增开发原则**：详见 `AGENTS.md`「防御性编程与状态安全原则」，包含副作用隔离、索引一致性、失败回滚、输入验证、跨阶段校验、配置统一、资源生命周期管理 7 条原则。

---

## 21. 为什么 thinking 参数必须始终传递？

**选择**：LLM Batch API 请求中无论 `LLM_THINKING_ENABLED` 配置为 `0` 还是 `1`，都在 body 中显式传递 `extra_body={"thinking": {"type": "enabled"/"disabled"}}`。

**原因**：
- **Kimi K2.6 等模型 thinking 默认开启**：如果不传递参数，模型始终使用 thinking 模式，用户配置 `thinking_enabled=0` 将无效
- **显式控制**：`{"type": "disabled"}` 可可靠关闭 thinking，`{"type": "enabled"}` 可确保开启（配合 `keep: "all"` 可实现多轮推理连贯）
- **Batch API 一致性**：同步对话客户端（`LLMChatClient`）和 Batch API 请求（`EntityExtractor._build_batch_requests()`）使用相同的参数传递逻辑

**实现**：
- `llm_chat_client.py`：`extra_body={"thinking": {"type": "enabled" if self.thinking_enabled else "disabled"}}`
- `doc_graph_pipeline.py`：在 `_build_batch_requests()` 的 request body 中同样添加 `extra_body`
- 用户编辑 `batch/{doc_id}.jsonl` 时若删除了 `extra_body`，stage3 读取后会自动补充

---

## 18. GapsFirstScanner：Caption-driven Linear Extractor

**架构定位**：`GapsFirstScanner` 是本地 PDF 解析器（`PDFParser` fallback）的页面级提取引擎，负责从单页 PDF 中提取图片（diagrams）和表格。它替代了早期基于 UZN（Unified Zone Recognition）的全页面扫描方案，核心设计思想是**"像人一样线性阅读页面"**——从上往下，利用天然存在的 y 轴分割信息（页眉、Caption、章节标题）来划分处理区间。

**历史演进**：
- **早期版本**（UZN 方案）：从全页面角度出发，统一识别 diagram 和 table 区域，代码复杂（2000+ 行），性能极差
- **当前版本**（GapsFirstScanner）：Caption-driven 线性提取，~600 行，性能提升 5-10x，提取质量显著改善

### 18.1 核心机制：Zone 模型与 Hard Separator

页面在 y 轴上被「硬分割」切分为多个区间（Zone），提取算法只在 Zone 内部搜索：

```
页面 y 轴：
├─ header ──┤  ← 硬分割（页眉区域）
│           │
├─ heading ─┤  ← 硬分割（章节标题，如 "B2.2 Channel fields"）
│           │
├─ caption ─┤  ← 硬分割（Figure/Table Caption）
│           │
├─ body_text┤  ← 硬分割（大段正文块，高>20pt 且 宽>60%页面宽度）
│           │
├─ caption ─┤  ← 硬分割（另一个 Caption）
│           │
├─ footer ──┤  ← 硬分割（页尾区域）
```

**Hard Separator 构建规则**（按优先级从高到低）：

| 类型 | 来源 | 合并规则 |
|------|------|----------|
| header/footer | 配置边距 | 固定区域，不可被其他 separator 覆盖 |
| figure_caption | 正则匹配 `^Figure\s+[A-Z]?\d+...` | 相邻 separator 重叠时，高优先级覆盖低优先级 |
| table_caption | 正则匹配 `^(Table|表)\s*[A-Z]?\d+...` | 同上 |
| heading | **TOC 优先匹配**，其次字号 > 阈值 或 粗体 | 同上 |
| body_text | 高>20pt 且 宽>60%页面宽度 | 低优先级，可被 caption/heading 覆盖 |

相邻硬分割之间的空隙就是一个 `_Zone`，包含：
- `y0`, `y1`：y 轴范围
- `drawings`：落在该 y 范围内的矢量绘图（线条、矩形）
- `text_block_count`：落在该范围内的文本块数量
- `consumed`：是否已被 Figure/Table Caption 消耗

### 18.2 body_text Hard Separator 的权衡

**设计意图**：大段正文块（如多行段落）本身构成天然的 y 轴分割——表格/图片不会跨越正文块出现。

**性能影响**（TI 219 页测试）：
- 无 body_text 硬分割：6.7s
- 有 body_text 硬分割：4.4s
- **body_text 硬分割减少了 Zone 数量**，使后续 figure/table 提取遍历的 zone 更少，反而**提升了性能**

**阈值选择**：`height > 20.0 and width > page_width * 0.6`
- 过宽：会误将短段落识别为硬分割，增加 zone 数量
- 过窄：遗漏真正的正文块，zone 过大导致搜索范围增加
- 当前阈值基于 TI/AMBA/SPRUI07 的实证调优

### 18.2a Zone 合并：跨分隔符的 Drawing

**问题**：图表内部的标注文本（如 "SYSCLK"、"Master Clock"）可能被误判为 heading，产生硬分隔符，将同一幅图切分为上下两个 zone。Drawing 的中心点落入 zone 之间的 gap，导致提取丢失。

**修复**：构建 zone 后，检测是否有 drawing 的 rect 同时跨越相邻两个 zone（中心点在 gap 中，且与两侧 zone 均有 >1pt 重叠）。若存在，合并这两个 zone 并重新分配所有 drawing。

### 18.3 Figure Caption 驱动提取

**核心规则**：不存在无 Figure Caption 的图。一个 Figure Caption 仅对应一个图片。

**搜索逻辑**：
1. 对每个 `figure_caption`，找到其上方和下方最近的、有 drawings 的 zone
2. 比较距离，选择更近的 zone 作为目标
3. 将目标 zone 内的 drawings 按空间邻近度聚类（`CLUSTER_PROXIMITY`）
4. **每个 cluster 渲染为一张独立图片**（早期版本将所有 cluster 扁平化为一张大图，导致过度合并）
5. 用 `_build_clip()` 构建最终裁剪区域：drawing-bounded X，zone-clamped Y，边缘标签扩展，padding

**关键修复**：AMBA page 21 的图被分割为 3 张提取的问题，通过 cluster-based 渲染修复。

### 18.4 Table Caption 驱动提取

**核心规则**：Table Caption 下方的 zone 包含表格。

**搜索逻辑**：
1. 对每个 `table_caption`，找到其下方第一个未被消耗的 zone（`zone.y0 >= caption.y1`，且在 200pt 范围内）
2. 在该 zone 内调用 `find_tables()` 提取表格

### 18.5 Orphan Zone 检测：无 Caption 表格

**设计意图**：可能存在无 Table Caption 的表格（如寄存器位域图、引脚配置表）。

**检测流程**：
1. Figure/Table Caption 驱动的提取会标记某些 zone 为 `consumed`
2. 剩余的、有 drawings 的 zone 就是 "orphan zones"
3. 对每个 orphan zone，调用 `_classify_table_style()` 判断是否是表格
4. 如果是，调用 `find_tables()` 提取

**`_classify_table_style` 算法**：

```python
def _classify_table_style(zone):
    if not zone.drawings:
        return None

    h_lines = sum(1 for r in zone.drawings if r.width > r.height * 3)
    v_lines = sum(1 for r in zone.drawings if r.height > r.width * 3)
    other = sum(1 for r in zone.drawings if not above)

    # 非线条元素占多数 → 框图/位域图，不是表格
    if other > (h_lines + v_lines) * 0.5:
        return None

    # Style A: Grid table（多横线 + 多竖线）
    if h_lines >= 4 and v_lines >= 3:
        if text_density < 0.02:
            return None  # 过滤位域图
        return "grid"

    # Style B: Horizontal / borderless table
    if v_lines <= 1 and h_lines >= 4 and zone.text_block_count >= 3:
        if text_density < 0.01:
            return None
        return "horizontal"   # 统一由 BorderlessTableExtractor 处理

    return None
```

**产出率验证**（TI 219 页）：
- 84 个 orphan zone 触发检测，76 个确实包含表格
- **命中率 90.5%**，误报率极低

### 18.6 自适应表格检测策略

**机制与策略分离原则**：`_classify_table_style` 识别表格风格（机制），`process_page` 根据风格选择 `find_tables` 策略（策略）。

**策略选择规则**：
- 页面有任何 orphan zone 被分类为 `horizontal`（含零高度横线占多数的无边框表格）→ 直接调用 `BorderlessTableExtractor`，不再走 `find_tables`
- 页面只有 `grid` 风格的 orphan zones → 整页使用 `lines_strict` 策略

**原因**：
- `lines_strict` 对 grid 表格精确（误报低），对 horizontal/无边框表格完全失效
- AMBA 零高度横线表格无法被 `find_tables` 任何策略识别，需要专门的几何提取器
- 实测 `find_tables(strategy="lines")` 在 AMBA 上零产出，且对 grid 表格误报更多，因此移除该策略
- 按页自适应、按风格直接路由，避免全文档统一策略的"一刀切"问题

**实测效果**（4 文档基准）：

| 文档 | 统一 `lines` | 自适应策略 | 差异 |
|------|-------------|-----------|------|
| TI (grid 为主) | 37.3s / 142 表格 | **33.7s** / 134 表格 | **-3.6s**，误报减少 |
| AMBA (horizontal 为主) | 20.0s / 14 表格 | **19.8s** / 14 表格 | 几乎相同 |
| SPRUI07 (混合) | 93.4s / 588 表格 | **88.3s** / 537 表格 | **-5.1s**，误报减少 51 个 |
| DC_UG (混合) | 12.9s / 90 表格 | **13.8s** / 114 表格 | +24 表格（horizontal 受益） |

### 18.7 预计算优化

**原始问题**：每个 Table Caption 和每个 orphan zone 都单独调用 `page.find_tables(clip=zone_rect)`，导致 142 次 PyMuPDF C 调用 → 耗时 ~20s。

**优化方案**：
1. `process_page` 开始时，根据页面 orphan zones 的风格决定 `table_strategy`
2. 调用一次 `page.find_tables(strategy=table_strategy)`，获取全页所有表格
3. `_find_tables_in_zone()` 遍历预计算列表，用 `tab_bbox.intersects(clip_rect)` 筛选落在当前 zone 内的表格
4. 不再调用 PyMuPDF C 代码，纯 Python 筛选开销可忽略

**效果**：TI 文档 `find_tables` 调用从 142 次降至 219 次全页扫描（但全页扫描比多次 clip 调用略快），整体性能显著改善。

### 18.8 表格检测底层限制与实验记录

本节记录 GapsFirstScanner 表格检测相关的底层库限制、踩坑过程和实验结论，防止后续重复无意义的工作。

#### 18.8.1 AMBA 零高度线：PyMuPDF 根本性限制

**实验日期**：2026-06-06

**现象**：即使自适应策略正确识别了 AMBA 表格为 `horizontal`，`find_tables(strategy="lines")` 仍然返回 **0 个表格**。

**实验过程**：
1. 渲染 AMBA page 44/47/140 整页图像，确认表格有水平线、无竖直线
2. 检查 `page.get_drawings()` 返回的 drawing 数据：
   ```
   Drawing 0: rect=(84.0,368.5,559.3,368.5), w=475.2, h=0.0
   Drawing 1: rect=(84.0,368.9,559.3,368.9), w=475.2, h=0.0
   ```
3. 对 AMBA page 28/33/34/35 验证：17-58 条 horizontal lines，**100% 是零高度线**（height=0.000000）
4. 测试 `lines_strict`/`lines`/`text` 三种策略，均无法有效提取

**根因**：这些线条的 **height=0.0**（y0 == y1），是数学意义上的**零高度线**。`find_tables` 的内部算法无法识别零高度线条。

**结论与修复**：这是 `find_tables` 的固有缺陷。为支持此类表格，新增 `scripts/parsers/borderless_table_extractor.py`：
- 以 caption 为锚点，在下方区域搜索水平线并去重，得到行边界；
- 用 header 行单词的 x 位置聚类出列中心；
- 按行区间 + 最近列中心分配 word，重建 Markdown 表格。
- `GapsFirstScanner._classify_table_style` 将零高度线/无边框表格统一归类为 `horizontal`；`process_page` 在识别到该风格时直接路由到 `BorderlessTableExtractor`，不再调用 `find_tables`。

**验证**：对 AMBA CHI page 28（`Table B1.1`）实测可正确提取 3 列 4 行的无边框表格。

#### 18.8.2 `tab.cells` 空伪表格：PyMuPDF 边界情况

**实验日期**：2026-06-10

**现象**：`parse()` 执行时出现大量 `ValueError: min() iterable argument is empty`：
```
Table detection failed on page 74: min() iterable argument is empty
Table detection failed on page 83: min() iterable argument is empty
...
```

**实验过程**：
1. 在 `_find_tables_in_zone` 中添加完整 traceback 捕获
2. traceback 指向 `tab.bbox` 属性访问：
   ```
   File ".../pymupdf/table.py", line 1534, in bbox
       min(map(itemgetter(0), c)),
   ValueError: min() iterable argument is empty
   ```
3. 确认 `find_tables()` 偶尔会返回 **cells 为空的伪表格**
4. 访问 `tab.bbox` 时，PyMuPDF 尝试从 cells 计算 bbox，但 cells 为空 → `min()` 失败

**修复**：在 `_find_tables_in_zone` 中访问 `tab.bbox` 前检查 `tab.cells`：
```python
for tab in tables:
    if not tab.cells:  # 防御性跳过空 cells 伪表格
        continue
    tab_bbox = fitz.Rect(tab.bbox)
    ...
```

**深层问题**：空 cells 伪表格的出现说明 orphan zone 检测过于宽松，大量非表格区域触发了 `find_tables`。后续通过收紧 `_classify_table_style` 条件减少了这种情况。

#### 18.8.3 `_fix_drawing_rect` 的隐蔽副作用

**实验日期**：2026-06-10

**现象**：在 `_classify_table_style` 中添加 `r.height == 0.0` 零高度线检测后，基准测试显示 `filtered_zero_height=0`，检测完全失效。

**实验过程**：
1. 检查 `_fix_drawing_rect` 实现：
   ```python
   if rect.height < cls.DRAWING_RECT_MIN_SIZE:  # 0.0 < 0.1
       return fitz.Rect(rect.x0, rect.y0 - 0.5, rect.x1, rect.y1 + 0.5)
   ```
2. 确认 `_fix_drawing_rect` 在 drawing 分配阶段已将零高度线扩展为 height=1.0
3. `_classify_table_style` 遍历的是 `zone.drawings`（已被扩展），所以 `r.height == 0.0` 永远为 False

**教训**：在 `_fix_drawing_rect` **之前**必须先统计原始 drawing 特征。解决方案：
1. `_Zone` 添加 `h_zero_height: int` 字段
2. `process_page` 中先遍历 `raw_drawings` 统计零高度线，再调用 `_fix_drawing_rect`
3. `_classify_table_style` 使用 `zone.h_zero_height` 而非自行计算

#### 18.8.4 `_classify_table_style` 演进与空跑率实验

**实验日期**：2026-06-10

**v1 算法**（旧版）：
```python
if len(zone.drawings) < 10: return None
if h_lines >= 4 and v_lines >= 3: return "grid"
if v_lines <= 1 and h_lines >= 6 and text >= 5: return "horizontal"
```

**v1 问题**：`len < 10` 硬性门槛过滤掉了一些真实表格；不区分 drawing 类型，位域图/框图被误判。

**v2 算法**（当前）：
```python
# 1. 去掉 len<10 门槛
# 2. 区分 horizontal / vertical / other 三类 drawing
# 3. other > lines*0.5 时过滤（非线条元素占多数）
# 4. grid 风格增加文本密度过滤（<0.02 text/pt 过滤位域图）
# 5. horizontal 风格统一路由到 BorderlessTableExtractor（不再使用 find_tables）
```

**空跑率对比实验**（ orphan zone 触发 find_tables 但返回 0 表格的比例）：

| 文档 | v1 空跑率 | v2 空跑率 | 主要过滤手段 |
|------|----------|----------|------------|
| TI | 15.9% | **1.4%** | low_density (~6), zero_height (~2) |
| AMBA | 92.9% | **0.0%** | **zero_height (~111)** — 全部过滤 |
| SPRUI07 | 10.4% | **8.8%** | low_density (~34) |
| DC_UG | 31.0% | **28.6%** | low_density (~29) |

**关键发现**：
- AMBA 的 100% 空跑由零高度线导致，v2 的 zero_height 过滤将其降为 0%
- DC_UG 仍有 ~28% 空跑，因为 DC_UG 的部分 orphan zones 确实是表格区域但 `find_tables` 未检出（非误判问题）
- 文本密度过滤（0.02 text/pt）有效识别了位域图（大量 lines + 极少 text）

#### 18.8.5 各文档 orphan zone 特征

| 特征 | TI | AMBA | SPRUI07 | DC_UG |
|------|-----|------|---------|-------|
| 空跑主因 | 位域图误判 | **零高度线** | 少量非表格区域 | 低文本密度 |
| success density | 5.65/100pt | N/A | 4.92/100pt | 3.38/100pt |
| empty density | 5.14/100pt | 3.12/100pt | 5.71/100pt | 1.99/100pt |
| success text | 23.2 | N/A | 10.8 | 12.6 |
| empty text | 7.0 | 14.6 | 11.5 | 7.5 |

#### 18.8.6 `pymupdf4llm` 导入副作用与依赖移除

**实验日期**：2026-06-12

**问题现象**：SPRUI07 page 80（PDF 页码 79，0-based）在本地 PDF fallback 解析中，**Figure 1-29 错误认领了 Table 1-28 的 zone，Figure 1-30 完全丢失**，最终只提取出 2 张图（期望 3 张）。现象仅在 `PDFParser` 被导入后出现；单独运行 `GapsFirstScanner.process_page()` 时结果正确。

**初步假设**：测试者曾怀疑 PyMuPDF 存在"内部状态错误"，因为调用 `_extract_horizontal_decorations()` 后再处理 page 80 会触发问题。

**实验方法**：

1. **最小复现**：构造独立脚本，分别测试以下场景对 page 80 的影响：
   - 不导入 `pdf_parser`，直接调用 `GapsFirstScanner.process_page()`
   - 导入 `parsers.pdf_parser` 后再调用 `process_page()`
   - 仅导入 `pymupdf4llm` 后再调用 `process_page()`
   - 分别单独调用 `page.get_drawings()` / `page.get_text("dict")` / `_extract_horizontal_decorations()` 后再处理

2. **对照实验**：对比 Layout 激活前后 `page.find_tables(strategy="lines_strict")` 的返回结果。

3. **根因定位**：检查 `pymupdf4llm` 导入时的副作用，确认其是否修改 `pymupdf` 模块状态。

**实验结果**：

| 前置操作 | page 80 图片数 | 是否正确 |
|---------|--------------|---------|
| 无前置 | 3 | ✅ |
| `import pymupdf4llm` | 2 | ❌ |
| `import parsers.pdf_parser` | 2 | ❌ |
| `_extract_horizontal_decorations(page)` | 2 | ❌ |
| `page.get_text("dict")` | 2 | ❌ |

关键发现：
- `page.get_drawings()` 和 `page.get_text("dict")` 在 Layout 激活前后返回**完全相同**的数据。
- 差异完全集中在 `page.find_tables()` 的返回结果：

| Layout 状态 | `find_tables` 返回的关键 bbox | 含义 |
|------------|---------------------------|------|
| 未激活 | `(54.1, 170.1, 557.8, 236.8)` | Table 1-28，边界清晰 |
| 未激活 | `(54.1, 350.8, 557.8, 439.1)` | Table 1-29，边界清晰 |
| 激活后 | `(65.3, 172.9, 544.4, 303.6)` | **Table 1-28 与 Figure 1-29 位域图被合并为一个大表格** |
| 激活后 | `(57.0, 89.7, 544.4, 135.5)` | Figure 1-28 位域图被识别为表格 |

**缺陷现场（调用链）**：

1. `pymupdf4llm/__init__.py` 导入时无条件执行：
   ```python
   try:
       import pymupdf.layout
   except ImportError:
       use_layout(False)
   else:
       use_layout(True)
   ```
2. `use_layout(True)` 调用 `pymupdf.layout.activate()`，将 `pymupdf._get_layout` 设置为一个 ONNX 文档布局分析器。
3. `fitz.Page.find_tables()` 内部会调用 `page.get_layout()`。
4. `page.get_layout()` 调用 `pymupdf._get_layout(self)`，触发 ONNX 模型推理。
5. Layout 信息被写入 `page.layout_information`，并**改变 `find_tables()` 的表格 bbox 计算**，导致 register bitfield diagram 与真实表格被错误合并。

**关键发现：fallback 是死代码**

在排查过程中进一步发现：`PDFParser.parse()` 中虽然定义了 `problem_pages` 字典和 `_call_pymupdf4llm_for_pages()` 等 fallback 方法，但 `_should_trigger_p4l_fallback()` **从未被调用**，`problem_pages` 也**从未被填充**。因此整个 PyMuPDF4LLM fallback 路径（Milestone 3）是**死代码**，不会在实际解析中产生任何输出。

**最终修复方案**：

既然 `pymupdf4llm` 的 layout 引擎会污染 `find_tables()`，且该依赖的 fallback 路径从未实际触发，**直接移除 `pymupdf4llm` 依赖**是最彻底的解决方案：

- 从 `pyproject.toml` 删除 `"pymupdf4llm>=1.27.0,<2.0.0"`
- 从 `scripts/parsers/pdf_parser.py` 删除：
  - `import pymupdf4llm` 和 `import pymupdf`
  - `_should_trigger_p4l_fallback()`
  - `_call_pymupdf4llm_for_pages()`
  - `_extract_tables_from_p4l_text()`
  - `_fuse_p4l_tables_into_pages()`
  - `_merge_image_lists()`
  - `parse()` 中的 `problem_pages` 相关逻辑
- 删除 `scripts/benchmark/run_pymupdf4llm.py`
- 清理 benchmark 脚本中所有 `pymupdf4llm` 对比项

**验证结果**：
- SPRUI07 page 80 正确提取 3 张图、3 张表。
- 全量测试通过。

**教训**：
- 外部库的导入级副作用（import-time side effect）可能 silently 改变同一进程中其他依赖库的行为。
- 当依赖库的行为与文档/直觉不符时，应优先检查**导入顺序和导入副作用**，而非假设底层库有状态错误。
- 引入依赖前应确认其实际使用路径；长期不触发且存在副作用的 fallback 代码应当及时清理，避免成为"隐患代码"。

### 18.9 历史演进与性能数据

**旧架构（UZN 方案，已删除）**：
- `unified_zone_recognition.py`，2037 行
- 从全页面角度统一识别 diagram 和 table 区域
- 代码复杂、性能极差、提取质量差

**当前架构（GapsFirstScanner）**：
- `gaps_first_scanner.py`，~600 行
- Caption-driven 线性提取，hard separator + zone 模型
- 已删除死代码：`_find_figure_regions`、`_detect_and_convert_tables`、`_strip_zone_text_blocks`、`_build_images_from_uzn`

**性能基准**（4 文档，GapsFirstScanner）：

| 文档 | 页数 | process_page | 每页耗时 | 图片 | 表格 |
|------|------|-------------|----------|------|------|
| TI TMS320F28335 | 219 | 30.6s | 140ms | 85 | 128 |
| AMBA CHI | 585 | 20.0s | 34ms | 218 | 14 |
| SPRUI07 | 868 | 76.4s | 88ms | 586 | 442 |
| DC_UG | 748 | 8.1s | 11ms | 1 | 61 |

**早期 Milestone 数据**（旧 PDFParser，lines_strict）：

| 文档 | 页数 | 改进前 | 版本 A（lines_strict） | 表格数 |
|------|------|--------|----------------------|--------|
| TI TMS320F28335 | 219 | 10.3s / 0表格 | 46.2s / 68表格 | 68 |
| AMBA CHI | 585 | 8.7s / 0表格 | 93.3s / 22表格 | 22 |

**注意**：GapsFirstScanner 的数据与旧 PDFParser **不可直接比较**，因为：
1. 架构完全不同（UZN vs Caption-driven）
2. 图片提取逻辑完全不同（旧版提取大量位域图/装饰线，新版只提取有 Caption 的图）
3. 表格检测范围不同（旧版全页扫描，新版 zone 级扫描）

### 18.10 多进程并行化结论

- 8 workers 加速比：TI 2.07x / AMBA 1.61x
- 但 find_tables 流程仅占解析总时间的 18-30%，Amdahl 定律限制整体收益仅 ~1.3x
- **即使 find_tables 无限加速，TI 仍需 28s+，AMBA 仍需 75s+**
- **否决实施**：代码复杂度增加不值得有限的收益

## 22. 实体提取 Prompt 设计

### 22.1 设计演进：从"验证导向提取规则"到"分步引导提取"

**初始方案的问题**：
早期 Prompt 采用"表格逐行提取"+"验证导向规则"设计：
- 要求 LLM 对每一行寄存器表都提取实体
- 要求验证每个提取的实体是否有明确来源
- 要求补全缺失的关系链

**过度提取的根因**：
1. **验证规则反噬**："每个实体必须有至少一条直接关系"的规则，导致 LLM 为代码示例中的临时变量（如 `EPwm1Regs.TBCTL.bit.PRDLD`）编造 `HAS_FIELD` 关系
2. **表格逐行提取的副作用**：LLM 将寄存器表格的表头/列名（如 "Offset"、"Size"）也提取为 RegisterField
3. **无内容分类引导**：代码示例、概述性文字、寄存器表格都被同等处理，导致代码中的变量被误提取为 Register

**解决方案：三步提取流程**：

```
Step 1: 内容分类（9 种类型）
  ├── 寄存器字段描述表格 → 提取 Register + RegisterField
  ├── 信号/参数表格 → 提取 Signal/Parameter
  ├── 代码示例 → 禁止提取任何实体（零容忍）
  ├── 架构/功能描述 → 提取 Module/Feature/Peripheral
  ├── 协议状态机 → 提取 State/Transition/Protocol
  ├── 勘误/电气规格 → 提取 Advisory/ElectricalSpec
  ├── 概述/介绍 → 提取 Document(类型A/B)
  ├── 指令参考 → 提取 Instruction
  └── 其他 → 提取 Product/Pin 等

Step 2: 按类型提取（严格范围限制）
  ├── RegisterField 只能从寄存器字段描述表格提取
  ├── Register 只能从寄存器表/概述中明确命名的寄存器提取
  ├── 禁止从代码示例提取任何 RegisterField
  └── 禁止将描述性短语（"Compare A"、"Phase registers"）提取为 Register

Step 3: 关系链补全 + 去重检查
  ├── 每个非 Document 实体至少一条直接关系
  ├── Document/Product/Module 去重（同文档内）
  └── 属性格式规范化
```

**效果**：
- 全局节点数从 350 → 151（-57%）
- 边数从 414 → 164（-60%）
- 代码示例误提取从 13 个降至 0 个

**第二次演进：双方向聚焦（2026-05-20）**

三步提取流程解决了"过度提取"问题，但核心实体类型列表仍包含大量与目标用途关联较弱的类型（Pin、Package、ElectricalSpec、Advisory、Workaround、Protocol、Function、BuildConfig 等），分散 LLM 注意力。同时，ISA 方向和外设寄存器方向的提取深度不足以支撑汇编/C 验证代码生成。

**聚焦决策**：
- **ISA 方向**（汇编代码验证）：Instruction、InstructionGroup、AddressingMode、Operand、ArchitectureState、PipelineStage、FunctionalUnit、Interrupt、Exception、CPU_Mode、CLA_Task
- **外设寄存器方向**（裸机C代码验证）：Peripheral、Module、Register、RegisterField、ShadowRegister、MemoryRegion、Signal
- **支撑类型**：Product、Document、Parameter、ClockDomain、ResetDomain
- **精简移除**：Pin、Package、ElectricalSpec、ApplicationDomain、OrderingInfo、Advisory、Workaround、SiliconRevision、UsageNote、Protocol、ProtocolLayer、TransactionType、Channel、MessageField、Function、DataStructure、Section、BuildConfig、CodeExample、Scenario、TestCase

**Step 1 分类从 9 个聚焦为 8 个**：

```
Step 1: 内容分类（8 种类型）
  ├── 寄存器表 → 提取 Register + RegisterField + ShadowRegister + MemoryRegion
  ├── 外设功能描述 → 提取 Peripheral + Module + Feature + Signal + Parameter
  ├── ISA 架构描述 → 提取 ArchitectureState + PipelineStage + FunctionalUnit + CPU_Mode + Interrupt + Exception + MemoryRegion
  ├── 指令详细参考 → 提取 Instruction + InstructionGroup + AddressingMode + Operand + Register
  ├── 指令集概览 → 提取 Instruction + InstructionGroup + AddressingMode
  ├── CLA 描述 → 提取 CLA_Task + Instruction + ArchitectureState
  ├── 概述/介绍 → 提取 Document(类型A/B) + Product
  └── 其他 → 不提取任何实体
```

**新增代码生成导向提取要求**：
- ISA 方向：提取 opcode/format/cycle_count/affected_flags/delay_slots/atomic/repeatable/pipeline_stall_condition，支撑汇编验证代码生成
- 外设寄存器方向：提取 write_semantics/read_semantics/shadow_of/update_condition/config_sequence/dependency/trigger_condition，支撑裸机C验证代码生成

### 22.2 代码示例排除规则（零容忍）

**禁止提取的内容**：
| 类型 | 示例 | 说明 |
|------|------|------|
| 局部变量 | `epwm1_tz_isr`、`temp_count` | C 函数内局部变量 |
| 代码标签 | `Epwm1_tz_isr:` | 汇编/C 标签 |
| 汇编地址常量 | `0x007010`、`0x6800` | 绝对地址 |
| 宏展开值 | `EPWM1_INT`、`SYSCTL_PERIPH_EPWM1` | 预处理器宏 |
| 临时计算结果 | `5 * EPWM_TIMER_TBPRD / 4` | 表达式结果 |
| **寄存器字段访问** | `EPwm1Regs.TBCTL.bit.PRDLD` | 代码中对寄存器字段的访问语法 |
| **位域赋值** | `.bit.XXX = YYY` | 位域读写操作 |

**关键洞察**：`EPwm1Regs.TBCTL.bit.PRDLD` 在代码示例中是**访问语法**（表示"读取 TBCTL 寄存器的 PRDLD 字段"），不是寄存器字段的定义。RegisterField 的定义只存在于寄存器字段描述表格中（如 "TBCTL Field Descriptions" 表格）。

### 22.3 实体类型优先级与互斥规则

**优先级链**：`Module > Register > Signal > Instruction > Parameter > Feature`

**互斥原则**：同一个概念在同一文档中只能属于一个类型。

| 冲突场景 | 正确选择 | 错误示例 |
|----------|---------|---------|
| 一个名称既像 Register 又像 Signal | 按文档上下文判断，通常是 Register | `TBCTL` 在寄存器表中为 Register，在信号描述中为 Signal |
| "Compare A" 是寄存器名还是描述短语 | 如果只是描述功能，不提取为 Register | 提取为 Register 会导致与 `CMPA` 混淆 |
| "Phase registers" 是模块还是寄存器 | 描述性短语，不提取 | 提取为 Register 导致语义丢失 |

### 22.4 属性格式规范

**必须在 Prompt 中显式规定格式**，否则 LLM 输出格式不统一（实测发现同一文档中 `bits` 字段出现 `15-4`、`15:8`、`7-0` 三种格式，`reset_value` 出现 `0` 和 `0x0000` 两种格式）。

| 属性 | 格式 | 示例 | 错误示例 |
|------|------|------|---------|
| `width` | 纯数字 | `16` | `"16 bits"` |
| `access` | R/RW/R-0/W1C 等 | `R/W` | `"Read/Write"` |
| `reset_value` | 十六进制字符串 | `"0x0000"` | `0`（数字） |
| `address_offset` | 十六进制字符串 | `"0x0002"` | `"2"` |
| `bits` | 冒号分隔 | `"15:0"` | `"15-0"` |
| `description` | 纯文本，无 HTML | `"Counter Compare A Register"` | `"<p>Counter...</p>"` |
| `size_in_words` | 纯数字 | `1` | 保留原始语义，不附会 shadow/active 解释 |

### 22.5 Document 实体规则

**类型 A（当前文档本身）**：
- 每个文档只提取一次
- 属性：`name`=文档标题, `doc_type_hint`=类型推断
- 建立 `Document --[HAS_MODULE]--> Module` 等关系

**类型 B（引用文档）**：
- 仅在 "Related Documentation" 等引用章节中提取
- 建立 `Document_A --[CITES]--> Document_B`
- 禁止在正文其他位置重复提取引用文档

### 22.6 Prompt 迭代方法论

**Prompt 版本化管理**：
- 每次修改后执行完整的 Stage 2-6 Pipeline
- 对比全局节点/边数量变化、具体实体差异
- 记录"预期效果 vs 实际效果"，修正 Prompt 表述

**质量评审 checklist**（每次修改后执行）：
1. 全局节点数变化是否符合预期（减少误提取 → 应下降）
2. 代码示例中是否仍有 Register/RegisterField 误提取
3. Register 属性是否完整（address_offset、width、access、reset_value）
4. bits/reset_value 格式是否统一
5. 是否有孤立节点（无关系的实体）
6. 跨文档合并后属性冲突是否合理

**Prompt 不测试原则**：
- **不通过代码测试约束 prompt 内容**：prompt 是策略，策略变化快，静态字符串测试（如 `assert "opcode" in system_msg`）会成为迭代阻力
- prompt 质量通过完整 Pipeline 运行后的全局节点/边变化、具体实体差异来评估
- prompt 的约束规范通过 DESIGN.md 本章记录，不通过单元测试断言

### 22.7 变更历史

| 日期 | 变更 | 效果 |
|------|------|------|
| 2026-05-18 | 引入 Step 1→2→3 三步提取流程 | 节点 350→151，边 414→164 |
| 2026-05-18 | 代码示例寄存器字段排除 | 代码示例误提取从 13 个降至 0 个 |
| 2026-05-18 | 属性格式统一规范 | bits/reset_value/format 格式不一致问题消除 |
| 2026-05-20 | **聚焦 ISA + 外设寄存器双方向** | 核心实体类型从 35 个精简到 ~18 个；新增 ISA 专属 13 项属性、外设寄存器专属 10 项属性；新增"代码生成导向提取"章节 |
| 2026-05-20 | **移除 prompt 静态测试** | `test_entity_extractor.py` 从 12 个测试精简到 3 个（仅保留代码逻辑测试）；prompt 约束通过 DESIGN.md 文档化 |

---

## 已知限制与权衡汇总

| 限制 | 原因 | 缓解措施 |
|------|------|----------|
| **仅支持 PDF** | DOCX/XLSX 解析器代码已存在，但尚未接入 `DocGraphPipeline` | 下一步开发计划 |
| **Batch API 分钟级延迟** | Batch API 的固有特性 | 离线任务，用户可接受；支持 `progress` 查询状态 |
| **FAISS 与 SQLite 非原子** | 两个独立存储系统，无分布式事务 | 已缓解：FAISS 操作加 `fcntl` 进程锁，修改前 `_reload()` 磁盘最新状态。仍可通过 `reprocess` 或手动删除索引重建 |
| **Embedding 维度不可变** | FAISS 索引创建后维度固定 | 更换模型时删除旧索引并重新处理 |
| **JSONL 100MB 限制** | Kimi/BigModel Batch API 的硬性限制 | `submit_parallel_batches()` 自动拆分并行提交 |
| **NetworkX 内存上限** | 全局图为内存存储，随文档数量增长可能达到 GB 级 | 当前单机目标规模可接受；预留 Neo4j 迁移接口 |
| **图片需先文字化** | embedding-3 不支持图片输入 | LLM 先生成 `image_descriptions`，再合并到 block 文本 |
| **本地 PDF 解析目录页换行** | 目录页多行条目保留为多行 | 目录页对知识提取影响小，接受此行为 |
| **本地 PDF 解析依赖 TOC** | 无 TOC 的 PDF 回退到启发式规则 | `_fallback_heading_detection` 提供降级方案 |
| **_is_heading 仅识别 Markdown #** | 已删除数字编号/中文编号匹配，目录条目（如 "1 Introduction 7"）不再被识别为 heading | 目录文本被收集为 heading 之间的 content，可能混入正文 batch；当前接受此行为，后续可通过机制层策略处理 |
| **AMBA 零高度线表格提取** | AMBA 表格的水平线为 height=0.0 的零高度矢量线，`find_tables` 的任何策略（lines_strict/lines/text）均无法识别 | 已实现 `BorderlessTableExtractor`，对 `horizontal` 风格直接路由，可正确提取此类表格；跨页续表仍按页独立 |
| **TABLE_CAPTION_RE 误匹配正文引用句** | `"Table X.X describes..."` 等正文引用句被正则匹配为 caption，导致所在页面触发 find_tables 但无对应表格 | AMBA 文档中发生 191 次误匹配；GapsFirstScanner 的预计算机制使误触发成本从"多次 clip 调用"降至"一次全页扫描"，影响已大幅降低 |
| **跨页续表碎片化** | `find_tables()` 按页独立处理，无法识别 `"Continued from previous page"` 的跨页上下文 | AMBA 中 93 页跨页续表；GapsFirstScanner 的 zone 模型使续页 caption 能正确定位表格 zone，但 `find_tables` 仍只能提取当前页的部分行 |
| **冲突日志无关系级记录** | `add_relation` 冲突只记录 property_key，不记录完整关系上下文 | 冲突日志包含 from/to/type 信息，可定位 |
| **反馈不反向更新 block metadata** | 全局图更新后，block metadata 中的 extracted_entities 仍是处理时快照 | `get_content_with_entities` 查全局图获取最新状态 |

---

## 技术栈速查

| 层级 | 依赖 | 版本要求 |
|------|------|----------|
| 文档解析 | BigModel Expert API（主）/ PyMuPDF（fallback） | — |
| 向量检索 | `faiss-cpu`、`numpy` | Python >= 3.11 |
| 数据库 | SQLite（标准库 `sqlite3`） | — |
| LLM API | `requests`（OpenAI-compatible HTTP API） | — |
| 图谱 | `networkx` | — |
| 环境配置 | `python-dotenv` | — |
| 测试 | `pytest` | — |
| 代码质量 | `ruff`、`pyright` | — |
