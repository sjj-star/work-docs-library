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

2. **Tokenizer 不透明导致切分策略复杂**：BigModel embedding-3 的 tokenizer 与 tiktoken 不存在固定比例关系（对数字/单独字母接近字符级编码，对自然语言使用子词编码），导致按 token 分组的逻辑复杂且容易触发 3072 token 上限错误。实验后改为纯字符数限制（`CHUNK_MAX_CHARS = 6000`），彻底解决了分组问题。

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
- 图片必须先由 LLM 文字化（`image_descriptions`），再合并到 chunk 文本中向量化

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
- 图片压缩：`LLM_VISION_MAX_EDGE=1024`、`LLM_VISION_QUALITY=85`，平衡清晰度与成本

**权衡**：
- 图片必须先由 LLM 文字化，才能进入 embedding；当前 embedding 模型（embedding-3）不支持图片输入
- base64 编码增加 JSONL 体积，超大文档更容易触发 100MB 拆分阈值

---

## 6. 为什么 SQLite 而不是 PostgreSQL？

**选择**：使用 SQLite（标准库 `sqlite3`）作为元数据存储。

**原因**：
- **单用户场景**：本项目为本地部署的 Kimi CLI Plugin，无多用户并发需求
- **零运维**：无需安装、配置、维护数据库服务，一个 `.db` 文件即完整数据库
- **事务简单**：`KnowledgeDB._connect()` 使用上下文管理器，自动 commit/close
- **足够当前需求**：当前 Schema 包含 `_schema_meta`（版本管理）、`documents`（文档元数据）、`chunks`（内容块）、`conflict_logs`（同名实体属性冲突日志）、`feedback`（实体/关系用户反馈）五张表，无复杂查询

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
- `###` → 子章节/小节，基本 chunk 单位，内容较短时与同层级兄弟合并
- `#` 和第一个 `##` 之间的文字 → **移动**到第一个 `##` 的 preface（父节点 content 清空，子节点 content 前置）

> 注：BatchBuilder 的硬边界与合并规则详见第15节。

**权衡**：
- 某些 `#` 章节可能内容极少，产生小 batch；但这是保证语义边界的必要代价
- 对解析器生成的 Markdown 标题质量有依赖（BigModel Expert 通常表现良好）

---

## 8. 配置系统：config.json + .env 双轨设计

**选择**：三层优先级配置系统——环境变量（Kimi CLI 注入）> `config.json` > `.env` > 代码默认值。

**原因**：
- **`config.json`**：用户持久化配置，适合存放模型选择、端点地址、维度等不敏感的参数；由 `plugin.json` 的 `config_file` 指定路径，Kimi CLI 可自动管理
- **`.env`**：适合存放 API Key 等凭证，gitignored，不进入版本控制
- **环境变量**：Kimi CLI 运行时动态注入，优先级最高，支持用户在不修改文件的情况下临时覆盖配置

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
- `Chunk.metadata` 存储 `content_hash`、`extracted_entities`、`extracted_relations`（⚠️ 当前未按章节过滤，缓存全文档关系）、`image_descriptions`（⚠️ 当前未按章节过滤）、`embedding`
- `_save_chunks_to_db()` 向量化阶段分离 `reuse_pairs`（未变章节复用 embedding）和 `reembed_pairs`（变更/新增章节重新调用 Embedding API）
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

## 14. 为什么 ChapterParser 构建真正的多级树？

**选择**：`parse_tree()` 从扁平化的"`#` 为根、`##+` 为子节点"改为使用栈结构构建真正的多级树。

**原因**：
- **父子关系准确**：`###` 应该是 `##` 的子节点，而非 `##` 的同级
- **按原文顺序保留 content**：每个节点保留自己的原始 content，不互相合并，结构清晰
- **标题路径前缀**：`collect_all_nodes()` 为每个 chunk 附加从根到当前节点的完整标题路径，LLM 获得完整层级上下文

**实现**：
- 遍历 flat heading 列表，使用栈确定父子关系（弹出 `level >= 当前 level` 的节点）
- `collect_all_nodes(node)` 递归收集所有有 content 的节点，为每个节点生成 `\# Title\n\## Section\n\### Sub\n\ncontent` 格式的完整路径前缀

**权衡**：
- chunk 数量可能略有增加（原本为空的中间节点不会产生 chunk），但增量更新更精确

---

## 15. 为什么 BatchBuilder 以 `##` 为硬边界、`###` 为 chunk 单位？

**选择**：
- `ChapterParser.collect_all_nodes()` 先扁平化树，收集所有有 content 的节点
- `BatchBuilder.build_batches()` 接收扁平化节点列表，按 `max_chars` 切分，超长内容按 **段落边界**（`\n\n+`）切分为 sub-batch

**原因**：
- **按原文顺序**：每个节点保留自己的 content，不互相合并，标题路径前缀提供完整层级上下文
- **段落边界更安全**：Markdown 空行是天然的语义边界，避免带编号的标题行（如 "Table 6. SFO Library Routines"）被句号误切开
- **粒度适中**：小节级别（如 "2.1.1 Features"）内容通常在几千字符，适合作为 LLM batch 的基本单位

**实现**：
- `collect_all_nodes(node, ancestors)` 递归收集所有有 content 的节点，生成带标题路径前缀的 chunk
- `build_batches` 直接遍历扁平化节点列表，content 超过 `max_chars` 时调用 `_split_by_sentences` 按段落边界切分

**权衡**：
- 单个超长段落（无空行）可能略超 `max_chars`，但代码有 fallback 不会截断
- chunk 数量可能略有增加，但增量更新更精确（父章节和子章节独立比对 content_hash）

---

## 16. Chunk 与 Sub-chunk 的生成链路

### 概念定义

| 术语 | 定义 | 示例 |
|------|------|------|
| **Chapter** | `ChapterParser` 按 Markdown 标题层级（`#`/`##`/`###`/`####`）解析出的树节点，包含 `title` 和 `content` | `{"title": "2.1 GPIO Configuration", "content": "..."}` |
| **Chunk** | 数据库 `chunks` 表中的一行，是**内容存储和向量检索的最小单位** | `Chunk(chunk_id="ch_3", content="...", db_id=42)` |
| **Sub-chunk** | 超长 chapter 被 `_maybe_split_chapter` 拆分后的产物，有自己的 `chunk_id` 和 `db_id` | `Chunk(chunk_id="ch_3_part_0", content="...前半", db_id=43)` |

### 生成链路

```
PDF → Markdown → ChapterParser.parse_tree() → collect_all_nodes()
    → _maybe_split_chapter() → _insert_chunk() → SQLite
```

1. **章节树**（`parse_tree()`）：按 `#`/`##`/`###`/`####` 构建多级树。`###` 级别的小节是"基本 chunk 单位"
2. **扁平化**（`collect_all_nodes()`）：递归收集所有有 content 的节点，为每个节点附加从根到当前节点的完整标题路径前缀
3. **拆分检查**（`_maybe_split_chapter()`）：
   ```python
   max_chars = Config.CHUNK_MAX_CHARS  # 默认 6000
   if len(content) <= max_chars:
       return [{**ch, "_chunk_id": f"ch_{base_idx}"}]        # 不拆分
   parts = _split_for_embedding(content, max_chars)          # 按段落→句子切分
   return [{**ch, "content": part, "_chunk_id": f"ch_{base_idx}_part_{i}"} for i, part in enumerate(parts)]
   ```
4. **入库**（`_insert_chunk()`）：每个 chunk/sub-chunk 成为数据库中的独立行

### `_split_for_embedding` 的切分策略

按优先级逐级 fallback：

1. **段落边界**（`\n\n+`）：保护代码块、HTML table、Markdown table 不被截断
2. **句子边界**（`[.!?。！？]\s+`）：标点后的空格是天然语义边界
3. **保留原样**：若单个句子仍超长，记录 warning 并保留原样（由 API 错误处理兜底）

### Sub-chunk 的继承与独立性

Sub-chunk 继承父 chapter 的所有属性，但有自己的 `db_id`：

| 属性 | 继承/独立 | 说明 |
|------|----------|------|
| `content` | **独立** | 拆分后的片段内容 |
| `chunk_id` | **独立** | `ch_N_part_M` 格式 |
| `db_id` | **独立** | SQLite 自增主键，FAISS 索引用它 |
| `chapter_title` | 继承 | 与父 chapter 相同 |
| `content_hash` | 继承 | 父 chapter 的 hash（用于增量更新比对） |
| `extracted_entities` | 继承 | 父 chapter 缓存的实体引用 |
| `extracted_relations` | 继承 | 父 chapter 缓存的关系 |
| `image_descriptions` | 继承 | 父 chapter 合并后的图片描述 |

### 为什么 Chunk 粒度 = 向量化粒度？

数据库中的一行 = 一个向量化单位。这意味着：
- 每个 chunk（包括 sub-chunk）独立向量化
- sub-chunk 在 SQLite、FAISS、`_EntityChunkBridge` 中都是**一等公民**
- 不存在"一个 chunk 对应多个向量"或"多个 chunk 合并为一个向量"的情况
- 这彻底解决了早期版本中 batch 路径 split-chunk 未聚合导致 FAISS 重复向量的问题

### `CHUNK_MAX_CHARS` 的设计意图

`CHUNK_MAX_CHARS`（默认 6000）控制的是**单个 chunk 的最大字符数上限**：
- 在 **stage4（入库阶段）** 决定 chapter 是否需要拆分
- **向量化阶段只是下游消费方**，因为 chunk 已被拆分到合适大小
- 默认值 6000 是基于项目实际处理的 PDF 技术文档文本分布的经验参数（自然语言+代码/LaTeX 混合内容约对应 2500-2800 actual tokens）
- 纯数字/特殊字符密集的段落可能在此限制下仍超过 3072 tokens，但这是可接受的风险——API 调用失败时会记录错误，用户可手动调低该值

---

## 为什么需要 EntityChunkBridge 双向桥接索引？

**选择**：在 `KnowledgeBaseService` 内部维护一个纯内存的 `_EntityChunkBridge`，建立 `chunk_db_id ↔ (entity_type, entity_name)` 的双向多对多映射。零 schema 变更、零数据模型变更，从 SQLite `chunk.metadata["extracted_entities"]` 构建。

**原因**：
- **补齐单向关联的缺失**：原有架构中 chunk → entity 的关联已存在（通过 `metadata.extracted_entities`），但 entity → chunk 的反向查询缺失。`graph_provenance` 此前通过逐文档遍历所有 chunks 做暴力扫描（O(N)），不可扩展
- **打通不同粒度空间**：FAISS 向量索引操作的是 chunk 粒度（文本片段 + embedding），NetworkX 图谱操作的是 entity 粒度（结构化实体 + 关系）。桥接索引使两者可以双向导航
- **支持策略层灵活组合**：机制层只提供原子操作（`_chunk_to_entities` / `_entity_to_chunks` / `_get_chunk` / `_semantic_hits` / `_get_subgraph`），策略层可自由组合出任意跨粒度查询（语义→图谱→语义闭环、路径文本证据链等）

**实现**：
```python
_forward:  dict[int, set[_EntityRef]]     # chunk_id → {EntityRef}
_reverse:  dict[_EntityRef, set[int]]     # EntityRef → {chunk_id}
```

**生命周期**：
- `KnowledgeBaseService.__init__` → `bridge.rebuild()` 全量构建
- `ingest_document` / `reprocess_document` 完成后 → `_sync_bridge_for_doc()` 增量同步
- `attach()` 幂等设计：同一 chunk 重复 attach 会先 detach 旧绑定，避免索引累积
- `detach()` 双向清理：同时清除 `_forward` 和 `_reverse`，空集合自动删除

**不同粒度关联矩阵**：

| 方向 | 粒度 | 已有/新增 | 机制 |
|------|------|----------|------|
| chunk → entity | 向量→图谱 | 已有 | `metadata.extracted_entities` |
| entity → chunk | 图谱→向量 | **新增** | `_entity_to_chunks()` O(1) |
| document → entity | 文档→图谱 | 已有 | `GraphEntity.source_doc_ids` |
| entity → document | 图谱→文档 | 已有 | `GraphEntity.source_doc_ids` |
| chapter → entity | 章节→图谱 | 已有 | `GraphEntity.source_chapter` |
| chunk → subgraph | 向量→子图 | 已有 | `search_with_graph()` |
| subgraph → chunk | 子图→向量 | **可组合** | 子图实体 → `_entity_to_chunks()` |

**权衡**：
- 内存索引，重启需重建（但 `KnowledgeBaseService` 为长生命周期单例，初始化成本可忽略：几百文档 × 几十实体）
- 不解决"实体级语义搜索"（如搜索 "GPIO" 也匹配 "General Purpose Input Output"），那是实体 embedding 的范畴，不在本次机制设计范围内

---

## 联合查询与 Agent 自主推理

**设计**：`search_with_graph()` 使用原子操作组合实现语义搜索与图谱查询的联合。

**原子操作层（机制，无策略参数）**：
- `_semantic_hits(query, top_k)` → FAISS 搜索，返回 `[(chunk_db_id, score), ...]`
- `_get_chunk(chunk_db_id)` → 深拷贝获取 Chunk 对象
- `_chunk_to_entities(chunk_db_id)` → 桥接索引正向查询，返回 `set[_EntityRef]`
- `_entity_to_chunks(entity_type, name)` → 桥接索引反向查询，返回 `set[chunk_db_id]`
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

**chunk+实体联合返回**：`get_content_with_entities(chunk_db_id)` 使用 `_get_chunk` + `_chunk_to_entities` + `get_entity` + `get_entity_relations`，返回 chunk 内容 + 全局图中最新状态的关联实体/关系（深拷贝隔离）。

**Agent 自主推理能力**：
- Agent 可先用 `search_with_graph` 找到相关文本和图谱实体
- 再用 `graph_neighbors`/`graph_path`/`graph_subgraph` 做多跳推理
- 通过 `find_chunks_by_entity` 从任意实体反向查找支撑它的原始文本 chunks
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
- **向量化解耦**：stage4 完成后 chunks 状态为 `embedded`，知识图谱已可查询；stage5/stage6 独立执行向量化，不阻塞图谱使用
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
- `embedded` → stage4 完成（chunks 已入库，图谱已构建，但未向量化）
- `DONE` → stage6 完成（向量化已入库）

**权衡**：
- 磁盘占用增加：每个文档额外保存 results.jsonl、incremental.json、embed.jsonl 等（通常总计几十 KB 到几十 MB）
- stage3/stage4 之间需要保持 `result.md` 不变，否则增量分析结果可能不一致（修改 result.md 后应重新走 stage2→stage3）

---

## 20. 审计修复经验总结

**背景**：2026-04 代码审计发现 21 项缺陷（9 个 P0 数据损坏风险、7 个 P1 可靠性缺陷、5 个 P2 代码质量问题），已全部修复并通过 283 个测试验证。

**关键教训**：

| 教训 | 来源缺陷 | 修复措施 |
|------|---------|---------|
| **JSON 修复正则必须验证** | `_safe_parse_json` 用 `\x01` 替代 `\1` | 单字符修复，新增边界测试 |
| **返回全局对象前必须深拷贝** | `_apply_doc_properties` 修改原始节点 | `copy.copy` + `dict()` 深拷贝属性 |
| **删除操作必须同步清理索引** | `remove_document_contributions` 遗漏索引 | 节点移除循环中调用 `_remove_from_property_index` |
| **多存储系统操作必须原子化** | SQLite 已插入但 FAISS 向量化失败 | 统一批量写入，失败时 `remove_doc` 回滚 |
| **全局状态修改前必须快照** | `reprocess_document` 失败后半空状态 | `copy.deepcopy` 备份 `_g` + `_property_index` |
| **函数参数声明后必须使用** | `_merge_image_descriptions` 忽略 `chapter_title` | 按 `chapter_title` 过滤图片 |
| **过滤逻辑必须用完整复合键** | `_build_metadata` 按名称过滤导致跨类型污染 | `(type, name)` 元组匹配 |
| **文件写入必须原子化** | `_save_global_graph` 直接覆写 | 临时文件 + `os.replace`；失败时从 `.bak` 恢复 |
| **跨阶段产物必须校验一致性** | Stage3/Stage4 增量分析结果不一致 | 持久化 `_incremental.json` + hash 校验 |
| **配置路径加载/保存必须一致** | `_save_global_graph` 硬编码 `"graphs"` | 统一使用 `Config.GRAPH_OUTPUT_DIR` |

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
| **图片需先文字化** | embedding-3 不支持图片输入 | LLM 先生成 `image_descriptions`，再合并到 chunk 文本 |
| **本地 PDF 解析目录页换行** | 目录页多行条目保留为多行 | 目录页对知识提取影响小，接受此行为 |
| **本地 PDF 解析依赖 TOC** | 无 TOC 的 PDF 回退到启发式规则 | `_fallback_heading_detection` 提供降级方案 |
| **_is_heading 仅识别 Markdown #** | 已删除数字编号/中文编号匹配，目录条目（如 "1 Introduction 7"）不再被识别为 heading | 目录文本被收集为 heading 之间的 content，可能混入正文 batch；当前接受此行为，后续可通过机制层策略处理 |
| **冲突日志无关系级记录** | `add_relation` 冲突只记录 property_key，不记录完整关系上下文 | 冲突日志包含 from/to/type 信息，可定位 |
| **反馈不反向更新 chunk metadata** | 全局图更新后，chunk metadata 中的 extracted_entities 仍是处理时快照 | `get_content_with_entities` 查全局图获取最新状态 |

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
