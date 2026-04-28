# work-docs-library 技术方案文档

> 记录每个核心设计决策的选择原因、已知限制与权衡。本文档面向开发者与 Agent，用于理解架构背后的思考。

---

## 1. 为什么 Batch API 优先？

**选择**：所有 LLM 调用（实体提取 + 向量化）默认走 Batch API（Kimi Batch API + BigModel Batch API）。

**原因**：
- **成本**：Batch API 价格为同步 API 的 50%，对于文档处理这类非实时场景，成本优势显著
- **吞吐量**：`submit_parallel_batches()` 按 100MB JSONL 限制自动切分，ThreadPoolExecutor 并行提交，可处理超大文档
- **可接受性**：技术文档导入是离线任务，分钟级排队延迟对用户体验影响有限

**权衡**：
- 延迟从秒级变为分钟级，不适合实时交互场景
- 需要实现轮询和超时逻辑（`LLM_BATCH_TIMEOUT` 默认 3600 秒）
- 单个 JSONL 不能超过 100MB，超大文档需自动拆分

---

## 2. 为什么 NetworkX 而不是 Neo4j？

**选择**：使用 `networkx.DiGraph` 作为图谱存储引擎，JSON 序列化持久化。

**原因**：
- **轻量**：无需引入外部数据库服务，零部署成本
- **JSON 序列化**：`nx.node_link_data()` / `nx.node_link_graph()` 支持完整的图结构导出/导入，便于版本控制和调试
- **当前需求足够**：支持邻居查询、子图提取、BFS 路径搜索（`find_path`，深度上限 6），NetworkX 的内存遍历性能完全满足
- **预留接口**：`GraphStore` 抽象基类已预留 Neo4j 迁移接口，未来需求变化时可无缝替换

**增强功能**：
- **数据质量标记**：`GraphEntity`/`GraphRelation` 新增 `confidence`/`verified`/`created_at`/`updated_at`/`feedback_score` 字段，支持置信度追踪和人工验证
- **动态 CRUD**：支持运行时 `update_entity`/`delete_entity`/`update_relation`/`delete_relation`/`verify_entity`，Agent 可直接修正图谱
- **冲突检测**：同名实体属性差异时自动记录 `conflict_logs`，保留旧值到列表形式而非静默覆盖
- **属性索引**：内部维护 `property_index: dict[(entity_type, key, value), set[nid]]`，`find_by_property()` 从 O(N) 降至 O(1)
- **关系过滤**：`find_path()` 和 `get_neighbors()` 支持按 `rel_types` 过滤

**权衡**：
- 内存存储，单图规模受限于可用内存（当前目标文档为几十到几百页，远未触及瓶颈）
- 不支持并发写入（当前为单用户本地部署，不成为问题）
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
- **足够当前需求**：当前 Schema 仅包含 `documents`、`chunks` 两张表，无复杂查询

**权衡**：
- 不支持高并发写入（当前为单进程顺序处理，不成为问题）
- 无内置全文检索（当前使用 FAISS 向量搜索 + 关键词 LIKE 查询，暂不依赖 SQLite FTS）
- 极端情况下（进程崩溃）FAISS 索引与 SQLite 元数据可能不一致，可通过 `reprocess` 重建

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
- `#` 和第一个 `##` 之间的文字 → 归入第一个 `##` 的 preface

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
- `_resolve_config(env_name, json_path, default)` 统一解析，优先级逻辑集中在一处
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

**选择**：文档更新时，按章节计算 `content_hash`（MD5）指纹比较。未变章节复用 `Chunk.metadata` 中缓存的 `extracted_entities` / `extracted_relations` / `embedding`，仅对变更/新增章节进行 LLM 提取和向量化。

**原因**：
- **万页级文档的更新成本**：全量重新处理一个万页文档的成本约 600元 + 1小时。若只变更 1 页，增量更新可将成本降至 ~10元 + 5分钟
- **技术文档的结构性**：技术文档按章节组织，章节内部高度内聚，跨章节依赖相对较少。章节是天然的最小增量单元
- **Batch API 的延迟**：避免对未变章节重复提交 Batch API，显著降低排队等待时间

**实现细节**：
- `Chunk.metadata` 存储 `content_hash`、`extracted_entities`、`extracted_relations`、`image_descriptions`、`embedding`
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
- **preface 传播统一化**：任何有子节点的节点，其 content 应归入第一个子节点。递归传播使这一规则适用于所有层级（`#` → `##` → `###` → `####`）
- **BatchBuilder 基础**：多级树是 BatchBuilder 按层级切分 content 的前提

**实现**：
- 遍历 flat heading 列表，使用栈确定父子关系（弹出 `level >= 当前 level` 的节点）
- 递归传播 preface：`_propagate_preface(node)` — 有子节点时，`node.content` 移入 `node.children[0]`

**权衡**：
- `_collect_content` 需要递归收集，略微增加计算量（对当前文档规模可忽略）

---

## 15. 为什么 BatchBuilder 以 `##` 为硬边界、`###` 为 chunk 单位？

**选择**：
- `#`（level=1）是文档标题（书名），**不作为硬边界**
- `##`（level=2）是章节，**作为硬边界**（不跨 `##` 合并）
- `###`（level=3）是基本 **chunk 单位**
- `####+` 的内容通过 `_collect_content` 递归归并到父 `###`

**原因**：
- **`#` 是书名而非章节**：技术文档中 `#` 通常是书名（如 "# TMS320x280x HRPWM Reference Guide"），跨书名合并无意义
- **`##` 是语义边界**：不同章节（如 "2.1 Overview" 和 "2.2 Implementation"）内容差异大，跨章节合并会割裂逻辑
- **`###` 粒度适中**：小节级别（如 "2.1.1 Features"）内容通常在几千字符，适合作为 LLM batch 的基本单位。若内容过短，相邻 `###` 可合并

**实现**：
- `_find_chunk_nodes(node, chunk_level=3)`：从节点开始递归，找到所有 `level >= 3` 的节点作为 chunk 候选
- `build_batches` 遍历 `root.children`（`##` 节点）作为硬边界，在每个 `##` 内部找到 `###` 作为 chunk 单位
- 合并规则：相邻同级 `###` 若每个 `< 50% max_chars`，可合并为一个 batch

**权衡**：
- `###` 内容可能很短（如只有一段），导致小 batch。但这是保证语义边界的必要代价
- 如果文档没有 `###`（只有 `#` 和 `##`），`##` 降级作为 chunk 单位

---

## 联合查询与 Agent 自主推理

**设计**：`search_with_graph()` 实现语义搜索与图谱查询的联合。

**流程**：
1. FAISS 语义搜索 → top_k 个最相关 chunk
2. 读取 chunk `metadata` 中缓存的 `extracted_entities`
3. 对每个实体在全局图上做 `get_subgraph(center, graph_depth)` 扩展
4. 返回 `{chunks, related_entities, subgraphs}`

**chunk+实体联合返回**：`get_content_with_entities(chunk_db_id)` 返回 chunk 内容 + 全局图中最新状态的关联实体/关系（而非 metadata 中的处理时快照）。

**Agent 自主推理能力**：
- Agent 可先用 `search_with_graph` 找到相关文本和图谱实体
- 再用 `graph_neighbors`/`graph_path`/`graph_subgraph` 做多跳推理
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

## 已知限制与权衡汇总

| 限制 | 原因 | 缓解措施 |
|------|------|----------|
| **仅支持 PDF** | DOCX/XLSX 解析器代码已存在，但尚未接入 `DocGraphPipeline` | 下一步开发计划 |
| **Batch API 分钟级延迟** | Batch API 的固有特性 | 离线任务，用户可接受；支持 `progress` 查询状态 |
| **FAISS 与 SQLite 非原子** | 两个独立存储系统，无分布式事务 | `reprocess` 可重建；极端情况手动删除索引 |
| **Embedding 维度不可变** | FAISS 索引创建后维度固定 | 更换模型时删除旧索引并重新处理 |
| **JSONL 100MB 限制** | Kimi/BigModel Batch API 的硬性限制 | `submit_parallel_batches()` 自动拆分并行提交 |
| **NetworkX 内存上限** | 全局图为内存存储，随文档数量增长可能达到 GB 级 | 当前单机目标规模可接受；预留 Neo4j 迁移接口 |
| **图片需先文字化** | embedding-3 不支持图片输入 | LLM 先生成 `image_descriptions`，再合并到 chunk 文本 |
| **本地 PDF 解析目录页换行** | 目录页多行条目保留为多行 | 目录页对知识提取影响小，接受此行为 |
| **本地 PDF 解析依赖 TOC** | 无 TOC 的 PDF 回退到启发式规则 | `_fallback_heading_detection` 提供降级方案 |
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
