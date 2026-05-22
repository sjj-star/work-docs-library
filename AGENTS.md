# work-docs-library Agent 自主式开发策略文档

> 本文档供 AI Coding Agent 阅读，是项目权威技术参考与行动准则。项目详细介绍见 `README.md`，技术方案见 `DESIGN.md`。

---

## 项目目标（一句话定义）

从 IC 前端设计技术文档（PDF）中自动提取结构化实体与关系，构建**跨文档互通**的知识图谱，同时具有文档内容块的向量检索能力增强知识库。

**目标规模**：数百个文档，每个文档几十到万页级。全局图谱通过 NetworkX 内存存储管理，知识图谱检索与文档内容向量检索互通关联。

---

## 核心架构决策及原因

| 决策 | 原因 |
|------|------|
| **Batch API 优先（实体提取）** | 实体提取走 Batch API（成本为同步 API 的 50%；超大 JSONL 自动拆分并行提交；分钟级延迟可接受）。向量化走同步单文本 API（BigModel Embedding Batch 处理时间高达 9 小时，不可接受） |
| **KnowledgeBaseService 统一服务层** | 封装 DB + VectorIndex + GraphStore，为 Plugin 工具和上层应用提供一致 API；禁止工具直接访问 `db._connect()` |
| **NetworkX 而非 Neo4j** | 轻量、JSON 序列化、当前查询深度不超过 3 跳，无需引入外部服务 |
| **零数据丢失** | 技术文档信息密度高，任何截断/过滤都可能导致关键寄存器/信号丢失 |
| **树形章节解析** | 按 Markdown 标题层级拆分是唯一允许的结构性拆分；禁止基于内容相关性的过滤 |
| **Multimodal 图片流式处理** | 图片（时序图、架构框图）是文档语义不可分割的一部分，必须按原文顺序嵌入文档流 |
| **SQLite + FAISS + NetworkX** | 单用户、本地部署、零运维；FAISS IndexFlatIP 经 L2 归一化后等效于余弦相似度 |
| **Prompt 外部化** | 所有 LLM 提示词在 `scripts/prompts/*.txt` 中，用户可编辑，无需改代码 |
| **EntityChunkBridge 跨粒度桥接** | 零 schema 变更的内存双向索引 `chunk_db_id ↔ (entity_type, entity_name)`，打通向量空间与图谱空间，支持 O(1) 双向导航 |
| **Prompt 分步引导提取** | 先内容分类（Step 1）→ 再按类型提取（Step 2）→ 最后关系补全（Step 3），比"表格逐行提取"+"验证导向规则"减少 57% 节点/60% 边误提取 |
| **config.json + .env 双轨配置** | `config.json` 存非敏感参数（模型、端点），`.env` 存 API Key；三层优先级（环境变量 > config.json > .env > 默认值） |

---

## 代码规范

### 强制工具链
- **ruff**：代码格式化与 lint（`ruff check scripts/` / `ruff format scripts/`）
- **pyright**：类型检查（`pyright scripts/`）
- **pytest**：测试执行（`PYTHONPATH=scripts pytest scripts/tests/ -v`）

### 日志规范
- 统一使用 `logging`，格式：`"%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"`
- 禁止随意使用 `print()` 输出调试信息

### 类型与模型
- 领域模型使用 `dataclass`（`Chunk`、`Document`、`GraphEntity`、`GraphRelation`、`SubGraphView`）
- `Chunk.status`：`pending` → `embedded` → `done`（含 `skipped`、`failed`，`ChunkStatus` StrEnum）
- `Chunk.chunk_type`：`text` / `table` / `image_desc`（`ChunkType` StrEnum）
- `Document.status`：`pending` → `processing` → `done` / `failed`（`DocumentStatus` StrEnum）
- `GraphEntity`/`GraphRelation` 字段：`entity_type`/`rel_type`、`name`、`properties`、`doc_properties`（按文档原始属性快照）、`source_doc_ids`、`source_chapter`、`confidence`、`verified`、`created_at`、`updated_at`、`feedback_score`

### Chunk 与 Sub-chunk 概念

**Chunk** 是知识库中**内容存储和向量检索的最小单位**，对应数据库 `chunks` 表中的一行。

**Chunk 生成链路**：
```
PDF → Markdown → ChapterParser(树形章节 #/##/###/####) → collect_all_nodes() → _maybe_split_chapter() → SQLite 入库
```

1. **章节树**：`ChapterParser.parse_tree()` 按 Markdown 标题层级构建树形结构
2. **扁平化**：`collect_all_nodes()` 收集所有有 content 的节点，每个节点称为一个 **chapter**
3. **拆分（可选）**：`_maybe_split_chapter()` 检查 chapter content 是否超过 `CHUNK_MAX_CHARS`（默认 6000）
   - 未超限：直接成为一个 chunk，`chunk_id = ch_N`
   - 超限：拆分为多个 **sub-chunks**，`chunk_id = ch_N_part_0`、`ch_N_part_1`...
4. **入库**：每个 chunk/sub-chunk 成为数据库中的独立行，有自己的 `db_id`

**Sub-chunk** 是超长 chapter 被 `_split_for_embedding`（段落→句子语义边界）拆分后的产物。它继承父 chapter 的所有属性（`content_hash`、`extracted_entities`、`extracted_relations`、`image_descriptions`），但有自己的 `db_id` 和独立 content。

**Chunk 粒度 = 向量化粒度**：数据库中的一行 = 一个向量化单位。每个 chunk（包括 sub-chunk）独立向量化，不存在"一个 chunk 对应多个向量"或"多个 chunk 合并为一个向量"的情况。

**`CHUNK_MAX_CHARS` 的含义**：控制的是**单个 chunk 的最大字符数上限**。它在 stage4（入库阶段）决定 chapter 是否需要拆分；向量化阶段只是下游消费方。默认值 6000 是基于项目实际文本分布的经验参数。

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

---

## 中间产物持久化原则

> 从源数据到最终知识库图谱的全流程中，每个阶段的输出都必须以人类可读、可编辑、可重新加载的格式持久化到磁盘。这是确保人为干预、质量检查、数据重建等特殊复杂需求得以高效稳定地贯穿项目持续开发的核心机制。

### 为什么必须持久化？

技术文档的信息密度极高，Pipeline 中任何一处的黑盒处理都可能导致关键寄存器、信号或时序参数的丢失。将每个中间结果显式落盘，使整条链路从"一次性黑盒"转变为"可审计、可回滚、可修复"的透明流水线：

- **人为干预**：任意阶段完成后，开发人员可直接审查、修改磁盘上的中间产物，再重新触发后续阶段，无需从头执行。
- **质量检查**：通过对比上下游产物（如 `result.md` 与 `results.jsonl` 中的提取结果）快速定位数据质量问题，明确是解析器失真、LLM 提取偏差还是图谱合并错误。
- **数据重建**：任一阶段失败后，基于已持久化的上游产物重新执行该阶段即可，无需回到源文件重新开始。
- **成本保护**：API 调用阶段（`stage3_submit_batches`、`stage6_submit_embed_batches`）与本地处理阶段（`stage4_ingest_results`）解耦。若入库失败，无需重新付费调用 Batch API，直接重试 stage4/stage6 即可。

### 当前 Pipeline 的持久化映射

| 阶段 | 产物路径 | 说明 |
|------|----------|------|
| Stage 1 (Parse) | `knowledge_base/parsed/{doc_id}/result.md` | 解析后的 Markdown（可人工编辑后重新触发 stage2） |
| Stage 2 (Build JSONL) | `knowledge_base/batch/{doc_id}.jsonl` | LLM Batch API 输入请求 |
| Stage 2 (Build JSONL) | `knowledge_base/batch/{doc_id}_batch_info.json` | request → chapter 映射，用于结果回填 |
| Stage 3 (Submit) | `knowledge_base/batch/{doc_id}_results.jsonl` | LLM Batch API 原始返回结果（可审查、可编辑后重新触发 stage4） |
| Stage 3 (Submit) | `knowledge_base/batch/{doc_id}_incremental.json` | 增量分析摘要 + result.md hash 校验（供 stage4 一致性校验） |
| Stage 4 (Ingest) | `knowledge_base/graphs/{doc_id}.json` | 文档级子图快照（可从任意子图重建全局图） |
| Stage 4 (Ingest) | `knowledge_base/workdocs.db` | SQLite 元数据（chunks、documents、conflict_logs、feedback） |
| Stage 5 (Build Embed JSONL) | `knowledge_base/batch/{doc_id}_embed.jsonl` | Embedding 同步 API 输入请求（单文本 input 格式，custom_id 直接编码 db_id） |
| Stage 6 (Submit Embed) | — | 同步调用 Embedding API 逐条处理 stage5 的 JSONL，结果直接入库（不再生成中间结果文件） |
| Stage 6 (Submit Embed) | `knowledge_base/faiss.index` + `knowledge_base/id_map.json` | FAISS 向量索引与 ID 映射 |

### 开发约束

- **新增 Pipeline 阶段时，必须同步定义该阶段的持久化产物**：包括文件格式、存储路径、以及从该产物重新加载并继续下游处理的接口。
- **禁止阶段间纯内存传递大规模数据**：阶段之间的数据流转必须通过磁盘文件完成；仅允许在内存中传递小对象（如状态标记、文件路径、轻量级元数据字典）。
- **产物格式优先人类可读**：优先使用 Markdown、JSONL、JSON 等人类可直接阅读、对比、diff、手动编辑的格式，避免二进制或高度压缩的私有格式。
- **向后兼容的加载接口**：读取中间产物的代码必须能优雅处理格式演进（如缺失字段、新增字段），通过默认值或版本检测保证旧产物仍可加载。

---

## 防御性编程与状态安全原则

> 基于 2026-04 审计修复 21 项缺陷的经验总结。这些原则与"中间产物持久化原则"互补，共同确保项目在持续开发中不因代码疏忽而产生数据损坏、状态不一致或资源泄漏。

### 1. 副作用隔离原则

**核心观点**：返回内存对象前必须深拷贝，禁止调用方直接修改全局状态。

**教训来源**：`_apply_doc_properties` 直接修改全局图节点的 `properties`，导致首次带 `doc_id` 查询后，不带 `doc_id` 的查询也返回快照值。

**开发约束**：
- 任何查询方法返回的实体/关系对象，如果会根据参数动态替换属性，必须返回 `copy.deepcopy` 后的对象
- 全局状态对象（如 `GraphStore` 中的节点）不得被外部直接修改
- 在方法签名中明确标注是否返回可变引用或不可变副本

### 2. 索引一致性原则

**核心观点**：任何删除/修改操作必须同步维护相关索引；索引 key 设计必须考虑值域边界。

**教训来源**：
- `remove_document_contributions` 移除节点但未清理 `_property_index`，导致 `find_by_property` 返回幽灵数据
- `_property_index` 使用 `(entity_type, key, value)` 作为 dict key，当 `value` 为 list/dict 时直接崩溃

**开发约束**：
- 删除节点/边时，必须同步清理所有相关索引（属性索引、反向索引等）
- 索引 key 中若包含用户可控的值，必须处理不可哈希类型（序列化为 JSON 字符串或跳过）
- 索引维护代码必须与数据操作代码放在同一文件中，避免遗漏

### 3. 失败回滚原则

**核心观点**：多步骤状态变更前必须保存快照；文件写入必须原子化；异常时必须能恢复到修改前状态。

**教训来源**：
- `reprocess_document` 先移除旧数据再重新处理，处理失败后全局图处于半空状态
- `_save_global_graph` 直接覆写文件，进程崩溃时可能导致 `global.json` 损坏
- SQLite 和 FAISS 的更新分两步执行，向量化失败后两者不一致
- FAISS 索引文件被多进程并发写入覆盖（如同时 reprocess 两个文档），导致向量丢失或错乱

**开发约束**：
- 修改全局状态前保存快照（`copy.deepcopy` 或临时文件备份）
- 文件写入采用"临时文件 + 原子重命名（`os.replace`）"策略
- 跨存储系统的操作（如 SQLite + FAISS）必须统一在 try 块中，失败时统一清理
- 对共享文件资源（FAISS 索引、全局图 JSON）加进程级文件锁（`fcntl.flock`），修改前重载磁盘最新状态
- 对于不可逆操作（如 API 调用），必须在操作前完成所有本地校验

### 4. 输入验证与 Schema 兼容原则

**核心观点**：对外部输入必须做严格校验；Schema 变更时必须确保所有生产端和消费端同步更新。

**教训来源**：
- `_safe_parse_json` 的 trailing comma 修复正则错误（`\x01` 而非 `\1`），导致 LLM 返回的 JSON 解析失败
- 冲突日志表只有 `entity_type`/`name` 字段，关系冲突写入时丢失完整上下文
- `_merge_image_descriptions` 接受 `chapter_title` 参数但从未使用，导致全文档图片污染每个 chunk
- `_build_metadata` 按名称过滤关系，同名不同类型的实体导致跨类型污染

**开发约束**：
- 解析 LLM/外部 API 返回的 JSON 时，必须做两级验证：语法校验（`json.loads`）+ 语义校验（必需字段、类型检查）
- 新增/修改数据模型字段时，必须同步更新所有序列化/反序列化路径（`to_dict`/`from_dict`、数据库插入、日志记录）
- 函数参数被声明后必须在实现中使用；未使用的参数应删除或添加 `TODO` 注释说明
- 过滤/匹配逻辑涉及复合键时，必须使用完整键（如 `(type, name)` 而非仅 `name`）

### 5. 跨阶段一致性校验原则

**核心观点**：跨阶段依赖的中间产物必须校验一致性；产物被修改后下游阶段必须拒绝或重新计算。

**教训来源**：`stage3_submit_batches` 和 `stage4_ingest_results` 独立调用 `_incremental_analysis`，若 `result.md` 在 stage3 之后被修改，stage4 的增量分析结果与 stage3 不匹配。

**开发约束**：
- 多阶段 Pipeline 中，下游阶段必须校验上游产物的完整性（文件 hash、内容指纹、行数校验）
- 如果校验失败，必须抛出异常或自动重新执行上游阶段，禁止静默使用不一致的数据
- 增量分析等阶段性结果应持久化到磁盘，供下游阶段读取复用而非重新计算

### 6. 配置统一原则

**核心观点**：所有路径、阈值、常量引用必须通过统一的配置项；加载路径和保存路径必须一致。

**教训来源**：`_load_all_graphs` 使用 `Config.GRAPH_OUTPUT_DIR`，但 `_save_global_graph` 硬编码 `"graphs"`，导致配置变更后加载和保存操作不同路径。

**开发约束**：
- 禁止在代码中硬编码路径字符串、目录名、文件名格式
- 加载和保存同一资源时，必须使用相同的配置变量
- 配置变更后，必须通过测试验证所有读写路径仍然一致

### 7. 资源生命周期管理原则

**核心观点**：所有资源必须有明确的关闭路径；`try/finally` 是管理资源的标准模式。

**教训来源**：`DocGraphPipeline.close()` 未关闭 `llm_batch` 和 `embed_batch`，导致 HTTP Session 泄漏。

**开发约束**：
- 任何持有 `requests.Session`、`文件句柄`、`数据库连接` 的类，必须实现 `close()` 方法
- 调用方必须将资源创建包装在 `try/finally` 中，确保 `close()` 被调用
- 资源类初始化失败时，已创建的部分资源必须清理，避免半初始化泄漏

---

### 8. Prompt 策略演进原则（2026-05 新增）

**核心观点**：Prompt 策略与代码机制分离；策略调整在 Prompt 层面完成，代码只负责加载和执行。

**教训来源**：
- `entity_extraction_system.txt` 从 431 行→180 行→300 行的演进中，零代码变更即可生效
- "验证导向提取规则"等硬编码策略导致代码变量、章节标题、格式说明都被提取为实体
- 分步引导（分类→提取→补全）将准确率从"全局规则驱动"改进为"上下文感知驱动"

**开发约束**：
- **所有提取策略在 Prompt 中描述**，禁止代码硬编码提取规则
- **不通过代码测试约束 prompt 内容**：prompt 是策略，策略变化快，静态字符串测试会成为迭代阻力
- **Prompt 约束通过 DESIGN.md 记录**：prompt 的实体类型、关系类型、属性规范、提取流程等设计意图和变更历史在 DESIGN.md 中详细记录
- **宁可漏提也不误提**：代码示例中的变量/标签/寄存器字段访问必须明确排除
- **表格列映射显式定义**：禁止 LLM 自行推断列含义
- **属性格式统一**：width/access/reset/address_offset 等高频属性在 Prompt 中给出正例/反例
- **体系结构差异显式说明**：不同芯片架构的地址单位、寄存器命名惯例等差异需告知 LLM

**跨文档属性差异处理**：
- `doc_properties[doc_id]` 保存每个文档的原始属性快照，`properties` 为全局合并属性（冲突时比较来源文档的信息完整性评分——非空属性数量，高分覆盖低分；互补属性取并集；平局保留旧值）
- 查询接口支持 `doc_id` 参数获取指定文档的原始属性（深拷贝替换，不修改全局图）
- 属性冲突自动记录 `conflict_logs` 表供人工审核

---

## 开发计划

### 当前阶段（已完成）
- ✅ DocGraphPipeline 重构：Batch API 架构、树形章节解析、multimodal 图片处理
- ✅ 零数据丢失：移除所有源数据过滤和截断
- ✅ Prompt 外部化：所有提示词在 `scripts/prompts/*.txt`
- ✅ 代码清理：删除 11 个旧文件、4 个旧 prompt 文件、7 个旧测试文件
- ✅ **API 接口重构**：新增 `KnowledgeBaseService` 统一服务层；Plugin 暴露 22 个优化工具。`semantic_search` 合并语义搜索与图谱扩展，`graph_query` 支持深度扩展（实体→邻居→子图），`graph_upsert_entity` 统一添加/更新/验证，`graph_feedback` 统一提交与查询反馈，新增 `graph_provenance` 实现实体→文档溯源
- ✅ **图谱查询增强**：`GraphStore.find_path()` BFS 路径搜索、`search_entities()` 模糊搜索
- ✅ **跨文档知识互通**：全局统一图谱 `global.json` + 文档子图快照 `{doc_id}.json`，同名同类型实体自动去重
- ✅ **章节级增量更新**：`content_hash` 指纹比较，未变章节复用实体缓存与 embedding，仅 LLM 提取变更/新增章节
- ✅ **本地 PDF 解析器 TOC 驱动式章节识别**：基于 PDF 内置 TOC 直接匹配定位 heading，解决编号/标题分离和表格噪声问题
- ✅ **本地 PDF 解析器 Markdown 格式修复**：解决标题重复和段落内换行被误分段问题，输出格式与 BigModel Expert 保持一致
- ✅ **完整章节层级构造**：使用 PDF TOC 真实层级生成 `#`/`##`/`###`/`####`，不再扁平化为两级
- ✅ **段落内换行保留原始 `\n`**：LLM 直接理解换行语义，零额外 token，无 HTML 标签污染
- ✅ **ChapterParser 多级树重构**：栈结构构建真正的多级树；去掉 preface 传播，每个节点保留自己的原始 content
- ✅ **BatchBuilder 接收扁平化节点**：`ChapterParser.collect_all_nodes()` 收集所有有 content 的节点，为每个 chunk 附加完整标题路径前缀；段落边界切分（`\n\n+`）替代句子边界，避免编号标题被误切开
- ✅ **数据模型清理**：移除 `Document.chunks`、`Document.metadata`（早期版本）、`Chunk.page_start/page_end` 等无功能实体字段；引入 `StrEnum` 约束状态值。注意：`Chunk.metadata` 在后续版本中重新引入，现承载 `content_hash`/`extracted_entities`/`extracted_relations`/`image_descriptions`/`embedding` 等核心缓存数据
- ✅ **数据库 schema 简化**：移除 `chapters_override`、`page_start`、`page_end` 列；新增 `_schema_meta` 版本管理表；新增 `query_by_doc()` 辅助方法
- ✅ **数据质量增强**：`GraphEntity`/`GraphRelation` 新增 `confidence`/`verified`/`created_at`/`updated_at`/`feedback_score` 字段
- ✅ **图谱动态更新接口**：Plugin 暴露 `graph_upsert_entity`/`graph_delete_entity`/`graph_upsert_relation`/`graph_delete_relation` 等 4 个写工具。`graph_update_relation` 可通过 `graph_delete_relation` + `graph_upsert_relation` 组合实现
- ✅ **冲突检测与日志**：同名实体属性差异自动记录 `conflict_logs` 表，供人工审核
- ✅ **语义-图谱联合查询**：`search_with_graph()` 先 FAISS 语义搜索再扩展关联子图
- ✅ **chunk+实体联合返回**：`get_content_with_entities()` 返回 chunk 及其关联的图谱实体/关系
- ✅ **用户反馈机制**：`graph_feedback` 工具支持对实体/关系打分（+1/-1），`feedback_score` 实时汇总
- ✅ **IC 关系扩展**：新增 `DRIVES`/`DRIVEN_BY`/`TIMING_PATH`/`CLOCK_GATED_BY`/`RESET_BY`/`PARAMETERIZED_BY`/`INSTANCE_OF`
- ✅ **处理器架构图谱扩展（C28x+CLA）**：新增 `Instruction`/`InstructionGroup`/`AddressingMode`/`Operand`/`ArchitectureState`/`PipelineStage`/`FunctionalUnit`/`Interrupt`/`Exception`/`MemoryRegion`/`ShadowRegister`/`CPU_Mode`/`CLA_Task`/`Peripheral` 等实体类型，以及 `ISA_HAS_INSTRUCTION`/`INSTRUCTION_READS_REGISTER`/`INSTRUCTION_WRITES_REGISTER`/`INSTRUCTION_MODIFIES_STATE`/`MODULE_IMPLEMENTS_INSTRUCTION`/`INTERRUPT_TRIGGERS`/`HAS_PERIPHERAL`/`CLA_HAS_TASK` 等跨层级关系类型，支持 ISA 层级与 RTL 层级的知识互通
- ✅ **多文档类型图谱扩展**：新增 `Document`/`Pin`/`Package`/`ElectricalSpec`/`ApplicationDomain`/`OrderingInfo`/`Advisory`/`Workaround`/`SiliconRevision`/`UsageNote`/`Protocol`/`ProtocolLayer`/`TransactionType`/`Channel`/`MessageField`/`StateMachine`/`State`/`Transition`/`Function`/`DataStructure`/`Section`/`CodeExample` 等实体类型，以及 `SUPERSEDES`/`EXTENDS`/`AFFECTS`/`HAS_WORKAROUND`/`CORRECTS`/`DEFINES_TRANSACTION`/`USES_CHANNEL`/`HAS_STATE`/`TRANSITIONS_FROM`/`TRANSITIONS_TO`/`TRIGGERED_BY` 等关系类型，覆盖 Datasheet/Errata/Protocol Spec/ISA Manual/App Note/SW Dev Manual 六种文档类型的核心结构
- ✅ **跨产品外设变体建模**：`GraphEntity`/`GraphRelation` 新增 `doc_properties` 字段，保存每个文档的原始属性快照；引入 `Product` 实体类型，文档解析时自动提取产品型号并建立 `Product --[HAS_MODULE]--> Module` 关系；查询接口支持 `doc_id` 参数以获取指定文档的原始属性
- ✅ **属性索引优化**：`NetworkXGraphStore` 内部维护 `property_index`，`find_by_property()` 从 O(N) 降至 O(1)
- ✅ **跨粒度桥接索引**：`_EntityChunkBridge` 机制层实现 `chunk_db_id ↔ (entity_type, entity_name)` 双向映射。`graph_provenance` 从 O(N) 暴力扫描优化为 O(1) 反向查询。`search_with_graph` / `get_content_with_entities` 重构为原子操作组合
- ✅ **Pipeline 六阶段拆分**（中间产物持久化原则的具体实践）：`_process_one` 拆分为 `stage1_parse` / `stage2_build_jsonl` / `stage3_submit_batches` / `stage4_ingest_results` / `stage5_build_embed_jsonl` / `stage6_submit_embed_batches`，每个中间产物（result.md / requests.jsonl / batch_info.json / results.jsonl / 子图谱 JSON / embed.jsonl / embed_results.jsonl）均可独立执行、人为干预、重新执行
- ✅ **Chunk 粒度与向量化粒度统一**：`_save_chunks_to_db`（stage4）中，超长 chapter 通过 `_maybe_split_chapter` 拆分为独立 sub-chunks（`ch_N_part_0`/`ch_N_part_1`），继承父 chapter 的 entities/relations/content_hash。Stage 5/6 不再处理 split/average，数据库一行 = 一个向量化单位，彻底解决 batch 路径 split-chunk 未聚合导致 FAISS 重复向量的问题
- ✅ 302 个测试全部通过

### 当前阶段（新进展 — 2026-05-08 BigModel Embedding Token 估算问题研究与修复）
- ✅ **实验验证 BigModel embedding-3 tokenizer 与 tiktoken 差异**：系统性对照实验证实两者不存在固定比例关系。BigModel embedding-3 对数字/单独字母接近字符级别编码，对自然语言使用子词编码（与 tiktoken 接近），对重复字符压缩率低于 tiktoken
- ✅ **确认限制为 3072 actual tokens**：通过 Embedding API 响应 `usage.prompt_tokens` 直接测量，失败点精确在 actual_tokens = 3072 处
- ✅ **添加 `CHUNK_MAX_CHARS` 配置与 `_split_for_embedding` 字符数保护**：以 6000 字符作为 chunk 最大字符数上限
- ✅ **修复 FAISS 重复向量问题**：删除旧索引，重建 41 个唯一向量
- ✅ **重建全局图谱**：324 节点 / 412 边

### 当前阶段（新进展 — 2026-05-08 回退 tiktoken + Embedding Batch）
- ✅ **去掉 tiktoken 依赖**：`_split_for_embedding` / `_maybe_split_chapter` 改为纯字符数限制，删除 `EMBED_MAX_TOKENS_PER_REQUEST` 配置
- ✅ **Embedding 改为同步单文本 API**：stage5 生成单文本 JSONL（`body.input` 为字符串，`custom_id` 编码 db_id，不再生成 `embed_map.json`）；stage6 读取 JSONL 后调用 `EmbeddingClient.embed_single()` 逐条同步处理，删除 fallback 路径和 `EMBED_ARRAY_MAX_SIZE` 配置
- ✅ **删除 `batch_clients.submit_embedding_batch`**：不再被任何代码调用
- ✅ **299 个测试全部通过**（删除 3 个废弃测试）

### 当前阶段（新进展 — 2026-05-08 并发覆盖与接口字段名修复审计）
- ✅ **FAISS 进程级文件锁**：`VectorIndex` 新增 `fcntl.flock` + `_reload()`，修改前重载磁盘最新状态，防止多进程并发写入覆盖
- ✅ **接口字段名标准化**：`plugin_router.py` 中 `_entity_to_dict` `"type"`→`"entity_type"`；`_relation_to_dict` `"type"`→`"rel_type"`, `"from"`/`"to"`→`"from_name"`/`"to_name"`；`tool_graph_path` 同步更新路径节点与边描述字段
- ✅ **全局图完整性校验**：`_load_all_graphs()` 启动时若 nodes<10 且 documents>0 则自动 `rebuild_global_graph()`；`reprocess_document()` 保存后若节点数低于处理前 50% 自动重建
- ✅ **305 个测试全部通过**

### 当前阶段（新进展 — 2026-05-18 Prompt 分步引导与实体提取质量修复）
- ✅ **Prompt 分步引导策略**：`entity_extraction_system.txt` 引入 Step 1（内容分类）→ Step 2（按类型提取）→ Step 3（关系链补全与去重）三步流程，LLM 先判断 chunk 类型再聚焦提取，显著减少跨类型过度提取
- ✅ **代码示例寄存器字段排除**：明确禁止提取 `EPwm1Regs.TBCTL.bit.PRDLD`、`.bit.XXX = YYY` 等代码中的寄存器字段访问，解决 17 个代码字段误提取为 RegisterField 的问题
- ✅ **属性格式统一规范**：`width`→数字、`access`→标准缩写（R/W）、`reset_value`→十六进制字符串、`address_offset`→十六进制字符串，消除格式不一致
- ✅ **表格列映射规则化**：寄存器汇总表格和字段描述表格分别定义列→属性映射，明确 `Size(x16)` 列映射为 `size_in_words`，禁止将表格格式信息误解析为属性
- ✅ **RegisterField 来源限制**：只能从"Field Descriptions"表格提取，禁止从代码示例、概述、附录、法律声明中提取
- ✅ **Document 实体规则细化**：明确区分"当前文档本身"（类型 A，每个文档只提取一次）和"引用文档"（类型 B，Related Documentation 章节中可提取，需建立 CITES 关系）
- ✅ **体系结构地址单位说明**：明确不同芯片架构的地址基本单位差异（C2000=16-bit word，ARM=8-bit byte），`width` 表示寄存器位宽与地址单位无关
- ✅ **实体提取规模收敛**：全局节点从 350→193→151→176，边从 414→227→164→184，过度提取问题得到系统性控制
- ✅ **LLM Batch → Chat 回退机制**：`WORKDOCS_LLM_MODE=chat` 切换到同步 Chat API，`_submit_via_chat()` 逐条调用并将结果以 Batch API 完全一致格式写入 `results.jsonl`，Stage 4 零修改复用。单条失败不中断流程，适合调试或 Batch API 不可用时回退
- ✅ **User-Agent 伪装**：`llm_chat_client.py` 从 `plugin.json` 的 `runtime.host_version` 读取版本，默认 `KimiCLI/1.44.0`，确保通过 Kimi Coding API 白名单校验
- ✅ **304 个测试全部通过**

### 当前阶段（新进展 — 2026-05-19 DESIGN.md 审计与代码修复）
- ✅ **DESIGN.md 严格审计**：对全部 22 章 + 3 个无编号章节逐一检查代码实现。14 项完全实现、5 项部分实现、1 项未实现/遗弃
- ✅ **全局图合并策略改为信息完整性优先**：`_add_entity_unsafe` / `_add_relation_unsafe` 冲突属性时比较来源文档完整性评分（非空属性数量），高分覆盖低分。互补属性始终取并集，平局保留旧值，无法推断来源时保留现有值
- ✅ **修复 config.py 硬编码 config.json 路径**：`_load_config_json()` 先读取 `plugin.json` 的 `config_file` 字段，再回退到默认 `"config.json"`
- ✅ **修复 llm_chat_client.py thinking 参数遗漏**：改为 `extra_body.setdefault("thinking", ...)`，确保 caller 提供 `extra_body` 时 thinking 仍被设置
- ✅ **修复 GraphRelation feedback_score 未同步**：新增 `db.get_relation_feedback_score()`，修复 `KnowledgeBaseService.submit_feedback()` 中关系反馈的同步逻辑
- ✅ **DESIGN.md 同步更新**：第 1 章更新 Embedding 策略说明（补充 9 小时 Batch 处理时间 + tokenizer 不透明导致字符数限制策略的经验教训）；第 2 章标注 Neo4j 接口当前无实现；第 19 章更新 Stage 6 中间产物清单（移除 `embed_map.json` / `embed_results.jsonl`）
- ✅ **309 个测试全部通过**（新增 1 个关系反馈同步测试）

### 下一阶段（精确到下一步）
1. **可视化**：图谱可视化导出（Graphviz / D3.js）
2. **评估体系**：实体提取准确率、关系提取召回率的自动化评估
3. **DOCX/XLSX 接入 pipeline**：当前解析器代码存在但未接入 `DocGraphPipeline`

---

## Agent 自主开发原则

### 可以自主决定的事项
- 代码重构（变量重命名、函数拆分、模块移动）
- 新增单元测试或调整测试结构
- 更新日志格式、错误提示信息
- 优化算法实现（不改变输入输出语义）
- 更新本文档（AGENTS.md）和 README.md、DESIGN.md
- 不过度考虑项目开发的版本兼容性

### 必须询问用户的事项
- 修改核心数据模型（`Chunk`、`Document`、`GraphEntity`、`GraphRelation` 的字段）
- 修改数据库 Schema（新增/删除/修改表或字段）
- 修改配置系统的优先级或解析逻辑
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
| `README.md` / `AGENTS.md` / `DESIGN.md` | ✅ 可改 | 文档必须随代码同步更新 |
| `knowledge_base/` | ❌ 禁止 | 运行时生成数据 |

---

## 测试策略

### 核心原则
- **Mock 优先**：所有涉及外部 API 的测试使用 Fake 客户端，**禁止调用真实 API**
- **回归即修复**：任何导致测试失败的变更必须当场修复
- **314 个测试用例必须全部通过**

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

## 配置系统说明

### 三层优先级（高 → 低）
```
环境变量（Kimi CLI 运行时注入，如 llm.api_key）
  ↓
config.json（用户持久化配置，项目根目录）
  ↓
.env 文件（环境变量回退，如 WORKDOCS_LLM_API_KEY）
  ↓
代码硬编码默认值
```

### 关键配置项速查

完整配置列表见 `scripts/core/config.py`。以下为全部活跃配置项：

| 环境变量 | config.json 路径 | 默认值 | 说明 |
|---------|-----------------|--------|------|
| **LLM 配置** | | | |
| `WORKDOCS_LLM_API_KEY` | `llm.api_key` | 空 | LLM Batch API Key（实体提取） |
| `WORKDOCS_LLM_BASE_URL` | `llm.endpoint` | `https://api.moonshot.cn/v1` | LLM Base URL |
| `WORKDOCS_LLM_MODEL` | `llm.model` | `kimi-k2.5` | 对话模型 |
| `WORKDOCS_LLM_THINKING_ENABLED` | `llm.thinking_enabled` | `0` | 是否启用 thinking 模式（`1`=`enabled`，`0`=`disabled`）。Kimi K2.6 等模型 thinking 默认开启，必须显式传递才能可靠关闭 |
| `WORKDOCS_LLM_MODE` | `llm.mode` | `batch` | LLM 实体提取模式：`batch`（Batch API，默认）或 `chat`（同步 Chat API，逐条调用，适合调试或 Batch API 不可用时回退）。Chat 模式结果以与 Batch 完全一致的格式写入 `results.jsonl`，Stage 4 零修改复用 |
| `WORKDOCS_LLM_BATCH_ENDPOINT` | `llm.batch_endpoint` | `/v1/chat/completions` | LLM Batch API endpoint |
| `WORKDOCS_LLM_BATCH_COMPLETION_WINDOW` | `llm.completion_window` | `24h` | Batch 完成窗口 |
| `WORKDOCS_LLM_BATCH_MAX_CHARS` | `llm.batch_max_chars` | `10000` | 每个 LLM batch 最大文本字符数 |
| `WORKDOCS_LLM_BATCH_TIMEOUT` | `llm.batch_timeout` | `3600` | LLM Batch API 轮询超时（秒） |
| `WORKDOCS_LLM_MAX_RETRIES` | `llm.max_retries` | `3` | LLM 同步请求最大重试次数 |
| `WORKDOCS_LLM_RETRY_BACKOFF` | `llm.retry_backoff` | `2` | LLM 重试退避系数（秒） |
| `WORKDOCS_LLM_TIMEOUT` | `llm.timeout` | `120` | LLM 同步请求超时（秒） |
| `WORKDOCS_LLM_VISION_MAX_EDGE` | `llm.vision_max_edge` | `1024` | 图片压缩最长边（px） |
| `WORKDOCS_LLM_VISION_QUALITY` | `llm.vision_quality` | `85` | JPEG 压缩质量 1-100 |
| **Embedding 配置** | | | |
| `WORKDOCS_EMBEDDING_API_KEY` | `embedding.api_key` | 空 | Embedding API Key（向量化） |
| `WORKDOCS_EMBEDDING_BASE_URL` | `embedding.endpoint` | `https://open.bigmodel.cn/api/paas/v4` | Embedding Base URL |
| `WORKDOCS_EMBEDDING_MODEL` | `embedding.model` | `embedding-3` | 向量化模型 |
| `WORKDOCS_EMBEDDING_DIMENSION` | `embedding.dimension` | `1024` | 向量维度 |
| `WORKDOCS_EMBEDDING_BATCH_ENDPOINT` | `embedding.batch_endpoint` | `/v4/embeddings` | ~~Embedding Batch API endpoint~~（已废弃，Embedding 改为同步单文本 API） |
| `WORKDOCS_EMBED_BATCH_TIMEOUT` | `embedding.batch_timeout` | `3600` | ~~Embedding Batch API 轮询超时（秒）~~（已废弃） |
| `WORKDOCS_CHUNK_MAX_CHARS` | `chunk.max_chars` | `6000` | **单个 chunk 的最大字符数上限**。在 stage4 入库时，若 chapter content 超过此值，`_maybe_split_chapter` 会将其拆分为多个 sub-chunks。这是一个基于项目文本分布的**经验参数**（自然语言+代码混合内容约对应 2500-2800 actual tokens），不保证对所有文本类型安全 |
| `WORKDOCS_EMBED_MAX_RETRIES` | `embedding.max_retries` | `3` | Embedding 同步请求最大重试次数 |
| `WORKDOCS_EMBED_RETRY_BACKOFF` | `embedding.retry_backoff` | `2` | Embedding 重试退避系数（秒） |
| `WORKDOCS_EMBED_TIMEOUT` | `embedding.timeout` | `120` | Embedding 同步请求超时（秒） |
| **Parser 配置** | | | |
| `WORKDOCS_PARSER_API_KEY` | `parser.api_key` | 空 | PDF 解析 API Key（BigModel 专用） |
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
| **Plugin 默认值** | | | |
| `WORKDOCS_PLUGIN_SEARCH_TOP_K` | `plugin.search_top_k` | `5` | 语义搜索默认返回条数 |
| `WORKDOCS_PLUGIN_QUERY_TOP_K` | `plugin.query_top_k` | `10` | 查询默认返回条数 |
| `WORKDOCS_PLUGIN_GRAPH_MAX_DEPTH` | `plugin.graph_max_depth` | `3` | 图谱查询默认最大深度 |
| `WORKDOCS_PLUGIN_SUBGRAPH_DEPTH` | `plugin.subgraph_depth` | `1` | 子图扩展默认深度 |
| `WORKDOCS_PLUGIN_DEFAULT_LIMIT` | `plugin.default_limit` | `100` | 默认分页限制 |
| **Pipeline / Graph** | | | |

| `WORKDOCS_GRAPH_MAX_PATH_DEPTH` | `graph.max_path_depth` | `6` | 图谱路径搜索最大深度 |
| `WORKDOCS_GRAPH_OUTPUT_DIR` | `graph.output_dir` | `graphs` | 图谱 JSON 输出目录 |

> 注：`config.json` 的 `config_file` 由 `plugin.json` 指定，Kimi CLI 可自动管理。`.env` 文件支持双路径加载：项目根目录和 `scripts/` 目录下的 `.env` 均会被读取（后者优先级更高）。

---

## 官方 API 开发文档

+ [Kimi Code CLI 插件](https://moonshotai.github.io/kimi-cli/zh/customization/plugins.html)
+ [Kimi API 概述](https://platform.kimi.com/docs/api/overview)
+ [Kimi 模型参数参考](https://platform.kimi.com/docs/api/models-overview)
+ [Kimi 使用 Batch API 批量处理任务](https://platform.kimi.com/docs/guide/use-batch-api)
+ [BigModel API 使用概述](https://docs.bigmodel.cn/cn/api/introduction)
+ [BigModel 结构化输出](https://docs.bigmodel.cn/cn/guide/capabilities/struct-output)
+ [BigModel Embedding-3](https://docs.bigmodel.cn/cn/guide/models/embedding/embedding-3)
+ [BigModel 批量处理](https://docs.bigmodel.cn/cn/guide/tools/batch)
+ [BigModel 新文件解析服务](https://docs.bigmodel.cn/cn/guide/tools/file-parser)
