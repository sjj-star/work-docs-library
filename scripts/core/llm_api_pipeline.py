"""
LLM API 驱动的处理管道
使用独立的 LLM 对话模型进行高质量文档总结和图像分析
"""
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Dict, Any
import shutil

from .config import Config
from .db import KnowledgeDB
from .vector_index import VectorIndex
from .models import Chunk, Document
from .llm_chat_client import LLMChatClient
from .embedding_client import EmbeddingClient
from .context_manager import get_context_manager, ContextWindowManager
from parsers.pdf_parser import PDFParser
from parsers.office_parser import OfficeParser

logger = logging.getLogger(__name__)


class LLMAPIIngestionPipeline:
    """LLM API 驱动的处理管道 - 使用独立配置的 LLM 模型"""
    
    def __init__(self, db: Optional[KnowledgeDB] = None, vec: Optional[VectorIndex] = None) -> None:
        self.db = db or KnowledgeDB()
        self.vec = vec or VectorIndex(dim=Config.EMBEDDING_DIMENSION)
        
        # 使用独立的 LLM 和 Embedding 客户端
        try:
            self.llm_client = LLMChatClient()
            logger.info(f"LLM 对话客户端已初始化 - 模型: {Config.LLM_MODEL} ({Config.LLM_PROVIDER})")
        except RuntimeError as e:
            logger.warning(f"LLM 客户端初始化失败: {e}")
            self.llm_client = None
        
        try:
            self.embedder = EmbeddingClient()
            logger.info(f"Embedding 客户端已初始化 - 模型: {Config.EMBEDDING_MODEL} ({Config.EMBEDDING_PROVIDER})")
        except RuntimeError as e:
            logger.warning(f"Embedding 客户端初始化失败: {e}")
            self.embedder = None
        
        # 上下文管理器
        self.context_manager = get_context_manager()
        logger.info("上下文窗口管理器已初始化")
        
        self.parsers = {
            ".pdf": PDFParser(),
            ".docx": OfficeParser(),
            ".xlsx": OfficeParser(),
        }
    
    def scan(self, path: str) -> List[str]:
        """扫描文件"""
        p = Path(path)
        files = []
        if p.is_file():
            files.append(str(p.resolve()))
        elif p.is_dir():
            for ext in self.parsers.keys():
                files.extend(str(f) for f in p.rglob(f"*{ext}"))
        return sorted(set(files))
    
    def ingest(self, path: str, dry_run: bool = False, auto_chapter: bool = False) -> List[str]:
        """主入口：处理文档集合"""
        files = self.scan(path)
        ingested: List[str] = []
        
        for f in files:
            doc_id = self._process_one(f, dry_run=dry_run, auto_chapter=auto_chapter)
            if doc_id:
                ingested.append(doc_id)
        
        return ingested
    
    def _process_one(self, file_path: str, dry_run: bool, auto_chapter: bool, force: bool = False) -> Optional[str]:
        """处理单个文档 - 两阶段：Phase A (Parse & Embed) + Phase B (LLM Enhance)"""
        suffix = Path(file_path).suffix.lower()
        parser = self.parsers.get(suffix)
        if not parser:
            return None
        
        # 强制重新处理时清理旧数据
        if force:
            file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
            old_img_dir = Config.DB_PATH.parent / "images" / file_hash
            if old_img_dir.exists():
                shutil.rmtree(old_img_dir)
        
        # 基础解析
        extract_images = True
        if suffix == ".pdf":
            raw_doc = parser.parse(file_path, extract_images=extract_images)
        else:
            raw_doc = parser.parse(file_path)
        
        if dry_run:
            logger.info("Dry-run | file=%s | pages=%s | chapters=%s", file_path, raw_doc.total_pages, len(raw_doc.chapters))
            return raw_doc.doc_id
        
        doc_id = raw_doc.doc_id
        existing = self.db.get_document_by_path(file_path)
        
        # 跳过已完成的文档
        if not force and existing and existing.file_hash == raw_doc.file_hash and existing.status == "done":
            logger.info("Skip unchanged | file=%s | doc_id=%s", file_path, existing.doc_id)
            return existing.doc_id
        
        # 检查 embedder
        if self.embedder is None:
            logger.warning("No embedder | file=%s will be ingested without embeddings", file_path)
        
        # --- Phase A: Parse & Embed ---
        need_phase_a = True
        if existing and not force:
            with self.db._connect() as conn:
                rows = conn.execute("SELECT status FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
            if rows:
                statuses = {r["status"] for r in rows}
                if statuses == {"done"}:
                    logger.info("Skip unchanged | doc_id=%s | all chunks done", doc_id)
                    return doc_id
                if "embedded" in statuses or "pending" in statuses:
                    need_phase_a = False
                    logger.info("Resume Phase B | doc_id=%s | existing chunks found", doc_id)
        
        if need_phase_a:
            self.db.upsert_document(raw_doc)
            self.db.update_document_status(doc_id, "processing")
            
            # 清理旧数据（如果存在）
            if existing:
                with self.db._connect() as conn:
                    old_chunks = conn.execute(
                        "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
                    ).fetchall()
                if old_chunks:
                    self.vec.remove_doc([r["id"] for r in old_chunks])
                    self.db.delete_chunks_by_doc(doc_id)
                    self.db.delete_chapter_summaries_by_doc(doc_id)
                    self.db.delete_concepts_by_doc(doc_id)
            
            # 处理 chunks（原始内容，不带 LLM 增强）
            chunks = raw_doc.chunks if raw_doc.chunks else []
            if not chunks:
                content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
                chunks = [Chunk(
                    doc_id=doc_id,
                    chunk_id="full",
                    content=content,
                    chunk_type="text",
                    page_start=1,
                    page_end=raw_doc.total_pages or 1,
                )]
            
            # 分配章节
            chapters = self.db.get_chapters(doc_id) if not auto_chapter else raw_doc.chapters
            for ck in chunks:
                ck.doc_id = doc_id
                for ch in chapters:
                    if ch.start_page <= ck.page_start <= ch.end_page:
                        ck.chapter_title = ch.title
                        break
            
            # 存储 chunks
            chunk_db_ids = []
            for ck in chunks:
                db_id = self.db.insert_chunk(ck)
                chunk_db_ids.append(db_id)
            
            # 向量化处理
            if self.embedder:
                texts = [ck.content for ck in chunks]
                for i in range(0, len(texts), Config.BATCH_SIZE):
                    batch_texts = texts[i:i + Config.BATCH_SIZE]
                    batch_ids = chunk_db_ids[i:i + Config.BATCH_SIZE]
                    try:
                        embeddings = self.embedder.embed(batch_texts)
                        # 批量更新 SQLite 和 FAISS
                        items = list(zip(batch_ids, embeddings))
                        self.db.update_chunks_embedded_batch(items)
                        self.vec.add_batch(items)
                    except Exception as e:
                        logger.error("Embed error | doc_id=%s | error=%s", doc_id, e)
                        self.db.update_document_status(doc_id, "failed")
                        raise
            else:
                for db_id in chunk_db_ids:
                    self.db.update_chunk_status(db_id, "embedded")
                logger.warning("No embedder | skipped vector index generation for doc_id=%s", doc_id)
        
        # --- Phase B: LLM Enhance ---
        if self.llm_client:
            try:
                doc_for_llm = self._load_document_with_chunks(doc_id)
                if not doc_for_llm:
                    doc_for_llm = raw_doc
                
                enhanced_doc = self._llm_enhance_document(doc_for_llm)
                self._persist_llm_outputs(doc_id, enhanced_doc)
            except Exception as e:
                logger.error("LLM enhance error | doc_id=%s | error=%s", doc_id, e)
                self.db.update_document_status(doc_id, "failed")
                raise
        else:
            with self.db._connect() as conn:
                rows = conn.execute(
                    "SELECT id FROM chunks WHERE doc_id = ? AND status = 'embedded'", (doc_id,)
                ).fetchall()
            for r in rows:
                self.db.update_chunk_status(r["id"], "done")
        
        self.db.update_document_status(doc_id, "done")
        logger.info("Ingestion complete | file=%s | doc_id=%s | mode=LLM_API_FLOW", file_path, doc_id)
        return doc_id
    
    def _load_document_with_chunks(self, doc_id: str) -> Optional[Document]:
        """从数据库加载文档及其 chunks（用于 Phase B 断点续传）"""
        doc = self.db.get_document(doc_id)
        if not doc:
            return None
        with self.db._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY page_start, chunk_id",
                (doc_id,)
            ).fetchall()
        if not rows:
            return None
        import json
        chunks = []
        for r in rows:
            kw_raw = r["keywords"] or ""
            try:
                keywords = json.loads(kw_raw)
            except json.JSONDecodeError:
                keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]
            chunks.append(Chunk(
                doc_id=r["doc_id"],
                chunk_id=r["chunk_id"],
                content=r["content"],
                chunk_type=r["chunk_type"],
                page_start=r["page_start"],
                page_end=r["page_end"],
                chapter_title=r["chapter_title"],
                keywords=keywords,
                summary=r["summary"] or "",
                status=r["status"] or "pending",
                metadata=json.loads(r["metadata"] or "{}"),
            ))
        doc.chunks = chunks
        return doc
    
    def _persist_llm_outputs(self, doc_id: str, enhanced_doc: Document) -> None:
        """持久化 LLM 增强结果到数据库（支持增量更新，跳过已 done 的 chunks）"""
        import json
        import re
        
        # 获取当前 chunk 状态映射
        with self.db._connect() as conn:
            rows = conn.execute(
                "SELECT id, chunk_id, status, metadata, page_start, content FROM chunks WHERE doc_id = ?",
                (doc_id,)
            ).fetchall()
        chunk_status_map = {r["chunk_id"]: {"db_id": r["id"], "status": r["status"], "metadata": json.loads(r["metadata"] or "{}"), "page_start": r["page_start"], "content": r["content"]} for r in rows}
        
        # 获取文档级结构化数据
        entities = enhanced_doc.metadata.get("entities", [])
        relationships = enhanced_doc.metadata.get("relationships", [])
        answered_questions = enhanced_doc.metadata.get("answered_questions", [])
        keywords = enhanced_doc.metadata.get("keywords", [])
        keywords_str = ",".join(keywords) if isinstance(keywords, list) else str(keywords)
        doc_summary = enhanced_doc.metadata.get("llm_summary", "")
        
        # 1. 写入章节摘要（包含结构化字段）
        chapter_summaries = enhanced_doc.metadata.get("chapter_summaries", [])
        chapter_summary_map = {}
        for cs in chapter_summaries:
            ch_title = cs.get("chapter_title", "")
            if not ch_title:
                continue
            chapter_summary_map[ch_title] = cs.get("summary", "")
            
            # 提取章节相关的 concepts（从 entities 转换）
            ch_concepts = []
            for ent in entities[:10]:
                if isinstance(ent, dict):
                    ch_concepts.append({
                        "name": ent.get("name", ""),
                        "definition": ent.get("definition", ent.get("description", "")),
                        "pages": [cs.get("start_page")] if cs.get("start_page") else []
                    })
            
            # 提取章节内容中的 key_figures 和 key_tables
            ch_text = ""
            for chunk in enhanced_doc.chunks:
                if chunk.chapter_title == ch_title and chunk.chunk_type == "text":
                    ch_text += chunk.content + "\n"
            
            key_figures = self._extract_figures_from_text(ch_text)
            key_tables = self._extract_tables_from_text(ch_text)
            
            self.db.upsert_chapter_summary(
                doc_id,
                ch_title,
                {
                    "start_page": cs.get("start_page"),
                    "end_page": cs.get("end_page"),
                    "summary": cs.get("summary", ""),
                    "concepts": ch_concepts,
                    "relationships": relationships[:10] if relationships else [],
                    "key_figures": key_figures,
                    "key_tables": key_tables,
                    "status": "done",
                }
            )
        
        # 2. 为每个 embedded chunk 写入 summary / keywords / status=done / metadata
        for ck in enhanced_doc.chunks:
            info = chunk_status_map.get(ck.chunk_id)
            if not info:
                continue
            db_id = info["db_id"]
            # 跳过已完成的 chunks（resume 场景）
            if info["status"] == "done":
                continue
            
            ch_summary = chapter_summary_map.get(ck.chapter_title or "", "")
            chunk_summary = ch_summary if ch_summary else doc_summary
            if chunk_summary:
                self.db.update_chunk_summary(db_id, chunk_summary)
            if keywords_str:
                self.db.update_chunk_keywords(db_id, keywords_str)
            
            # 写入结构化元数据到 chunk metadata
            meta = info["metadata"]
            if entities:
                meta["entities"] = entities
            if relationships:
                meta["relationships"] = relationships
            if answered_questions:
                meta["answered_questions"] = answered_questions
            if meta:
                self.db.update_chunk_metadata(db_id, meta)
            
            self.db.update_chunk_status(db_id, "done")
        
        # 3. 持久化图像分析结果到 chunk metadata
        image_analyses = enhanced_doc.metadata.get("image_analyses", [])
        for ia in image_analyses:
            chunk_id = ia.get("chunk_id")
            if not chunk_id or chunk_id not in chunk_status_map:
                continue
            info = chunk_status_map[chunk_id]
            db_id = info["db_id"]
            meta = info["metadata"]
            imgs = meta.get("images", [])
            for img in imgs:
                if str(img.get("path")) == str(ia.get("path", "")):
                    img["vision_desc"] = ia.get("analysis", "")
                    break
            else:
                imgs.append({"path": ia.get("path", ""), "vision_desc": ia.get("analysis", "")})
            meta["images"] = imgs
            if "vision_insights" in ia:
                meta["vision_insights"] = ia["vision_insights"]
            self.db.update_chunk_metadata(db_id, meta)
        
        # 4. 从 entities 生成高质量概念索引（回退到关键词）
        concept_sources = []
        for ent in entities:
            if isinstance(ent, dict) and ent.get("name"):
                concept_sources.append({
                    "name": ent["name"],
                    "definition": ent.get("definition", ent.get("description", f"实体: {ent['name']}")),
                })
        if not concept_sources:
            for kw in keywords:
                if kw and isinstance(kw, str):
                    concept_sources.append({"name": kw, "definition": f"关键词: {kw}"})
        
        # 计算 first_mentioned_page
        for concept in concept_sources:
            first_page = None
            for ck in enhanced_doc.chunks:
                if ck.chunk_type == "text" and concept["name"] in ck.content:
                    info = chunk_status_map.get(ck.chunk_id)
                    if info and info.get("page_start") is not None:
                        first_page = info["page_start"]
                        break
            
            # 计算 related_concepts（从 relationships 推导）
            related = []
            for rel in relationships:
                if isinstance(rel, dict):
                    if rel.get("from") == concept["name"] and rel.get("to"):
                        related.append(rel["to"])
                    elif rel.get("to") == concept["name"] and rel.get("from"):
                        related.append(rel["from"])
            
            self.db.upsert_concept(
                doc_id=doc_id,
                concept_name=concept["name"],
                definition=concept["definition"],
                first_mentioned_page=first_page,
                related_concepts=related[:5],
            )
        
        logger.info("LLM output persisted | doc_id=%s | chapters=%s | images=%s | keywords=%s | entities=%s | concepts=%s",
                   doc_id, len(chapter_summaries), len(image_analyses), len(keywords), len(entities), len(concept_sources))
    
    def _llm_enhance_document(self, raw_doc: Document) -> Document:
        """使用 LLM 增强文档内容（四大模块并发执行）"""
        if not self.llm_client:
            logger.warning("LLM 客户端不可用，跳过增强处理")
            return raw_doc
        
        logger.info(f"开始 LLM 增强处理 | doc_id={raw_doc.doc_id}")
        
        # 四大模块无数据依赖，使用线程池并发执行
        max_workers = min(4, Config.BATCH_SIZE)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_summary = executor.submit(self._hierarchical_summarize, raw_doc)
            future_images = executor.submit(self._analyze_document_images, raw_doc)
            future_chapters = executor.submit(self._generate_chapter_summaries, raw_doc)
            future_structured = executor.submit(self._extract_structured_metadata, raw_doc)
            
            text_summary = future_summary.result()
            image_analyses = future_images.result()
            chapter_summaries = future_chapters.result()
            structured = future_structured.result()
        
        # 5. 更新文档元数据
        raw_doc.metadata.update({
            "llm_summary": text_summary,
            "image_analyses": image_analyses,
            "chapter_summaries": chapter_summaries,
            "keywords": structured["keywords"],
            "entities": structured["entities"],
            "relationships": structured["relationships"],
            "answered_questions": structured["answered_questions"],
            "processing_mode": "LLM_API_FLOW",
            "llm_model": Config.LLM_MODEL,
            "llm_provider": Config.LLM_PROVIDER,
            "thinking_enabled": Config.LLM_THINKING_ENABLED
        })
        
        logger.info(f"LLM 增强处理完成 | doc_id={raw_doc.doc_id} | summary_length={len(text_summary)}")
        return raw_doc
    
    def _hierarchical_summarize(self, doc: Document) -> str:
        """层次化文档总结 - 使用上下文管理器优化"""
        if not self.llm_client:
            logger.warning("LLM 客户端不可用，跳过文档总结")
            return ""
        
        # 收集所有文本内容
        all_text = []
        for chunk in doc.chunks:
            if chunk.chunk_type == "text":
                all_text.append(chunk.content)
        
        if not all_text:
            return ""
        
        full_text = "\n\n".join(all_text)
        
        # 估算总 token 数
        estimated_tokens = self.context_manager.estimate_tokens(full_text)
        max_tokens = self.context_manager.model_params["max_tokens"]
        
        if estimated_tokens <= max_tokens * 0.8:
            # 文本较短，直接总结
            return self.llm_client.summarize(full_text)["summary"]
        
        # 智能分块（按段落边界）
        chunks = self._smart_chunking(full_text, max_size=6000)
        
        if len(chunks) == 1:
            return self.llm_client.summarize(chunks[0])["summary"]
        
        # 层次化总结：块级并发
        max_workers = min(4, Config.BATCH_SIZE)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.llm_client.summarize, chunk): i for i, chunk in enumerate(chunks)}
            results = {}
            for future in futures:
                i = futures[future]
                try:
                    results[i] = future.result()["summary"]
                except Exception as e:
                    logger.error(f"块总结失败 | chunk={i} | error={e}")
                    results[i] = ""
        
        chunk_summaries = [f"部分{i+1}: {results[i]}" for i in range(len(chunks)) if results[i]]
        
        if not chunk_summaries:
            logger.error("所有块的总结都为空")
            return ""
        
        # 最终汇总
        combined = "\n\n".join(chunk_summaries)
        return self.llm_client.summarize(combined)["summary"]
    
    def _smart_chunking(self, text: str, max_size: int = 6000) -> List[str]:
        """智能分块，优先按章节/段落边界分割"""
        chunks = []
        current_chunk = ""
        
        # 按段落分割
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            if len(current_chunk + para) < max_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _analyze_document_images(self, doc: Document) -> List[Dict[str, Any]]:
        """分析文档中的所有图像"""
        if not self.llm_client:
            return []
        
        image_analyses = []
        image_count = 0
        
        # 收集所有图像任务
        image_tasks = []
        for chunk in doc.chunks:
            imgs = chunk.metadata.get("images", []) if chunk.metadata else []
            for img in imgs:
                image_tasks.append((chunk, img))
        
        if not image_tasks:
            return image_analyses
        
        # 图像分析并发执行
        max_workers = min(4, len(image_tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._analyze_single_image, chunk, img, i): i for i, (chunk, img) in enumerate(image_tasks)}
            for future in futures:
                i = futures[future]
                try:
                    result = future.result()
                    image_analyses.append(result)
                    if "error" not in result:
                        image_count += 1
                        logger.info(f"图像分析完成 | image_id=img_{image_count} | length={len(result['analysis'])}")
                except Exception as e:
                    logger.error(f"图像分析失败 | task={i} | error={e}")

        logger.info(f"图像分析完成 | total_images={image_count}")
        return image_analyses
    
    def _analyze_single_image(self, chunk, img, idx: int) -> Dict[str, Any]:
        """分析单个图像（用于线程池并发）"""
        prompt = """分析这个技术图表/图像，提供：
1. 图表类型和主要内容
2. 关键元素和标注
3. 技术含义和作用
4. 与上下文的可能关联

请用中文回答，保持简洁专业。"""
        try:
            analysis = self.llm_client.vision_describe(img["path"], prompt)
            return {
                "image_id": f"img_{idx + 1}",
                "path": str(img["path"]),
                "analysis": analysis,
                "chunk_id": chunk.chunk_id,
                "page": chunk.page_start
            }
        except Exception as e:
            logger.error(f"图像分析失败 | path={img.get('path', 'unknown')} | error={e}")
            return {
                "image_id": f"img_{idx + 1}",
                "path": str(img.get("path", "unknown")),
                "analysis": f"图像分析失败: {e}",
                "chunk_id": chunk.chunk_id,
                "page": chunk.page_start,
                "error": str(e)
            }
        
        logger.info(f"图像分析完成 | total_images={image_count}")
        return image_analyses
    
    def _generate_chapter_summaries(self, doc: Document) -> List[Dict[str, Any]]:
        """生成章节摘要 - 使用递归章节总结"""
        if not self.llm_client or not doc.chapters:
            return []
        
        if len(doc.chapters) < 2:
            # 章节数较少，使用普通章节摘要
            return self._generate_simple_chapter_summaries(doc)
        
        # 使用递归章节总结
        try:
            # 调用递归章节总结
            final_summary = self._recursive_chapter_summarization(doc)
            
            if not final_summary:
                logger.warning("递归章节总结返回空结果，使用普通章节摘要")
                return self._generate_simple_chapter_summaries(doc)
            
            # 从元数据获取章节摘要
            chapter_summaries = doc.metadata.get("recursive_chapter_summaries", [])
            
            if not chapter_summaries:
                logger.warning("未找到递归章节摘要，使用普通章节摘要")
                return self._generate_simple_chapter_summaries(doc)
            
            return chapter_summaries
            
        except Exception as e:
            logger.error(f"递归章节总结失败: {e}，回退到普通章节摘要")
            return self._generate_simple_chapter_summaries(doc)
    
    def _generate_simple_chapter_summaries(self, doc: Document) -> List[Dict[str, Any]]:
        """生成普通章节摘要（不使用递归）"""
        chapter_summaries = []
        
        # 准备章节任务
        chapter_tasks = []
        for chapter in doc.chapters:
            chapter_texts = []
            for chunk in doc.chunks:
                if chunk.chapter_title == chapter.title and chunk.chunk_type == "text":
                    chapter_texts.append(chunk.content)
            chapter_tasks.append((chapter, chapter_texts))
        
        # 普通章节摘要并发执行
        max_workers = min(4, len(chapter_tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._summarize_single_chapter, chapter, texts): chapter for chapter, texts in chapter_tasks}
            for future in futures:
                try:
                    result = future.result()
                    chapter_summaries.append(result)
                    if "error" not in result:
                        logger.info(f"章节摘要生成完成 | chapter={result['chapter_title']} | length={len(result['summary'])}")
                except Exception as e:
                    chapter = futures[future]
                    logger.error(f"章节摘要生成失败 | chapter={chapter.title} | error={e}")
                    chapter_summaries.append({
                        "chapter_title": chapter.title,
                        "start_page": chapter.start_page,
                        "end_page": chapter.end_page,
                        "summary": f"章节摘要生成失败: {e}",
                        "error": str(e)
                    })
        
        return chapter_summaries
    
    def _summarize_single_chapter(self, chapter, chapter_texts: List[str]) -> Dict[str, Any]:
        """总结单个章节（用于线程池并发）"""
        if not chapter_texts:
            return {
                "chapter_title": chapter.title,
                "start_page": chapter.start_page,
                "end_page": chapter.end_page,
                "summary": "",
                "text_length": 0
            }
        chapter_content = "\n\n".join(chapter_texts)
        chapter_summary = self.llm_client.summarize(chapter_content)["summary"]
        return {
            "chapter_title": chapter.title,
            "start_page": chapter.start_page,
            "end_page": chapter.end_page,
            "summary": chapter_summary,
            "text_length": len(chapter_content)
        }
        
        return chapter_summaries
    
    def _extract_figures_from_text(self, text: str) -> List[str]:
        """从文本中提取图表引用（Figure/Fig./图）"""
        if not text:
            return []
        import re
        figures = []
        # 匹配 Figure/Fig./图 后跟编号和标题
        for m in re.finditer(r'(?:Figure|Fig\.?|图)\s*[\d\-\.]+(?:\s*[:：\-]\s*|\s+)([^\n\.]+)', text, re.IGNORECASE):
            caption = m.group(0).strip()
            if caption and caption not in figures:
                figures.append(caption)
        return figures[:5]
    
    def _extract_tables_from_text(self, text: str) -> List[str]:
        """从文本中提取表格引用（Table/表）"""
        if not text:
            return []
        import re
        tables = []
        for m in re.finditer(r'(?:Table|表)\s*[\d\-\.]+(?:\s*[:：\-]\s*|\s+)([^\n\.]+)', text, re.IGNORECASE):
            caption = m.group(0).strip()
            if caption and caption not in tables:
                tables.append(caption)
        return tables[:5]
    
    def _parse_llm_json(self, text: str) -> Optional[Dict[str, Any]]:
        """安全解析 LLM 返回的 JSON，支持 markdown 代码块包裹"""
        import json
        cleaned = text.strip()
        # 移除可能的 markdown 代码块标记
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
    
    def _extract_structured_metadata(self, doc: Document) -> Dict[str, Any]:
        """提取结构化元数据：关键词、实体、关系、已回答问题"""
        if not self.llm_client:
            return {"keywords": [], "entities": [], "relationships": [], "answered_questions": []}
        
        try:
            sources = [doc.title] if doc.title else []
            for chapter in doc.chapters:
                sources.append(chapter.title)
            
            text = "\n".join(sources)
            if len(text) < 50:
                for chunk in doc.chunks[:3]:
                    if chunk.chunk_type == "text":
                        text += "\n" + chunk.content[:500]
                        if len(text) > 1000:
                            break
            
            from .llm_chat_client import BaseLLMClient
            prompt_template = BaseLLMClient._load_prompt("structural_summarize")
            prompt = prompt_template.replace("{{text}}", text[:2000])
            
            response = self.llm_client.chat([{"role": "user", "content": prompt}])
            data = self._parse_llm_json(response)
            if data and isinstance(data, dict):
                return {
                    "keywords": data.get("keywords", [])[:10] if isinstance(data.get("keywords"), list) else [],
                    "entities": data.get("entities", []) if isinstance(data.get("entities"), list) else [],
                    "relationships": data.get("relationships", []) if isinstance(data.get("relationships"), list) else [],
                    "answered_questions": data.get("answered_questions", [])[:5] if isinstance(data.get("answered_questions"), list) else [],
                }
            
            # 回退：只提取关键词
            keywords = []
            for line in response.splitlines():
                if "," in line:
                    keywords = [k.strip() for k in line.split(",") if k.strip()]
                    break
            return {"keywords": keywords[:10], "entities": [], "relationships": [], "answered_questions": []}
            
        except Exception as e:
            logger.error(f"结构化元数据提取失败 | error={e}")
            return {"keywords": [], "entities": [], "relationships": [], "answered_questions": []}
    
    def _recursive_chapter_summarization(self, doc: Document) -> str:
        """
        递归式章节总结 - 使用上下文管理器优化章节感知的总结
        """
        if not self.llm_client:
            logger.warning("LLM 客户端不可用，退回到普通层次化总结")
            return self._hierarchical_summarize(doc)
        
        if not doc.chapters or len(doc.chapters) < 2:
            logger.debug("文档章节数 < 2，使用普通层次化总结")
            return self._hierarchical_summarize(doc)
        
        logger.info(f"开始递归章节总结 | doc_id={doc.doc_id} | chapters={len(doc.chapters)}")
        
        # 1. 按章节收集内容
        chapter_contents = self._collect_chapter_contents(doc)
        
        # 2. 生成章节摘要
        chapter_summaries = []
        previous_summaries = []
        
        for i, (chapter, content) in enumerate(chapter_contents):
            if not content.strip():
                previous_summaries.append("")
                continue
            
            # 构建上下文
            context = self.context_manager.calculate_optimal_context(
                content=content,
                chapter_index=i,
                previous_summaries=previous_summaries
            )
            
            # 生成章节摘要
            chapter_summary = self._generate_chapter_summary(
                chapter_title=chapter.title,
                context=context,
                chapter_index=i,
                total_chapters=len(chapter_contents)
            )
            
            if chapter_summary:
                chapter_summaries.append({
                    "chapter_title": chapter.title,
                    "summary": chapter_summary,
                    "index": i
                })
                previous_summaries.append(chapter_summary)
        
        if not chapter_summaries:
            return self._hierarchical_summarize(doc)
        
        # 3. 递归聚合
        final_summary = self._recursive_summary_aggregation(chapter_summaries)
        
        # 4. 保存到元数据
        doc.metadata.update({
            "recursive_chapter_summaries": chapter_summaries
        })
        
        return final_summary
    
    def _collect_chapter_contents(self, doc: Document) -> List[tuple]:
        """按章节收集内容"""
        if not doc.chapters:
            return []
        
        chapter_contents = []
        for chapter in doc.chapters:
            chapter_texts = []
            for chunk in doc.chunks:
                if (chunk.chapter_title == chapter.title and 
                    chunk.chunk_type == "text" and 
                    chunk.content.strip()):
                    chapter_texts.append(chunk.content)
            
            combined_content = "\n\n".join(chapter_texts)
            chapter_contents.append((chapter, combined_content))
        
        return chapter_contents
    
    def _generate_chapter_summary(self, chapter_title: str, context: str,
                                chapter_index: int, total_chapters: int) -> str:
        """生成章节摘要，利用前面章节的摘要作为上下文理解当前章节。"""
        if not self.llm_client:
            return ""
        
        prompt = f"""你对一本技术书籍进行逐章总结。当前是第 {chapter_index + 1}/{total_chapters} 章。

**任务要求**：
1. 总结当前章节的核心内容、关键机制和设计 rationale。
2. 如果上下文中提供了前面章节的摘要，请利用它们来理解当前章节：识别因果关系、前置依赖、概念演进或主题转折。
3. 明确指出当前章节与前面哪些内容有直接关联（例如："本章的 DMA 配置依赖于第 3 章介绍的时钟树"）。
4. 如果当前章节是后续内容的基础，也请指出。

**章节标题**：{chapter_title}

**上下文（前面章节的摘要，如有）**：
{context}

**请输出当前章节的结构化摘要**："""
        
        try:
            return self.llm_client.chat([{"role": "user", "content": prompt}]).strip()
        except Exception as e:
            logger.error(f"章节摘要生成失败: {e}")
            return ""
    
    def _recursive_summary_aggregation(self, chapter_summaries: List[Dict]) -> str:
        """递归生成摘要的摘要"""
        if len(chapter_summaries) <= 3:
            return self._direct_summary_aggregation(chapter_summaries)
        
        return self._grouped_recursive_aggregation(chapter_summaries)
    
    def _direct_summary_aggregation(self, chapter_summaries: List[Dict]) -> str:
        """直接汇总多个章节摘要，产生更高层次的'摘要的摘要'。"""
        if not self.llm_client:
            return ""
        
        combined = "\n\n".join([
            f"第{s['index'] + 1}章 {s.get('chapter_title', s.get('chapter', ''))}:\n{s['summary']}"
            for s in chapter_summaries
        ])
        
        prompt = f"""你正在对一本书的多个章节摘要进行高层次汇总，产生"摘要的摘要"。

**输入**：以下是一组章节的详细摘要。

{combined}

**任务要求**：
1. 不要简单复述各章内容，而是提炼出跨章节的整体脉络、核心架构或主线逻辑。
2. 识别章节之间的层次关系：哪些是基础概念、哪些是进阶应用、哪些是具体实现。
3. 如果存在因果关系或依赖链，请明确阐述（例如："因为 A 章定义了协议，B 章才能在此基础上描述传输机制"）。
4. 输出应是一份简洁的、可作为全书概览的高层摘要。

**请输出高层汇总摘要**："""
        
        try:
            return self.llm_client.chat([{"role": "user", "content": prompt}]).strip()
        except Exception as e:
            logger.error(f"汇总失败: {e}")
            return "\n\n".join([s["summary"] for s in chapter_summaries])
    
    def _grouped_recursive_aggregation(self, chapter_summaries: List[Dict]) -> str:
        """分组递归聚合（组间并发）"""
        group_size = 4
        groups = [chapter_summaries[i:i+group_size] for i in range(0, len(chapter_summaries), group_size)]
        
        max_workers = min(4, len(groups))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._direct_summary_aggregation, group): i for i, group in enumerate(groups)}
            group_summaries = []
            for future in futures:
                i = futures[future]
                try:
                    group_summary = future.result()
                    group_summaries.append({
                        "chapter_title": f"组{i+1}",
                        "summary": group_summary,
                        "index": i
                    })
                except Exception as e:
                    logger.error(f"分组聚合失败 | group={i} | error={e}")
                    group_summaries.append({
                        "chapter_title": f"组{i+1}",
                        "summary": "",
                        "index": i
                    })
        
        if len(group_summaries) > 3:
            return self._grouped_recursive_aggregation(group_summaries)
        else:
            return self._direct_summary_aggregation(group_summaries)
    
    def close(self) -> None:
        """关闭资源"""
        if self.llm_client:
            self.llm_client.close()
        if self.embedder:
            self.embedder.close()
        if self.context_manager:
            self.context_manager.close()