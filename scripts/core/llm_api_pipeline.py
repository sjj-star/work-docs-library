"""
LLM API 驱动的处理管道
使用独立的 LLM 对话模型进行高质量文档总结和图像分析
"""
import hashlib
import logging
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
        self.vec = vec or VectorIndex()
        
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
        """处理单个文档"""
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
        
        # 检查是否需要处理
        existing = self.db.get_document_by_path(file_path)
        if not force and existing and existing.file_hash == raw_doc.file_hash and existing.status == "done":
            logger.info("Skip unchanged | file=%s | doc_id=%s", file_path, existing.doc_id)
            return existing.doc_id
        
        # 检查 embedder
        if self.embedder is None:
            logger.warning("No embedder | file=%s will be ingested without embeddings", file_path)
        
        # 存储文档基础信息
        self.db.upsert_document(raw_doc)
        self.db.update_document_status(raw_doc.doc_id, "processing")
        
        # 清理旧数据（如果存在）
        if existing:
            with self.db._connect() as conn:
                old_chunks = conn.execute(
                    "SELECT id FROM chunks WHERE doc_id = ?", (raw_doc.doc_id,)
                ).fetchall()
            if old_chunks:
                self.vec.remove_doc([r["id"] for r in old_chunks])
                self.db.delete_chunks_by_doc(raw_doc.doc_id)
                self.db.delete_chapter_summaries_by_doc(raw_doc.doc_id)
                self.db.delete_concepts_by_doc(raw_doc.doc_id)
        
        # LLM 增强处理
        enhanced_doc = self._llm_enhance_document(raw_doc)
        
        # 处理 chunks
        chunks = enhanced_doc.chunks if enhanced_doc.chunks else []
        if not chunks:
            # 回退：全文作为单个 chunk
            content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            chunks = [Chunk(
                doc_id=enhanced_doc.doc_id,
                chunk_id="full",
                content=content,
                chunk_type="text",
                page_start=1,
                page_end=enhanced_doc.total_pages or 1,
            )]
        
        # 分配章节
        chapters = self.db.get_chapters(enhanced_doc.doc_id) if not auto_chapter else enhanced_doc.chapters
        for ck in chunks:
            ck.doc_id = enhanced_doc.doc_id
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
                    for db_id, emb in zip(batch_ids, embeddings):
                        self.db.update_chunk_embedding(db_id, emb)
                        self.vec.add(db_id, emb)
                        self.db.update_chunk_status(db_id, "embedded")
                except Exception as e:
                    logger.error("Embed error | doc_id=%s | error=%s", enhanced_doc.doc_id, e)
        else:
            logger.warning("No embedder | skipped vector index generation for doc_id=%s", enhanced_doc.doc_id)
        
        # 更新状态
        self.db.update_document_status(enhanced_doc.doc_id, "done")
        logger.info("Ingestion complete | file=%s | doc_id=%s | mode=LLM_API_FLOW", file_path, enhanced_doc.doc_id)
        return enhanced_doc.doc_id
    
    def _llm_enhance_document(self, raw_doc: Document) -> Document:
        """使用 LLM 增强文档内容"""
        if not self.llm_client:
            logger.warning("LLM 客户端不可用，跳过增强处理")
            return raw_doc
        
        logger.info(f"开始 LLM 增强处理 | doc_id={raw_doc.doc_id}")
        
        # 1. 层次化文本总结
        text_summary = self._hierarchical_summarize(raw_doc)
        
        # 2. 图像详细分析（如果有图像）
        image_analyses = self._analyze_document_images(raw_doc)
        
        # 3. 生成章节摘要（如果有章节）
        chapter_summaries = self._generate_chapter_summaries(raw_doc)
        
        # 4. 智能关键词提取
        keywords = self._extract_keywords(raw_doc)
        
        # 5. 更新文档元数据
        raw_doc.metadata.update({
            "llm_summary": text_summary,
            "image_analyses": image_analyses,
            "chapter_summaries": chapter_summaries,
            "keywords": keywords,
            "processing_mode": "LLM_API_FLOW",
            "llm_model": Config.LLM_MODEL,
            "llm_provider": Config.LLM_PROVIDER,
            "thinking_enabled": Config.LLM_THINKING_ENABLED
        })
        
        # 6. 增强 chunk 内容（在原始内容前添加 LLM 总结）
        if text_summary and raw_doc.chunks:
            summary_prefix = f"[LLM 总结] {text_summary}\n\n[原文内容]\n\n"
            for chunk in raw_doc.chunks:
                if chunk.chunk_type == "text":
                    chunk.content = summary_prefix + chunk.content
        
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
        
        # 层次化总结
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"总结块 {i+1}/{len(chunks)}")
            summary = self.llm_client.summarize(chunk)["summary"]
            chunk_summaries.append(f"部分{i+1}: {summary}")
        
        # 验证每个摘要非空
        chunk_summaries = [s for s in chunk_summaries if s]
        
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
        
        for chunk in doc.chunks:
            imgs = chunk.metadata.get("images", []) if chunk.metadata else []
            if not imgs:
                continue
            
            for img in imgs:
                try:
                    prompt = f"""分析这个技术图表/图像，提供：
1. 图表类型和主要内容
2. 关键元素和标注
3. 技术含义和作用
4. 与上下文的可能关联

请用中文回答，保持简洁专业。"""
                    
                    analysis = self.llm_client.vision_describe(img["path"], prompt)
                    
                    image_analyses.append({
                        "image_id": f"img_{image_count + 1}",
                        "path": str(img["path"]),
                        "analysis": analysis,
                        "chunk_id": chunk.chunk_id,
                        "page": chunk.page_start
                    })
                    
                    image_count += 1
                    logger.info(f"图像分析完成 | image_id=img_{image_count} | length={len(analysis)}")
                    
                except Exception as e:
                    logger.error(f"图像分析失败 | path={img.get('path', 'unknown')} | error={e}")
                    image_analyses.append({
                        "image_id": f"img_{image_count + 1}",
                        "path": str(img.get("path", "unknown")),
                        "analysis": f"图像分析失败: {e}",
                        "chunk_id": chunk.chunk_id,
                        "page": chunk.page_start,
                        "error": str(e)
                    })
        
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
        
        for chapter in doc.chapters:
            try:
                # 收集章节相关的文本内容
                chapter_texts = []
                for chunk in doc.chunks:
                    if chunk.chapter_title == chapter.title and chunk.chunk_type == "text":
                        chapter_texts.append(chunk.content)
                
                if not chapter_texts:
                    continue
                
                # 章节内容总结
                chapter_content = "\n\n".join(chapter_texts)
                chapter_summary = self.llm_client.summarize(chapter_content)["summary"]
                
                chapter_summaries.append({
                    "chapter_title": chapter.title,
                    "start_page": chapter.start_page,
                    "end_page": chapter.end_page,
                    "summary": chapter_summary,
                    "text_length": len(chapter_content)
                })
                
                logger.info(f"章节摘要生成完成 | chapter={chapter.title} | length={len(chapter_summary)}")
                
            except Exception as e:
                logger.error(f"章节摘要生成失败 | chapter={chapter.title} | error={e}")
                chapter_summaries.append({
                    "chapter_title": chapter.title,
                    "start_page": chapter.start_page,
                    "end_page": chapter.end_page,
                    "summary": f"章节摘要生成失败: {e}",
                    "error": str(e)
                })
        
        return chapter_summaries
    
    def _extract_keywords(self, doc: Document) -> List[str]:
        """智能关键词提取"""
        if not self.llm_client:
            return []
        
        try:
            # 收集文档标题和章节标题作为关键词基础
            keyword_sources = [doc.title] if doc.title else []
            for chapter in doc.chapters:
                keyword_sources.append(chapter.title)
            
            # 使用 LLM 提取关键词
            keyword_text = "\n".join(keyword_sources)
            if len(keyword_text) < 50:
                # 如果内容太少，添加一些文本内容
                for chunk in doc.chunks[:3]:  # 只取前3个chunk
                    if chunk.chunk_type == "text":
                        keyword_text += "\n" + chunk.content[:500]
                        if len(keyword_text) > 1000:
                            break
            
            keyword_prompt = f"""从技术文档中提取5-8个最重要的关键词，用逗号分隔：

{keyword_text[:2000]}"""
            
            keyword_response = self.llm_client.chat([{"role": "user", "content": keyword_prompt}])
            
            # 解析关键词
            keywords = []
            for line in keyword_response.splitlines():
                if "," in line:
                    keywords = [k.strip() for k in line.split(",") if k.strip()]
                    break
            
            return keywords[:10]  # 限制数量
            
        except Exception as e:
            logger.error(f"关键词提取失败 | error={e}")
            return []
    
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
                    "chapter": chapter.title,
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
        """生成章节摘要"""
        if not self.llm_client:
            return ""
        
        prompt = f"""请对第 {chapter_index + 1}/{total_chapters} 章进行总结：

章节标题：{chapter_title}

上下文：
{context}

总结："""
        
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
        """直接汇总"""
        if not self.llm_client:
            return ""
        
        combined = "\n\n".join([
            f"第{s['index'] + 1}章 {s['chapter']}:\n{s['summary']}"
            for s in chapter_summaries
        ])
        
        prompt = f"""对以下章节摘要进行汇总总结：

{combined}

整体总结："""
        
        try:
            return self.llm_client.chat([{"role": "user", "content": prompt}]).strip()
        except Exception as e:
            logger.error(f"汇总失败: {e}")
            return "\n\n".join([s["summary"] for s in chapter_summaries])
    
    def _grouped_recursive_aggregation(self, chapter_summaries: List[Dict]) -> str:
        """分组递归聚合"""
        group_size = 4
        groups = [chapter_summaries[i:i+group_size] for i in range(0, len(chapter_summaries), group_size)]
        
        group_summaries = []
        for i, group in enumerate(groups):
            group_summary = self._direct_summary_aggregation(group)
            group_summaries.append({
                "chapter": f"组{i+1}",
                "summary": group_summary,
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