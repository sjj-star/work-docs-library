"""
兼容性管道 - 保持现有行为，使用新的独立 Embedding 客户端
用于 Agent Skill Flow 模式
"""
import hashlib
import logging
from pathlib import Path
from typing import List, Optional
import shutil

from .config import Config
from .db import KnowledgeDB
from .vector_index import VectorIndex
from .models import Chunk, Document
from .embedding_client import EmbeddingClient  # 使用新的独立客户端
from parsers.pdf_parser import PDFParser
from parsers.office_parser import OfficeParser

logger = logging.getLogger(__name__)


class CompatibilityIngestionPipeline:
    """兼容性管道 - 保持现有行为，仅使用 Embedding 向量化"""
    
    def __init__(self, db: Optional[KnowledgeDB] = None, vec: Optional[VectorIndex] = None) -> None:
        self.db = db or KnowledgeDB()
        self.vec = vec or VectorIndex(dim=Config.EMBEDDING_DIMENSION)
        self.embedder: Optional[EmbeddingClient] = None
        try:
            # 使用新的独立 Embedding 客户端
            self.embedder = EmbeddingClient()
            logger.info(f"Embedding 客户端已初始化 - 模型: {Config.EMBEDDING_MODEL} ({Config.EMBEDDING_PROVIDER})")
        except RuntimeError as e:
            logger.warning(f"Embedding 客户端初始化失败: {e}")
            self.embedder = None
        
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
        """处理单个文档 - 保持现有逻辑"""
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
            logger.info("Dry-run | file=%s | pages=%s | chapters=%s", 
                       file_path, raw_doc.total_pages, len(raw_doc.chapters))
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
        
        # 处理 chunks
        chunks = raw_doc.chunks if raw_doc.chunks else []
        if not chunks:
            # 回退：全文作为单个 chunk
            content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            chunks = [Chunk(
                doc_id=raw_doc.doc_id,
                chunk_id="full",
                content=content,
                chunk_type="text",
                page_start=1,
                page_end=raw_doc.total_pages or 1,
            )]
        
        # 自动 Vision 描述（如果启用）
        if Config.AUTO_VISION:
            self._auto_vision_describe(chunks)
        
        # 分配章节
        chapters = self.db.get_chapters(raw_doc.doc_id) if not auto_chapter else raw_doc.chapters
        for ck in chunks:
            ck.doc_id = raw_doc.doc_id
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
                    logger.error("Embed error | doc_id=%s | error=%s", raw_doc.doc_id, e)
        else:
            logger.warning("No embedder | skipped vector index generation for doc_id=%s", raw_doc.doc_id)
        
        # 更新状态
        self.db.update_document_status(raw_doc.doc_id, "done")
        logger.info("Ingestion complete | file=%s | doc_id=%s | mode=AGENT_SKILL_FLOW", file_path, raw_doc.doc_id)
        return raw_doc.doc_id
    
    def _auto_vision_describe(self, chunks: List[Chunk]) -> None:
        """自动 Vision 描述（保持现有逻辑）"""
        if not Config.AUTO_VISION:
            return
        
        # 使用旧的 LLM 客户端进行 vision 描述（保持兼容性）
        try:
            from .llm_client import ChatClient
            vision_client = ChatClient()
            
            for ck in chunks:
                imgs = ck.metadata.get("images", []) if ck.metadata else []
                if not imgs:
                    continue
                
                descriptions = []
                for img in imgs:
                    desc = vision_client.vision_describe(
                        img["path"],
                        prompt="Describe this technical diagram in detail. If it is a block diagram, architecture diagram, or timing chart, explain the components and relationships.",
                    )
                    descriptions.append(f"- Image: {desc}")
                    img["vision_desc"] = desc
                
                # 替换图像引用块
                marker = "\n\n[IMAGES ON THIS PAGE]"
                idx = ck.content.find(marker)
                if idx != -1:
                    base_text = ck.content[:idx]
                else:
                    base_text = ck.content
                ck.content = base_text + "\n\n[IMAGES ON THIS PAGE]\n" + "\n".join(descriptions)
            
            vision_client.close()
            
        except Exception as e:
            logger.error("Vision error | failed to auto-describe images: %s", e)
    
    def close(self) -> None:
        """关闭资源"""
        if self.embedder:
            self.embedder.close()