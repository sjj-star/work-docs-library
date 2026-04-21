import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import shutil

from .config import Config
from .db import KnowledgeDB
from .llm_chat_client import LLMChatClient as ChatClient
from .embedding_client import EmbeddingClient
from .models import Chunk, Document
from .vector_index import VectorIndex
from parsers.pdf_parser import PDFParser
from parsers.office_parser import OfficeParser

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, db: Optional[KnowledgeDB] = None, vec: Optional[VectorIndex] = None) -> None:
        self.db = db or KnowledgeDB()
        self._vec_override = vec  # 用户提供的 VectorIndex 实例
        self._vec: Optional[VectorIndex] = None  # 延迟初始化
        self.embedder: Optional[EmbeddingClient] = None
        try:
            # 使用新的独立 Embedding 客户端
            self.embedder = EmbeddingClient()
        except RuntimeError:
            self.embedder = None
        
        self.parsers = {
            ".pdf": PDFParser(),
            ".docx": OfficeParser(),
            ".xlsx": OfficeParser(),
        }
    
    @property
    def vec(self) -> VectorIndex:
        """延迟初始化的 VectorIndex，在首次访问时创建"""
        if self._vec is None:
            if self._vec_override is not None:
                self._vec = self._vec_override
            elif self.embedder is not None and self.embedder._dim_validated:
                # 使用 EmbeddingClient 探测到的实际维度
                self._vec = VectorIndex(dim=self.embedder.get_embedding_dimension())
            else:
                # 无法探测维度时的回退：使用配置中的维度
                self._vec = VectorIndex(dim=Config.EMBEDDING_DIMENSION)
        return self._vec
    
    @vec.setter
    def vec(self, value: VectorIndex) -> None:
        self._vec = value

    def scan(self, path: str) -> List[str]:
        p = Path(path)
        files = []
        if p.is_file():
            files.append(str(p.resolve()))
        elif p.is_dir():
            for ext in self.parsers.keys():
                files.extend(str(f) for f in p.rglob(f"*{ext}"))
        return sorted(set(files))

    def ingest(self, path: str, dry_run: bool = False, auto_chapter: bool = False) -> List[str]:
        files = self.scan(path)
        ingested: List[str] = []
        for f in files:
            doc_id = self._process_one(f, dry_run=dry_run, auto_chapter=auto_chapter)
            if doc_id:
                ingested.append(doc_id)
        return ingested

    def _process_one(self, file_path: str, dry_run: bool, auto_chapter: bool, force: bool = False) -> Optional[str]:
        suffix = Path(file_path).suffix.lower()
        parser = self.parsers.get(suffix)
        if not parser:
            return None

        # When forcing re-process, delete any old images first so the parser
        # can write fresh images safely.
        if force:
            file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
            old_img_dir = Config.DB_PATH.parent / "images" / file_hash
            if old_img_dir.exists():
                shutil.rmtree(old_img_dir)

        extract_images = True
        if suffix == ".pdf":
            raw_doc = parser.parse(file_path, extract_images=extract_images)
        else:
            raw_doc = parser.parse(file_path)

        if dry_run:
            logger.info("Dry-run | file=%s | pages=%s | chapters=%s", file_path, raw_doc.total_pages, len(raw_doc.chapters))
            return raw_doc.doc_id

        existing = self.db.get_document_by_path(file_path)
        if not force and existing and existing.file_hash == raw_doc.file_hash and existing.status == "done":
            logger.info("Skip unchanged | file=%s | doc_id=%s", file_path, existing.doc_id)
            return existing.doc_id

        if self.embedder is None:
            logger.warning("No embedder | file=%s will be ingested without embeddings", file_path)

        self.db.upsert_document(raw_doc)
        self.db.update_document_status(raw_doc.doc_id, "processing")

        if existing:
            with self.db._connect() as conn:
                old_chunks = conn.execute(
                    "SELECT id FROM chunks WHERE doc_id = ?", (raw_doc.doc_id,)
                ).fetchall()
            self.vec.remove_doc([r["id"] for r in old_chunks])
            self.db.delete_chunks_by_doc(raw_doc.doc_id)
            self.db.delete_chapter_summaries_by_doc(raw_doc.doc_id)
            self.db.delete_concepts_by_doc(raw_doc.doc_id)

        chunks = raw_doc.chunks if raw_doc.chunks else []
        if not chunks:
            content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            chunks = [Chunk(
                doc_id=raw_doc.doc_id,
                chunk_id="full",
                content=content,
                chunk_type="text",
                page_start=1,
                page_end=raw_doc.total_pages or 1,
            )]

        # Scenario A: script-level automatic Vision description for images
        if Config.AUTO_VISION:
            vision_client = None
            try:
                vision_client = ChatClient()
                for ck in chunks:
                    imgs = ck.metadata.get("images", [])
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
                    # Replace image reference block with Vision descriptions
                    marker = "\n\n[IMAGES ON THIS PAGE]"
                    idx = ck.content.find(marker)
                    if idx != -1:
                        base_text = ck.content[:idx]
                    else:
                        base_text = ck.content
                    ck.content = base_text + "\n\n[IMAGES ON THIS PAGE]\n" + "\n".join(descriptions)
            except Exception as e:
                logger.error("Vision error | failed to auto-describe images: %s", e)
            finally:
                if vision_client is not None:
                    vision_client.close()

        chapters = self.db.get_chapters(raw_doc.doc_id) if not auto_chapter else raw_doc.chapters
        for ck in chunks:
            ck.doc_id = raw_doc.doc_id
            for ch in chapters:
                if ch.start_page <= ck.page_start <= ch.end_page:
                    ck.chapter_title = ch.title
                    break

        chunk_db_ids = []
        for ck in chunks:
            db_id = self.db.insert_chunk(ck)
            chunk_db_ids.append(db_id)

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

        self.db.update_document_status(raw_doc.doc_id, "done")
        logger.info("Ingestion complete | file=%s | doc_id=%s", file_path, raw_doc.doc_id)
        return raw_doc.doc_id

    def close(self) -> None:
        if self.embedder:
            self.embedder.close()
