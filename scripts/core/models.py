from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class Chapter:
    title: str
    start_page: int
    end_page: int
    level: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    content: str
    chunk_type: str  # text, table, image_desc, summary
    page_start: int = 0
    page_end: int = 0
    chapter_title: str = ""
    keywords: List[str] = field(default_factory=list)
    summary: str = ""
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Document:
    doc_id: str
    title: str
    source_path: str
    file_type: str
    total_pages: int = 0
    chunks: List[Chunk] = field(default_factory=list)
    chapters: List[Chapter] = field(default_factory=list)
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    file_hash: str = ""
    status: str = "pending"  # pending, processing, done, failed

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["chapters"] = [c.to_dict() for c in self.chapters]
        d["chunks"] = [c.to_dict() for c in self.chunks]
        return d
