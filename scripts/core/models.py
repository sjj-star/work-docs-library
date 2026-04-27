"""models 模块."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from .enums import ChunkStatus, ChunkType, DocumentStatus


@dataclass
class Chapter:
    """Chapter 类."""

    title: str
    start_page: int
    end_page: int
    level: int = 1

    def to_dict(self) -> dict[str, Any]:
        """to_dict 函数."""
        return asdict(self)


@dataclass
class Chunk:
    """章节内容块（Chunk）.

    Chunk 是按文档章节拆分的内容单元。
    """

    doc_id: str
    chunk_id: str
    content: str
    chunk_type: ChunkType | str = ChunkType.TEXT
    chapter_title: str = ""
    keywords: list[str] = field(default_factory=list)
    summary: str = ""
    status: ChunkStatus | str = ChunkStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    id: int = 0  # database primary key

    def to_dict(self) -> dict[str, Any]:
        """to_dict 函数."""
        return asdict(self)


@dataclass
class Document:
    """Document 类 — 文档元数据（不含 chunks，chunks 独立存储于 SQLite）."""

    doc_id: str
    title: str
    source_path: str
    file_type: str
    total_pages: int = 0
    chapters: list[Chapter] = field(default_factory=list)
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    file_hash: str = ""
    status: DocumentStatus | str = DocumentStatus.PENDING

    def to_dict(self) -> dict[str, Any]:
        """to_dict 函数."""
        d = asdict(self)
        d["chapters"] = [c.to_dict() for c in self.chapters]
        return d
