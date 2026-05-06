"""领域枚举定义."""

from enum import StrEnum


class ChunkStatus(StrEnum):
    """Chunk 处理状态."""

    PENDING = "pending"
    EMBEDDED = "embedded"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


class DocumentStatus(StrEnum):
    """Document 处理状态."""

    PENDING = "pending"
    PROCESSING = "processing"
    BATCH_SUBMITTED = "batch_submitted"
    DONE = "done"
    FAILED = "failed"


class ChunkType(StrEnum):
    """Chunk 内容类型."""

    TEXT = "text"
    TABLE = "table"
    IMAGE_DESC = "image_desc"
    SUMMARY = "summary"
