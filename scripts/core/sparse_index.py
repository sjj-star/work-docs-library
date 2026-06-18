"""Sparse (BM25) index over content_blocks."""

import logging
import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .db import KnowledgeDB

logger = logging.getLogger(__name__)


class BM25SparseIndex:
    """基于 content_blocks 的内存 BM25 索引."""

    def __init__(self, db: KnowledgeDB | None = None) -> None:
        """初始化 BM25 索引."""
        self.db = db or KnowledgeDB()
        self._corpus: list[tuple[int, str]] = []
        self._index: BM25Okapi | None = None
        self._build()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """分词：英文/数字标识符 + CJK 连续字符 2-gram + 其他符号."""
        text = text.lower()
        tokens: list[str] = []
        # 1. English words, numbers, identifiers
        tokens.extend(re.findall(r"[a-zA-Z0-9_]+", text))
        # 2. CJK runs: emit single chars and 2-grams within each contiguous run
        for run in re.findall(r"[\u4e00-\u9fff]+", text):
            tokens.extend(run)
            for i in range(len(run) - 1):
                tokens.append(run[i] + run[i + 1])
        # 3. Other symbols
        tokens.extend(re.findall(r"[^\s\w\u4e00-\u9fff]", text))
        return tokens

    @classmethod
    def from_blocks(cls, blocks: list[dict]) -> "BM25SparseIndex":
        """Build a BM25 index from an existing list of block dicts."""
        instance = cls.__new__(cls)
        instance.db = None
        tokenized: list[list[str]] = []
        instance._corpus = []
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError(f"Block must be a dict, got {type(block)}")
            block_id = block.get("id")
            content = block.get("content")
            if block_id is None:
                raise ValueError("Block is missing required field 'id'")
            if not isinstance(content, str):
                raise ValueError(f"Block {block_id} has invalid content type: {type(content)}")
            tokens = cls._tokenize(content)
            tokenized.append(tokens)
            instance._corpus.append((block_id, content))
        if tokenized:
            instance._index = BM25Okapi(tokenized)
        else:
            instance._index = None
        return instance

    def _build(self) -> None:
        """从所有文档的 content_blocks 构建 BM25 索引."""
        docs = self.db.list_documents()
        tokenized: list[list[str]] = []
        self._corpus = []
        for doc in docs:
            for block in self.db.query_blocks_by_doc(doc.doc_id):
                tokens = self._tokenize(block["content"])
                tokenized.append(tokens)
                self._corpus.append((block["id"], block["content"]))
        if tokenized:
            self._index = BM25Okapi(tokenized)
            logger.info(f"BM25 索引构建完成 | blocks={len(tokenized)}")
        else:
            logger.info("BM25 索引为空 | 无文档")

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """搜索 BM25 索引，返回 (block_db_id, score) 列表."""
        if self._index is None or not self._corpus:
            return []
        scores = self._index.get_scores(self._tokenize(query))
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(self._corpus[i][0]), float(scores[i])) for i in top_idx if scores[i] > 0]

    def index_info(self) -> dict[str, Any]:
        """返回索引元信息."""
        return {
            "index_type": "BM25Okapi",
            "num_blocks": len(self._corpus),
            "built": self._index is not None,
        }
