"""Embedding 客户端（BigModel / OpenAI-compatible）.

基于统一 APIClient 构建，支持：
- 按官方错误码分类重试
- 超长文本自动拆分后 embed 并平均
- 单条失败隔离（不影响其他 texts）
"""

from __future__ import annotations

import numpy as np

from .api_client import APIClient, APIError, BigModelProvider, ContentTooLargeError
from .config import Config


class EmbeddingClient:
    """向外部 Embedding API 发送文本并获取向量."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """初始化 EmbeddingClient.

        Args:
            api_key: API Key，默认使用 Config.EMBEDDING_API_KEY
            base_url: 基础 URL，默认使用 Config.EMBEDDING_BASE_URL
            model: 模型名，默认使用 Config.EMBEDDING_MODEL
            timeout: 请求超时（秒）
        """
        self.model = model or Config.EMBEDDING_MODEL
        provider = BigModelProvider(api_key=api_key, base_url=base_url)
        self._client = APIClient(provider, timeout=timeout)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """返回向量维度.

        必须在至少成功调用一次 embed() 之后才能获取准确值。
        """
        if self._dimension is None:
            raise RuntimeError(
                "Embedding dimension not yet detected. Call embed() at least once first."
            )
        return self._dimension

    def _split_text(self, text: str, max_chars: int) -> list[str]:
        """按段落、行、句子边界拆分超长文本."""
        max_chars = max(2, max_chars)
        if len(text) <= max_chars:
            return [text]

        # 优先按双换行（段落）拆分
        chunks: list[str] = []
        for paragraph in text.split("\n\n"):
            if len(paragraph) <= max_chars:
                chunks.append(paragraph)
                continue
            # 段落仍超长，按单换行或句子边界拆分
            for line in paragraph.split("\n"):
                if len(line) <= max_chars:
                    chunks.append(line)
                    continue
                # 行仍超长，按句子边界硬拆
                start = 0
                while start < len(line):
                    end = start + max_chars
                    chunks.append(line[start:end])
                    start = end

        # 合并过小的碎片，减少 API 调用次数
        merged: list[str] = []
        current = ""
        for piece in chunks:
            if len(current) + len(piece) + 2 <= max_chars:
                current = f"{current}\n\n{piece}" if current else piece
            else:
                if current:
                    merged.append(current)
                current = piece
        if current:
            merged.append(current)
        return merged if merged else [text]

    def _call_embedding_api(self, inputs: list[str]) -> list[list[float]]:
        """调用 Embedding API，返回与 inputs 顺序一致的向量列表."""
        if not inputs:
            return []

        response = self._client.post(
            Config.EMBEDDING_ENDPOINT,
            json={"model": self.model, "input": inputs},
        )
        data = response.json().get("data", [])
        embeddings = [item.get("embedding", []) for item in data]
        if len(embeddings) != len(inputs):
            raise APIError(
                f"Embedding response length mismatch: expected {len(inputs)}, "
                f"got {len(embeddings)}",
                status_code=response.status_code,
            )

        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])
        return embeddings

    def _embed_chunks(self, chunks: list[str]) -> list[float] | None:
        """对拆分后的 chunks 分别 embed 并取平均."""
        try:
            embeddings = self._call_embedding_api(chunks)
        except ContentTooLargeError:
            # 拆分后仍然超长（极少见），继续二次拆分
            if Config.EMBED_SPLIT_OVERLONG and any(
                len(c) > Config.EMBED_MAX_CHARS_PER_TEXT for c in chunks
            ):
                smaller: list[str] = []
                half = max(1, Config.EMBED_MAX_CHARS_PER_TEXT // 2)
                for c in chunks:
                    if len(c) > Config.EMBED_MAX_CHARS_PER_TEXT:
                        for i in range(0, len(c), half):
                            smaller.append(c[i : i + half])
                    else:
                        smaller.append(c)
                return self._embed_chunks(smaller)
            raise

        if not embeddings:
            return None
        arr = np.array(embeddings, dtype=np.float32)
        avg = arr.mean(axis=0)
        # L2 归一化，保持与原始 embedding 相似的度量方式
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        return avg.tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量获取文本向量.

        超长文本会被拆分后 embed 再平均；任何单条失败都会抛出 APIError，
        由调用方决定是否重试或降级，避免零向量污染索引。
        """
        results: list[list[float]] = []
        for text in texts:
            text = str(text)
            try:
                if Config.EMBED_SPLIT_OVERLONG and len(text) > Config.EMBED_MAX_CHARS_PER_TEXT:
                    chunks = self._split_text(text, Config.EMBED_MAX_CHARS_PER_TEXT)
                    embedding = self._embed_chunks(chunks)
                else:
                    embedding = self._call_embedding_api([text])[0]

                if embedding is None:
                    raise APIError("Empty embedding returned")
                results.append(embedding)
            except APIError:
                raise
            except Exception as exc:
                raise APIError(str(exc)) from exc
        return results

    def embed_single(self, text: str) -> list[float]:
        """获取单条文本向量."""
        return self.embed([text])[0]

    def close(self) -> None:
        """关闭底层 HTTP 客户端."""
        self._client.close()
